"""PROMPTPEEK token extraction (Section IV-A).

One probe surrounds a batch of candidate next-tokens with identical dummy
requests:

    [pre dummies] [candidates] [post dummies]

Pre-dummies are admitted first, so the dummy prefix is stored. LPM then
sorts whatever is still waiting. The attacker never sees the cache. The
only observation is the order in which their own request ids complete.

Under unmodified LPM the paper's pattern is:

* hit:  exactly one candidate finishes before every post-dummy
         (it ties the dummy's match length and keeps its arrival slot,
         which is ahead of the post-dummies; the misses fall behind);
* miss: every candidate finishes after every post-dummy;
* anything else (all candidates ahead of the post-dummies, or several
  of them ahead) is not a unique hit. FCFS and LPM-with-margin produce
  this ambiguous order, which is how those defenses blunt the attack.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from promptpeek.scheduler import ServingEngine


@dataclass
class ProbeResult:
    hit_token: int | None
    promoted: list[int]
    order: list[str]
    batches: list[list[str]]
    batch_match_lens: list[list[int]]
    batch_priorities: list[list[int]]
    candidate_tokens: list[int]
    n_requests: int
    n_candidate_requests: int
    n_dummy_requests: int


@dataclass
class RecoveryStep:
    token: int | None
    rounds: int
    requests: int
    probes: list[ProbeResult] = field(default_factory=list)


@dataclass
class Recovery:
    tokens: list[int]
    steps: list[RecoveryStep]
    requests: int
    stopped_reason: str

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)


def probe(
    engine: ServingEngine,
    tenant: str,
    prefix: list[int],
    candidates: list[int],
    dummy: int,
    n_pre: int | None = None,
    n_post: int | None = None,
    tag: str = "p",
) -> ProbeResult:
    """Send one candidate batch and read the completion-order side channel.

    ``n_pre`` defaults to ``max_batch`` so the running batch is filled with
    dummies before any candidate is enqueued (Figure 6). ``n_post`` defaults
    to the same value so a miss cannot share a batch with the post-dummies
    in a way that hides the order.
    """
    if dummy in candidates:
        raise ValueError("dummy token must not also be a candidate")
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidates must be unique")
    if len(candidates) < 2:
        raise ValueError("need at least two candidates so FCFS is ambiguous")

    n_pre = engine.max_batch if n_pre is None else n_pre
    n_post = engine.max_batch if n_post is None else n_post
    if n_pre < engine.max_batch or n_post < 1:
        raise ValueError("n_pre must be >= max_batch so candidates wait")

    for i in range(n_pre):
        engine.enqueue(tenant, prefix + [dummy], f"{tag}-pre{i}")
    # Admit the first wave by arrival. This is the batch that is already
    # running in Figure 6; LPM has not sorted the candidates yet.
    engine.fill_running_if_idle(policy_override="fcfs")

    for i, tok in enumerate(candidates):
        engine.enqueue(tenant, prefix + [tok], f"{tag}-c{i}")
    for i in range(n_post):
        engine.enqueue(tenant, prefix + [dummy], f"{tag}-post{i}")

    engine.complete_running()
    engine.drain()

    order = [req.req_id for req in engine.completed]
    batches = [batch.req_ids for batch in engine.batches]
    batch_match_lens = [list(batch.match_lens) for batch in engine.batches]
    batch_priorities = [list(batch.priorities) for batch in engine.batches]
    # Reset the observation buffers but keep the KV cache. The next probe
    # must still see the victim prefix (and the dummies just stored).
    engine.completed.clear()
    engine.batches.clear()

    cand_pos = {f"{tag}-c{i}": i for i in range(len(candidates))}
    post_positions = [order.index(f"{tag}-post{i}") for i in range(n_post)]
    first_post = min(post_positions)
    promoted_idx = [cand_pos[req_id] for req_id in order if req_id in cand_pos and order.index(req_id) < first_post]
    promoted = [candidates[i] for i in promoted_idx]
    hit = promoted[0] if len(promoted) == 1 else None
    return ProbeResult(
        hit_token=hit,
        promoted=promoted,
        order=order,
        batches=batches,
        batch_match_lens=batch_match_lens,
        batch_priorities=batch_priorities,
        candidate_tokens=list(candidates),
        n_requests=n_pre + n_post + len(candidates),
        n_candidate_requests=len(candidates),
        n_dummy_requests=n_pre + n_post,
    )


def recover(
    engine: ServingEngine,
    tenant: str,
    known_prefix: list[int],
    propose,
    max_new_tokens: int,
    k: int = 8,
    max_rounds: int = 4,
) -> Recovery:
    """Walk forward one token at a time (Section IV-B).

    ``propose(prefix, k, exclude) -> (candidates, dummy)`` is the local model
    from Section IV-A1. It must not return a token in ``exclude`` (those were
    already stored by an earlier miss, and a second probe would look like a
    hit against the attacker's own cache).
    """
    prefix = list(known_prefix)
    steps: list[RecoveryStep] = []
    total_requests = 0
    reason = "max_new_tokens"

    for step_i in range(max_new_tokens):
        exclude: set[int] = set()
        step = RecoveryStep(token=None, rounds=0, requests=0)
        found: int | None = None
        for round_i in range(max_rounds):
            candidates, dummy = propose(prefix, k, exclude)
            if len(candidates) < 2:
                # A dry proposer after at least one miss is the paper's
                # partial failure (the secret was never a unique hit).
                # Exhaustion with no observation is a separate, earlier stop.
                reason = "no_unique_hit" if step.rounds else "proposer_exhausted"
                break
            overlap = [tok for tok in candidates if tok in exclude]
            if overlap or dummy in exclude or dummy in candidates:
                raise ValueError(
                    "proposer reused an excluded token or used the dummy as a candidate"
                )
            exclude.update(candidates)
            exclude.add(dummy)
            result = probe(
                engine,
                tenant,
                prefix,
                candidates,
                dummy,
                tag=f"s{step_i}r{round_i}",
            )
            step.rounds += 1
            step.requests += result.n_requests
            step.probes.append(result)
            if result.hit_token is not None:
                found = result.hit_token
                break
        total_requests += step.requests
        step.token = found
        steps.append(step)
        if found is None:
            if reason == "max_new_tokens":
                reason = "no_unique_hit"
            break
        prefix.append(found)

    if reason == "max_new_tokens":
        reason = "recovered"
    return Recovery(
        tokens=prefix[len(known_prefix) :],
        steps=steps,
        requests=total_requests,
        stopped_reason=reason,
    )


def plant_victim(engine: ServingEngine, tenant: str, tokens: list[int], req_id: str = "victim") -> None:
    """Serve the victim once and leave the prefix stored.

    The request is admitted by arrival and then completes. Its KV stays
    until eviction, which is condition 1 of Section VII.
    """
    engine.enqueue(tenant, tokens, req_id)
    engine.fill_running_if_idle(policy_override="fcfs")
    engine.complete_running()
    engine.completed.clear()
    engine.batches.clear()
