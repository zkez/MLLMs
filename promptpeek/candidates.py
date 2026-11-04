"""Candidate generators for PROMPTPEEK (Section IV-A1).

The scheduler never reads a KV tensor. It only needs the next-token ids
the attacker chooses to send. Two proposers are deterministic and are what
the unit tests use:

* ``OracleProposer`` knows the true continuation and emits it first, when
  that id is still inside the allowed alphabet and has not been excluded.
* ``AlphabetProposer`` does not know the secret. It walks a small alphabet
  in order. This is the brute-force proposer behind scenario 2 (a known
  template, a short secret such as a gender, an age, or account digits).

``QwenLogitProposer`` is the paper's local model: same tokenizer family as
the target, top-k next tokens from the logits, plus one least-likely id
used as the dummy. On this machine that model is Qwen3-4B-Base, GGUF,
CPU only. It proposes candidates. It is not the multi-tenant server.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OracleProposer:
    """Emit the true next id first, then the rest of ``alphabet``.

    ``target`` is the full victim sequence, including the known prefix.
    The next id is ``target[len(prefix)]``. If that id is not in
    ``alphabet`` (a rare token the local model would not propose) it is
    simply omitted, and recovery stops. ``exclude`` drops ids whose KV
    was stored by an earlier miss in the same step.
    """

    target: list[int]
    alphabet: list[int]
    dummy_pool: list[int]
    calls: int = 0

    def __call__(
        self,
        prefix: list[int],
        k: int,
        exclude: set[int],
    ) -> tuple[list[int], int]:
        self.calls += 1
        banned = set(exclude)
        nxt = self.target[len(prefix)] if len(prefix) < len(self.target) else None
        chosen: list[int] = []
        if nxt is not None and nxt in self.alphabet and nxt not in banned:
            chosen.append(nxt)
        for tok in self.alphabet:
            if len(chosen) >= k:
                break
            if tok in banned or tok in chosen:
                continue
            chosen.append(tok)
        dummy = _pick_dummy(self.dummy_pool, banned, chosen)
        return chosen, dummy


@dataclass
class AlphabetProposer:
    """Brute-force proposer: the next unseen alphabet ids, in order."""

    alphabet: list[int]
    dummy_pool: list[int]
    calls: int = 0
    history: list[list[int]] = field(default_factory=list)

    def __call__(
        self,
        prefix: list[int],
        k: int,
        exclude: set[int],
    ) -> tuple[list[int], int]:
        del prefix
        self.calls += 1
        banned = set(exclude)
        chosen = [tok for tok in self.alphabet if tok not in banned][:k]
        self.history.append(list(chosen))
        dummy = _pick_dummy(self.dummy_pool, banned, chosen)
        return chosen, dummy


def _pick_dummy(pool: list[int], banned: set[int], chosen: list[int]) -> int:
    chosen_set = set(chosen)
    for tok in pool:
        if tok not in banned and tok not in chosen_set:
            return tok
    tok = 1_000_000
    while tok in banned or tok in chosen_set:
        tok += 1
    return tok


class QwenLogitProposer:
    """Top-k / least-likely proposer backed by a llama.cpp model.

    The model object is anything with ``reset``, ``eval``, ``n_tokens``
    and ``scores`` (llama-cpp-python's ``Llama``). Importing llama-cpp
    is delayed so the scheduler tests do not need it.
    """

    def __init__(self, llm):
        self.llm = llm
        self.calls = 0

    def __call__(
        self,
        prefix: list[int],
        k: int,
        exclude: set[int],
    ) -> tuple[list[int], int]:
        import numpy as np

        self.calls += 1
        banned = set(exclude)
        logits = self._logits(prefix)
        work = np.array(logits, dtype=np.float32, copy=True)
        vocab = int(work.shape[0])
        for tok in banned:
            if 0 <= tok < vocab:
                work[tok] = -np.inf
        finite = np.isfinite(work)
        n_finite = int(finite.sum())
        if n_finite <= 0 or k <= 0:
            return [], _pick_dummy([], banned, [])
        take = min(k, n_finite)
        # argpartition is enough; the returned order is re-sorted by logit.
        part = np.argpartition(-work, take - 1)[:take]
        order = part[np.argsort(-work[part])]
        chosen = [int(i) for i in order if np.isfinite(work[int(i)])]
        dummy = _least_likely(logits, banned, set(chosen))
        return chosen, dummy

    def _logits(self, prefix: list[int]):
        if not prefix:
            raise ValueError("Qwen proposer needs a non-empty prefix")
        self.llm.reset()
        self.llm.eval(list(prefix))
        row = self.llm.n_tokens - 1
        return self.llm.scores[row]

    def topk_ids(self, prefix: list[int], k: int) -> list[int]:
        """Top-k with nothing excluded. Does not count as an attack round."""
        import numpy as np

        logits = np.asarray(self._logits(prefix))
        take = min(k, int(logits.shape[0]))
        part = np.argpartition(-logits, take - 1)[:take]
        order = part[np.argsort(-logits[part])]
        return [int(i) for i in order]


def _least_likely(logits, banned: set[int], chosen: set[int]) -> int:
    import numpy as np

    logits = np.asarray(logits)
    # Look at a slice of the smallest logits. A full sort of a 150k vocab
    # is unnecessary for picking one dummy id.
    window = min(256, int(logits.shape[0]))
    part = np.argpartition(logits, window - 1)[:window]
    order = part[np.argsort(logits[part])]
    for idx in order:
        tok = int(idx)
        if tok not in banned and tok not in chosen:
            return tok
    return _pick_dummy([], banned, list(chosen))


def load_qwen(model_path: str, n_ctx: int = 64, n_threads: int = 4):
    """Load a GGUF model on CPU. Raises whatever llama.cpp raises.

    ``n_gpu_layers=0`` keeps every tensor on CPU. ``logits_all=True`` is
    required: this llama-cpp-python build only writes ``scores`` when
    that flag is set.
    """
    from llama_cpp import Llama

    return Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_batch=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=0,
        logits_all=True,
        verbose=False,
        use_mmap=True,
        use_mlock=False,
    )


def make_oracle(known_len: int, secret: list[int], distractors: list[int], dummy: int = 10_000):
    """Proposer that puts the true next secret id first.

    Used by the unit tests and by the talk when we want to show the
    channel itself, separate from whether a language model guessed well.
    """

    def propose(prefix: list[int], k: int, exclude: set[int]) -> tuple[list[int], int]:
        idx = len(prefix) - known_len
        pool: list[int] = []
        if 0 <= idx < len(secret):
            pool.append(secret[idx])
        pool.extend(distractors)
        banned = set(exclude)
        chosen: list[int] = []
        seen: set[int] = set()
        for tok in pool:
            if tok in banned or tok in seen:
                continue
            chosen.append(tok)
            seen.add(tok)
            if len(chosen) == k:
                break
        return chosen, _pick_dummy([dummy], banned, chosen)

    return propose


def make_alphabet(known_len: int, alphabet: list[int], dummy: int = 10_001):
    """Brute-force a small token alphabet. This is scenario 2 without a model.

    ``known_len`` is accepted so the call shape matches ``make_oracle``.
    The alphabet does not depend on the prefix.
    """
    del known_len

    def propose(prefix: list[int], k: int, exclude: set[int]) -> tuple[list[int], int]:
        del prefix
        banned = set(exclude)
        chosen = [tok for tok in alphabet if tok not in banned][:k]
        return chosen, _pick_dummy([dummy], banned, chosen)

    return propose


def make_blind(pool: list[int], dummy: int = 10_002):
    """A proposer that never emits tokens outside ``pool`` (rare-token case)."""

    def propose(prefix: list[int], k: int, exclude: set[int]) -> tuple[list[int], int]:
        del prefix
        banned = set(exclude)
        chosen = [tok for tok in pool if tok not in banned][:k]
        return chosen, _pick_dummy([dummy], banned, chosen)

    return propose


class QwenBaseProposer:
    """Qwen3-4B-Base (GGUF) as the local model in Section IV-A1.

    Only next-token logits are used. The order channel does not read
    these tensors; the model only ranks candidate token ids. ``n_ctx``
    defaults to 256, which is enough for the short prompt in the demo.
    """

    def __init__(self, model_path: str, n_ctx: int = 256, n_threads: int | None = None):
        import os

        if n_threads is None:
            n_threads = min(8, os.cpu_count() or 4)
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.llm = load_qwen(model_path, n_ctx=n_ctx, n_threads=n_threads)
        self._proposer = QwenLogitProposer(self.llm)

    def tokenize(self, text: str) -> list[int]:
        ids = self.llm.tokenize(text.encode("utf-8"), add_bos=False, special=False)
        return [int(i) for i in ids]

    def detokenize(self, token_ids: list[int]) -> str:
        if not token_ids:
            return ""
        raw = self.llm.detokenize([int(i) for i in token_ids])
        if isinstance(raw, str):
            return raw
        return raw.decode("utf-8", errors="replace")

    def piece(self, token_id: int) -> str:
        return self.detokenize([int(token_id)])

    def propose(self, prefix: list[int], k: int, exclude: set[int]) -> tuple[list[int], int]:
        return self._proposer(prefix, k, exclude)
