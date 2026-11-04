"""Minimal multi-tenant serving scheduler.

This models the three policies the paper actually depends on (Section II-B
and III-A), not a full SGLang runtime:

* prefix KV cache: a token's cache is reusable only when every preceding
  token matches (radix tree);
* admission: at most ``max_batch`` requests run at once, and a request that
  is admitted stores its prefix in the cache;
* scheduling of the *waiting* queue: longest-prefix match (LPM), FCFS, or
  LPM with a minimum gain of M tokens (the mitigation sketched in Section VIII).

Per-tenant isolation (vLLM's ``cache_salt`` / separate radix trees) is a
separate switch. It keeps LPM on, but a tenant can only hit prefixes that
tenant itself stored.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Request:
    req_id: str
    tenant: str
    tokens: tuple[int, ...]
    arrival: int


@dataclass
class BatchRecord:
    """One running batch, in the order those requests start (and, with
    max_tokens=1, the order their first token is returned)."""

    index: int
    req_ids: list[str]
    match_lens: list[int]
    priorities: list[int]


@dataclass
class _Node:
    children: dict[int, _Node] = field(default_factory=dict)
    parent: _Node | None = None
    token: int | None = None
    last_used: int = 0


class PrefixCache:
    """Radix-tree prefix cache.

    ``isolated=True`` gives every tenant a private tree, which is the
    practical form of "do not share KV across mutually distrustful users".
    ``capacity`` counts stored tokens (edges). When an insert would exceed
    it, every entry except ``protected`` prefixes is dropped. That matches
    the paper's eviction description: once memory is exhausted, all KV
    except the current batch is released at once (Section III-A, PE).
    """

    def __init__(self, capacity: int = 100_000, isolated: bool = False):
        self.capacity = capacity
        self.isolated = isolated
        self.roots: dict[str, _Node] = {}
        self.used = 0
        self._clock = 0
        self.evictions = 0

    def _key(self, tenant: str) -> str:
        return tenant if self.isolated else "*"

    def _root(self, tenant: str) -> _Node:
        key = self._key(tenant)
        root = self.roots.get(key)
        if root is None:
            root = _Node()
            self.roots[key] = root
        return root

    def match_len(self, tenant: str, tokens: tuple[int, ...] | list[int]) -> int:
        node = self._root(tenant)
        matched = 0
        for tok in tokens:
            child = node.children.get(tok)
            if child is None:
                break
            node = child
            matched += 1
        if matched:
            self._clock += 1
            node.last_used = self._clock
        return matched

    def insert(
        self,
        tenant: str,
        tokens: tuple[int, ...] | list[int],
        protected: list[tuple[str, tuple[int, ...]]] | None = None,
    ) -> None:
        """Insert ``tokens`` for ``tenant``.

        Walk by index, not by ``list.index``, so a repeated token id stays
        on the correct path (the sequence ``(7, 7, 8)`` is not the same
        node as ``(7, 8)``). Eviction drops the whole tree once, reinserts
        only ``protected`` prefixes inline, and never calls ``insert``
        again. If the protected set still fills the cache, the rest of
        this insert is abandoned instead of evicting forever.
        """
        tokens = tuple(tokens)
        if protected is None:
            protected = [(tenant, tokens)]
        self._clock += 1
        node = self._root(tenant)
        evicted = False
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            child = node.children.get(tok)
            if child is None and self.used + 1 > self.capacity and not evicted:
                self._evict_except(protected)
                evicted = True
                node = self._rewind(tenant, tokens[:i])
                if node is None:
                    return
                child = node.children.get(tok)
            if child is None:
                if self.used + 1 > self.capacity:
                    return
                child = _Node(parent=node, token=tok)
                node.children[tok] = child
                self.used += 1
            node = child
            node.last_used = self._clock
            i += 1

    def _rewind(self, tenant: str, prefix: tuple[int, ...]) -> _Node | None:
        """Walk ``prefix`` on the post-eviction tree, creating a node only
        when capacity remains. Returns None if the prefix itself does not fit.
        """
        node = self._root(tenant)
        for tok in prefix:
            child = node.children.get(tok)
            if child is None:
                if self.used + 1 > self.capacity:
                    return None
                child = _Node(parent=node, token=tok)
                node.children[tok] = child
                self.used += 1
            node = child
            node.last_used = self._clock
        return node

    def _evict_except(self, protected: list[tuple[str, tuple[int, ...]]]) -> None:
        """Drop every cached token, then reinsert ``protected`` directly.

        The reinsertion does not go through ``insert``, so it cannot evict
        again. Protected prefixes are the batch that is running; the paper
        keeps that batch even when it is what filled the memory.
        """
        self.evictions += 1
        self.roots.clear()
        self.used = 0
        for tenant, toks in protected:
            node = self._root(tenant)
            self._clock += 1
            for tok in toks:
                child = node.children.get(tok)
                if child is None:
                    child = _Node(parent=node, token=tok)
                    node.children[tok] = child
                    self.used += 1
                node = child
                node.last_used = self._clock


class ServingEngine:
    """Continuous batching with a configurable waiting-queue policy.

    The attack relies on a timing detail from Figure 6: the first wave of
    dummy requests is *already running* (so its KV is stored) before the
    candidates are sorted. ``fill_running_if_idle`` admits that wave.
    Later waves are chosen by ``policy``.
    """

    def __init__(
        self,
        policy: str = "lpm",
        max_batch: int = 4,
        capacity: int = 100_000,
        isolated: bool = False,
        lpm_margin: int = 1,
    ):
        if policy not in {"lpm", "fcfs"}:
            raise ValueError(f"unknown policy {policy}")
        if lpm_margin < 1:
            raise ValueError("lpm_margin must be >= 1")
        self.policy = policy
        self.max_batch = max_batch
        self.lpm_margin = lpm_margin
        self.cache = PrefixCache(capacity=capacity, isolated=isolated)
        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self.completed: list[Request] = []
        self.batches: list[BatchRecord] = []
        self._arrival = 0

    def enqueue(self, tenant: str, tokens: list[int] | tuple[int, ...], req_id: str) -> Request:
        self._arrival += 1
        req = Request(req_id=req_id, tenant=tenant, tokens=tuple(tokens), arrival=self._arrival)
        self.waiting.append(req)
        return req

    def match_len(self, req: Request) -> int:
        return self.cache.match_len(req.tenant, req.tokens)

    def priority(self, req: Request, policy: str | None = None) -> int:
        """Higher is served sooner.

        ``lpm_margin`` is the paper's M (Section VIII). Bucketing by
        ``match_len // M`` means a request must gain M extra shared tokens
        to move into the next priority bucket, instead of winning by a
        single token. M=1 is unmodified LPM.

        Floor division has a boundary leak: when the shared length L
        satisfies ``L % M == M - 1``, one extra token still changes the
        bucket. Callers that want the mitigation to hold must pick a
        prefix length that is not on that boundary, or compare match
        lengths with a real gap of M rather than a bucket id.
        """
        policy = self.policy if policy is None else policy
        if policy == "fcfs":
            return 0
        return self.match_len(req) // self.lpm_margin

    def fill_running_if_idle(self, policy_override: str | None = None) -> list[Request]:
        if self.running or not self.waiting:
            return []
        policy = policy_override or self.policy
        batch = self._take(policy, self.max_batch)
        # Record the hit length before admission writes these requests into
        # the cache. After insert, every admitted request would "match"
        # itself and the Figure 6 lengths would disappear.
        match_lens = [self.match_len(req) for req in batch]
        if policy == "fcfs":
            priorities = [0 for _ in batch]
        else:
            priorities = [matched // self.lpm_margin for matched in match_lens]
        self.running = batch
        protected = [(req.tenant, req.tokens) for req in batch]
        for req in batch:
            self.cache.insert(req.tenant, req.tokens, protected=protected)
        self.batches.append(
            BatchRecord(
                index=len(self.batches),
                req_ids=[req.req_id for req in batch],
                match_lens=match_lens,
                priorities=priorities,
            )
        )
        return batch

    def complete_running(self) -> list[Request]:
        done = list(self.running)
        self.completed.extend(done)
        self.running = []
        return done

    def drain(self) -> None:
        while self.running or self.waiting:
            if not self.running:
                self.fill_running_if_idle()
            self.complete_running()

    def _take(self, policy: str, k: int) -> list[Request]:
        if not self.waiting:
            return []
        if policy == "fcfs":
            chosen = self.waiting[:k]
        else:
            indexed = list(enumerate(self.waiting))
            indexed.sort(key=lambda item: (-self.priority(item[1], policy), item[0]))
            chosen = [item[1] for item in indexed[:k]]
        chosen_ids = {id(req) for req in chosen}
        self.waiting = [req for req in self.waiting if id(req) not in chosen_ids]
        return chosen
