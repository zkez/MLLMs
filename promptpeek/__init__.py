"""PROMPTPEEK reproduction: KV-cache sharing side channel in multi-tenant LLM serving.

The leak reproduced here is the one in Wu et al., NDSS 2025
("I Know What You Asked"). It comes from the scheduler, not from the
model weights: a shared prefix cache plus longest-prefix-match scheduling
makes a cache hit visible in the order the attacker's own requests return.
"""

__all__ = ["scheduler", "attack", "candidates"]
