"""Scheduler and PROMPTPEEK order-channel tests.

These tests never load Qwen. They check the side channel the paper
actually uses: the order in which the attacker's own requests complete
under longest-prefix-match, and the ways the defenses break that pattern.
"""

from __future__ import annotations

import unittest

from promptpeek.attack import plant_victim, probe, recover
from promptpeek.candidates import AlphabetProposer, OracleProposer
from promptpeek.scheduler import PrefixCache, Request, ServingEngine


def _engine(**kwargs) -> ServingEngine:
    kwargs.setdefault("max_batch", 4)
    kwargs.setdefault("capacity", 10_000)
    return ServingEngine(**kwargs)


class PrefixCacheTests(unittest.TestCase):
    def test_duplicate_token_ids_stay_on_the_path(self):
        cache = PrefixCache(capacity=10)
        cache.insert("t", [7, 7, 8])
        self.assertEqual(cache.used, 3)
        self.assertEqual(cache.match_len("t", [7, 7, 8]), 3)
        # A wrong shortcut (7 -> 8) would report 2 here.
        self.assertEqual(cache.match_len("t", [7, 8]), 1)

    def test_shared_prefix_is_not_counted_twice(self):
        cache = PrefixCache()
        cache.insert("t", [1, 2, 3])
        cache.insert("t", [1, 2, 4])
        self.assertEqual(cache.used, 4)
        self.assertEqual(cache.match_len("t", [1, 2, 3]), 3)
        self.assertEqual(cache.match_len("t", [1, 2, 4]), 3)

    def test_eviction_replays_duplicate_tokens_without_hanging(self):
        cache = PrefixCache(capacity=2)
        cache.insert("t", [7, 7, 8, 9])
        self.assertEqual(cache.evictions, 1)
        self.assertEqual(cache.match_len("t", [7, 7, 8, 9]), 4)
        self.assertEqual(cache.match_len("t", [7, 8]), 1)

    def test_eviction_drops_unprotected_victim(self):
        cache = PrefixCache(capacity=4)
        cache.insert("v", [1, 2, 3, 4])
        cache.insert("a", [1, 2, 9], protected=[("a", (1, 2, 9))])
        self.assertEqual(cache.evictions, 1)
        self.assertEqual(cache.match_len("a", [1, 2, 9]), 3)
        self.assertLess(cache.match_len("v", [1, 2, 3, 4]), 4)


class OrderChannelTests(unittest.TestCase):
    def test_lpm_hit_promotes_exactly_one(self):
        engine = _engine(policy="lpm")
        plant_victim(engine, "victim", [1, 2, 3, 4, 7, 8])
        result = probe(engine, "attacker", [1, 2, 3], [4, 5, 6], dummy=99)
        self.assertEqual(result.hit_token, 4)
        self.assertEqual(result.promoted, [4])
        self.assertEqual(
            result.order,
            [
                "p-pre0",
                "p-pre1",
                "p-pre2",
                "p-pre3",
                "p-c0",
                "p-post0",
                "p-post1",
                "p-post2",
                "p-post3",
                "p-c1",
                "p-c2",
            ],
        )
        # The waiting-queue batch (index 1) is where LPM actually sorts.
        # Match lengths follow Figure 6: the hit and the post-dummies share
        # one more token than the misses.
        self.assertEqual(result.batch_match_lens[1][0], 4)
        self.assertIn(3, result.batch_match_lens[-1])

    def test_lpm_miss_promotes_none(self):
        engine = _engine(policy="lpm")
        plant_victim(engine, "victim", [1, 2, 3, 4, 7, 8])
        result = probe(engine, "attacker", [1, 2, 3], [5, 6, 8], dummy=99)
        self.assertIsNone(result.hit_token)
        self.assertEqual(result.promoted, [])
        self.assertEqual(
            result.order,
            [
                "p-pre0",
                "p-pre1",
                "p-pre2",
                "p-pre3",
                "p-post0",
                "p-post1",
                "p-post2",
                "p-post3",
                "p-c0",
                "p-c1",
                "p-c2",
            ],
        )

    def test_isolation_blocks_the_unique_hit(self):
        engine = _engine(policy="lpm", isolated=True)
        plant_victim(engine, "victim", [1, 2, 3, 4, 7, 8])
        self.assertEqual(engine.cache.match_len("attacker", [1, 2, 3, 4]), 0)
        result = probe(engine, "attacker", [1, 2, 3], [4, 5, 6], dummy=99)
        self.assertIsNone(result.hit_token)
        self.assertEqual(result.promoted, [])
        self.assertLess(engine.cache.match_len("attacker", [1, 2, 3, 4, 7, 8]), 5)

    def test_fcfs_is_not_a_unique_hit(self):
        engine = _engine(policy="fcfs")
        plant_victim(engine, "victim", [1, 2, 3, 4, 7, 8])
        result = probe(engine, "attacker", [1, 2, 3], [4, 5, 6], dummy=99)
        self.assertIsNone(result.hit_token)
        self.assertEqual(result.promoted, [4, 5, 6])
        self.assertEqual(
            result.order,
            [
                "p-pre0",
                "p-pre1",
                "p-pre2",
                "p-pre3",
                "p-c0",
                "p-c1",
                "p-c2",
                "p-post0",
                "p-post1",
                "p-post2",
                "p-post3",
            ],
        )

    def test_margin_blocks_one_extra_token_off_the_boundary(self):
        # Prefix length 4, M=2. 4//2 == 5//2, so one extra token does not
        # change the bucket. The secret is still in the cache; LPM-with-margin
        # just refuses to reorder for it.
        engine = _engine(policy="lpm", lpm_margin=2)
        secret = [1, 2, 3, 4, 8, 9]
        plant_victim(engine, "victim", secret)
        longer = Request("hit", "attacker", (1, 2, 3, 4, 8), arrival=1)
        shorter = Request("miss", "attacker", (1, 2, 3, 4, 6), arrival=2)
        self.assertEqual(engine.priority(longer), engine.priority(shorter))
        result = probe(engine, "attacker", [1, 2, 3, 4], [8, 6, 5], dummy=99)
        self.assertIsNone(result.hit_token)
        self.assertGreater(len(result.promoted), 1)

    def test_margin_floor_division_leaks_on_the_boundary(self):
        # Prefix length 3, M=2. 3 % 2 == 1, so 3//2 != 4//2. One extra
        # token still changes the bucket. This is the honest limit of
        # implementing M as match_len // M.
        engine = _engine(policy="lpm", lpm_margin=2)
        plant_victim(engine, "victim", [1, 2, 3, 8, 9])
        longer = Request("hit", "attacker", (1, 2, 3, 8), arrival=1)
        shorter = Request("miss", "attacker", (1, 2, 3, 6), arrival=2)
        self.assertEqual(engine.priority(shorter), 1)
        self.assertEqual(engine.priority(longer), 2)
        result = probe(engine, "attacker", [1, 2, 3], [8, 6, 5], dummy=99)
        self.assertEqual(result.hit_token, 8)
        self.assertEqual(result.promoted, [8])

    def test_stored_miss_becomes_a_false_hit_if_retried(self):
        engine = _engine(policy="lpm")
        plant_victim(engine, "victim", [1, 2, 3, 7])
        first = probe(engine, "attacker", [1, 2, 3], [0, 1], dummy=99, tag="r0")
        self.assertIsNone(first.hit_token)
        # Token 0 is now in the attacker's own cache. Retrying it next to a
        # fresh wrong token looks like the unique-hit pattern.
        second = probe(engine, "attacker", [1, 2, 3], [0, 8], dummy=98, tag="r1")
        self.assertEqual(second.hit_token, 0)
        self.assertNotEqual(second.hit_token, 7)


class RecoveryTests(unittest.TestCase):
    def test_oracle_recovers_multiple_tokens(self):
        secret = [1, 2, 3, 7, 0, 5]
        engine = _engine(policy="lpm")
        plant_victim(engine, "victim", secret)
        proposer = OracleProposer(
            target=secret,
            alphabet=list(range(10)),
            dummy_pool=[100, 101, 102, 103],
        )
        recovered = recover(
            engine,
            "attacker",
            [1, 2, 3],
            proposer,
            max_new_tokens=3,
            k=4,
        )
        self.assertEqual(recovered.tokens, [7, 0, 5])
        self.assertEqual(recovered.stopped_reason, "recovered")
        self.assertEqual(recovered.requests, 3 * (4 + 4 + 4))

    def test_alphabet_brute_force_excludes_previous_misses(self):
        # Account digits 7, 0, 5 over alphabet 0-9, four candidates at a
        # time. Token 7 and token 5 each miss a round before they hit.
        # Request count is 5 probes * (4 pre + 4 post + 4 candidates) = 60.
        secret = [10, 11, 12, 7, 0, 5]
        engine = _engine(policy="lpm")
        plant_victim(engine, "victim", secret)
        proposer = AlphabetProposer(
            alphabet=list(range(10)),
            dummy_pool=list(range(100, 120)),
        )
        recovered = recover(
            engine,
            "attacker",
            [10, 11, 12],
            proposer,
            max_new_tokens=3,
            k=4,
            max_rounds=4,
        )
        self.assertEqual(recovered.tokens, [7, 0, 5])
        self.assertEqual(recovered.stopped_reason, "recovered")
        self.assertEqual([step.rounds for step in recovered.steps], [2, 1, 2])
        self.assertEqual(recovered.requests, 60)
        candidate_requests = sum(step.rounds * 4 for step in recovered.steps)
        self.assertEqual(candidate_requests, 20)

    def test_truth_outside_the_candidate_list_stops_recovery(self):
        secret = [1, 2, 3, 42]
        engine = _engine(policy="lpm")
        plant_victim(engine, "victim", secret)
        proposer = OracleProposer(
            target=secret,
            alphabet=[0, 1, 2, 3],
            dummy_pool=[100, 101],
        )
        recovered = recover(
            engine,
            "attacker",
            [1, 2, 3],
            proposer,
            max_new_tokens=2,
            k=4,
            max_rounds=4,
        )
        self.assertEqual(recovered.tokens, [])
        # The alphabet was searched and missed, then the proposer ran dry.
        # That is the paper's partial failure, not an empty first batch.
        self.assertEqual(recovered.stopped_reason, "no_unique_hit")
        self.assertIsNone(recovered.steps[0].token)
        self.assertGreaterEqual(recovered.steps[0].rounds, 1)

    def test_eviction_clears_the_victim_and_the_next_probe_misses(self):
        victim = [1, 2, 3, 4, 5]
        tight = _engine(policy="lpm", capacity=len(victim))
        plant_victim(tight, "victim", victim)
        self.assertEqual(tight.cache.match_len("victim", victim), len(victim))
        evicting = probe(tight, "attacker", [1, 2, 3], [4, 6], dummy=99, tag="ev")
        self.assertGreaterEqual(tight.cache.evictions, 1)
        self.assertIsNone(evicting.hit_token)
        self.assertLess(tight.cache.match_len("victim", victim), len(victim))

        # The following token of the victim is 5. It was not stored by the
        # attacker, and eviction already dropped it. A later probe misses.
        follow = probe(tight, "attacker", [1, 2, 3, 4], [5, 8], dummy=98, tag="nx")
        self.assertIsNone(follow.hit_token)

        roomy = _engine(policy="lpm", capacity=10_000)
        plant_victim(roomy, "victim", victim)
        control = probe(roomy, "attacker", [1, 2, 3], [4, 6], dummy=99, tag="ok")
        self.assertEqual(control.hit_token, 4)


if __name__ == "__main__":
    unittest.main()
