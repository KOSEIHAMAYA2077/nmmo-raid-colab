"""Protect matching and confirmation semantics of the frozen last-picker protocol."""
import unittest

import torch

from .last_picker import make_schedule, prepare, run_arm, summary
from .ready_battle import Battle
from .ready_policy import MemberBatch, Policies


class LastPickerTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.schedule = make_schedule(16, 1, set())

    def test_schedule_balances_people_and_seats_and_changes_one_peer(self):
        self.assertEqual(len(self.schedule), 128 * 6)
        self.assertEqual({(s['focal_id'], s['focal_seat']) for s in self.schedule},
                         {(i, j) for i in range(128) for j in range(6)})
        a, b = prepare(self.schedule, 'A'), prepare(self.schedule, 'B')
        self.assertTrue(((a.jobs != b.jobs).sum(1) == 1).all())
        self.assertTrue((a.ready == b.ready).all())
        self.assertTrue((a.ready.sum(1) == 5).all())
        self.assertTrue(((a.jobs == 0).sum(1) == 1).all())
        self.assertTrue(((a.jobs == 1).sum(1) == 0).all())
        self.assertTrue(((b.jobs == 1).sum(1) == 1).all())
        own_a, peers_a, mask_a = a.observations()
        own_b, peers_b, mask_b = b.observations()
        for i, s in enumerate(self.schedule):
            seat = s['focal_seat']
            self.assertTrue(torch.equal(own_a[i, seat], own_b[i, seat]))
            self.assertTrue(torch.equal(mask_a[i, seat], mask_b[i, seat]))
            difference = peers_a[i, seat] != peers_b[i, seat]
            self.assertEqual(int(difference.sum()), 2)  # Only D/H one-hot bits.
            self.assertFalse(difference[:, 3:].any())

    def test_identical_observation_ignoring_policy_has_identical_paired_outcomes(self):
        model = Policies(width=6)
        for parameter in model.parameters():
            parameter.data.zero_()
        scenarios = self.schedule[:7]
        ids = torch.tensor([[s['focal_id']] for s in scenarios])
        batch = MemberBatch(ids, model.members, model.shapes)
        draws = torch.rand(16, 2, len(scenarios), 1, generator=torch.Generator().manual_seed(4))
        a = run_arm(batch, scenarios, draws, 6, 'A', False)
        b = run_arm(batch, scenarios, draws, 6, 'B', False)
        for left, right in zip(a, b):
            self.assertEqual(left['steps'], right['steps'])
            self.assertEqual(left['confirmed_job'], right['confirmed_job'])

    def test_timeouts_are_not_counted_as_committed_roles(self):
        rows = [dict(focal_id=0, A=dict(confirmed_job=-1, hoped_job=1, departed=False, won=False),
                     B=dict(confirmed_job=2, hoped_job=2, departed=True, won=True)),
                dict(focal_id=1, A=dict(confirmed_job=1, hoped_job=1, departed=True, won=True),
                     B=dict(confirmed_job=2, hoped_job=2, departed=True, won=True))]
        result = summary(rows)
        self.assertEqual(result['A_H'], 1)
        self.assertEqual(result['correct'], 1)
        self.assertEqual(result['unconfirmed_either'], 1)
        self.assertEqual(result['delta_H_pp'], 50)

    def test_last_legal_turn_counts_and_rule_completes_both_battles(self):
        battle = Battle()
        try:
            for arm in ('A', 'B'):
                scenarios = self.schedule[:6]  # All 6 focal seats represented by one person.
                prep = prepare(scenarios, arm)
                seats = torch.tensor([s['focal_seat'] for s in scenarios])
                rows = torch.arange(len(scenarios))
                _, peers, _ = prep.observations()
                healer_count = peers[rows, seats, :, 1].sum(-1)
                chosen = torch.where(healer_count == 0, 1, 2)
                actions = torch.zeros_like(prep.jobs)
                actions[rows, seats] = 4 + chosen
                prep.advance(actions, torch.zeros_like(actions))
                self.assertFalse(prep.departed.any())
                for _ in range(14):
                    prep.advance(torch.zeros_like(actions), torch.zeros_like(actions))
                actions[rows, seats] = 7
                prep.advance(actions, torch.zeros_like(actions))
                self.assertTrue(prep.departed.all())
                for scenario, jobs in zip(scenarios, prep.jobs.tolist()):
                    self.assertEqual(sorted(jobs), [0, 1, 2, 2, 2, 2])
                    self.assertTrue(battle.evaluate(jobs, scenario['env_seed'])[0])
        finally:
            battle.close()


if __name__ == '__main__':
    unittest.main()
