import unittest

import torch

from .ready_prep import Prep


class ReadyPrepTest(unittest.TestCase):
    def test_last_turn_departure_and_timeout(self):
        p = Prep(3, cap=4)
        msgs = torch.arange(6).remainder(4).expand(3, -1)
        choices = torch.full((3, 6), 6)
        self.assertFalse(p.observations()[2][:, :, 7].any())
        p.advance(choices, msgs)
        _, public, _ = p.observations()
        self.assertEqual(public[1, 0, :, 6:10].argmax(-1).tolist(), [1, 2, 3, 0, 1])
        choices[:] = 7
        choices[1:, 0] = 0
        p.advance(choices, msgs)
        self.assertEqual(p.departed.tolist(), [True, False, False])
        choices[:] = 0
        choices[1, 1] = 4  # A ready player changes role.
        choices[2, 1] = 7  # Another ready player explicitly revokes.
        p.advance(choices, msgs)
        self.assertFalse(p.ready[1, 1])
        self.assertFalse(p.ready[2, 1])
        choices[1, :2] = 7
        p.advance(choices, msgs)
        self.assertEqual(p.departed.tolist(), [True, True, False])
        self.assertEqual(p.turns.tolist(), [2, 4, 4])
        self.assertFalse(p.active.any())
        self.assertEqual(p.jobs[0].tolist(), [2] * 6)

    def test_locked_member_remains_visible_and_can_talk(self):
        p = Prep(1, cap=6, lock_ready=True)
        messages = torch.zeros(1, 6, dtype=torch.long)
        actions = torch.tensor([[4, 5, 6, 6, 6, 6]])
        p.advance(actions, messages)
        actions[:] = 0
        actions[0, 0] = 7
        p.advance(actions, messages)
        own, peers, mask = p.observations()
        self.assertEqual(mask[0, 0].tolist(), [True] + [False]*7)
        self.assertTrue(mask[0, 1, 7])
        self.assertTrue(own[0, 0, 24])
        self.assertTrue(peers[0, 1, 0, 10])
        self.assertEqual(peers[0, 1, 0, :3].tolist(), [1, 0, 0])
        for invalid, token in [(5, 2), (7, 3)]:
            actions[0, 0] = invalid  # Bypass mask to verify environment enforcement.
            messages[0, 0] = token
            p.advance(actions, messages)
            self.assertTrue(p.ready[0, 0])
            self.assertEqual(p.jobs[0, 0], 0)
            _, peers, _ = p.observations()
            self.assertEqual(int(peers[0, 1, 0, 6:10].argmax()), token)
        actions[:] = 7
        p.advance(actions, messages)
        self.assertTrue(p.departed[0])
        self.assertEqual(p.turns[0], 5)
