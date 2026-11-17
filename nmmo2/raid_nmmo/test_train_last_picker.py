"""Verify balance, single-member updates, and observation-only role interventions."""
import unittest

import numpy as np
import torch

from .last_picker import make_schedule, run_arm
from .ready_battle import Battle
from .ready_policy import Policies
from .train_last_picker import collect, learning_reward, training_schedule


class CountHealers:
    """Test policy reads peer observations and commits the corresponding role."""
    def __init__(self,count):
        self.rows=count
        self.columns=1

    def forward(self,own,peers,hidden,mask):
        desired=torch.where(peers[:,:,:,1].sum(-1)==0,5,6)
        assigned=own[:,:,9:12].sum(-1)>0
        action=torch.where(assigned,7,desired)
        logits=torch.full((*action.shape,8),-100.0)
        logits.scatter_(-1,action.unsqueeze(-1),100.0)
        return logits,torch.zeros(*action.shape,4),torch.zeros_like(action,dtype=torch.float32),hidden


class TrainingProtocolTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_equal_contexts_for_every_person_and_heldout_parties(self):
        rng=np.random.default_rng(91)
        heldout={tuple(range(6))}
        scenarios=training_schedule(rng,0,2,heldout)
        self.assertEqual(len(scenarios),256)
        for person in range(128):
            self.assertEqual(sum(s['focal_id']==person for s in scenarios),2)
        self.assertFalse(any(tuple(sorted(s['ids'])) in heldout for s in scenarios))
        for scenario in scenarios:
            self.assertEqual(scenario['ids'][scenario['focal_seat']],scenario['focal_id'])
            self.assertEqual(sorted(scenario['A_jobs']),[-1,0,2,2,2,2])

    def test_display_swap_changes_choice_but_keeps_real_teammates(self):
        scenarios=make_schedule(7,1,set())[:6]
        batch=CountHealers(len(scenarios))
        draws=torch.rand(16,2,len(scenarios),1)
        for arm,expected in [('A',1),('B',2)]:
            normal=run_arm(batch,scenarios,draws,6,arm,True)
            swapped=run_arm(batch,scenarios,draws,6,arm,True,swap_role_display=True)
            for scenario,a,b in zip(scenarios,normal,swapped):
                focal=scenario['focal_seat']
                self.assertEqual(a['confirmed_job'],expected)
                self.assertEqual(b['confirmed_job'],3-expected)
                self.assertEqual(a['turns'],2)
                self.assertTrue(b['departed'])
                for seat in range(6):
                    if seat!=focal:
                        self.assertEqual(a['final_jobs'][seat],b['final_jobs'][seat])

    def test_diagnostic_only_changes_two_failed_rewards(self):
        equalized = (.4730113636363636 + .8977272727272729) / 2
        cases = [('A', 0, False, .6306818181818183),
                 ('A', 1, True, 2.0), ('A', 2, False, .4730113636363636),
                 ('B', 0, False, .8863636363636365),
                 ('B', 1, False, .8977272727272729), ('B', 2, True, 2.0)]
        for arm, job, won, actual in cases:
            self.assertEqual(learning_reward(arm, job, True, won, actual), actual)
            diagnostic = learning_reward(arm, job, True, won, actual, equalized)
            expected = equalized if (arm, job) in [('A', 2), ('B', 1)] else actual
            self.assertEqual(diagnostic, expected)
        self.assertEqual(learning_reward('A', 2, False, False, 0, equalized), 0)
        self.assertEqual(learning_reward('B', 1, True, True, 2, equalized), 2)

    def test_only_focal_players_receive_updates(self):
        torch.manual_seed(17)
        model=Policies(width=6)
        optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
        scenarios=make_schedule(31,1,set())[:2]
        battle=Battle()
        untouched=model.members[127].detach().clone()
        initial=model.members[0].detach().clone()
        try:
            records,rewards,outcomes=collect(model,scenarios,battle)
        finally:
            battle.close()
        self.assertEqual([r['arm'] for r in outcomes],['A','A','B','B'])
        self.assertTrue(all((r['ids']==0).all() for r in records))
        loss=model.update(records,rewards,optimizer,epochs=1)
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(torch.equal(untouched,model.members[127]))
        self.assertFalse(torch.equal(initial,model.members[0]))


if __name__=='__main__':
    unittest.main()
