"""The cached stochastic branch requires an exchangeable, one-hit DPS population."""
from types import SimpleNamespace
import unittest

from .certified_battle import CertifiedBattle


class CertificateTest(unittest.TestCase):
    def battle(self):
        game = SimpleNamespace(_alive=[True]*3, _jobs=[2]*3, _hp=[100.0]*3,
                               _mp=[50]*3, _job_cd=[0]*3, _skill_cd=[1]*3,
                               _skill2_cd=[0]*3, _threat=[100]*3)
        cfg = SimpleNamespace(boss_aoe_interval=0, boss_atk=105, dps_hp=100)
        battle = CertifiedBattle.__new__(CertifiedBattle)
        battle.env = SimpleNamespace(game=game, cfg=cfg)
        return battle

    def test_only_exchangeable_dps_are_certified(self):
        self.assertTrue(self.battle().exchangeable_dps([0, 1, 2]))
        for attribute, value in [('_jobs', 1), ('_hp', 99), ('_mp', 40),
                                 ('_job_cd', 1), ('_skill_cd', 0),
                                 ('_skill2_cd', 1), ('_threat', 90)]:
            with self.subTest(attribute=attribute):
                battle = self.battle()
                getattr(battle.env.game, attribute)[0] = value
                self.assertFalse(battle.exchangeable_dps([0, 1, 2]))
        battle = self.battle()
        battle.env.cfg.boss_atk = 99
        self.assertFalse(battle.exchangeable_dps([0, 1, 2]))
        battle = self.battle()
        battle.env.cfg.boss_aoe_interval = 5
        self.assertFalse(battle.exchangeable_dps([0, 1, 2]))
        self.assertFalse(self.battle().exchangeable_dps([0, 1]))


if __name__ == '__main__':
    unittest.main()
