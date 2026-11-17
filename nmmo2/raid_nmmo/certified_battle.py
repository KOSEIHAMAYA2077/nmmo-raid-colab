"""Reuse aggregate battle outcomes when random targets are exchangeable DPS."""
from .combat_policies import by_current_job
from .raid_game import ACT_JOB_TANK, JOB_DPS
from .ready_battle import Battle


class CertifiedBattle(Battle):
    """The caller observes win and reward, so equivalent target identities suffice."""

    def exchangeable_dps(self, tied):
        game, cfg = self.env.game, self.env.cfg
        living = [i for i, alive in enumerate(game._alive) if alive]
        if cfg.boss_aoe_interval or cfg.boss_atk < cfg.dps_hp or sorted(tied) != living:
            return False
        if not all(game._jobs[i] == JOB_DPS and game._hp[i] == cfg.dps_hp for i in living):
            return False
        # All surviving policies have identical damage, MP and cooldown inputs.
        # A boss strike removes one whole, identical DPS. Remaining players are
        # still exchangeable, so subsequent damage and the final reward agree.
        fields = (game._mp, game._job_cd, game._skill_cd, game._skill2_cd, game._threat)
        return all(all(values[i] == values[living[0]] for i in living) for values in fields)

    def _play_native(self, key, seed):
        uncertain = False

        def choose_tie(tied):
            nonlocal uncertain
            uncertain = uncertain or not self.exchangeable_dps(tied)
            return self.env.game._np_random.choice(tied)

        self.env.reset(seed=seed)
        obs, total, done, info = self.env.step([ACT_JOB_TANK + job for job in key])
        if obs['jobs'] != list(key) or obs['in_prep']:
            raise RuntimeError('Lineup was not installed')
        while not done:
            obs, reward, done, info = self.env.step(by_current_job(self.env, obs), tie_choice_fn=choose_tie)
            total += reward
        outcome = bool(info['won']), float(total)
        if not uncertain:
            self._unambiguous[key] = outcome
        return outcome
