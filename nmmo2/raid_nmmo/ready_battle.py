"""Play the agreed six-person lineup through the existing boss 01 battle."""

from .combat_policies import by_current_job
from .raid_game import ACT_JOB_TANK, JOB_DPS, JOB_HEALER, JOB_TANK, RaidNmmoEnv
from boss_catalog import make_boss_config


class Battle:
    """The one setup step installs a lineup; every party has 30 combat turns."""

    def __init__(self):
        cfg = make_boss_config("01_std_single", 6, prep_turns=1)
        self.env = RaidNmmoEnv(cfg)
        self._unambiguous = {}

    def evaluate(self, jobs, seed):
        lineup = list(jobs)
        if len(lineup) != self.env.cfg.num_agents or any(
            job not in (JOB_TANK, JOB_HEALER, JOB_DPS) for job in lineup
        ):
            raise ValueError("Battle needs six assigned T/H/D jobs")
        key = tuple(lineup)
        if key in self._unambiguous:
            return self._unambiguous[key]
        return self._play_native(key, int(seed))

    def _play_native(self, key, seed):
        # Only a tied threat score allows the seed to enter raid combat. Keep
        # seat order: action resolution and healer target ties use that order.
        used_tiebreak = False

        def choose_tie(tied):
            nonlocal used_tiebreak
            used_tiebreak = True
            return self.env.game._np_random.choice(tied)

        lineup = list(key)
        self.env.reset(seed=seed)
        obs, total_reward, done, info = self.env.step([ACT_JOB_TANK + int(job) for job in lineup])
        if obs["jobs"] != lineup or obs["in_prep"]:
            raise RuntimeError("The agreed lineup did not reach the combat phase")
        while not done:
            obs, reward, done, info = self.env.step(by_current_job(self.env, obs), tie_choice_fn=choose_tie)
            total_reward += reward
        outcome = bool(info["won"]), float(total_reward)
        if not used_tiebreak:
            self._unambiguous[key] = outcome
        return outcome

    def close(self):
        self.env.close()
