"""Evaluate scripted role compositions (T:H:D) on MinimalRaidEnv to verify balance."""
from typing import Dict, List, Tuple
from env import (
    ACT_BASIC,
    ACT_JOB_DPS,
    ACT_JOB_HEALER,
    ACT_JOB_TANK,
    ACT_SKILL,
    ACT_WAIT,
    JOB_DPS,
    JOB_HEALER,
    JOB_TANK,
    MinimalRaidEnv,
    RaidConfig,
)


def scripted_actions(obs: Dict, target_jobs: List[int], heal_threshold: float = 0.85) -> List[int]:
    actions = []
    n = len(obs["jobs"])
    for i in range(n):
        if not obs["alive"][i]:
            actions.append(ACT_WAIT)
            continue
        want = target_jobs[i]
        if obs["jobs"][i] != want and (obs["in_prep"] or obs["job_cd"][i] == 0):
            actions.append(ACT_JOB_TANK + want)
            continue
        if obs["in_prep"]:
            actions.append(ACT_WAIT)
            continue

        job = obs["jobs"][i]
        if job == JOB_TANK:
            # Use Provoke when off cooldown and enough MP, otherwise shield bash
            if obs["skill_cd"][i] == 0 and obs["mp"][i] >= 15:
                actions.append(ACT_SKILL)
            else:
                actions.append(ACT_BASIC)
        elif job == JOB_HEALER:
            # Heal only if someone is below heal_threshold of max HP (avoid overheal & MP waste)
            need_heal = False
            for k in range(n):
                if not obs["alive"][k]:
                    continue
                max_hp = 250.0 if obs["jobs"][k] == JOB_TANK else 100.0
                if obs["hp"][k] / max_hp < heal_threshold:
                    need_heal = True
                    break
            if need_heal and obs["skill_cd"][i] == 0 and obs["mp"][i] >= 20:
                actions.append(ACT_SKILL)
            else:
                actions.append(ACT_BASIC)
        else:  # DPS
            if obs["skill_cd"][i] == 0 and obs["mp"][i] >= 25:
                actions.append(ACT_SKILL)
            else:
                actions.append(ACT_BASIC)
    return actions


def run_comp(cfg: RaidConfig, n_tank: int, n_heal: int, n_dps: int, seeds: int = 5) -> Tuple[float, float, float, float]:
    target_jobs = [JOB_TANK] * n_tank + [JOB_HEALER] * n_heal + [JOB_DPS] * n_dps
    wins, rets, rem_hp, alive_end = [], [], [], []
    for s in range(seeds):
        env = MinimalRaidEnv(cfg)
        obs = env.reset(seed=s)
        total_r = 0.0
        while not env.done:
            acts = scripted_actions(obs, target_jobs)
            obs, r, done, info = env.step(acts)
            total_r += r
        wins.append(1.0 if info["won"] else 0.0)
        rets.append(total_r)
        rem_hp.append(info["boss_hp"])
        alive_end.append(info["alive_count"])
    return (
        sum(wins) / seeds,
        sum(rets) / seeds,
        sum(rem_hp) / seeds,
        sum(alive_end) / seeds,
    )


def main() -> None:
    cfg = RaidConfig()
    print(f"Config: N={cfg.num_agents}, boss_hp={cfg.boss_hp}, boss_atk={cfg.boss_atk}, battle_turns={cfg.battle_turns}")
    print(f"{'T:H:D':>7} | {'win_rate':>8} | {'return':>7} | {'boss_hp_left':>12} | {'alive_end':>9}")
    print("-" * 55)
    for t in range(cfg.num_agents + 1):
        for h in range(cfg.num_agents + 1 - t):
            d = cfg.num_agents - t - h
            win, ret, rem, alv = run_comp(cfg, t, h, d)
            print(f"{t}:{h}:{d}".rjust(7) + f" | {win:8.2f} | {ret:7.3f} | {rem:12.1f} | {alv:9.2f}")


if __name__ == "__main__":
    main()
