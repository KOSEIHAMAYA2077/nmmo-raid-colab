"""同じ固定編成・無通信で2ボスを倒せるか、回復方策だけ変えて比較する。"""

from env import ACT_BASIC, ACT_SKILL, ACT_SKILL2, JOB_DPS, JOB_HEALER, JOB_TANK, RaidConfig
from sweep_checks import make_comp_policy, run_policy


# 準備中に固定職に就き、戦闘中はそのまま。run_policy は全員の通信を0とする。
JOBS = [JOB_TANK, JOB_HEALER] + [JOB_DPS] * 8
standard = make_comp_policy(JOBS, healer_mode="smart_aoe")


def reserve_for_group_heal(env, obs, seed):
    actions = standard(env, obs, seed)
    if obs["in_prep"] or not obs["alive"][1]:
        return actions

    # DPS が大きく削れたら、MP が貯まるまで通常攻撃して全体回復を優先する。
    wounded_dps = any(0 < hp <= 65 for hp in obs["hp"][2:])
    healer_mp = obs["mp"][1]
    actions[1] = ACT_BASIC
    if wounded_dps:
        if healer_mp >= env.cfg.healer_aoe_mp:
            actions[1] = ACT_SKILL2
    elif 0 < obs["hp"][0] <= 200 and healer_mp >= env.cfg.healer_heal_mp:
        actions[1] = ACT_SKILL
    return actions


if __name__ == "__main__":
    seeds = 100  # 両方策・両ボスで同じ環境 seed 0..99
    for boss_hp, period in [(5500, 4), (7500, 0)]:
        config = RaidConfig(num_agents=10, boss_hp=boss_hp, boss_atk=110,
                            boss_aoe_interval=period, boss_aoe_dmg=35)
        for name, policy in [("smart_aoe", standard), ("reserve_mp", reserve_for_group_heal)]:
            result = run_policy(config, policy, seeds=seeds)
            wins = round(result["win"] * seeds)
            print(f"HP={boss_hp} AoE_period={period} {name:10s} "
                  f"wins={wins}/{seeds} mean_alive={result['alive']:.1f} "
                  f"mean_boss_hp_left={result['boss_left']:.1f}")
