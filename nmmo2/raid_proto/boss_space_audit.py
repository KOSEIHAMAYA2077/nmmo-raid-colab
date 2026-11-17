"""現在のプレイヤー設定 (RaidConfig) に対して、ボスのパラメータがどこまで適切かを境界調査するスクリプト。

調べる 3 つの軸：
1. 単体攻撃力 boss_atk の境界 (60..340)：
   - 何以上なら「タンク無し (0:1)」が防げるか？
   - 何以上なら「1タンク (1:1)」が落ち、交互挑発「タンクスイッチ (2:1_swap)」や「2ヒーラー (1:2)」が必要になるか？
2. 全体攻撃 (AoE威力 × AoE周期) の 2次元マップ (N=4, 6, 10, 16)：
   - どの範囲が「1H余裕」「1H要工夫(MP温存)」「2H必須」「3H必須」「攻略不能」になるか？
3. ボス HP (1人あたり換算 (N-2) 倍数 500..1000) の境界：
   - 何以下だと全員DPS (0:0:N) が暴れてしまうか？何以上だと時間切れになるか？
"""
from typing import Dict, List, Tuple

from env import (
    ACT_BASIC,
    ACT_JOB_TANK,
    ACT_SKILL,
    ACT_WAIT,
    JOB_DPS,
    JOB_HEALER,
    JOB_TANK,
    MinimalRaidEnv,
    RaidConfig,
)
from sweep_checks import make_comp_policy, run_policy
from calibrate_bosses import make_reserve_mp_policy


def make_tank_swap_policy(
    target_jobs: List[int],
    healer_mode: str = "smart_aoe",
    use_reserve: bool = False,
):
    """複数タンク時に交互で挑発（タンクスイッチ）を行う方策。

    env.py では挑発したターンのみ追加50%軽減（計75%カット）が入るため、
    T=2 で奇数ターンに T0・偶数ターンに T1 が挑発すると、
    ボス攻撃を受ける側のタンクに毎ターン 75% 軽減を乗せられる。
    """
    if use_reserve:
        base_fn = make_reserve_mp_policy(
            target_jobs, dps_hp_thresh=70.0, tank_hp_thresh=190.0, min_wounded_for_aoe=2, tank_emergency_hp=120.0
        )
    else:
        base_fn = make_comp_policy(target_jobs, healer_mode=healer_mode)

    def fn(env: MinimalRaidEnv, obs: Dict, seed: int) -> List[int]:
        actions = base_fn(env, obs, seed)
        if obs["in_prep"]:
            return actions
        n = env.cfg.num_agents
        living_tanks = [k for k in range(n) if obs["alive"][k] and obs["jobs"][k] == JOB_TANK]
        if len(living_tanks) >= 2:
            battle_turn = env.turn - env.cfg.prep_turns
            active_tank = living_tanks[battle_turn % len(living_tanks)]
            for t_id in living_tanks:
                if t_id == active_tank:
                    if obs["skill_cd"][t_id] == 0 and obs["mp"][t_id] >= env.cfg.tank_provoke_mp:
                        actions[t_id] = ACT_SKILL
                    else:
                        actions[t_id] = ACT_BASIC
                else:
                    # 非番タンクは挑発を温存して通常攻撃
                    actions[t_id] = ACT_BASIC
        return actions

    return fn


def audit_boss_atk_boundaries() -> None:
    """1. boss_atk の全域スキャン (60..340)：タンク不要ラインとタンクスイッチ有効域を確認。"""
    print("=== 1. boss_atk 境界スキャン (N=6 HP=3000, N=10 HP=5800, AoE=0) ===")
    for n, bhp in [(6, 3000), (10, 5800)]:
        print(f"\n--- N={n}, boss_hp={bhp}, AoEなし ---")
        atk_list = [60, 80, 95, 100, 105, 125, 145, 165, 185, 205, 230, 260, 300, 340]
        header = (
            f"  {'ATK':>4s} | {'0:0:N':>6s} | {'0:1:N-1':>7s} | {'1:1(通常)':>9s} | "
            f"{'1:2(通常)':>9s} | {'2:1(同発)':>9s} | {'2:1(交代)':>9s} | {'2:2(交代)':>9s}"
        )
        print(header)
        for atk in atk_list:
            cfg = RaidConfig(num_agents=n, boss_hp=bhp, boss_atk=atk)
            cases = [
                ("0:0", [JOB_DPS] * n, "normal", False),
                ("0:1", [JOB_HEALER] + [JOB_DPS] * (n - 1), "normal", False),
                ("1:1", [JOB_TANK, JOB_HEALER] + [JOB_DPS] * (n - 2), "normal", False),
                ("1:2", [JOB_TANK] + [JOB_HEALER] * 2 + [JOB_DPS] * (n - 3), "normal", False),
                ("2:1s", [JOB_TANK] * 2 + [JOB_HEALER] + [JOB_DPS] * (n - 3), "normal", False),
                ("2:1w", [JOB_TANK] * 2 + [JOB_HEALER] + [JOB_DPS] * (n - 3), "swap", False),
                ("2:2w", [JOB_TANK] * 2 + [JOB_HEALER] * 2 + [JOB_DPS] * (n - 4), "swap", False),
            ]
            cols = []
            for _, jobs, tmode, _ in cases:
                if tmode == "swap":
                    fn = make_tank_swap_policy(jobs, healer_mode="smart_aoe")
                else:
                    fn = make_comp_policy(jobs, healer_mode="smart_aoe")
                res = run_policy(cfg, fn, seeds=3)
                cols.append(f"{res['win']:.2f}({res['alive']:2.0f})")
            print(f"  {atk:4d} | " + " | ".join(f"{c:>9s}" if i >= 2 else f"{c:>7s}" if i == 1 else f"{c:>6s}" for i, c in enumerate(cols)))


def audit_aoe_2d_map() -> None:
    """2. AoE 威力 × 周期の 2D マップ：全4サイズで「1H可 / 要工夫 / 2H必須 / 3H必須 / 不可」を分類。"""
    print("\n=== 2. AoE 2次元マップ (dmg × interval) @ N=4, 6, 10, 16 ===")
    # 各サイズ、1H/2H/3H の余力を残すため HP = 650*(N-2)（N=4のみ2H時は別計上）で基準を見る
    # 代表編成を絞って高速にチェック
    dmgs = [20, 25, 30, 35, 45, 55, 70]
    intervals = [5, 4, 3, 2]
    for n in (4, 6, 10, 16):
        bhp = 650 * (n - 2) if n >= 6 else 1400
        print(f"\n--- N={n}, HP={bhp}, ATK=105 ---")
        print("  記号: [1]=1:1可(smart), [1*]=1:1可(MP温存のみ), [2]=2H必須(1:2可), [3]=3H必須, [X]=不可")
        row_hdr = "  dmg\\int | " + " | ".join(f"{iv}T{'':13s}" for iv in intervals)
        print(row_hdr)
        for dmg in dmgs:
            cells = []
            for iv in intervals:
                cfg = RaidConfig(num_agents=n, boss_hp=bhp, boss_atk=105, boss_aoe_dmg=dmg, boss_aoe_interval=iv)
                # チェックする編成リスト（H=1, H=2, H=3, 2T2H）
                tag = "X"
                alive_best = 0
                for h in range(1, min(4, n - 1)):
                    d = n - 1 - h
                    if d <= 0:
                        continue
                    jobs = [JOB_TANK] + [JOB_HEALER] * h + [JOB_DPS] * d
                    r_smart = run_policy(cfg, make_comp_policy(jobs, healer_mode="smart_aoe"), seeds=2)
                    r_res65 = run_policy(cfg, make_reserve_mp_policy(jobs, 65.0, 200.0, 1, 0.0), seeds=2)
                    r_tri75 = run_policy(cfg, make_reserve_mp_policy(jobs, 75.0, 185.0, 2, 110.0), seeds=2)
                    if r_smart["win"] == 1.0:
                        tag = f"{h}H(生{int(round(r_smart['alive']))})"
                        break
                    elif r_res65["win"] == 1.0 or r_tri75["win"] == 1.0:
                        best_a = max(
                            r_res65["alive"] if r_res65["win"] == 1.0 else 0,
                            r_tri75["alive"] if r_tri75["win"] == 1.0 else 0,
                        )
                        tag = f"{h}H*(生{int(round(best_a))})"
                        break
                cells.append(f"{tag:15s}")
            print(f"  {dmg:7d} | " + " | ".join(cells))


def audit_hp_range_boundaries() -> None:
    """3. HP スケーリング係数 (N-2)*k の下限 (全員DPS暴走) と上限 (時間切れ) を検証。"""
    print("\n=== 3. ボス HP 適正範囲スキャン (HP = k * (N - 2), ATK=105, AoE=0) ===")
    for n in (4, 6, 10, 16):
        print(f"\n--- N={n} ---")
        for mult in (300, 450, 600, 750, 850, 900, 950, 1000, 1050):
            bhp = mult * (n - 2)
            cfg = RaidConfig(num_agents=n, boss_hp=bhp, boss_atk=105)
            r_all_dps = run_policy(cfg, make_comp_policy([JOB_DPS] * n), seeds=3)
            r_no_t = run_policy(cfg, make_comp_policy([JOB_HEALER] + [JOB_DPS] * (n - 1), healer_mode="smart_aoe"), seeds=3)
            r_11 = run_policy(cfg, make_comp_policy([JOB_TANK, JOB_HEALER] + [JOB_DPS] * (n - 2), healer_mode="smart_aoe"), seeds=3)
            h2_str = "-"
            if n >= 6:
                r_12 = run_policy(cfg, make_comp_policy([JOB_TANK, JOB_HEALER, JOB_HEALER] + [JOB_DPS] * (n - 3), healer_mode="smart_aoe"), seeds=3)
                h2_str = f"{r_12['win']:.0f}"
            print(
                f"  mult={mult:4d} (HP={bhp:5d}) | 全DPS勝={r_all_dps['win']:.0f}(生{r_all_dps['alive']:.0f}) "
                f"| T無勝={r_no_t['win']:.0f} | 1:1勝={r_11['win']:.0f}(残HP={r_11['boss_left']:.0f}) | 1:2勝={h2_str}"
            )


if __name__ == "__main__":
    audit_boss_atk_boundaries()
    audit_aoe_2d_map()
    audit_hp_range_boundaries()
