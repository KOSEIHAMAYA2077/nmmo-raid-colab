"""4・6・10・16人レイド用のボス設定（HP / 単体攻撃力 / AoE周期・威力）を精密検証するスクリプト。

調べるポイント：
1. 単体攻撃ボス：全4サイズ (4, 6, 10, 16) で「全員DPS・Tなし・Hなし・挑発サボり」が全敗し、
   1:1:(N-2) が安定勝利する HP スケーリング式（850*(N-2) 〜 920*(N-2)）の比較。
2. AoE ボス（全4サイズ対応）：N=4, 6 でも全滅せず、かつ回復負荷・人数差が出る設定の探索
   （穏やか AoE: 1Hでも工夫で勝てる / 強AoE: 複数Hが必要）。
3. 選抜した候補ボスセットに対する 30-seed 厳密確認（勝率・残ターン・生存数・全サボり条件）。
"""
from typing import Callable, Dict, List, Tuple
import numpy as np

from env import (
    ACT_BASIC,
    ACT_SKILL,
    ACT_SKILL2,
    JOB_DPS,
    JOB_HEALER,
    JOB_TANK,
    MinimalRaidEnv,
    RaidConfig,
)
from sweep_checks import make_comp_policy, run_policy


def make_reserve_mp_policy(
    target_jobs: List[int],
    dps_hp_thresh: float = 65.0,
    tank_hp_thresh: float = 200.0,
    min_wounded_for_aoe: int = 1,
    tank_emergency_hp: float = 110.0,
):
    """汎用人数対応版の MP 温存ヒーラー方策。

    - タンクが緊急ライン（tank_emergency_hp 以下）なら単体回復を最優先して即死を防ぐ。
    - そうでなく、dps_hp_thresh 以下の非タンク味方が min_wounded_for_aoe 人以上いれば、
      全体回復の MP (35) が貯まるまで通常攻撃で待ってから全体回復を撃つ。
    - そうでなくタンクが tank_hp_thresh 以下なら単体回復する。
    """
    base_fn = make_comp_policy(target_jobs, healer_mode="smart_aoe")

    def fn(env: MinimalRaidEnv, obs: Dict, seed: int) -> List[int]:
        actions = base_fn(env, obs, seed)
        if obs["in_prep"]:
            return actions

        cfg = env.cfg
        n = cfg.num_agents
        wounded_cnt = sum(
            1
            for k in range(n)
            if obs["alive"][k] and obs["jobs"][k] != JOB_TANK and obs["hp"][k] <= dps_hp_thresh
        )
        living_tanks = [k for k in range(n) if obs["alive"][k] and obs["jobs"][k] == JOB_TANK]
        min_tank_hp = min((obs["hp"][k] for k in living_tanks), default=9999.0)

        for i in range(n):
            if not obs["alive"][i] or obs["jobs"][i] != JOB_HEALER:
                continue
            mp = obs["mp"][i]
            actions[i] = ACT_BASIC
            if min_tank_hp <= tank_emergency_hp and obs["skill_cd"][i] == 0 and mp >= cfg.healer_heal_mp:
                actions[i] = ACT_SKILL
            elif wounded_cnt >= min_wounded_for_aoe:
                if obs["skill2_cd"][i] == 0 and mp >= cfg.healer_aoe_mp:
                    actions[i] = ACT_SKILL2
            elif min_tank_hp <= tank_hp_thresh and obs["skill_cd"][i] == 0 and mp >= cfg.healer_heal_mp:
                actions[i] = ACT_SKILL
        return actions

    return fn


def get_policies(jobs: List[int]) -> Dict[str, Callable]:
    """比較する標準ヒーラー方策 4 種類を返す。"""
    return {
        "smart_aoe": make_comp_policy(jobs, healer_mode="smart_aoe"),
        "reserve_65": make_reserve_mp_policy(jobs, dps_hp_thresh=65.0, tank_hp_thresh=200.0, tank_emergency_hp=0.0),
        "triage_65": make_reserve_mp_policy(jobs, dps_hp_thresh=65.0, tank_hp_thresh=180.0, min_wounded_for_aoe=2, tank_emergency_hp=110.0),
        "triage_75": make_reserve_mp_policy(jobs, dps_hp_thresh=75.0, tank_hp_thresh=180.0, min_wounded_for_aoe=2, tank_emergency_hp=110.0),
    }


def eval_all_comps(cfg: RaidConfig, seeds: int = 5) -> Dict[str, List[Tuple[str, float, float]]]:
    """全 (T, H, D) 構成を 4 方策で実行し、全勝 (win == 1.0) 構成を返す。"""
    n = cfg.num_agents
    out: Dict[str, List[Tuple[str, float, float]]] = {
        "smart_aoe": [],
        "reserve_65": [],
        "triage_65": [],
        "triage_75": [],
    }
    for t in range(n + 1):
        for h in range(n + 1 - t):
            d = n - t - h
            jobs = [JOB_TANK] * t + [JOB_HEALER] * h + [JOB_DPS] * d
            label = f"{t}:{h}:{d}"
            for mode, fn in get_policies(jobs).items():
                res = run_policy(cfg, fn, seeds=seeds)
                if res["win"] == 1.0:
                    out[mode].append((label, res["alive"], res["ret"]))
    return out


def sweep_single_target_formulas() -> None:
    """1. 単体ボス（AoEなし）：850*(N-2) vs 880*(N-2) vs 920*(N-2) と 1:1:(N-2) の余裕度。"""
    print("=== 1. 単体攻撃ボス（AoE なし）スケーリング比較 ===")
    for k in (850, 880, 920):
        print(f"\n--- 式: boss_hp = {k} * (N - 2), boss_atk = 105, AoEなし ---")
        for n in (4, 6, 10, 16):
            bhp = k * (n - 2)
            cfg = RaidConfig(num_agents=n, boss_hp=bhp, boss_atk=105)
            out = eval_all_comps(cfg, seeds=5)
            winners = [f"{c}(生{a:.0f})" for c, a, _ in out["smart_aoe"]]
            # 代表 1:1:(N-2) とその次点 (1:2:(N-3) など) の残り HP/余裕を確認
            bal_jobs = [JOB_TANK, JOB_HEALER] + [JOB_DPS] * (n - 2)
            bal_res = run_policy(cfg, make_comp_policy(bal_jobs, healer_mode="smart_aoe"), seeds=5)
            no_prov = run_policy(cfg, make_comp_policy(bal_jobs, tank_mode="no_provoke", healer_mode="smart_aoe"), seeds=5)
            all_dps = run_policy(cfg, make_comp_policy([JOB_DPS] * n), seeds=5)
            print(
                f"  N={n:2d} HP={bhp:5d} | 全勝構成={winners} "
                f"| 1:1:{n-2}勝={bal_res['win']:.0f}(生{bal_res['alive']:.1f}) "
                f"| 挑発なし勝={no_prov['win']:.0f} 全DPS勝={all_dps['win']:.0f}"
            )


def sweep_unified_aoe_formulas() -> None:
    """2. N=4, 6, 10, 16 すべてが攻略可能な AoE ボス条件の探索。"""
    print("\n=== 2. AoE ボス探索（N=4, 6, 10, 16 すべてを 4 方策でチェック） ===")
    # A. 穏やか〜中程度 AoE: 1:1:(N-2) でも工夫すれば全サイズ勝てるか？
    unified_candidates = [
        ("弱AoE_25/5T_HP820*(N-2)", lambda n: (820 * (n - 2), 105, 25, 5)),
        ("中AoE_30/5T_HP780*(N-2)", lambda n: (780 * (n - 2), 105, 30, 5)),
        ("中AoE_30/4T_HP740*(N-2)", lambda n: (740 * (n - 2), 105, 30, 4)),
        ("強AoE_35/4T_HP680*(N-2)", lambda n: (680 * (n - 2), 105, 35, 4)),
        ("強AoE_40/4T_HP600*(N-2)", lambda n: (600 * (n - 2), 105, 40, 4)),
        # N=4 の 1:2:1 (最大ダメ ~1050) も通る低HP強AoE (ただし 0:0:4 の最大ダメは ~350)
        ("2H許容AoE_35/4T_HP=250*N-50", lambda n: (max(950, 720 * (n - 2) - 500), 105, 35, 4)),
    ]
    for title, fn_p in unified_candidates:
        print(f"\n--- {title} ---")
        for n in (4, 6, 10, 16):
            bhp, batk, admg, aint = fn_p(n)
            cfg = RaidConfig(
                num_agents=n, boss_hp=bhp, boss_atk=batk,
                boss_aoe_dmg=admg, boss_aoe_interval=aint,
            )
            out = eval_all_comps(cfg, seeds=3)
            all_dps = run_policy(cfg, make_comp_policy([JOB_DPS] * n), seeds=3)
            print(f"  N={n:2d} (HP={bhp:5d}, ATK={batk}, AoE={admg}/{aint}T) 全DPS勝={all_dps['win']:.0f}")
            for m in ("smart_aoe", "reserve_65", "triage_65", "triage_75"):
                # ヒートマップ代わりに最小 H 構成と代表例を最大 6 つ表示
                w_str = ", ".join(f"{c}(生{a:.0f})" for c, a, _ in out[m][:6])
                if len(out[m]) > 6:
                    w_str += f" ... 全{len(out[m])}通り"
                print(f"    {m:10s}: {w_str if w_str else '勝利なし'}")


def verify_recommended_presets() -> None:
    """3. 最終候補セット（サイズごとの「単体ボス」と「AoEボス」）を 20 seed で精密検証。"""
    print("\n=== 3. 推奨ボス候補セットの 20-seed 精密検証 ===")
    # 候補A：統一単体ボス  HP = 880*(N-2), ATK = 105
    # 候補B：中AoEボス（1Hでも戦術で補える／2Hで全員生存）HP = 780*(N-2), ATK = 105, AoE = 30/5T (or 6,10,16は35/4T)
    # 候補C：サイズ別「2H要求強AoE（6/10/16人用）」HP = 2700 / 5500 / 11000, ATK = 110〜125, AoE = 45/4T
    presets = [
        ("N= 4 単体     ", RaidConfig(num_agents=4, boss_hp=1760, boss_atk=105)),
        ("N= 4 弱AoE    ", RaidConfig(num_agents=4, boss_hp=1640, boss_atk=105, boss_aoe_dmg=25, boss_aoe_interval=5)),
        ("N= 4 中AoE    ", RaidConfig(num_agents=4, boss_hp=1560, boss_atk=105, boss_aoe_dmg=30, boss_aoe_interval=5)),
        ("N= 6 単体     ", RaidConfig(num_agents=6, boss_hp=3520, boss_atk=105)),
        ("N= 6 中AoE    ", RaidConfig(num_agents=6, boss_hp=3120, boss_atk=105, boss_aoe_dmg=30, boss_aoe_interval=5)),
        ("N= 6 強AoE(2H)", RaidConfig(num_agents=6, boss_hp=2700, boss_atk=110, boss_aoe_dmg=45, boss_aoe_interval=4)),
        ("N=10 単体     ", RaidConfig(num_agents=10, boss_hp=7040, boss_atk=105)),
        ("N=10 中AoE    ", RaidConfig(num_agents=10, boss_hp=5500, boss_atk=110, boss_aoe_dmg=35, boss_aoe_interval=4)),
        ("N=10 強AoE(2H)", RaidConfig(num_agents=10, boss_hp=5500, boss_atk=125, boss_aoe_dmg=45, boss_aoe_interval=4)),
        ("N=16 単体     ", RaidConfig(num_agents=16, boss_hp=12320, boss_atk=105)),
        ("N=16 中AoE    ", RaidConfig(num_agents=16, boss_hp=10000, boss_atk=105, boss_aoe_dmg=35, boss_aoe_interval=4)),
        ("N=16 強AoE(2H)", RaidConfig(num_agents=16, boss_hp=11000, boss_atk=125, boss_aoe_dmg=45, boss_aoe_interval=4)),
    ]
    for name, cfg in presets:
        n = cfg.num_agents
        out = eval_all_comps(cfg, seeds=20)
        # ロール欠如・サボり確認 (1:1:(N-2) 基準)
        j_11 = [JOB_TANK, JOB_HEALER] + [JOB_DPS] * (n - 2)
        all_dps = run_policy(cfg, make_comp_policy([JOB_DPS] * n), seeds=20)
        no_t = run_policy(cfg, make_comp_policy([JOB_HEALER] + [JOB_DPS] * (n - 1), healer_mode="smart_aoe"), seeds=20)
        no_h = run_policy(cfg, make_comp_policy([JOB_TANK] + [JOB_DPS] * (n - 1)), seeds=20)
        no_prov = run_policy(cfg, make_comp_policy(j_11, tank_mode="no_provoke", healer_mode="smart_aoe"), seeds=20)
        print(
            f"\n[{name}] N={n:2d} HP={cfg.boss_hp:5d} ATK={cfg.boss_atk} "
            f"AoE={cfg.boss_aoe_dmg}/{cfg.boss_aoe_interval}T | "
            f"異常系勝率(全DPS={all_dps['win']:.0f}/T無={no_t['win']:.0f}/H無={no_h['win']:.0f}/挑発無={no_prov['win']:.0f})"
        )
        for m in ("smart_aoe", "reserve_65", "triage_65", "triage_75"):
            # H 人数別に整理して見やすく出力（代表で最小 T のみ、または全勝すべて）
            items = [(c, int(round(a))) for c, a, _ in out[m]]
            print(f"  {m:10s} ({len(items):2d}): {items[:8]}{' ...' if len(items)>8 else ''}")


if __name__ == "__main__":
    sweep_single_target_formulas()
    sweep_unified_aoe_formulas()
    verify_recommended_presets()
