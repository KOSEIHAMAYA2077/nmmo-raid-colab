"""16種類のボスカタログ（N=4, 6, 10, 16 の計64設定）を現在の RaidConfig に対して全数検証するスクリプト。

検証内容：
1. 各 (boss_id, N) について全 (T, H, D) 構成を 5 種類の戦術で走査する：
   - smart_aoe    : 標準の全体・単体回復
   - reserve_65   : MP温存 (HP<=65でAoE優先)
   - triage_75    : タンク緊急保護付き MP温存 (2人以上HP<=75でAoE)
   - swap_smart   : 複数タンク時に交互挑発（タンクスイッチ）＋ smart_aoe
   - swap_reserve : 複数タンク時に交互挑発＋ triage
2. 勝利編成が 10-seed で安定して勝てるか（環境seedによるタゲ揺れがないか）再確認する。
3. 異常系チェック：全DPS (0:0:N)、タンク無 (0:1:N-1)、ヒーラー無 (1:0:N-1)、挑発サボりが全敗 (0%) になるか確認。
4. 最後に「今のプレイヤー設定（RaidConfig）に対するボス群の適正診断」を集計表示する。
"""
from typing import Dict, List, Tuple

from env import JOB_DPS, JOB_HEALER, JOB_TANK, RaidConfig
from sweep_checks import make_comp_policy, run_policy
from calibrate_bosses import make_reserve_mp_policy
from boss_space_audit import make_tank_swap_policy
from boss_catalog import BOSS_ARCHETYPES


def make_five_tactics(jobs: List[int]) -> Dict[str, callable]:
    """任意の (T, H, D) リストに対して 5 種類の基準スクリプト方策を返す。"""
    return {
        "smart": make_comp_policy(jobs, healer_mode="smart_aoe"),
        "reserve65": make_reserve_mp_policy(jobs, 65.0, 200.0, 1, 0.0),
        "triage75": make_reserve_mp_policy(jobs, 75.0, 185.0, 2, 110.0),
        "swap_smart": make_tank_swap_policy(jobs, healer_mode="smart_aoe", use_reserve=False),
        "swap_res": make_tank_swap_policy(jobs, healer_mode="smart_aoe", use_reserve=True),
    }


def evaluate_encounter(cfg: RaidConfig) -> Dict:
    """1 つの (boss, N) 設定に対して全構成 × 5戦術＋異常系をチェックする。"""
    n = cfg.num_agents
    winners_by_mode: Dict[str, List[Tuple[str, float, float]]] = {
        "smart": [],
        "reserve65": [],
        "triage75": [],
        "swap_smart": [],
        "swap_res": [],
    }
    comp_to_jobs: Dict[str, List[int]] = {}

    # 1次スクリーニング (seeds=1: 確定挙動の絞り込み)
    for t in range(n + 1):
        for h in range(n + 1 - t):
            d = n - t - h
            if d == 0:
                continue  # 火力0は討伐不可なので省略
            jobs = [JOB_TANK] * t + [JOB_HEALER] * h + [JOB_DPS] * d
            label = f"{t}:{h}:{d}"
            comp_to_jobs[label] = jobs
            for mode, fn in make_five_tactics(jobs).items():
                # T<=1 なら swap 系は smart/reserve と同じなので再計算をスキップしてコピー
                if t <= 1 and mode == "swap_smart":
                    if winners_by_mode["smart"] and winners_by_mode["smart"][-1][0] == label:
                        winners_by_mode["swap_smart"].append(winners_by_mode["smart"][-1])
                    continue
                res = run_policy(cfg, fn, seeds=1)
                if res["win"] == 1.0:
                    winners_by_mode[mode].append((label, res["alive"], res["ret"]))

    # 2次確認 (見つかった勝利構成と異常系を seeds=10 で確証)
    verified: Dict[str, List[Tuple[str, float]]] = {m: [] for m in winners_by_mode}
    for mode, items in winners_by_mode.items():
        for label, _, _ in items:
            jobs = comp_to_jobs[label]
            fn = make_five_tactics(jobs)[mode]
            r10 = run_policy(cfg, fn, seeds=10)
            if r10["win"] == 1.0:
                verified[mode].append((label, r10["alive"]))

    # 異常系確認（10 seed）
    all_dps_w = run_policy(cfg, make_comp_policy([JOB_DPS] * n), seeds=10)["win"]
    no_tank_w = run_policy(cfg, make_comp_policy([JOB_HEALER] + [JOB_DPS] * (n - 1), healer_mode="smart_aoe"), seeds=10)["win"]
    no_heal_w = run_policy(cfg, make_comp_policy([JOB_TANK] + [JOB_DPS] * (n - 1)), seeds=10)["win"]
    no_prov_w = run_policy(
        cfg,
        make_comp_policy([JOB_TANK, JOB_HEALER] + [JOB_DPS] * (n - 2), tank_mode="no_provoke", healer_mode="smart_aoe"),
        seeds=10,
    )["win"]

    # サマリー抽出：全体で最も少サポート・標準構成での勝ち方、タンクスイッチ専用勝ち、戦術差
    all_win_comps = sorted({c for lst in verified.values() for c, _ in lst}, key=lambda x: tuple(map(int, x.split(":"))))
    t1_only = [c for c in all_win_comps if c.startswith("1:")]
    t2_swap_comps = [c for c, _ in verified["swap_smart"] + verified["swap_res"] if not c.startswith("1:") and c not in dict(verified["smart"])]
    zero_healer_wins = [c for c in all_win_comps if c.split(":")[1] == "0"]
    zero_tank_wins = [c for c in all_win_comps if c.split(":")[0] == "0"]

    return {
        "verified": verified,
        "all_win_comps": all_win_comps,
        "smart_comps": [c for c, _ in verified["smart"]],
        "reserve_only": [c for c, _ in verified["reserve65"] + verified["triage75"] if c not in dict(verified["smart"])],
        "swap_only": sorted(set(t2_swap_comps)),
        "zero_healer_wins": zero_healer_wins,
        "zero_tank_wins": zero_tank_wins,
        "t1_min_h": min((int(c.split(":")[1]) for c in t1_only), default=99),
        "bad_wins": (all_dps_w, no_tank_w, no_heal_w, no_prov_w),
    }


def main() -> None:
    sizes = (4, 6, 10, 16)
    print("=== ボスカタログ (16種 × 4サイズ = 64条件) 全数検証 ===")
    summary_stats = {
        n: {
            "total": 0, "winnable": 0, "intended": 0, "bad_ok": 0,
            "t1h1": 0, "tactic_diff": 0, "h2plus": 0, "swap_win": 0,
            "zero_h_cheese": 0, "zero_t_cheese": 0,
        }
        for n in sizes
    }

    for arch in BOSS_ARCHETYPES:
        print(f"\n[{arch.boss_id}] {arch.name_ja} ({arch.category}, split={arch.split}, 想定N>={arch.min_n_intended})")
        print(f"  式: HP={arch.hp_per_dps_slot}*(N-2), ATK={arch.boss_atk}, AoE={arch.aoe_dmg}/{arch.aoe_interval}T, T_limit={arch.battle_turns}")
        for n in sizes:
            cfg = arch.make_config(n)
            ev = evaluate_encounter(cfg)
            bad = ev["bad_wins"]
            bad_ok = all(x == 0.0 for x in bad)

            summary_stats[n]["total"] += 1
            if n >= arch.min_n_intended:
                summary_stats[n]["intended"] += 1
            if ev["all_win_comps"]:
                summary_stats[n]["winnable"] += 1
            if bad_ok:
                summary_stats[n]["bad_ok"] += 1
            if f"1:1:{n-2}" in ev["all_win_comps"]:
                summary_stats[n]["t1h1"] += 1
            if ev["reserve_only"]:
                summary_stats[n]["tactic_diff"] += 1
            if ev["all_win_comps"] and f"1:1:{n-2}" not in ev["all_win_comps"] and any(int(c.split(":")[1]) >= 2 for c in ev["all_win_comps"]):
                summary_stats[n]["h2plus"] += 1
            if ev["swap_only"]:
                summary_stats[n]["swap_win"] += 1
            if ev["zero_healer_wins"]:
                summary_stats[n]["zero_h_cheese"] += 1
            if ev["zero_tank_wins"]:
                summary_stats[n]["zero_t_cheese"] += 1

            # 見やすく1行要約を表示
            smart_s = [f"{c}(生{a:.0f})" for c, a in ev["verified"]["smart"][:4]]
            triage_s = sorted({f"{c}" for c, _ in ev["verified"]["reserve65"] + ev["verified"]["triage75"]} - set(ev["smart_comps"]))
            swap_s = ev["swap_only"][:4]
            status_tag = "OK" if (ev["all_win_comps"] and bad_ok) else ("非想定N" if n < arch.min_n_intended else "要調整!")
            cheese_s = f" 0H勝={ev['zero_healer_wins'][:2]}" if ev["zero_healer_wins"] else ""
            print(
                f"  N={n:2d} HP={cfg.boss_hp:5d} [{status_tag:4s}] | 異常系全敗={bad_ok} "
                f"| smart勝={len(ev['smart_comps']):2d}{smart_s} "
                f"| 温存追加={triage_s[:3]} | 交代追加={swap_s}{cheese_s}"
            )

    print("\n=== 今のプレイヤー設定 (RaidConfig) に対する適正診断サマリー ===")
    for n in sizes:
        st = summary_stats[n]
        print(
            f"  N={n:2d}: 攻略可能={st['winnable']:2d}/{st['total']} (想定内={st['intended']}) "
            f"| 異常系(全DPS等)全敗={st['bad_ok']}/{st['total']} "
            f"| 1:1勝={st['t1h1']} | 温存で勝敗変化={st['tactic_diff']} "
            f"| 2H+必須={st['h2plus']} | 2T交代専用={st['swap_win']} "
            f"| H=0複数T抜道={st['zero_h_cheese']} | T=0抜道={st['zero_t_cheese']}"
        )


if __name__ == "__main__":
    main()
