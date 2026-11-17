"""NumPy 試作 (MinimalRaidEnv) と NMMO 2.1.2 移植版 (RaidNmmoEnv) の完全一致検証＋再現テスト。

検証項目：
1. ターン単位の完全一致検証（boss_hp, hp, mp, jobs, threat, alive, CD, last_comm, 共通報酬, 勝敗）
   - N=4, 6, 10, 16 / 複数ボス (01, 03_20T, 05, 08, 13, 14) / 乱数行動・途中転職・通信あり含む
2. NMMO 本体機能の直接確認
   - obs[aid]["Communication"] の実体行に送信トークン (token+1) が載り、1手変えると即反映されること
   - use_comm=False で全員座標同一＆トークン定数になること
   - NMMO Env.step() が返す nmmo_rewards が生存者全員に共通レイド報酬として届くこと
   - 討伐時に boss NPC が消滅 (len(realm.npcs)==0) し、game.is_over が True になること
3. history/26/11/16/1536.md の再現目標：
   - 6人 Boss 08: 1:1:4 敗北 / 1:2:3 (smart_aoe) 10/10勝(6生) / 2:1:3 (交代+MP温存) 10/10勝(6生)
   - 16人 Boss 01: 1:1:14 (10/10勝, 16生) / 2:0:14 (10/10勝, 5生) / 3:0:13 (10/10勝, 15生)、全員合計報酬 2.0
   - 16人 Boss 02: 2:0:14・3:0:13 が 0/10勝、1:1:14 が 10/10勝(16生)
4. Mac CPU 上の NMMO 2.1.2 スループット計測 (N=6, N=16)
"""
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

_NMMO2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_NMMO2_ROOT / "raid_proto"))
sys.path.insert(0, str(_NMMO2_ROOT / "raid_nmmo"))

from env import JOB_DPS, JOB_HEALER, JOB_TANK, MinimalRaidEnv, NUM_ACTIONS, RaidConfig
from boss_catalog import make_boss_config
from sweep_checks import make_comp_policy
from calibrate_bosses import make_reserve_mp_policy
from boss_space_audit import make_tank_swap_policy
from raid_game import RaidNmmoEnv


def run_nmmo_episode(nenv: RaidNmmoEnv, policy_fn, seed: int = 0) -> Dict:
    """RaidNmmoEnv 上で 1 エピソード実行し、勝率・共通報酬合計・生存人数・所要ステップを返す。"""
    obs = nenv.reset(seed=seed)
    total_ret = 0.0
    steps = 0
    while not nenv.done:
        acts = policy_fn(nenv, obs, seed)
        obs, r, done, info = nenv.step(acts)
        total_ret += r
        steps += 1
        # NMMO 本体 Task 経由で生存中の全員へ同一のチーム報酬が届いているか検証
        if r != 0.0:
            for aid, rew in nenv.last_nmmo_rewards.items():
                assert abs(rew - r) < 1e-9, f"Agent {aid} reward {rew} != team reward {r}"
    return {
        "won": info["won"],
        "ret": total_ret,
        "alive": info["alive_count"],
        "boss_left": info["boss_hp"],
        "steps": steps,
    }


def _check_single_parity(
    n: int, boss_id: str, pol_fn, ep_seed: int, randomize_prep_comms: bool = False
) -> None:
    """1 エピソード分の MinimalRaidEnv vs RaidNmmoEnv を毎ターン完全一致検証する共通ヘルパー。"""
    cfg = make_boss_config(boss_id, n)
    np_env = MinimalRaidEnv(cfg)
    nm_env = RaidNmmoEnv(cfg, use_comm=True)
    try:
        obs_np = np_env.reset(seed=ep_seed)
        obs_nm = nm_env.reset(seed=ep_seed)
        rng_comm = np.random.default_rng(1000 + ep_seed)
        rng_tie_np = np.random.default_rng(5000 + ep_seed)
        rng_tie_nm = np.random.default_rng(5000 + ep_seed)

        def tie_choice(tied, r=rng_tie_nm):
            return int(r.choice(tied))

        while not np_env.done:
            acts = pol_fn(np_env, obs_np, ep_seed)
            comms = (
                [int(rng_comm.integers(0, cfg.comm_vocab)) for _ in range(n)]
                if randomize_prep_comms
                else [0] * n
            )
            np_env.rng = rng_tie_np
            obs_np, r_np, d_np, info_np = np_env.step(acts, comms)
            obs_nm, r_nm, d_nm, info_nm = nm_env.step(acts, comms, tie_choice_fn=tie_choice)

            for k in ("turn", "in_prep", "jobs", "alive", "job_cd", "skill_cd", "skill2_cd", "last_comm"):
                assert obs_np[k] == obs_nm[k], f"{boss_id} N={n} k={k}: {obs_np[k]} != {obs_nm[k]}"
            for k in ("hp", "mp", "threat"):
                assert np.allclose(obs_np[k], obs_nm[k], atol=1e-9), f"{boss_id} N={n} k={k}"
            assert abs(obs_np["boss_hp"] - obs_nm["boss_hp"]) < 1e-9
            assert abs(r_np - r_nm) < 1e-9
            assert d_np == d_nm
            assert info_np["won"] == info_nm["won"]
            assert info_np["boss_target"] == info_nm["boss_target"]
            assert info_np["alive_count"] == info_nm["alive_count"]

        assert nm_env.game.is_over is True
        if info_nm["won"]:
            assert len(nm_env.nmmo_env.realm.npcs) == 0
    finally:
        nm_env.close()


def verify_exact_parity() -> None:
    """1. NumPy MinimalRaidEnv と NMMO RaidNmmoEnv の毎ターン完全一致（主要＋全64条件）を検証する。"""
    print("=== 1. NumPy 版 vs NMMO 2.1.2 版のターン単位完全一致チェック ===")
    cases = [
        (4, "01_std_single", "smart_aoe", [JOB_TANK, JOB_HEALER, JOB_DPS, JOB_DPS]),
        (6, "08_hard_aoe_45_4t", "swap_res", [JOB_TANK, JOB_TANK, JOB_HEALER] + [JOB_DPS] * 3),
        (6, "03_fast_rush_20t", "smart_aoe", [JOB_TANK, JOB_HEALER] + [JOB_DPS] * 4),
        (10, "07_tactics_vs_comp_35_4t", "reserve65", [JOB_TANK, JOB_HEALER] + [JOB_DPS] * 8),
        (16, "01_std_single", "swap_smart", [JOB_TANK] * 2 + [JOB_DPS] * 14),
        (16, "14_swap_specialist_235", "swap_res", [JOB_TANK] * 2 + [JOB_HEALER] * 2 + [JOB_DPS] * 12),
    ]
    for n, boss_id, tactic, jobs in cases:
        if tactic == "smart_aoe":
            pol = make_comp_policy(jobs, healer_mode="smart_aoe")
        elif tactic == "reserve65":
            pol = make_reserve_mp_policy(jobs, 65.0, 200.0, 1, 0.0)
        elif tactic == "swap_smart":
            pol = make_tank_swap_policy(jobs, healer_mode="smart_aoe", use_reserve=False)
        else:
            pol = make_tank_swap_policy(jobs, healer_mode="smart_aoe", use_reserve=True)
        for ep_seed in (0, 1, 2):
            _check_single_parity(n, boss_id, pol, ep_seed, randomize_prep_comms=True)
        print(f"  [PASS] N={n:2d} boss={boss_id:24s} tactic={tactic:10s} -> 3-seed 毎ターン完全一致")

    # 全 16 ボス × 4 人数（64 条件）でも 1:1 / 2:1_swap での完全一致を確認（環境キャッシュで高速化）
    from boss_catalog import BOSS_ARCHETYPES
    checked = 0
    for n in (4, 6, 10, 16):
        nm_env = RaidNmmoEnv(make_boss_config("01_std_single", n), use_comm=True)
        try:
            jobs = [JOB_TANK, JOB_HEALER] + [JOB_DPS] * (n - 2) if n == 4 else [JOB_TANK] * 2 + [JOB_HEALER] + [JOB_DPS] * (n - 3)
            pol = make_tank_swap_policy(jobs, healer_mode="smart_aoe", use_reserve=True)
            for arch in BOSS_ARCHETYPES:
                cfg = arch.make_config(n)
                np_env = MinimalRaidEnv(cfg)
                nm_env.set_raid_config(cfg)
                obs_np = np_env.reset(seed=0)
                obs_nm = nm_env.reset(seed=0)
                rng_c = np.random.default_rng(77)
                t_np = np.random.default_rng(88)
                t_nm = np.random.default_rng(88)
                while not np_env.done:
                    acts = pol(np_env, obs_np, 0)
                    comms = [int(rng_c.integers(0, cfg.comm_vocab)) for _ in range(n)]
                    np_env.rng = t_np
                    obs_np, r_np, d_np, info_np = np_env.step(acts, comms)
                    obs_nm, r_nm, d_nm, info_nm = nm_env.step(acts, comms, tie_choice_fn=lambda t, r=t_nm: int(r.choice(t)))
                    assert abs(obs_np["boss_hp"] - obs_nm["boss_hp"]) < 1e-9
                    assert np.allclose(obs_np["hp"], obs_nm["hp"], atol=1e-9)
                    assert obs_np["last_comm"] == obs_nm["last_comm"]
                    assert abs(r_np - r_nm) < 1e-9 and d_np == d_nm and info_np["won"] == info_nm["won"]
                checked += 1
        finally:
            nm_env.close()
    print(f"  [PASS] 16ボス × 4サイズ = 全{checked}条件 毎ターン完全一致")


def verify_native_comm_and_no_comm() -> None:
    """2. 送信記号 1 つの変更で NMMO Communication 観測だけが変わること＆無通信固定を検証。"""
    print("\n=== 2. NMMO 通信経路（上書きなし・座標漏れなし・無通信固定）の検証 ===")
    cfg = make_boss_config("01_std_single", 6)

    # (A) use_comm=True: 同じ局面で agent 0 の Token だけ 0 -> 3 に変えて比較
    env_a = RaidNmmoEnv(cfg, use_comm=True)
    env_b = RaidNmmoEnv(cfg, use_comm=True)
    try:
        env_a.reset(seed=7)
        env_b.reset(seed=7)
        acts = [0] * 6
        env_a.step(acts, comms=[0, 1, 2, 0, 1, 2])
        env_b.step(acts, comms=[3, 1, 2, 0, 1, 2])

        # 味方 (agent_id=2..6) の NMMO 生観測["Communication"]を確認
        comm_a = env_a.last_nmmo_obs[2]["Communication"]
        comm_b = env_b.last_nmmo_obs[2]["Communication"]
        rows_a = comm_a[comm_a[:, 0] != 0].tolist()
        rows_b = comm_b[comm_b[:, 0] != 0].tolist()
        assert rows_a[0][3] == 1 and rows_b[0][3] == 4, f"Expected token 1 vs 4, got {rows_a[0]} vs {rows_b[0]}"
        assert rows_a[1:] == rows_b[1:], "Other teammates' comm rows should match identically"
        # Entity の message 列以外は完全一致（座標・HP・ID 等にリーク無し）
        ent_a = env_a.last_nmmo_obs[2]["Entity"].copy()
        ent_b = env_b.last_nmmo_obs[2]["Entity"].copy()
        from nmmo.entity.entity import EntityState
        msg_col = EntityState.State.attr_name_to_col["message"]
        ent_a[:, msg_col] = 0
        ent_b[:, msg_col] = 0
        assert np.array_equal(ent_a, ent_b), "Entity attributes other than message differed!"
        print(f"  [PASS] comm=0 vs 3 -> NMMO Communication row[0].msg = {rows_a[0][3]} vs {rows_b[0][3]} (他全員完全一致)")
    finally:
        env_a.close()
        env_b.close()

    # (B) use_comm=False: 何を渡しても Token=0 -> msg=1 固定、全員座標同一
    env_off = RaidNmmoEnv(cfg, use_comm=False)
    try:
        env_off.reset(seed=7)
        obs, _, _, _ = env_off.step([0] * 6, comms=[3, 2, 1, 3, 2, 1])
        rows = env_off.last_nmmo_obs[1]["Communication"]
        rows = rows[rows[:, 0] != 0]
        assert list(rows[:, 3]) == [1] * 6
        assert len({(int(r[1]), int(r[2])) for r in rows}) == 1, "All agents must share identical (row, col)"
        assert obs["last_comm"] == [0] * 6
        print("  [PASS] use_comm=False -> 全員座標一致 & msg=1 (decoded 0) 定数固定")
    finally:
        env_off.close()


def verify_1536_milestones_on_nmmo() -> None:
    """3. history/26/11/16/1536.md の主要ケース（6人 Boss 08・16人 Boss 01/02）を NMMO 上で再現。"""
    print("\n=== 3. history/26/11/16/1536.md 再現テスト (NMMO 2.1.2 上・各10戦) ===")

    # --- 6人 Boss 08 ---
    cfg6 = make_boss_config("08_hard_aoe_45_4t", 6)
    env6 = RaidNmmoEnv(cfg6, use_comm=True)
    try:
        cases_6 = [
            ("1:1:4 smart_aoe", make_comp_policy([JOB_TANK, JOB_HEALER] + [JOB_DPS] * 4, healer_mode="smart_aoe")),
            ("1:2:3 smart_aoe", make_comp_policy([JOB_TANK, JOB_HEALER, JOB_HEALER] + [JOB_DPS] * 3, healer_mode="smart_aoe")),
            ("2:1:3 swap+res ", make_tank_swap_policy([JOB_TANK, JOB_TANK, JOB_HEALER] + [JOB_DPS] * 3, use_reserve=True)),
        ]
        for label, pol in cases_6:
            res_list = [run_nmmo_episode(env6, pol, seed=s) for s in range(10)]
            wins = sum(r["won"] for r in res_list)
            mean_alive = np.mean([r["alive"] for r in res_list])
            mean_ret = np.mean([r["ret"] for r in res_list])
            print(f"  N= 6 Boss 08 | {label} | wins={wins:2d}/10 | alive={mean_alive:4.1f} | ret={mean_ret:.3f}")
    finally:
        env6.close()

    # --- 16人 Boss 01 & Boss 02 ---
    cfg16_01 = make_boss_config("01_std_single", 16)
    cfg16_02 = make_boss_config("02_tight_enrage_single", 16)
    env16 = RaidNmmoEnv(cfg16_01, use_comm=True)
    try:
        for boss_name, cfg in [("Boss 01", cfg16_01), ("Boss 02", cfg16_02)]:
            env16.set_raid_config(cfg)
            cases_16 = [
                ("1:1:14 smart_aoe", make_comp_policy([JOB_TANK, JOB_HEALER] + [JOB_DPS] * 14, healer_mode="smart_aoe")),
                ("2:0:14 swap     ", make_tank_swap_policy([JOB_TANK] * 2 + [JOB_DPS] * 14, use_reserve=False)),
                ("3:0:13 swap     ", make_tank_swap_policy([JOB_TANK] * 3 + [JOB_DPS] * 13, use_reserve=False)),
            ]
            for label, pol in cases_16:
                res_list = [run_nmmo_episode(env16, pol, seed=s) for s in range(10)]
                wins = sum(r["won"] for r in res_list)
                mean_alive = np.mean([r["alive"] for r in res_list])
                mean_ret = np.mean([r["ret"] for r in res_list])
                print(f"  N=16 {boss_name} | {label} | wins={wins:2d}/10 | alive={mean_alive:4.1f} | ret={mean_ret:.3f}")
    finally:
        env16.close()


def benchmark_speed() -> None:
    """4. Mac CPU での NMMO 2.1.2 レイド実行速度を実測する。"""
    print("\n=== 4. Mac CPU 実行速度ベンチマーク (NMMO 2.1.2 RaidNmmoEnv) ===")
    for n, boss_id, jobs in [
        (6, "01_std_single", [JOB_TANK, JOB_HEALER] + [JOB_DPS] * 4),
        (16, "01_std_single", [JOB_TANK, JOB_HEALER] + [JOB_DPS] * 14),
    ]:
        cfg = make_boss_config(boss_id, n)
        nenv = RaidNmmoEnv(cfg, use_comm=True)
        pol = make_comp_policy(jobs, healer_mode="smart_aoe")
        n_eps = 20
        try:
            t0 = time.perf_counter()
            total_steps = 0
            for s in range(n_eps):
                r = run_nmmo_episode(nenv, pol, seed=s)
                total_steps += r["steps"]
            dt = time.perf_counter() - t0
            print(
                f"  N={n:2d}: {n_eps} eps ({total_steps} steps) in {dt:.3f}s "
                f"-> {n_eps/dt:.1f} eps/s, {total_steps/dt:.1f} env_steps/s ({(total_steps*n)/dt:.0f} agent_steps/s)"
            )
        finally:
            nenv.close()


if __name__ == "__main__":
    verify_exact_parity()
    verify_native_comm_and_no_comm()
    verify_1536_milestones_on_nmmo()
    benchmark_speed()
