"""MinimalRaidEnv の手書き方策による大量検証スクリプト（PyTorch 導入前の環境テスト）。

確認する項目：
1. 人数スケーリング（N=4, 5, 6, 8, 10）と最適編成比
2. 敵視・サボり・過剰ヒール（挑発なし、タンク放置、回復連打など）
3. ジョブチェンジ CD（0..999）と途中交代
4. ランダム行動ベースラインのスコア
5. ボス AoE ＋ ヒーラーの弱い AoE 回復の使い分け
6. 回復スキルに CD あり（例: 単体 CD=1, AoE CD=3）vs CD なし（MP のみで制御）の比較
"""
from typing import Dict, List
import numpy as np

from env import (
    ACT_BASIC,
    ACT_JOB_TANK,
    ACT_SKILL,
    ACT_SKILL2,
    ACT_WAIT,
    JOB_DPS,
    JOB_HEALER,
    JOB_TANK,
    MinimalRaidEnv,
    NUM_ACTIONS,
    RaidConfig,
)


def max_hp_of(cfg: RaidConfig, job: int) -> float:
    """ジョブごとの最大 HP を返す補助関数。"""
    if job == JOB_TANK:
        return float(cfg.tank_hp)
    if job == JOB_HEALER:
        return float(cfg.healer_hp)
    return float(cfg.dps_hp)


def run_policy(cfg: RaidConfig, policy_fn, seeds: int = 10) -> Dict[str, float]:
    """指定方策を複数 seed で実行し、勝率・共通報酬・残HP・タゲ移動回数などを集計する。"""
    wins, rets, boss_left, alive_end, target_switches = [], [], [], [], []
    tank_deaths, healer_deaths, dps_deaths = [], [], []
    for s in range(seeds):
        env = MinimalRaidEnv(cfg)
        obs = env.reset(seed=s)
        total_r = 0.0
        prev_target = -1
        switches = 0
        while not env.done:
            acts = policy_fn(env, obs, s)
            obs, r, done, info = env.step(acts)
            total_r += r
            bt = info["boss_target"]
            if bt != -1 and prev_target != -1 and bt != prev_target:
                switches += 1
            if bt != -1:
                prev_target = bt

        wins.append(1.0 if info["won"] else 0.0)
        rets.append(total_r)
        boss_left.append(info["boss_hp"])
        alive_end.append(info["alive_count"])
        target_switches.append(switches)
        tank_deaths.append(sum(1 for i, a in enumerate(obs["alive"]) if not a and obs["jobs"][i] == JOB_TANK))
        healer_deaths.append(sum(1 for i, a in enumerate(obs["alive"]) if not a and obs["jobs"][i] == JOB_HEALER))
        dps_deaths.append(sum(1 for i, a in enumerate(obs["alive"]) if not a and obs["jobs"][i] == JOB_DPS))

    return {
        "win": float(np.mean(wins)),
        "ret": float(np.mean(rets)),
        "boss_left": float(np.mean(boss_left)),
        "alive": float(np.mean(alive_end)),
        "switches": float(np.mean(target_switches)),
        "td": float(np.mean(tank_deaths)),
        "hd": float(np.mean(healer_deaths)),
        "dd": float(np.mean(dps_deaths)),
    }


def make_comp_policy(
    target_jobs: List[int],
    heal_threshold: float = 0.85,
    tank_mode: str = "normal",   # normal | lazy | no_provoke
    healer_mode: str = "smart",  # smart | smart_aoe | spam_aoe | spam_heal | attack_only
    fallback_tank_idx: int = -1,
):
    """固定編成で動く手書き方策を返す（行動ルールは条件ごとに切り替え可能）。"""
    def fn(env: MinimalRaidEnv, obs: Dict, seed: int) -> List[int]:
        cfg = env.cfg
        n = cfg.num_agents
        living_tanks = sum(1 for k in range(n) if obs["alive"][k] and obs["jobs"][k] == JOB_TANK)
        acts = []
        for i in range(n):
            if not obs["alive"][i]:
                acts.append(ACT_WAIT)
                continue

            # 目標ジョブと違えば準備中（または CD 明け）に転職する
            want = target_jobs[i]
            if fallback_tank_idx == i and not obs["in_prep"] and living_tanks == 0:
                want = JOB_TANK
            if obs["jobs"][i] != want and (obs["in_prep"] or obs["job_cd"][i] == 0):
                acts.append(ACT_JOB_TANK + want)
                continue

            if obs["in_prep"]:
                acts.append(ACT_WAIT)
                continue

            job = obs["jobs"][i]
            if job == JOB_TANK:
                if tank_mode == "lazy":
                    acts.append(ACT_WAIT)
                elif tank_mode == "no_provoke":
                    acts.append(ACT_BASIC)
                else:
                    # 挑発が撃てれば挑発、CD 中や MP 不足なら通常攻撃で敵視を稼ぐ
                    if obs["skill_cd"][i] == 0 and obs["mp"][i] >= cfg.tank_provoke_mp:
                        acts.append(ACT_SKILL)
                    else:
                        acts.append(ACT_BASIC)
            elif job == JOB_HEALER:
                if healer_mode == "attack_only":
                    acts.append(ACT_BASIC)
                elif healer_mode == "spam_aoe":
                    if obs["skill2_cd"][i] == 0 and obs["mp"][i] >= cfg.healer_aoe_mp:
                        acts.append(ACT_SKILL2)
                    else:
                        acts.append(ACT_BASIC)
                elif healer_mode == "spam_heal":
                    if obs["skill_cd"][i] == 0 and obs["mp"][i] >= cfg.healer_heal_mp:
                        acts.append(ACT_SKILL)
                    else:
                        acts.append(ACT_BASIC)
                elif healer_mode == "smart_aoe":
                    # 3 人以上削れていれば AoE 回復、1〜2 人なら単体回復、誰も減っていなければ攻撃
                    hurt = [
                        k for k in range(n)
                        if obs["alive"][k] and (obs["hp"][k] / max_hp_of(cfg, obs["jobs"][k])) < heal_threshold
                    ]
                    if len(hurt) >= 3 and obs["skill2_cd"][i] == 0 and obs["mp"][i] >= cfg.healer_aoe_mp:
                        acts.append(ACT_SKILL2)
                    elif len(hurt) >= 1 and obs["skill_cd"][i] == 0 and obs["mp"][i] >= cfg.healer_heal_mp:
                        acts.append(ACT_SKILL)
                    else:
                        acts.append(ACT_BASIC)
                else:  # smart (単体回復のみ)
                    need = any(
                        obs["alive"][k] and (obs["hp"][k] / max_hp_of(cfg, obs["jobs"][k])) < heal_threshold
                        for k in range(n)
                    )
                    if need and obs["skill_cd"][i] == 0 and obs["mp"][i] >= cfg.healer_heal_mp:
                        acts.append(ACT_SKILL)
                    else:
                        acts.append(ACT_BASIC)
            else:  # DPS：強撃が撃てれば強撃、そうでなければ通常攻撃
                if obs["skill_cd"][i] == 0 and obs["mp"][i] >= cfg.dps_skill_mp:
                    acts.append(ACT_SKILL)
                else:
                    acts.append(ACT_BASIC)
        return acts

    return fn


def check_1_population_scaling() -> None:
    print("\n=== 1. 人数スケーリング (N=4, 5, 6, 8, 10, boss_atk=105, AoE なし) ===")
    cases = [
        (4, 2100, 105, 30),
        (5, 2800, 105, 30),
        (6, 3500, 105, 30),
        (8, 5200, 105, 30),
        (10, 6800, 105, 30),
    ]
    for n, bhp, batk, bturns in cases:
        cfg = RaidConfig(num_agents=n, boss_hp=bhp, boss_atk=batk, battle_turns=bturns)
        winners = []
        best_nonwin = None
        for t in range(n + 1):
            for h in range(n + 1 - t):
                d = n - t - h
                jobs = [JOB_TANK] * t + [JOB_HEALER] * h + [JOB_DPS] * d
                res = run_policy(cfg, make_comp_policy(jobs), seeds=5)
                if res["win"] > 0:
                    winners.append((f"{t}:{h}:{d}", res["win"], res["ret"], res["alive"]))
                elif best_nonwin is None or res["ret"] > best_nonwin[1]:
                    best_nonwin = (f"{t}:{h}:{d}", res["ret"], res["boss_left"], res["alive"])
        print(f"N={n:2d} boss_hp={bhp} atk={batk}: winners={winners} | best_fail={best_nonwin}")


def check_2_threat_and_misbehavior() -> None:
    print("\n=== 2. 敵視・サボり・過剰ヒール (N=6, boss_hp=3500, boss_atk=105) ===")
    cfg = RaidConfig()
    jobs_114 = [JOB_TANK] + [JOB_HEALER] + [JOB_DPS] * 4
    tests = [
        ("1:1:4 normal", make_comp_policy(jobs_114, tank_mode="normal", healer_mode="smart")),
        ("1:1:4 tank_no_provoke", make_comp_policy(jobs_114, tank_mode="no_provoke", healer_mode="smart")),
        ("1:1:4 tank_lazy(wait)", make_comp_policy(jobs_114, tank_mode="lazy", healer_mode="smart")),
        ("1:1:4 healer_spam_heal", make_comp_policy(jobs_114, tank_mode="normal", healer_mode="spam_heal")),
        ("1:1:4 healer_attack_only", make_comp_policy(jobs_114, tank_mode="normal", healer_mode="attack_only")),
    ]
    for name, fn in tests:
        res = run_policy(cfg, fn, seeds=5)
        print(
            f"{name:28s} | win={res['win']:.2f} ret={res['ret']:.3f} "
            f"left={res['boss_left']:6.1f} alive={res['alive']:.2f} "
            f"switches={res['switches']:.1f} deaths(T/H/D)={res['td']:.1f}/{res['hd']:.1f}/{res['dd']:.1f}"
        )


def check_4_random_baselines() -> None:
    print("\n=== 4. ランダム & 無行動ベースライン (N=6, boss_hp=3500, boss_atk=105) ===")
    cfg = RaidConfig()
    rng = np.random.default_rng(42)
    tests = [
        ("noop", lambda env, obs, s: [ACT_WAIT] * env.cfg.num_agents),
        ("uniform_random_all", lambda env, obs, s: [int(rng.integers(0, NUM_ACTIONS)) for _ in range(env.cfg.num_agents)]),
        ("random_combat_only(DPS)", lambda env, obs, s: [int(rng.integers(0, 4)) for _ in range(env.cfg.num_agents)]),
    ]
    for name, fn in tests:
        res = run_policy(cfg, fn, seeds=20)
        print(f"{name:25s} | win={res['win']:.2f} ret={res['ret']:.3f} left={res['boss_left']:6.1f} alive={res['alive']:.2f}")


def check_6_healer_cd_comparison() -> None:
    """回復スキルに CD がある場合（単体 CD=1, AoE CD=3）vs ない場合（MP のみ）を比較する。"""
    print("\n=== 6. 回復 CD なし (0,0) vs CD あり (単体1, AoE3) の比較 ===")
    cd_settings = [
        ("CDなし(0,0)", 0, 0),
        ("CDあり(単1,AoE3)", 1, 3),
        ("AoEのみCD(単0,AoE3)", 0, 3),
    ]

    # (A) N=6 ボス AoE あり
    print("--- (A) N=6 AoE あり (boss_hp=3200, atk=105, aoe=30/5T) ---")
    for label, h_cd, a_cd in cd_settings:
        cfg = RaidConfig(
            num_agents=6, boss_hp=3200, boss_atk=105,
            boss_aoe_interval=5, boss_aoe_dmg=30,
            healer_heal_cd=h_cd, healer_aoe_cd=a_cd,
        )
        jobs_114 = [JOB_TANK] + [JOB_HEALER] + [JOB_DPS] * 4
        for mode in ["smart", "smart_aoe", "spam_aoe"]:
            res = run_policy(cfg, make_comp_policy(jobs_114, healer_mode=mode), seeds=5)
            print(
                f"  {label:18s} 1:1:4 {mode:10s} | win={res['win']:.2f} ret={res['ret']:.3f} "
                f"alive={res['alive']:.2f} switches={res['switches']:.1f}"
            )

    # (B) N=10 ボス AoE あり（1:1:8 と 1:2:7 の違いを見る）
    print("--- (B) N=10 AoE あり (boss_hp=5500, atk=110, aoe=35/4T) ---")
    for label, h_cd, a_cd in cd_settings:
        cfg = RaidConfig(
            num_agents=10, boss_hp=5500, boss_atk=110,
            boss_aoe_interval=4, boss_aoe_dmg=35,
            healer_heal_cd=h_cd, healer_aoe_cd=a_cd,
        )
        for comp_name, n_t, n_h, n_d in [("1:1:8", 1, 1, 8), ("1:2:7", 1, 2, 7)]:
            jobs = [JOB_TANK] * n_t + [JOB_HEALER] * n_h + [JOB_DPS] * n_d
            res = run_policy(cfg, make_comp_policy(jobs, healer_mode="smart_aoe"), seeds=5)
            print(
                f"  {label:18s} {comp_name} smart_aoe | win={res['win']:.2f} ret={res['ret']:.3f} "
                f"left={res['boss_left']:6.1f} alive={res['alive']:.2f} switches={res['switches']:.1f}"
            )


if __name__ == "__main__":
    check_1_population_scaling()
    check_2_threat_and_misbehavior()
    check_4_random_baselines()
    check_6_healer_cd_comparison()
