"""マップ移動なしの最小レイド環境（IPPO・役割分担・会話の研究用プロトタイプ）。

FF14 / WoW の敵視・回復比率および NMMO 2.1.2 の基礎パラメータ（初期 HP 100 等）を参考に、
・戦闘前（準備フェーズ：ジョブチェンジ CD 0）＋ 戦闘中（ジョブチェンジに CD あり）の 2 フェーズ制
・敵視（攻撃 1x、タンクスタンス倍率、有効回復 0.5x、挑発＝1位＋α）
・MP 管理とスキル CD（単体回復・弱い AoE 回復）
・全員共通のレイド報酬（削ったボス HP 割合 ＋ 討伐ボーナス）
を備えている。
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


# --- ジョブ ID ---
JOB_TANK = 0
JOB_HEALER = 1
JOB_DPS = 2
JOB_NAMES = ("TANK", "HEALER", "DPS")

# --- 行動 ID（各ターン 1 つ選択） ---
ACT_WAIT = 0        # 何もしない（待機）
ACT_BASIC = 1       # 通常攻撃（タンクは高敵視、DPS は主力火力）
ACT_SKILL = 2       # 固有スキル1（タンク:挑発 / ヒーラー:単体回復 / DPS:強撃）
ACT_SKILL2 = 3      # 固有スキル2（ヒーラー専用の弱い全体回復：FF14のメディカ相当）
ACT_JOB_TANK = 4    # タンクへ転職
ACT_JOB_HEALER = 5  # ヒーラーへ転職
ACT_JOB_DPS = 6     # DPSへ転職
NUM_ACTIONS = 7


@dataclass
class RaidConfig:
    # --- パーティ・フェーズ・会話 ---
    num_agents: int = 6       # パーティ人数（5, 6, 8, 10 等に変更可能）
    prep_turns: int = 3       # 戦闘前の準備ターン数（ジョブチェンジ CD=0 で相談できる）
    battle_turns: int = 30    # 戦闘の制限ターン数（0 になる前に削り切れなければ敗北）
    job_change_cd: int = 10   # 戦闘中のジョブチェンジ CD（準備中は 0）
    comm_vocab: int = 4       # 会話トークンの種類数（0..comm_vocab-1）

    # --- ボスの強さ ---
    boss_hp: int = 3500
    boss_atk: int = 105             # 毎ターンの単体攻撃（敵視 1 位を狙う）
    boss_aoe_interval: int = 0      # 全体攻撃の周期（0=なし、5=戦闘 5 ターンごとに発動）
    boss_aoe_dmg: int = 30          # 全体攻撃のダメージ

    # --- タンクの性能 ---
    tank_hp: int = 250
    tank_dr: float = 0.50                  # 常時被ダメージ軽減（50%カット）
    tank_basic_dmg: int = 5
    tank_threat_mult: float = 5.0          # 通常攻撃の敵視倍率（タンクスタンス相当）
    tank_provoke_mp: int = 15              # 挑発の消費 MP
    tank_provoke_cd: int = 2               # 挑発の CD
    tank_provoke_bonus_threat: float = 100.0  # 挑発時：現在の1位＋この値に敵視を引き上げる
    tank_provoke_extra_dr: float = 0.50    # 挑発したターンのみ追加で受けるダメージ軽減

    # --- ヒーラーの性能（FF14 ケアル vs メディカ比率を参考） ---
    healer_hp: int = 100
    healer_basic_dmg: int = 5
    healer_heal_amount: int = 55   # 単体回復の回復量
    healer_heal_mp: int = 15       # 単体回復の消費 MP（少ない）
    healer_heal_cd: int = 0        # 単体回復の CD（0=毎ターン可、1=1ターンおき）
    healer_aoe_amount: int = 25    # 全体回復の 1 人あたり回復量（単体の約0.45倍と弱い）
    healer_aoe_mp: int = 35        # 全体回復の消費 MP（単体の約2.3倍と重い）
    healer_aoe_cd: int = 0         # 全体回復の CD（個別管理）
    healer_threat_mult: float = 0.5  # 有効回復量に対する敵視倍率（FF14/WoW準拠 0.5倍）

    # --- DPS の性能 ---
    dps_hp: int = 100
    dps_basic_dmg: int = 20
    dps_skill_dmg: int = 45
    dps_skill_mp: int = 25
    dps_skill_cd: int = 2

    # --- MP・共通報酬 ---
    max_mp: int = 100
    mp_regen: int = 10        # 戦闘中、毎ターンの MP 自然回復
    win_bonus: float = 1.0    # 討伐成功時に全員へ加算される共通ボーナス


class MinimalRaidEnv:
    """N 人 vs 1 ボスのターン制レイド環境。"""

    def __init__(self, config: Optional[RaidConfig] = None):
        self.cfg = config or RaidConfig()
        self.rng = np.random.default_rng(0)
        self.reset(seed=0)

    def _job_max_hp(self, job: int) -> int:
        if job == JOB_TANK:
            return self.cfg.tank_hp
        if job == JOB_HEALER:
            return self.cfg.healer_hp
        return self.cfg.dps_hp

    def reset(self, seed: int = 0) -> Dict:
        self.rng = np.random.default_rng(seed)
        n = self.cfg.num_agents
        self.turn = 0
        self.boss_hp = float(self.cfg.boss_hp)
        # 全員同じ初期ジョブ（DPS）から開始し、準備フェーズで相談して分担する
        self.jobs = [JOB_DPS] * n
        self.hp = [float(self._job_max_hp(j)) for j in self.jobs]
        self.alive = [True] * n
        self.mp = [float(self.cfg.max_mp)] * n
        self.job_cd = [0] * n
        self.skill_cd = [0] * n   # 固有スキル1（挑発 / 単体回復 / 強撃）の残り CD
        self.skill2_cd = [0] * n  # 固有スキル2（ヒーラー全体回復）の残り CD
        self.threat = [0.0] * n
        self.provoke_buff = [False] * n
        self.last_comm = [0] * n
        self.done = False
        self.won = False
        return self._get_obs()

    @property
    def in_prep(self) -> bool:
        """現在が戦闘前の準備フェーズか"""
        return self.turn < self.cfg.prep_turns

    def _get_obs(self) -> Dict:
        return {
            "turn": self.turn,
            "in_prep": self.in_prep,
            "boss_hp": self.boss_hp,
            "jobs": list(self.jobs),
            "hp": list(self.hp),
            "mp": list(self.mp),
            "alive": list(self.alive),
            "job_cd": list(self.job_cd),
            "skill_cd": list(self.skill_cd),
            "skill2_cd": list(self.skill2_cd),
            "threat": list(self.threat),
            "last_comm": list(self.last_comm),
        }

    def _switch_job(self, i: int, new_job: int) -> None:
        """ジョブ変更。HP は現在割合を維持し、転職による全回復の悪用を防ぐ。"""
        if self.jobs[i] == new_job:
            return
        if not self.in_prep and self.job_cd[i] > 0:
            return
        old_max = self._job_max_hp(self.jobs[i])
        new_max = self._job_max_hp(new_job)
        ratio = self.hp[i] / max(old_max, 1.0)
        self.jobs[i] = new_job
        self.hp[i] = max(1.0, round(ratio * new_max))
        self.skill_cd[i] = 0
        self.skill2_cd[i] = 0
        if not self.in_prep:
            self.job_cd[i] = self.cfg.job_change_cd

    def step(
        self, actions: List[int], comms: Optional[List[int]] = None
    ) -> Tuple[Dict, float, bool, Dict]:
        assert not self.done, "Episode is already done"
        n = self.cfg.num_agents
        assert len(actions) == n
        if comms is None:
            comms = [0] * n

        prev_boss_hp = self.boss_hp
        self.provoke_buff = [False] * n

        # 1. 生存者の会話トークンを更新（次ターンの観測に入る）
        for i in range(n):
            if self.alive[i]:
                self.last_comm[i] = int(comms[i]) % self.cfg.comm_vocab

        # 2. 各プレイヤーの行動処理
        for i in range(n):
            if not self.alive[i]:
                continue
            act = int(actions[i])
            if act in (ACT_JOB_TANK, ACT_JOB_HEALER, ACT_JOB_DPS):
                self._switch_job(i, act - ACT_JOB_TANK)
                continue

            if self.in_prep:
                # 準備フェーズではジョブ変更と会話のみ有効（攻撃・スキルは発動しない）
                continue

            job = self.jobs[i]
            if act == ACT_BASIC:
                if job == JOB_TANK:
                    dmg = self.cfg.tank_basic_dmg
                    self.boss_hp = max(0.0, self.boss_hp - dmg)
                    self.threat[i] += dmg * self.cfg.tank_threat_mult
                elif job == JOB_HEALER:
                    dmg = self.cfg.healer_basic_dmg
                    self.boss_hp = max(0.0, self.boss_hp - dmg)
                    self.threat[i] += dmg
                else:
                    dmg = self.cfg.dps_basic_dmg
                    self.boss_hp = max(0.0, self.boss_hp - dmg)
                    self.threat[i] += dmg

            elif act == ACT_SKILL and self.skill_cd[i] == 0:
                if job == JOB_TANK and self.mp[i] >= self.cfg.tank_provoke_mp:
                    # タンク挑発：現在の最高敵視に追いついた上でボーナスを加算＋1ターン軽減
                    self.mp[i] -= self.cfg.tank_provoke_mp
                    self.skill_cd[i] = self.cfg.tank_provoke_cd
                    top_threat = max((self.threat[k] for k in range(n) if self.alive[k]), default=0.0)
                    self.threat[i] = max(self.threat[i], top_threat) + self.cfg.tank_provoke_bonus_threat
                    self.provoke_buff[i] = True
                elif job == JOB_HEALER and self.mp[i] >= self.cfg.healer_heal_mp:
                    # ヒーラー単体回復：HP割合が最も低い生存者を回復（オーバーヒールは敵視0）
                    candidates = [k for k in range(n) if self.alive[k]]
                    if candidates:
                        target = min(candidates, key=lambda k: self.hp[k] / self._job_max_hp(self.jobs[k]))
                        max_t_hp = float(self._job_max_hp(self.jobs[target]))
                        effective = min(float(self.cfg.healer_heal_amount), max_t_hp - self.hp[target])
                        if effective > 0:
                            self.mp[i] -= self.cfg.healer_heal_mp
                            self.skill_cd[i] = self.cfg.healer_heal_cd
                            self.hp[target] += effective
                            self.threat[i] += effective * self.cfg.healer_threat_mult
                elif job == JOB_DPS and self.mp[i] >= self.cfg.dps_skill_mp:
                    # DPS強撃
                    self.mp[i] -= self.cfg.dps_skill_mp
                    self.skill_cd[i] = self.cfg.dps_skill_cd
                    dmg = self.cfg.dps_skill_dmg
                    self.boss_hp = max(0.0, self.boss_hp - dmg)
                    self.threat[i] += dmg

            elif act == ACT_SKILL2 and self.skill2_cd[i] == 0:
                if job == JOB_HEALER and self.mp[i] >= self.cfg.healer_aoe_mp:
                    # ヒーラー弱い全体回復：全員を少しずつ回復し、合計有効回復×0.5の敵視を得る
                    living = [k for k in range(n) if self.alive[k]]
                    total_effective = 0.0
                    for k in living:
                        max_k_hp = float(self._job_max_hp(self.jobs[k]))
                        eff = min(float(self.cfg.healer_aoe_amount), max_k_hp - self.hp[k])
                        if eff > 0:
                            self.hp[k] += eff
                            total_effective += eff
                    self.mp[i] -= self.cfg.healer_aoe_mp
                    self.skill2_cd[i] = self.cfg.healer_aoe_cd
                    self.threat[i] += total_effective * self.cfg.healer_threat_mult

        # 3. 討伐判定（このターンで削り切ったらボスの反撃はない）
        boss_target = -1
        if self.boss_hp <= 0.0:
            self.won = True
            self.done = True
        elif not self.in_prep:
            # 4a. ボスの単体攻撃（敵視1位の生存者を殴る。同値はランダム）
            living = [k for k in range(n) if self.alive[k]]
            if living:
                max_th = max(self.threat[k] for k in living)
                tied = [k for k in living if self.threat[k] == max_th]
                boss_target = int(self.rng.choice(tied))
                dmg = float(self.cfg.boss_atk)
                if self.jobs[boss_target] == JOB_TANK:
                    dmg *= (1.0 - self.cfg.tank_dr)
                if self.provoke_buff[boss_target]:
                    dmg *= (1.0 - self.cfg.tank_provoke_extra_dr)
                self.hp[boss_target] = max(0.0, self.hp[boss_target] - dmg)
                if self.hp[boss_target] <= 0.0:
                    self.alive[boss_target] = False
                    self.threat[boss_target] = 0.0

            # 4b. ボスの周期的な全体攻撃（AoE）
            battle_turn = self.turn - self.cfg.prep_turns + 1
            if self.cfg.boss_aoe_interval > 0 and battle_turn % self.cfg.boss_aoe_interval == 0:
                for k in range(n):
                    if not self.alive[k]:
                        continue
                    admg = float(self.cfg.boss_aoe_dmg)
                    if self.jobs[k] == JOB_TANK:
                        admg *= (1.0 - self.cfg.tank_dr)
                    self.hp[k] = max(0.0, self.hp[k] - admg)
                    if self.hp[k] <= 0.0:
                        self.alive[k] = False
                        self.threat[k] = 0.0

        # 5. クールダウン経過と MP 自然回復
        if not self.done:
            for i in range(n):
                if not self.alive[i]:
                    continue
                if self.job_cd[i] > 0:
                    self.job_cd[i] -= 1
                if self.skill_cd[i] > 0:
                    self.skill_cd[i] -= 1
                if self.skill2_cd[i] > 0:
                    self.skill2_cd[i] -= 1
                if not self.in_prep:
                    self.mp[i] = min(float(self.cfg.max_mp), self.mp[i] + self.cfg.mp_regen)

        self.turn += 1
        max_turns = self.cfg.prep_turns + self.cfg.battle_turns
        if not self.done and (not any(self.alive) or self.turn >= max_turns):
            self.done = True

        # 6. 全員共通のレイド報酬（削ったHP割合 ＋ 勝利ボーナス）
        dmg_dealt = prev_boss_hp - self.boss_hp
        reward = dmg_dealt / float(self.cfg.boss_hp)
        if self.won:
            reward += self.cfg.win_bonus

        info = {
            "won": self.won,
            "boss_hp": self.boss_hp,
            "alive_count": sum(self.alive),
            "boss_target": boss_target,
        }
        return self._get_obs(), reward, self.done, info
