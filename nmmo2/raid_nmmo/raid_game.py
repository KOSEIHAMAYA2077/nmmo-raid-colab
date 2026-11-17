"""Neural MMO 2.1.2 本体（nmmo.Env + TeamBattle API）上の協力レイド実装。

設計方針（report/2026-11-16_meta_mmo_ippo_design_review.md および history/26/11/16/1536.md 準拠）：
- NumPy 版 env.py の二重実行（別環境での裏回し）はせず、NMMO 2.1.2 の Realm / Player / NPC /
  Communication / Task 上で直接 1 試合を進行させる。
- 位置取りや回避は入れず、全員をマップ中央付近に配置し、発動した攻撃は必中・全体回復は全生存者対象とする。
- 通信（Comm.Token）は著者版 TeamWrapper のような手書き上書きを行わず、エージェントの選択トークンを
  そのまま nmmo.Env.step() へ渡し、受信も NMMO の obs[aid]["Communication"] から取得する。
- NMMO 標準の NPC 毎tick自然回復(+1)や不要な個人報酬・死亡罰(-1)は抑止し、
  RaidConfig / boss_catalog.py の 16 ボス設定と同じダメージ・敵視・回復・共通報酬を再現する。
"""
import sys
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import nmmo
import numpy as np
from nmmo.core.config import Combat, Communication, Config, NPC, Terrain
from nmmo.core.game_api import TeamBattle
from nmmo.lib.team_helper import TeamLoader
from nmmo.task.task_api import Task

# raid_proto の定数・RaidConfig をそのまま再利用し、仕様の重複定義を防ぐ
_RAID_PROTO_DIR = Path(__file__).resolve().parents[1] / "raid_proto"
if str(_RAID_PROTO_DIR) not in sys.path:
    sys.path.insert(0, str(_RAID_PROTO_DIR))

from env import (  # noqa: E402
    ACT_BASIC,
    ACT_JOB_DPS,
    ACT_JOB_HEALER,
    ACT_JOB_TANK,
    ACT_SKILL,
    ACT_SKILL2,
    ACT_WAIT,
    JOB_DPS,
    JOB_HEALER,
    JOB_NAMES,
    JOB_TANK,
    NUM_ACTIONS,
    RaidConfig,
)


class RaidNmmoConfig(Config, Terrain, Combat, NPC, Communication):
    """NMMO 2.1.2 上のレイド専用設定（地形・戦闘・NPC・通信のみ有効化）。"""

    MAP_N = 1
    MAP_SIZE = 32
    MAP_CENTER = 16
    TERRAIN_RESET_TO_GRASS = True
    MAP_RESET_FROM_FRACTAL = True
    DEATH_FOG_ONSET = None

    NPC_N = 1
    NPC_DEFAULT_REFILL_DEAD_NPCS = False

    PLAYER_BASE_HEALTH = 20000  # ボス HP (最大13,000超) と タンク HP (250) を保持できる上限
    PLAYER_HEALTH_INCREMENT = 0
    ALLOW_MOVE_INTO_OCCUPIED_TILE = True
    COMBAT_SPAWN_IMMUNITY = 9999  # 独自レイド戦闘と NMMO 標準自動攻撃が二重発生しないよう保護

    TERRAIN_SYSTEM_ENABLED = True
    COMBAT_SYSTEM_ENABLED = True
    NPC_SYSTEM_ENABLED = True
    COMMUNICATION_SYSTEM_ENABLED = True
    COMMUNICATION_N_OBS = 32


class RaidBossTask(Task):
    """毎ターンのボスHP減少割合＋討伐ボーナスを全員の共通報酬として返す NMMO Task。"""

    def __init__(self, game: "CooperativeRaidGame", assignee: List[int]):
        self._game = game
        super().__init__(
            eval_fn=lambda gs: 0.0,
            assignee=assignee,
            reward_multiplier=1.0,
            spec_name="RaidBossClear",
            reward_to="team",
        )

    def _map_progress_to_reward(self, gs) -> float:
        if self._game.won:
            self._progress = 1.0
            self._completed_tick = gs.current_tick
        else:
            self._progress = max(0.0, min(1.0, self._game.boss_damage_fraction))
        return float(self._game.last_team_reward)


class CooperativeRaidGame(TeamBattle):
    """NMMO 2.1.2 の TeamBattle を継承した協力ボスレイド Game。"""

    required_systems = ["TERRAIN", "COMBAT", "NPC", "COMMUNICATION"]

    def __init__(self, env: nmmo.Env, raid_cfg: RaidConfig):
        super().__init__(env)
        # NMMO の Game.__init__ がインスタンス属性 self.assign_dead_reward=True を上書きして
        # TeamGameTemplate.assign_dead_reward=False を隠すため、明示的に False に戻して -1 死亡罰を無効化する
        self.assign_dead_reward = False
        self.raid_cfg = raid_cfg
        self._pending_raid_actions: List[int] = [ACT_WAIT] * raid_cfg.num_agents
        self._tie_choice_fn: Optional[Callable[[List[int]], int]] = None

        self.boss = None
        self.turn = 0
        self._boss_hp = float(raid_cfg.boss_hp)
        self._jobs: List[int] = [JOB_DPS] * raid_cfg.num_agents
        self._hp: List[float] = [float(raid_cfg.dps_hp)] * raid_cfg.num_agents
        self._mp: List[float] = [float(raid_cfg.max_mp)] * raid_cfg.num_agents
        self._alive: List[bool] = [True] * raid_cfg.num_agents
        self._job_cd: List[int] = [0] * raid_cfg.num_agents
        self._skill_cd: List[int] = [0] * raid_cfg.num_agents
        self._skill2_cd: List[int] = [0] * raid_cfg.num_agents
        self._threat: List[float] = [0.0] * raid_cfg.num_agents
        self._provoke_buff: List[bool] = [False] * raid_cfg.num_agents

        self.won = False
        self.last_team_reward = 0.0
        self.last_boss_target = -1
        self.last_events: List[Dict] = []

    def set_raid_config(self, raid_cfg: RaidConfig) -> None:
        """次回 reset() 以降に適用するボス／パーティ設定を差し替える。"""
        assert raid_cfg.num_agents == self.config.original["PLAYER_N"], (
            f"num_agents={raid_cfg.num_agents} must match env PLAYER_N={self.config.original['PLAYER_N']}"
        )
        self.raid_cfg = raid_cfg

    def queue_actions(
        self,
        actions: List[int],
        tie_choice_fn: Optional[Callable[[List[int]], int]] = None,
    ) -> None:
        """env.step() の直前に各エージェントのレイド行動（0..6）をセットする。"""
        assert len(actions) == self.raid_cfg.num_agents
        self._pending_raid_actions = [int(a) for a in actions]
        self._tie_choice_fn = tie_choice_fn

    @property
    def in_prep(self) -> bool:
        return self.turn < self.raid_cfg.prep_turns

    @property
    def boss_damage_fraction(self) -> float:
        return (self.raid_cfg.boss_hp - self._boss_hp) / float(self.raid_cfg.boss_hp)

    @property
    def is_over(self) -> bool:
        max_turns = self.raid_cfg.prep_turns + self.raid_cfg.battle_turns
        return self.won or (not any(self._alive)) or (self.turn >= max_turns) or (self.realm.num_players == 0)

    def _set_config(self):
        self.config.reset()
        self.config.toggle_systems(self.required_systems)
        self.config.set_for_episode("MAP_RESET_FROM_FRACTAL", True)
        self.config.set_for_episode("TERRAIN_RESET_TO_GRASS", True)
        self.config.set_for_episode("NPC_DEFAULT_REFILL_DEAD_NPCS", False)
        self.config.set_for_episode("ALLOW_MOVE_INTO_OCCUPIED_TILE", True)
        self.config.set_for_episode("PLAYER_HEALTH_INCREMENT", 0)
        self.config.set_for_episode("DEATH_FOG_ONSET", None)
        self.config.set_for_episode("HORIZON", self.raid_cfg.prep_turns + self.raid_cfg.battle_turns)
        self.config.set_for_episode(
            "TEAMS", {0: list(range(1, self.raid_cfg.num_agents + 1))}
        )

    def _set_realm(self, map_dict):
        self.realm.reset(self._np_random, map_dict, custom_spawn=True)
        row, col = self.realm.map.center_coord
        # NMMO can scatter water onto grass, including the otherwise fixed spawn tiles.
        self.realm.map.make_spawnable(row, col, radius=1)
        loader = TeamLoader(self.config, self._np_random, [[(row, col)]])
        self.realm.players.spawn(loader)

        # ボスは隣接マスに Passive (danger=0) として配置し、勝手な徘徊・標準攻撃を抑止
        self.boss = self.realm.npcs.spawn_npc(
            row, col + 1, danger=0.0, name="RaidBoss", apply_beta_to_danger=False
        )
        assert self.boss is not None, "Failed to spawn RaidBoss NPC on map"
        self.boss.health.update(float(self.raid_cfg.boss_hp))

        for aid in self.config.POSSIBLE_AGENTS:
            self.realm.players[aid].health.update(float(self.raid_cfg.dps_hp))

    def _define_tasks(self):
        return [RaidBossTask(self, list(self.config.TEAMS[0]))]

    def _reset_stats(self):
        super()._reset_stats()
        n = self.raid_cfg.num_agents
        self.turn = 0
        self._boss_hp = float(self.raid_cfg.boss_hp)
        self._jobs = [JOB_DPS] * n
        self._hp = [float(self.raid_cfg.dps_hp)] * n
        self._mp = [float(self.raid_cfg.max_mp)] * n
        self._alive = [True] * n
        self._job_cd = [0] * n
        self._skill_cd = [0] * n
        self._skill2_cd = [0] * n
        self._threat = [0.0] * n
        self._provoke_buff = [False] * n
        self.won = False
        self.last_team_reward = 0.0
        self.last_boss_target = -1
        self.last_events = []
        self._pending_raid_actions = [ACT_WAIT] * n

    def _job_max_hp(self, job: int) -> float:
        if job == JOB_TANK:
            return float(self.raid_cfg.tank_hp)
        if job == JOB_HEALER:
            return float(self.raid_cfg.healer_hp)
        return float(self.raid_cfg.dps_hp)

    def _switch_job(self, i: int, new_job: int) -> None:
        if self._jobs[i] == new_job:
            return
        if not self.in_prep and self._job_cd[i] > 0:
            return
        old_max = self._job_max_hp(self._jobs[i])
        new_max = self._job_max_hp(new_job)
        ratio = self._hp[i] / max(old_max, 1.0)
        self._jobs[i] = new_job
        self._hp[i] = float(max(1.0, round(ratio * new_max)))
        self.realm.players[i + 1].health.update(self._hp[i])
        self._skill_cd[i] = 0
        self._skill2_cd[i] = 0
        if not self.in_prep:
            self._job_cd[i] = self.raid_cfg.job_change_cd

    def _cull_dead_player(self, i: int, terminated: Dict[int, bool]) -> None:
        """ProtectTheKing と同じ正規手順でプレイヤーを死亡処理・削除する。"""
        aid = i + 1
        self._alive[i] = False
        self._hp[i] = 0.0
        self._threat[i] = 0.0
        if aid in self.realm.players:
            player = self.realm.players[aid]
            player.health.update(0)
            self.realm.players.dead_this_tick[aid] = player
            self.realm.players.cull_entity(player)
            player.datastore_record.delete()
        terminated[aid] = True

    def _cull_boss_npc(self) -> None:
        if self.boss is not None and self.boss.ent_id in self.realm.npcs:
            self.boss.health.update(0)
            self.realm.npcs.dead_this_tick[self.boss.ent_id] = self.boss
            self.realm.npcs.cull_entity(self.boss)
            self.boss.datastore_record.delete()

    def update(self, terminated, dead_players, dead_npcs):
        cfg = self.raid_cfg
        n = cfg.num_agents
        prev_boss_hp = self._boss_hp
        self._provoke_buff = [False] * n
        in_prep_now = self.in_prep
        # One row per local seat. Empty rows mean no event, including after death.
        events = [{} for _ in range(n)]
        self.last_events = events

        # NMMO の NPC.update() が毎 tick +1 自動回復するため、ターン開始時の実 HP へ戻す
        if self.boss is not None and self.boss.ent_id in self.realm.npcs:
            self.boss.health.update(self._boss_hp)

        # 1. 各プレイヤーの行動解決（ID 昇順）
        for i in range(n):
            if not self._alive[i]:
                continue
            aid = i + 1
            player = self.realm.players[aid]
            act = int(self._pending_raid_actions[i])

            if act in (ACT_JOB_TANK, ACT_JOB_HEALER, ACT_JOB_DPS):
                old_job = self._jobs[i]
                self._switch_job(i, act - ACT_JOB_TANK)
                if self._jobs[i] != old_job:
                    events[i]["switch"] = 1
                continue

            if in_prep_now:
                continue

            job = self._jobs[i]
            if act == ACT_BASIC:
                if job == JOB_TANK:
                    dmg = float(cfg.tank_basic_dmg)
                    self._boss_hp = max(0.0, self._boss_hp - dmg)
                    self._threat[i] += dmg * cfg.tank_threat_mult
                elif job == JOB_HEALER:
                    dmg = float(cfg.healer_basic_dmg)
                    self._boss_hp = max(0.0, self._boss_hp - dmg)
                    self._threat[i] += dmg
                else:
                    dmg = float(cfg.dps_basic_dmg)
                    self._boss_hp = max(0.0, self._boss_hp - dmg)
                    self._threat[i] += dmg
                player.history.damage_inflicted += dmg

            elif act == ACT_SKILL and self._skill_cd[i] == 0:
                if job == JOB_TANK and self._mp[i] >= cfg.tank_provoke_mp:
                    self._mp[i] -= cfg.tank_provoke_mp
                    self._skill_cd[i] = cfg.tank_provoke_cd
                    top_threat = max(
                        (self._threat[k] for k in range(n) if self._alive[k]), default=0.0
                    )
                    self._threat[i] = max(self._threat[i], top_threat) + cfg.tank_provoke_bonus_threat
                    self._provoke_buff[i] = True
                    events[i]["provoke"] = 1
                elif job == JOB_HEALER and self._mp[i] >= cfg.healer_heal_mp:
                    candidates = [k for k in range(n) if self._alive[k]]
                    if candidates:
                        target = min(
                            candidates, key=lambda k: self._hp[k] / self._job_max_hp(self._jobs[k])
                        )
                        max_t_hp = self._job_max_hp(self._jobs[target])
                        effective = min(float(cfg.healer_heal_amount), max_t_hp - self._hp[target])
                        events[i]["single_heal_attempted"] = float(cfg.healer_heal_amount)
                        events[i]["single_heal_effective"] = effective
                        if effective > 0:
                            self._mp[i] -= cfg.healer_heal_mp
                            self._skill_cd[i] = cfg.healer_heal_cd
                            self._hp[target] += effective
                            self.realm.players[target + 1].health.update(self._hp[target])
                            self._threat[i] += effective * cfg.healer_threat_mult
                elif job == JOB_DPS and self._mp[i] >= cfg.dps_skill_mp:
                    self._mp[i] -= cfg.dps_skill_mp
                    self._skill_cd[i] = cfg.dps_skill_cd
                    dmg = float(cfg.dps_skill_dmg)
                    self._boss_hp = max(0.0, self._boss_hp - dmg)
                    self._threat[i] += dmg
                    player.history.damage_inflicted += dmg
                    events[i]["strong_hit"] = 1

            elif act == ACT_SKILL2 and self._skill2_cd[i] == 0:
                if job == JOB_HEALER and self._mp[i] >= cfg.healer_aoe_mp:
                    living = [k for k in range(n) if self._alive[k]]
                    total_effective = 0.0
                    for k in living:
                        max_k_hp = self._job_max_hp(self._jobs[k])
                        eff = min(float(cfg.healer_aoe_amount), max_k_hp - self._hp[k])
                        if eff > 0:
                            self._hp[k] += eff
                            self.realm.players[k + 1].health.update(self._hp[k])
                            total_effective += eff
                    self._mp[i] -= cfg.healer_aoe_mp
                    self._skill2_cd[i] = cfg.healer_aoe_cd
                    self._threat[i] += total_effective * cfg.healer_threat_mult
                    events[i]["aoe_heal_attempted"] = float(cfg.healer_aoe_amount * len(living))
                    events[i]["aoe_heal_effective"] = total_effective

        if self.boss is not None and self.boss.ent_id in self.realm.npcs:
            self.boss.health.update(self._boss_hp)

        # 2. 討伐判定とボスの反撃（単体 → 周期AoE）
        boss_target = -1
        if self._boss_hp <= 0.0:
            self.won = True
            self._cull_boss_npc()
        elif not in_prep_now:
            living = [k for k in range(n) if self._alive[k]]
            if living:
                max_th = max(self._threat[k] for k in living)
                tied = [k for k in living if self._threat[k] == max_th]
                if len(tied) == 1:
                    boss_target = tied[0]
                elif self._tie_choice_fn is not None:
                    boss_target = int(self._tie_choice_fn(tied))
                else:
                    boss_target = int(self._np_random.choice(tied))

                dmg = float(cfg.boss_atk)
                events[boss_target]["boss_targeted"] = 1
                if self._jobs[boss_target] == JOB_TANK:
                    dmg *= (1.0 - cfg.tank_dr)
                if self._provoke_buff[boss_target]:
                    dmg *= (1.0 - cfg.tank_provoke_extra_dr)
                self._hp[boss_target] = max(0.0, self._hp[boss_target] - dmg)
                self.realm.players[boss_target + 1].history.damage_received += dmg
                if self._hp[boss_target] <= 0.0:
                    self._cull_dead_player(boss_target, terminated)
                else:
                    self.realm.players[boss_target + 1].health.update(self._hp[boss_target])

            battle_turn = self.turn - cfg.prep_turns + 1
            if cfg.boss_aoe_interval > 0 and battle_turn % cfg.boss_aoe_interval == 0:
                for k in range(n):
                    if not self._alive[k]:
                        continue
                    admg = float(cfg.boss_aoe_dmg)
                    if self._jobs[k] == JOB_TANK:
                        admg *= (1.0 - cfg.tank_dr)
                    self._hp[k] = max(0.0, self._hp[k] - admg)
                    self.realm.players[k + 1].history.damage_received += admg
                    if self._hp[k] <= 0.0:
                        self._cull_dead_player(k, terminated)
                    else:
                        self.realm.players[k + 1].health.update(self._hp[k])

        self.last_boss_target = boss_target

        # 3. クールダウン経過と MP 自然回復
        if not self.is_over:
            for i in range(n):
                if not self._alive[i]:
                    continue
                if self._job_cd[i] > 0:
                    self._job_cd[i] -= 1
                if self._skill_cd[i] > 0:
                    self._skill_cd[i] -= 1
                if self._skill2_cd[i] > 0:
                    self._skill2_cd[i] -= 1
                if not in_prep_now:
                    self._mp[i] = min(float(cfg.max_mp), self._mp[i] + cfg.mp_regen)

        # 4. ターン進行・共通報酬・勝者確定
        self.turn += 1
        dmg_dealt = prev_boss_hp - self._boss_hp
        reward = dmg_dealt / float(cfg.boss_hp)
        if self.won:
            reward += cfg.win_bonus
        self.last_team_reward = reward

        super().update(terminated, dead_players, dead_npcs)

    def _check_winners(self, terminated):
        if self.won:
            return list(self.config.POSSIBLE_AGENTS)
        return None


class RaidNmmoEnv:
    """NMMO 2.1.2 上で動く CooperativeRaidGame を MinimalRaidEnv と同じ操作感で扱うラッパ。

    - received_comm[receiver][sender] は各受信者の NMMO Communication 表の復号値（未受信は -1）。
    - last_comm は NumPy 試作との互換のために保持する各自の最終送信履歴（死亡後も保持）。
    - 通信を無効化 (use_comm=False) したい場合は全員の Token を定数 0 に固定し、
      全員同座標のため位置情報からの漏洩も防ぐ。
    """

    def __init__(self, cfg: Optional[RaidConfig] = None, use_comm: bool = True):
        self.cfg = cfg or RaidConfig()
        self.use_comm = use_comm
        self._map_dir = tempfile.TemporaryDirectory(prefix="nmmo2_raid_")

        nmmo_cfg = RaidNmmoConfig()
        nmmo_cfg.set("PATH_MAPS", self._map_dir.name)
        nmmo_cfg.set("PLAYER_N", self.cfg.num_agents)
        nmmo_cfg.set("TEAMS", {0: list(range(1, self.cfg.num_agents + 1))})
        nmmo_cfg.set("COMMUNICATION_NUM_TOKENS", self.cfg.comm_vocab)
        nmmo_cfg.set("COMMUNICATION_N_OBS", max(32, self.cfg.num_agents))
        nmmo_cfg.set("HORIZON", self.cfg.prep_turns + self.cfg.battle_turns)

        self.nmmo_env = nmmo.Env(nmmo_cfg, seed=0)
        self.game = CooperativeRaidGame(self.nmmo_env, self.cfg)
        self.last_nmmo_obs: Dict = {}
        self.last_nmmo_rewards: Dict[int, float] = {}
        self.last_terminated: Dict[int, bool] = {}
        self.last_truncated: Dict[int, bool] = {}
        self._last_comm: List[int] = [0] * self.cfg.num_agents
        self.done = False

    def close(self) -> None:
        self._map_dir.cleanup()

    @property
    def turn(self) -> int:
        return self.game.turn

    @property
    def in_prep(self) -> bool:
        return self.game.in_prep

    @property
    def won(self) -> bool:
        return self.game.won

    def set_raid_config(self, cfg: RaidConfig) -> None:
        """同じ人数 N で別ボス設定へ切り替える（マップ再生成なしで即 reset 可能）。"""
        self.game.set_raid_config(cfg)
        self.cfg = cfg

    def reset(self, seed: int = 0) -> Dict:
        self.last_nmmo_obs, _ = self.nmmo_env.reset(game=self.game, seed=int(seed))
        self.last_nmmo_rewards = {aid: 0.0 for aid in self.nmmo_env.possible_agents}
        self.last_terminated = {aid: False for aid in self.nmmo_env.possible_agents}
        self.last_truncated = {aid: False for aid in self.nmmo_env.possible_agents}
        self._last_comm = [0] * self.cfg.num_agents
        self.done = False
        return self._get_obs()

    def _received_comm(self) -> np.ndarray:
        """Decode native rows for each receiver. A missing message differs from token zero."""
        n = self.cfg.num_agents
        received = np.full((n, n), -1, dtype=np.int16)
        for aid, native in self.last_nmmo_obs.items():
            if self.game._alive[aid - 1]:
                rows = native["Communication"]
                senders, messages = rows[:, 0], rows[:, 3]
                present = (senders > 0) & (senders <= n) & (messages > 0)
                received[aid - 1, senders[present] - 1] = messages[present] - 1
        return received

    def _get_obs(self) -> Dict:
        return {
            "turn": self.game.turn,
            "in_prep": self.game.in_prep,
            "boss_hp": self.game._boss_hp,
            "jobs": list(self.game._jobs),
            "hp": list(self.game._hp),
            "mp": list(self.game._mp),
            "alive": list(self.game._alive),
            "job_cd": list(self.game._job_cd),
            "skill_cd": list(self.game._skill_cd),
            "skill2_cd": list(self.game._skill2_cd),
            "threat": list(self.game._threat),
            "last_comm": list(self._last_comm),
            "received_comm": self._received_comm(),
        }

    def step(
        self,
        actions: List[int],
        comms: Optional[List[int]] = None,
        tie_choice_fn: Optional[Callable[[List[int]], int]] = None,
    ) -> Tuple[Dict, float, bool, Dict]:
        assert not self.done, "Episode is already done"
        n = self.cfg.num_agents
        if comms is None or not self.use_comm:
            comms_norm = [0] * n
        else:
            comms_norm = [int(c) % self.cfg.comm_vocab for c in comms]

        # ターン開始時に生きているメンバーの直近通信と NMMO Comm 行動を準備
        nmmo_actions: Dict[int, Dict] = {}
        pre_alive = list(self.game._alive)
        for i in range(n):
            if pre_alive[i]:
                self._last_comm[i] = comms_norm[i]
                nmmo_actions[i + 1] = {"Comm": {"Token": comms_norm[i]}}

        self.game.queue_actions(actions, tie_choice_fn=tie_choice_fn)
        (
            self.last_nmmo_obs,
            self.last_nmmo_rewards,
            self.last_terminated,
            self.last_truncated,
            _,
        ) = self.nmmo_env.step(nmmo_actions)

        self.done = self.game.is_over

        info = {
            "won": self.game.won,
            "boss_hp": self.game._boss_hp,
            "alive_count": sum(self.game._alive),
            "boss_target": self.game.last_boss_target,
            "game_is_over": bool(self.game.is_over),
            "events": self.game.last_events,
        }
        return self._get_obs(), float(self.game.last_team_reward), self.done, info
