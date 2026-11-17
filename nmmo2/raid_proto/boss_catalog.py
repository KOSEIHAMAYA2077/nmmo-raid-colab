"""4・6・10・16人レイド用の多様なボス原型（ボスカタログ）定義。

各ボスは「環境パラメータ（HP係数 / 単体攻撃 / AoE威力・周期 / 制限ターン）」を
パーティ人数 N に応じて設定できるようになっており、
- 訓練用 / 検証用 / 未知テスト用 の分割（過学習チェック用）
- 役割バランス（1T1H型 / MP温存型 / 多ヒーラー型 / タンクスイッチ型 / 複合型）
の切り分け実験に使える。
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

from env import RaidConfig


@dataclass(frozen=True)
class BossArchetype:
    """1 種類のボス原型定義。"""

    boss_id: str
    name_ja: str
    category: str          # single | tactical_aoe | multi_healer | tank_buster | hybrid
    split: str             # train | val | test_unseen
    hp_per_dps_slot: int   # 基準 HP = hp_per_dps_slot * max(1, N - 2) (hp_override が無ければ使用)
    boss_atk: int          # 単体攻撃ダメージ
    aoe_dmg: int           # 全体攻撃ダメージ
    aoe_interval: int      # 全体攻撃周期（0 = なし）
    battle_turns: int = 30 # 戦闘制限ターン数（エンレイジ）
    min_n_intended: int = 4  # 想定最小人数（例：2T2H型は N>=6 想定）
    hp_floor_by_n: Optional[Dict[int, int]] = None  # Nごとの例外HP指定（N=4での1DPS制限対策など）
    notes: str = ""

    def make_config(self, num_agents: int, **config_overrides) -> RaidConfig:
        """指定人数 N 用の RaidConfig を生成する。"""
        if self.hp_floor_by_n and num_agents in self.hp_floor_by_n:
            bhp = self.hp_floor_by_n[num_agents]
        else:
            bhp = self.hp_per_dps_slot * max(1, num_agents - 2)
        kwargs = dict(
            num_agents=num_agents,
            boss_hp=bhp,
            boss_atk=self.boss_atk,
            boss_aoe_dmg=self.aoe_dmg,
            boss_aoe_interval=self.aoe_interval,
            battle_turns=self.battle_turns,
        )
        kwargs.update(config_overrides)
        return RaidConfig(**kwargs)


# 16種類の多様なボス原型カタログ
# - train (8種): 基本の学習・多様性付与に使う標準〜中難度セット
# - val   (4種): ハイパーパラメータやチェックポイント選定用
# - test_unseen (4種): 未知の周期・ギミック組み合わせに対する汎化テスト用
BOSS_ARCHETYPES: List[BossArchetype] = [
    # --- A. 単体・DPSレース系 ---
    BossArchetype(
        boss_id="01_std_single",
        name_ja="標準単体ボス",
        category="single",
        split="train",
        hp_per_dps_slot=880,
        boss_atk=105,
        aoe_dmg=0,
        aoe_interval=0,
        notes="1:1:(N-2) 基準ボス。タンク維持＋単体回復の基本形。",
    ),
    BossArchetype(
        boss_id="02_tight_enrage_single",
        name_ja="タイトDPSチェック単体",
        category="single",
        split="val",
        hp_per_dps_slot=940,
        boss_atk=105,
        aoe_dmg=0,
        aoe_interval=0,
        notes="余剰タンク・余剰ヒーラーを許さない高HPレース。",
    ),
    BossArchetype(
        boss_id="03_fast_rush_20t",
        name_ja="短期決戦ラッシュ(20T)",
        category="single",
        split="test_unseen",
        hp_per_dps_slot=450,
        boss_atk=115,
        aoe_dmg=35,
        aoe_interval=4,
        battle_turns=20,
        notes="20ターン制限の短期戦。序盤から高火力＋AoEケアを要求し、0H抜道を塞ぐ設定。",
    ),

    # --- B. 弱〜中 AoE・MP温存／戦術切り分け系 ---
    BossArchetype(
        boss_id="04_light_aoe_5t",
        name_ja="弱AoE入門(25/5T)",
        category="tactical_aoe",
        split="train",
        hp_per_dps_slot=820,
        boss_atk=105,
        aoe_dmg=25,
        aoe_interval=5,
        notes="5Tごとの軽い全体攻撃。全体回復を使えば 1:1 で完勝。",
    ),
    BossArchetype(
        boss_id="05_mod_aoe_5t",
        name_ja="中AoE標準(30/5T)",
        category="tactical_aoe",
        split="train",
        hp_per_dps_slot=780,
        boss_atk=105,
        aoe_dmg=30,
        aoe_interval=5,
        notes="単体回復だけではDPSが落ちる標準全体攻撃ボス。",
    ),
    BossArchetype(
        boss_id="06_freq_aoe_mp_3t",
        name_ja="高頻度弱AoE(25/3T)",
        category="tactical_aoe",
        split="val",
        hp_per_dps_slot=720,
        boss_atk=105,
        aoe_dmg=25,
        aoe_interval=3,
        notes="3T周期。雑に撃つとMPが枯れるため MP温存 or 2H が有効。",
    ),
    BossArchetype(
        boss_id="07_tactics_vs_comp_35_4t",
        name_ja="戦術vs編成AoE(35/4T)",
        category="tactical_aoe",
        split="train",
        hp_per_dps_slot=650,
        boss_atk=105,
        aoe_dmg=35,
        aoe_interval=4,
        notes="1HはMP温存で犠牲を出しつつ辛勝、2Hなら全員生還する代表例。",
    ),

    # --- C. 強AoE・複数ヒーラー要求系 ---
    BossArchetype(
        boss_id="08_hard_aoe_45_4t",
        name_ja="強AoE複数H圧力(45/4T)",
        category="multi_healer",
        split="train",
        hp_per_dps_slot=620,
        boss_atk=110,
        aoe_dmg=45,
        aoe_interval=4,
        min_n_intended=6,
        notes="1Hの回復総量を超える圧力。N>=6 では 2H 以上の編成を要求。",
    ),
    BossArchetype(
        boss_id="09_fast_hard_aoe_3t",
        name_ja="高速強AoE(35/3T)",
        category="multi_healer",
        split="test_unseen",
        hp_per_dps_slot=600,
        boss_atk=105,
        aoe_dmg=35,
        aoe_interval=3,
        min_n_intended=6,
        notes="未知の3T高頻度強AoE。即時の全体回復連携が必要。",
    ),
    BossArchetype(
        boss_id="10_heavy_aoe_55_4t",
        name_ja="激重AoE(55/4T)",
        category="multi_healer",
        split="val",
        hp_per_dps_slot=580,
        boss_atk=105,
        aoe_dmg=55,
        aoe_interval=4,
        min_n_intended=6,
        notes="2HでもMP管理が必要、大人数では3Hも有効になる極圧AoE。",
    ),
    BossArchetype(
        boss_id="11_spike_aoe_65_5t",
        name_ja="スパイク大AoE(65/5T)",
        category="multi_healer",
        split="test_unseen",
        hp_per_dps_slot=620,
        boss_atk=105,
        aoe_dmg=65,
        aoe_interval=5,
        min_n_intended=6,
        notes="一撃65ダメで瀕死になるため、5T間隔の戻しが必須なスパイク型。",
    ),

    # --- D. 単体強打・タンクスイッチ系 ---
    BossArchetype(
        boss_id="12_tank_pressure_155",
        name_ja="単体強打(ATK155+軽AoE)",
        category="tank_buster",
        split="train",
        hp_per_dps_slot=760,
        boss_atk=155,
        aoe_dmg=20,
        aoe_interval=5,
        notes="タンク被ダメが重く、軽AoEで0H抜道を封じつつ1T1Hのヒーラー配分を試す。",
    ),
    BossArchetype(
        boss_id="13_tank_buster_205",
        name_ja="タンクバスター(ATK205)",
        category="tank_buster",
        split="train",
        hp_per_dps_slot=680,
        boss_atk=205,
        aoe_dmg=25,
        aoe_interval=5,
        min_n_intended=6,
        notes="1T1Hは確実に崩壊。1T2H または 2T交代挑発(スイッチ)が必須。",
    ),
    BossArchetype(
        boss_id="14_swap_specialist_235",
        name_ja="2Tスイッチ特化(ATK235)",
        category="tank_buster",
        split="test_unseen",
        hp_per_dps_slot=620,
        boss_atk=235,
        aoe_dmg=25,
        aoe_interval=5,
        min_n_intended=6,
        notes="N=6では2T交代(2:1:3)のみ勝利可。大人数でも2T交代が大きく有利。",
    ),

    # --- E. 単体強打＋全体攻撃 複合系 ---
    BossArchetype(
        boss_id="15_hybrid_145_30_4t",
        name_ja="単体+AoE複合(ATK145+30/4T)",
        category="hybrid",
        split="val",
        hp_per_dps_slot=640,
        boss_atk=145,
        aoe_dmg=30,
        aoe_interval=4,
        min_n_intended=6,
        notes="タンク介護と全体戻しの同時処理が求められ、分担・MP配分が効く。",
    ),
    BossArchetype(
        boss_id="16_endurance_200_25_4t",
        name_ja="総合試験(ATK200+AoE25/4T)",
        category="hybrid",
        split="train",
        hp_per_dps_slot=600,
        boss_atk=200,
        aoe_dmg=25,
        aoe_interval=4,
        min_n_intended=6,
        notes="タンクスイッチ/2H と全体ケアの両方を要求（0H抜道なし）。",
    ),
]


BOSS_BY_ID: Dict[str, BossArchetype] = {b.boss_id: b for b in BOSS_ARCHETYPES}


def list_boss_ids(split: Optional[str] = None) -> List[str]:
    """指定 split（省略時はすべて）の boss_id 一覧を返す。"""
    return [b.boss_id for b in BOSS_ARCHETYPES if split is None or b.split == split]


def make_boss_config(boss_id: str, num_agents: int, **overrides) -> RaidConfig:
    """boss_id と参加人数 num_agents から RaidConfig を構築する。"""
    return BOSS_BY_ID[boss_id].make_config(num_agents=num_agents, **overrides)
