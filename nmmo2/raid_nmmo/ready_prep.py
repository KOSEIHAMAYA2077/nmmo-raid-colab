"""Six-seat synchronous role selection with revocable ready checks and broadcasts."""
import torch
from torch.nn.functional import one_hot

from .raid_game import ACT_JOB_TANK, ACT_JOB_DPS
from boss_catalog import make_boss_config


class Prep:
    seats = 6

    def __init__(self, batch, cap=16, *, lock_ready=False):
        if batch < 1 or cap < 1:
            raise ValueError("batch and cap must be positive")
        self.cap = cap
        self.lock_ready = lock_ready
        self.cfg = make_boss_config("01_std_single", self.seats)
        self.jobs = torch.full((batch, self.seats), -1, dtype=torch.long)
        self.ready = torch.zeros_like(self.jobs, dtype=torch.bool)
        self.last_msg = torch.zeros_like(self.jobs)
        self.turns = torch.zeros(batch, dtype=torch.long)
        self.active = torch.ones(batch, dtype=torch.bool)
        self.departed = torch.zeros_like(self.active)
        self.peer_seats = torch.arange(self.seats).expand(self.seats, -1)[
            ~torch.eye(self.seats, dtype=torch.bool)].reshape(self.seats, self.seats - 1)

    def observations(self):
        """Global+self, each peer's latest public state, and legal action mask."""
        batch, n = self.jobs.shape
        cfg = self.cfg
        x = torch.zeros(batch, n, 25)
        x[:, :, 0] = 1
        x[:, :, 1] = (self.turns / (self.cap + cfg.battle_turns))[:, None]
        x[:, :, 2:9] = torch.tensor([1, 1, n / 16, cfg.boss_hp / 16000,
                                    cfg.boss_atk / 300, cfg.boss_aoe_dmg / 100,
                                    1 / cfg.boss_aoe_interval if cfg.boss_aoe_interval else 0])
        roles = one_hot(self.jobs.clamp_min(0), 3) * (self.jobs >= 0).unsqueeze(-1)
        symbols = one_hot(self.last_msg, 4)
        x[:, :, 9:12] = roles
        x[:, :, 12:15] = 1  # Full HP/MP, alive. Cooldowns and threat start at zero.
        x[:, :, 20:24] = symbols
        x[:, :, 24] = self.ready
        public = torch.zeros(batch, n, 11)
        public[:, :, :3] = roles
        public[:, :, 3:5] = 1
        public[:, :, 6:10] = symbols * (self.turns > 0)[:, None, None]
        public[:, :, 10] = self.ready
        mask = torch.zeros(batch, n, 8, dtype=torch.bool)
        mask[:, :, 0] = True
        can_choose = self.active[:, None].expand_as(self.ready)
        if self.lock_ready:
            can_choose = can_choose & ~self.ready
        mask[:, :, ACT_JOB_TANK:ACT_JOB_DPS+1] = can_choose.unsqueeze(-1)
        mask[:, :, 7] = can_choose & (self.jobs >= 0)
        return x, public[:, self.peer_seats], mask

    def advance(self, action, msg):
        """Apply simultaneous choices once; a departure on the last legal turn counts."""
        live = self.active[:, None]
        request = (action >= ACT_JOB_TANK) & (action <= ACT_JOB_DPS) & live
        if self.lock_ready:
            request = request & ~self.ready
        new_jobs = torch.where(request, action - ACT_JOB_TANK, self.jobs)
        toggle = (action == 7) & (self.jobs >= 0) & live
        if self.lock_ready:
            self.ready = self.ready | toggle
        else:
            self.ready = (self.ready ^ toggle) & (new_jobs == self.jobs)
        self.jobs = new_jobs
        self.last_msg = torch.where(live, msg, self.last_msg)
        self.turns = self.turns + self.active
        self.departed = self.departed | (self.active & self.ready.all(dim=1))
        self.active = self.active & ~self.departed & (self.turns < self.cap)
