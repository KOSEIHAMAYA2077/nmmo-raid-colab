"""Independent recurrent PPO policies for six-person ready-check parties."""
import math

import torch
from torch import nn


class MemberBatch:
    """Gather seats playing the same policy into a single batched matrix multiply."""

    def __init__(self, ids, members, shapes):
        self.party_shape = ids.shape
        people, owner, counts = ids.flatten().unique(sorted=True, return_inverse=True, return_counts=True)
        order = owner.argsort(stable=True)
        offsets = torch.arange(owner.numel(), device=ids.device) - (counts.cumsum(0) - counts)[owner[order]]
        self.columns = int(counts.max())
        self.rows = len(people)
        self.positions = (owner[order] * self.columns + offsets)[order.argsort()]
        self.members = [members[k] for k in people.tolist()]
        flat = torch.stack(self.members)
        self.device = flat.device
        slices = flat.split([math.prod(shape) for shape in shapes], dim=1)
        self.weights = [part.reshape(self.rows, *shape) for part, shape in zip(slices, shapes)]

    def pack(self, x):
        seats = x.flatten(0, 1)
        grouped = seats.new_zeros(self.rows * self.columns, *seats.shape[1:])
        grouped[self.positions] = seats
        return grouped.reshape(self.rows, self.columns, *seats.shape[1:])

    def unpack(self, x):
        seats = x.flatten(0, 1)[self.positions]
        return seats.reshape(*self.party_shape, *x.shape[2:])

    def forward(self, obs, allies, h, mask):
        w_a, b_a, w_x, b_x, w_i, b_i, w_h, b_h, w_o, b_o = self.weights
        ally_rows = self.pack(allies)
        e = torch.bmm(ally_rows.flatten(1, 2), w_a) + b_a
        mean = e.tanh().reshape(self.rows, self.columns, allies.shape[-2], -1).mean(2)
        x = torch.cat((self.pack(obs), mean), dim=-1)
        x = (torch.bmm(x, w_x) + b_x).tanh()
        i_r, i_z, i_n = (torch.bmm(x, w_i) + b_i).chunk(3, -1)
        h_r, h_z, h_n = (torch.bmm(h, w_h) + b_h).chunk(3, -1)
        z = (i_z + h_z).sigmoid()
        candidate = (i_n + (i_r + h_r).sigmoid() * h_n).tanh()
        h = candidate + z * (h - candidate)
        out = self.unpack(torch.bmm(h, w_o) + b_o)
        act, msg, value = out.split((8, 4, 1), dim=-1)
        return act.masked_fill(~mask, torch.finfo(act.dtype).min), msg, value.squeeze(-1), h


class Policies(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.width = width
        # One Parameter per member, so absent members have no gradient or Adam update.
        self.shapes = [(11, 16), (1, 16), (41, width), (1, width),
                       (width, 3 * width), (1, 3 * width), (width, 3 * width),
                       (1, 3 * width), (width, 13), (1, 13)]
        self.members = nn.ParameterList()
        for _ in range(128):
            parts = []
            for i, shape in enumerate(self.shapes):
                bound = 1 / math.sqrt(self.shapes[i - i % 2][0])
                parts.append(torch.empty(shape).uniform_(-bound, bound).flatten())
            self.members.append(nn.Parameter(torch.cat(parts)))

    def step(self, ids, x_self, x_ally, hidden, mask, greedy=False):
        batch = MemberBatch(ids, self.members, self.shapes)
        action, msg, value, h = batch.forward(x_self, x_ally, batch.pack(hidden), mask)
        action = torch.distributions.Categorical(logits=action)
        msg = torch.distributions.Categorical(logits=msg)
        a = action.logits.argmax(-1) if greedy else action.sample()
        m = msg.logits.argmax(-1) if greedy else msg.sample()
        return a, m, action.log_prob(a) + msg.log_prob(m), value, batch.unpack(h)

    def update(self, records, returns, optimizer, epochs=3):
        """Undiscounted terminal outcome, recurrent PPO with per-member normalization."""
        ids = records[0]["ids"]
        valid = torch.stack([r["active"] for r in records])[:, :, None].expand(-1, -1, ids.size(1))
        owner = ids.expand(valid.size(0), -1, -1)[valid]
        count = torch.bincount(owner, minlength=len(self.members)).clamp_min(1)
        weights = 1 / count[owner]
        old_logp = torch.stack([r["logp"] for r in records])[valid].detach()
        target = returns.reshape(1, -1, 1).expand_as(valid)[valid].detach()
        advantage = None
        for _ in range(epochs):
            batch = MemberBatch(ids, self.members, self.shapes)
            h = returns.new_zeros(batch.rows, batch.columns, self.width)
            logps, entropies, values = [], [], []
            for r in records:
                a, m, v, h = batch.forward(r["x_self"], r["x_ally"], h, r["mask"])
                ad = torch.distributions.Categorical(logits=a)
                md = torch.distributions.Categorical(logits=m)
                logps.append(ad.log_prob(r["action"]) + md.log_prob(r["msg"]))
                entropies.append(.02 * ad.entropy() + .01 * md.entropy())
                values.append(v)
            value = torch.stack(values)[valid]
            if advantage is None:
                adv = (target - value).detach()
                mean = adv.new_zeros(len(self.members)).scatter_add_(0, owner, adv * weights)
                centered = adv - mean[owner]
                var = adv.new_zeros(len(self.members)).scatter_add_(0, owner, centered.square() * weights)
                advantage = centered / (var[owner] + 1e-8).sqrt()
            ratio = (torch.stack(logps)[valid] - old_logp).exp()
            policy = torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage)
            loss = (weights * (.5 * (value - target).square() - policy - torch.stack(entropies)[valid])).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for member in batch.members:
                nn.utils.clip_grad_norm_([member], .5)
            optimizer.step()
        return float(loss.detach()) / len(batch.members)
