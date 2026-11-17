"""Frozen H-versus-DPS role intervention with paired action/message draws."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .random_ready import draw
from .ready_policy import MemberBatch, Policies
from .ready_prep import Prep


def sample_logits(logits, uniform):
    cdf = logits.softmax(-1).cumsum(-1)
    return (uniform.unsqueeze(-1) >= cdf / cdf[..., -1:]).sum(-1).clamp_max(logits.size(-1) - 1)


@torch.no_grad()
def run_arm(model_batch, ids, seats, role, draws, width, lock_ready=False):
    prep = Prep(len(ids), lock_ready=lock_ready)
    h = model_batch.pack(torch.zeros(*ids.shape, width))
    row = torch.arange(len(ids))
    for turn, (ua, um) in enumerate(draws):
        obs, peers, mask = prep.observations()
        a, m, _, h = model_batch.forward(obs, peers, h, mask)
        picked = sample_logits(a, ua)
        msgs = sample_logits(m, um)
        # Seat still communicates and controls readiness. Substitute WAIT only
        # for a subsequent job-change request; keep original action probabilities.
        requests = picked[row, seats]
        if turn == 0:
            picked[row, seats] = 4 + role
        else:
            switch = (requests >= 4) & (requests <= 6) & (requests != 4 + role)
            picked[row, seats] = torch.where(switch, 0, requests)
        prep.advance(picked, msgs)
        if not prep.active.any():
            break
    assert (prep.jobs[row, seats] == role).all()
    return prep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--pairs", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=83491)
    args = parser.parse_args()
    torch.set_num_threads(1)
    lock_ready = json.loads((args.run / "settings.json").read_text()).get("lock_ready", False)
    model = Policies(width=64)
    model.load_state_dict(torch.load(args.run/"final.pt", weights_only=True, map_location="cpu"))
    model.eval()
    rng = np.random.default_rng(args.seed)
    random_stream = torch.Generator().manual_seed(args.seed)
    parties, _ = draw(rng, args.pairs)
    all_seats = rng.integers(0, 6, args.pairs)
    output = args.run/f"partner_HD_{args.seed}_{args.pairs}.csv"
    counts = Counter()
    with torch.no_grad(), output.open("x") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pair","ids","forced_seat","H_arm_jobs","D_arm_jobs","H_arm_turns","D_arm_turns",
                         "H_arm_departed","D_arm_departed","H_arm_other_healers","D_arm_other_healers"])
        for start in range(0, len(parties), 32):
            ids = torch.tensor(parties[start:start+32])
            seats = torch.tensor(all_seats[start:start+32])
            batch = MemberBatch(ids, model.members, model.shapes)
            draws = torch.rand(16, 2, *ids.shape, generator=random_stream)
            h, d = (run_arm(batch, ids, seats, role, draws, model.width, lock_ready) for role in (1, 2))
            for i, slot in enumerate(seats.tolist()):
                nh = int((h.jobs[i] == 1).sum()) - 1
                nd = int((d.jobs[i] == 1).sum())
                counts[nh > 0, nd > 0] += 1
                writer.writerow([start+i, ids[i].tolist(), slot, h.jobs[i].tolist(), d.jobs[i].tolist(),
                                 int(h.turns[i]), int(d.turns[i]), int(h.departed[i]), int(d.departed[i]), nh, nd])
    print(output, 'paired (another H in H-arm, another H in D-arm):', dict(counts))


if __name__ == "__main__":
    main()
