"""Counterfactual teammate-job cue sensitivity at naturally reached ready-check decisions."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from .random_ready import draw
from .ready_policy import MemberBatch, Policies
from .ready_prep import Prep


@torch.no_grad()
def probe(folder, games, schedule_seed):
    cfg = json.loads((folder / 'settings.json').read_text())
    torch.manual_seed(cfg['seed'])
    initial = Policies()
    final = Policies()
    final.load_state_dict(torch.load(folder / 'final.pt', weights_only=True))
    parties, _ = draw(np.random.default_rng(schedule_seed), games)
    totals = defaultdict(lambda: np.zeros(6))
    torch.manual_seed(57291)
    for start in range(0, games, 32):
        ids = torch.tensor(parties[start:start+32])
        prep = Prep(len(ids), lock_ready=cfg['lock_ready'])
        hidden = {name: torch.zeros(*ids.shape, 64) for name in ('initial', 'final')}
        batches = {name: MemberBatch(ids, model.members, model.shapes) for name, model in
                   [('initial', initial), ('final', final)]}
        while prep.active.any():
            own, allies, mask = prep.observations()
            healer_counts = allies[..., 1].sum(-1)
            eligible = prep.active[:, None] & ~prep.ready
            for name, batch in batches.items():
                h = batch.pack(hidden[name])
                logits, msg, _, new_h = batch.forward(own, allies, h, mask)
                factual = logits.softmax(-1)
                # Each focal seat sees one modified sender; other seats' inputs
                # never enter its policy. Never advance the counterfactual branch.
                for peer in range(5):
                    observation = allies.clone()
                    observation[:, :, peer, [1, 2]] = allies[:, :, peer, [2, 1]]
                    changed_logits = batch.forward(own, observation, h, mask)[0]
                    changed = changed_logits.softmax(-1)
                    for column, direction in [(1, 'H_to_D'), (2, 'D_to_H')]:
                        natural = eligible & allies[:, :, peer, column].bool()
                        no_other = healer_counts == int(column == 1)
                        for subset, select in [
                            ('all', natural),
                            ('no_other_peer_H', natural & no_other),
                            ('no_other_peer_H_and_focal_nonH', natural & no_other & (prep.jobs != 1)),
                            ('no_other_peer_H_and_focal_T_or_D', natural & no_other & (prep.jobs >= 0) & (prep.jobs != 1)),
                        ]:
                            if not select.any():
                                continue
                            a, b = factual[select], changed[select]
                            change = b[:, 5] - a[:, 5]
                            totals[name, direction, subset] += [len(a), change.double().sum().item(),
                                change.abs().double().sum().item(), (a.argmax(1) != b.argmax(1)).sum().item(),
                                a[:, 5].double().sum().item(), b[:, 5].double().sum().item()]
                hidden[name] = batch.unpack(new_h)
                if name == 'final':
                    action = torch.distributions.Categorical(logits=logits).sample()
                    message = torch.distributions.Categorical(logits=msg).sample()
            prep.advance(action, message)
    output = {}
    for (model, direction, subset), numbers in totals.items():
        n, delta, abs_delta, flips, before, after = numbers.tolist()
        output.setdefault(model, {}).setdefault(direction, {})[subset] = dict(
            probes=int(n), greedy_flips=int(flips), mean_delta_pH=delta/n,
            mean_abs_delta_pH=abs_delta/n, mean_factual_pH=before/n, mean_counterfactual_pH=after/n)
    return dict(parties=games, party_seed=schedule_seed, action_seed=57291, data=output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--games', type=int, default=256)
    parser.add_argument('--seed', type=int, default=92917)
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(probe(args.folder, args.games, args.seed), indent=2))
