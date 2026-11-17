"""Train independent IPPO players in random six-player ready-check parties."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .ready_battle import Battle
from .ready_policy import Policies
from .ready_prep import Prep


def run_batch(model, battle, parties, seeds, *, greedy=False, training=False, lock_ready=False):
    ids = torch.tensor(parties, dtype=torch.long)
    batch = len(ids)
    prep = Prep(batch, lock_ready=lock_ready)
    hidden = torch.zeros(batch, 6, 64)
    records, trajectory = [], []
    with torch.no_grad():
        while prep.active.any():
            own, allies, mask = prep.observations()
            active = prep.active.clone()
            action, msg, logp, _, hidden = model.step(ids, own, allies, hidden, mask, greedy=greedy)
            if training:
                records.append(dict(ids=ids, x_self=own.clone(), x_ally=allies.clone(), mask=mask.clone(),
                                    action=action, msg=msg, logp=logp, active=active))
            prep.advance(action, msg)
            trajectory.append(prep.jobs.clone())
    paths = torch.stack(trajectory).transpose(0, 1).tolist()
    rewards = torch.zeros(batch)
    rows = []
    for i, path in enumerate(paths):
        path = path[:int(prep.turns[i])]
        departed = bool(prep.departed[i])
        won = False
        if departed:
            won, rewards[i] = battle.evaluate(path[-1], int(seeds[i]))
        # Anchor at the first fully assigned party. A replacement must change
        # an already assigned job, not merely finish an initial selection.
        first = next((jobs for jobs in path if -1 not in jobs), None)
        missing = [] if first is None else [role for role in (0, 1) if role not in first]
        fills = {role: [int(parties[i][j]) for j, job in enumerate(path[-1]) if job == role]
                 for role in missing}
        rows.append(dict(ids=list(map(int, parties[i])), seed=int(seeds[i]), departed=departed, won=bool(won),
                         turns=len(path), first=first, final=path[-1], missing=missing, fillers=fills,
                         recovered=bool(missing and departed and all(fills.values())), jobs=path))
    return records, rewards, rows


def draw(rng, n):
    parties = [rng.choice(128, size=6, replace=False).tolist() for _ in range(n)]
    return parties, rng.integers(0, 2**31, n)


def summarize(rows):
    return dict(games=len(rows), departures=sum(r['departed'] for r in rows), wins=sum(r['won'] for r in rows),
                shortages=sum(bool(r['missing']) for r in rows), filled_at_departure=sum(r['recovered'] for r in rows),
                filled_and_won=sum(r['recovered'] and r['won'] for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--updates', type=int, default=100)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--eval-games', type=int, default=256)
    parser.add_argument('--lock-ready', action='store_true', help='Commit the chosen job permanently on ready')
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False, parents=True)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = Policies(width=64)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    battle = Battle()
    counts = np.zeros(128, dtype=int)
    seen = set()
    def evaluate(label):
        schedule = np.random.default_rng(20261116)
        parties, seeds = draw(schedule, args.eval_games)
        for greedy in (False, True):
            rows = []
            with torch.random.fork_rng():
                torch.manual_seed(1910)
                for s in range(0, len(parties), 32):
                    _, _, chunk = run_batch(model, battle, parties[s:s+32], seeds[s:s+32], greedy=greedy, lock_ready=args.lock_ready)
                    rows.extend(chunk)
            for r in rows:
                r['party_seen_in_training'] = tuple(sorted(r['ids'])) in seen
            tag = label + ('_greedy' if greedy else '_sampled')
            with (args.out / (tag+'.jsonl')).open('w') as handle:
                for row in rows:
                    print(json.dumps(row), file=handle)
            print(tag, summarize(rows), flush=True)
    try:
        evaluate('initial')
        with (args.out/'training.csv').open('w') as handle:
            writer = None
            for u in range(1, args.updates+1):
                parties, seeds = draw(rng, args.batch)
                for party in parties:
                    counts[party] += 1
                    seen.add(tuple(sorted(party)))
                records, rewards, rows = run_batch(model, battle, parties, seeds, training=True, lock_ready=args.lock_ready)
                model.update(records, rewards, opt)
                stats = dict(update=u, **summarize(rows))
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=stats)
                    writer.writeheader()
                writer.writerow(stats)
                handle.flush()
                if u % 10 == 0:
                    print(stats, flush=True)
        torch.save(model.state_dict(), args.out/'final.pt')
        (args.out/'settings.json').write_text(json.dumps(dict(seed=args.seed, updates=args.updates, batch=args.batch,
              eval_games=args.eval_games, lock_ready=args.lock_ready, appearances=counts.tolist(), parameters=sum(p.numel() for p in model.parameters()))))
        evaluate('final')
    finally:
        battle.close()


if __name__ == '__main__':
    main()
