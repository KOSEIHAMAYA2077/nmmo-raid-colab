"""Recover historical ready states and verify the complete saved role trajectories."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .random_ready import draw
from .ready_policy import Policies
from .ready_prep import Prep


def roles(jobs):
    return ''.join('THD-'[j] for j in jobs)


def replay(path, greedy):
    mode = 'greedy' if greedy else 'sampled'
    saved = [json.loads(line) for line in (path / f'final_{mode}.jsonl').read_text().splitlines()]
    parties, seeds = draw(np.random.default_rng(20261116), len(saved))
    lock_ready = json.loads((path / "settings.json").read_text()).get("lock_ready", False)
    model = Policies(width=64)
    model.load_state_dict(torch.load(path / 'final.pt', weights_only=True))
    model.eval()
    torch.manual_seed(1910)
    stats = Counter()
    peaks = Counter()
    examples = {}
    hands = []
    for start in range(0, len(saved), 32):
        ids = torch.tensor(parties[start:start+32])
        prep = Prep(len(ids), lock_ready=lock_ready)
        hidden = torch.zeros(len(ids), 6, 64)
        traces = [[] for _ in ids]
        while prep.active.any():
            active = prep.active.tolist()
            own, allies, mask = prep.observations()
            action, msg, _, _, hidden = model.step(ids, own, allies, hidden, mask, greedy=greedy)
            old_ready, old_jobs = prep.ready, prep.jobs
            prep.advance(action, msg)
            for j, live in enumerate(active):
                if not live:
                    continue
                trial = start+j
                turn = len(traces[j])+1
                jobs = prep.jobs[j].tolist()
                assert jobs == saved[trial]['jobs'][turn-1]
                ready = prep.ready[j].tolist()
                revoked = old_ready[j] & ~prep.ready[j]
                events = dict(trial=trial, turn=turn, ids=' '.join(map(str,parties[trial])),
                              jobs=roles(jobs), ready=''.join(str(int(r)) for r in ready),
                              ready_count=sum(ready), actions=''.join(map(str, action[j].tolist())),
                              messages=''.join(map(str, msg[j].tolist())),
                              explicit_revoke=int((revoked & (action[j]==7)).sum()),
                              job_change_unready=int((revoked & (old_jobs[j]!=prep.jobs[j])).sum()))
                traces[j].append(events)
                hands.append(events)
        for j, trace in enumerate(traces):
            idx = start+j
            row = saved[idx]
            assert (row['ids'],row['seed']) == (parties[idx],int(seeds[idx]))
            assert row['jobs'][-1] == row['final'] == prep.jobs[j].tolist()
            assert row['turns'] == len(trace) and row['departed'] == bool(prep.departed[j])
            exact = lambda s: sorted(s) == sorted('THDDDD')
            tail = next((i for i in range(len(trace)) if all(
                t['jobs'] == trace[-1]['jobs'] for t in trace[i:])), len(trace)-1)
            stable_len = len(trace)-tail
            stats['games'] += 1
            stats['departed'] += row['departed']
            stats['ever_exact'] += any(exact(t['jobs']) for t in trace)
            stats['ever_exact_timeout'] += (not row['departed'] and any(exact(t['jobs']) for t in trace))
            stats['final_exact_timeout'] += not row['departed'] and exact(trace[-1]['jobs'])
            if not row['departed'] and stable_len >= 4 and '-' not in trace[-1]['jobs']:
                stats['stable_assigned_final4_timeout'] += 1
                if exact(trace[-1]['jobs']):
                    stats['stable_exact_final4_timeout'] += 1
                    examples.setdefault('stable_exact', idx)
            stats['total_turns'] += len(trace)
            stats['zero_ready_all_turns'] += all(t['ready_count']==0 for t in trace)
            stats['explicit_revokes'] += sum(t['explicit_revoke'] for t in trace)
            stats['role_change_resets'] += sum(t['job_change_unready'] for t in trace)
            peaks[max(t['ready_count'] for t in trace)] += 1
    with (path / f'final_{mode}_hands.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(hands[0]))
        w.writeheader()
        w.writerows(hands)
    return dict(stats=stats, peak_ready_hist=peaks, examples=examples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+')
    args = parser.parse_args()
    torch.set_num_threads(1)
    with torch.no_grad():
        for path in args.runs:
            summary = {m:replay(path,m=='greedy') for m in ('sampled','greedy')}
            (path/'trace_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
            print(path, json.dumps(summary),flush=True)
