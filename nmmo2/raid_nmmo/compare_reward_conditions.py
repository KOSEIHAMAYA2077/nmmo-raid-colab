"""Compare exploration and matched role completion under two training rewards."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .last_picker import sha256


def paired_member_gain(original, diagnostic):
    before = {r['focal_id']: r for r in original['by_member']}
    after = {r['focal_id']: r for r in diagnostic['by_member']}
    assert before.keys() == after.keys()
    differences = []
    for person in sorted(before):
        assert before[person]['pairs'] == after[person]['pairs']
        differences.append(100 * (after[person]['correct'] - before[person]['correct']) / before[person]['pairs'])
    differences = np.array(differences)
    rng = np.random.default_rng(20261117)
    samples = differences[rng.integers(0, len(differences), (5000, len(differences)))].mean(1)
    return dict(mean_gain_pp=float(differences.mean()),
                paired_member_bootstrap_95_interval_pp=np.quantile(samples, [.025, .975]).tolist())


def metrics(result):
    n = result['pairs']
    return dict(pairs=n, paired_correct=result['correct'], paired_correct_percent=100*result['correct']/n,
                paired_inverse=result['inverse'], unconfirmed_either=result['unconfirmed_either'],
                A_correct_percent=100*result['role_history']['A']['correct_confirmed']/n,
                B_correct_percent=100*result['role_history']['B']['correct_confirmed']/n,
                average_accuracy_percent=50*(result['role_history']['A']['correct_confirmed']+
                                             result['role_history']['B']['correct_confirmed'])/n,
                mean_H_confirmation_percent=50*(result['A_H']+result['B_H'])/n,
                delta_H_pp=result['delta_H_pp'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--baseline-exploration', type=Path,
                        default=Path('runs/2026-11-16_last_picker_exploration/summary.json'))
    parser.add_argument('--diagnostic-exploration', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    baseline_exploration = {r['seed']: r for r in json.loads(args.baseline_exploration.read_text()).values()}
    diagnostic_exploration = {r['seed']: r for r in json.loads(args.diagnostic_exploration.read_text()).values()}
    args.out.mkdir(exist_ok=False, parents=True)
    results, csv_rows, hashes = {}, [], {}
    for folder in args.runs:
        settings = json.loads((folder/'settings.json').read_text())
        reference = Path(settings['reference_run'])
        baseline = json.loads((reference/'analysis.json').read_text())
        diagnostic = json.loads((folder/'analysis.json').read_text())
        seed = settings['seed']
        assert diagnostic['training']['reference_schedules_and_first_update_matched']
        item = dict(reference=str(reference), diagnostic=str(folder), exploration={}, evaluation={})
        for phase in ('early', 'middle', 'late'):
            old = baseline_exploration[seed]['phases'][phase]
            new = diagnostic_exploration[seed]['phases'][phase]
            assert (old['first_update'], old['last_update']) == (new['first_update'], new['last_update'])
            assert old['arms']['B']['trials'] == new['arms']['B']['trials']
            item['exploration'][phase] = dict(original=old['arms']['B'], diagnostic=new['arms']['B'])
        for mode in ('sampled', 'greedy'):
            old = baseline['evaluation']['final_'+mode]
            new = diagnostic['evaluation']['final_'+mode]
            item['evaluation'][mode] = dict(original=metrics(old), diagnostic=metrics(new),
                                            **paired_member_gain(old, new))
            for condition, values in [('original', metrics(old)), ('diagnostic', metrics(new))]:
                csv_rows.append(dict(seed=seed, mode=mode, condition=condition, **values))
        item['display_intervention'] = diagnostic['display_intervention']
        results[seed] = item
        for path in (reference/'analysis.json', reference/'initial.pt', folder/'settings.json',
                     folder/'analysis.json', args.baseline_exploration, args.diagnostic_exploration):
            hashes[str(path)] = sha256(path)
        print(seed, json.dumps(item['evaluation']), flush=True)
    (args.out/'comparison.json').write_text(json.dumps(results, indent=2)+'\n')
    (args.out/'input_hashes.json').write_text(json.dumps(hashes, indent=2)+'\n')
    with (args.out/'evaluation.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(csv_rows)


if __name__ == '__main__':
    main()
