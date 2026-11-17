"""Measure role exploration over training without loading or updating policies."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path


def episode_metrics(row):
    jobs = [step[2] for step in row['history']]
    assigned = [job for job in jobs if job >= 0]
    first_d = next((i for i, job in enumerate(jobs) if job == 2), None)
    ever_d = first_d is not None
    return Counter(
        trials=1,
        decisions=len(jobs),
        first_role_D=int(bool(assigned) and assigned[0] == 2),
        ever_D=int(ever_d),
        D_confirmed=int(row['confirmed_job'] == 2),
        H_confirmed=int(row['confirmed_job'] == 1),
        T_confirmed=int(row['confirmed_job'] == 0),
        unconfirmed=int(not row['departed']),
        ever_D_then_H=int(ever_d and 1 in jobs[first_d + 1:]),
        ever_D_and_end_H_confirmed=int(ever_d and row['confirmed_job'] == 1),
        final_D_unconfirmed=int(row['hoped_job'] == 2 and not row['departed']),
        D_requests=sum(step[0] == 6 for step in row['history']),
    )


def describe(counts):
    result = dict(counts)
    trials = counts['trials']
    for name in ('first_role_D', 'ever_D', 'D_confirmed', 'H_confirmed', 'T_confirmed',
                 'unconfirmed', 'ever_D_then_H', 'ever_D_and_end_H_confirmed', 'final_D_unconfirmed'):
        result[name + '_percent'] = 100 * counts[name] / trials
    result['mean_decisions'] = counts['decisions'] / trials
    result['D_requests_per_decision_percent'] = 100 * counts['D_requests'] / counts['decisions']
    result['D_confirmed_given_ever_D_percent'] = 100 * counts['D_confirmed'] / counts['ever_D'] if counts['ever_D'] else None
    result['end_H_given_ever_D_percent'] = 100 * counts['ever_D_and_end_H_confirmed'] / counts['ever_D'] if counts['ever_D'] else None
    return result


def analyze(folder):
    settings = json.loads((folder / 'settings.json').read_text())
    updates = settings['updates']
    boundaries = [updates // 3, 2 * updates // 3, updates]
    sums = defaultdict(Counter)
    members = defaultdict(Counter)
    per_update = defaultdict(Counter)
    with gzip.open(folder / 'training.jsonl.gz', 'rt') as handle:
        for line in handle:
            row = json.loads(line)
            phase = next(i for i, end in enumerate(boundaries) if row['update'] <= end)
            values = episode_metrics(row)
            sums[phase, row['arm']].update(values)
            members[phase, row['arm'], row['focal_id']].update(values)
            per_update[row['update'], row['arm']].update(values)
    # Reconcile final-role counts with the original training CSV, independently
    # aggregated from the individual episodes rather than from this new analysis.
    with (folder / 'training.csv').open() as handle:
        for original in csv.DictReader(handle):
            for arm in ('A', 'B'):
                values = per_update[int(original['update']), arm]
                target = 'H_confirmed' if arm == 'A' else 'D_confirmed'
                assert values[target] == int(original[arm + '_correct'])
                assert values['trials'] - values['unconfirmed'] == int(original[arm + '_confirmed'])
                assert values['trials'] == 128 * settings['pairs_per_member_per_update']
    phases = {}
    for phase, name in enumerate(('early', 'middle', 'late')):
        phases[name] = dict(first_update=1 if phase == 0 else boundaries[phase - 1] + 1,
                            last_update=boundaries[phase],
                            arms={arm: describe(sums[phase, arm]) for arm in ('A', 'B')},
                            members={arm: {str(member): describe(members[phase, arm, member])
                                           for member in range(128)} for arm in ('A', 'B')})
    for arm in ('A', 'B'):
        assert sum(sums[i, arm]['trials'] for i in range(3)) == settings['updates'] * settings['pairs_per_member_per_update'] * 128
    def member_rate(phase, member, field):
        counts = members[phase, 'B', member]
        return counts[field] / counts['trials']
    for phase, data in phases.items():
        expected_trials = (data['last_update'] - data['first_update'] + 1) * settings['pairs_per_member_per_update']
        assert all(counts['trials'] == expected_trials for arm in ('A', 'B')
                   for counts in data['members'][arm].values())
    comparison = {}
    for field in ('ever_D', 'first_role_D', 'D_confirmed'):
        difference = [member_rate(2, member, field) - member_rate(0, member, field) for member in range(128)]
        comparison[field] = dict(decreased=sum(d < 0 for d in difference),
                                 unchanged=sum(d == 0 for d in difference),
                                 increased=sum(d > 0 for d in difference))
    return dict(seed=settings['seed'], phases=phases, member_early_late_B_comparison=comparison,
                validation=dict(all_training_updates_reconciled=True, all_member_context_counts_matched=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    result = {str(folder): analyze(folder) for folder in args.runs}
    (args.out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    columns = ['seed', 'phase', 'first_update', 'last_update', 'trials', 'first_role_D_percent',
               'ever_D_percent', 'D_confirmed_percent', 'ever_D_and_end_H_confirmed_percent',
               'D_confirmed_given_ever_D_percent', 'end_H_given_ever_D_percent',
               'mean_decisions', 'D_requests_per_decision_percent']
    with (args.out / 'B_summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        for folder, item in result.items():
            print(folder)
            for phase, data in item['phases'].items():
                values = data['arms']['B']
                row = dict(seed=item['seed'], phase=phase,
                           first_update=data['first_update'], last_update=data['last_update'],
                           **{column: values[column] for column in columns[4:]})
                writer.writerow(row)
                print(json.dumps(row))
            print(item['member_early_late_B_comparison'])


if __name__ == '__main__':
    main()
