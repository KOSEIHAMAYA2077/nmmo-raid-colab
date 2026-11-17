"""Audit per-member exposure and distinguish role choice from ready-check failures."""
import argparse
from collections import Counter
from contextlib import ExitStack
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from .last_picker import sha256, summary
from .ready_policy import Policies
from .train_last_picker import learning_reward, read_schedule


def read_results(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def role_metrics(rows):
    result = {}
    for arm, desired in [('A', 1), ('B', 2)]:
        outcomes = [row[arm] for row in rows]
        result[arm] = dict(
            trials=len(outcomes),
            ever_correct_hope=sum(any(step['job'] == desired for step in r['steps']) for r in outcomes),
            final_correct_hope=sum(r['hoped_job'] == desired for r in outcomes),
            correct_confirmed=sum(r['confirmed_job'] == desired for r in outcomes),
            correct_hope_but_unconfirmed=sum(r['hoped_job'] == desired and not r['departed'] for r in outcomes),
            confirmed_wrong=sum(r['departed'] and r['confirmed_job'] != desired for r in outcomes),
            never_assigned=sum(r['hoped_job'] < 0 for r in outcomes))
    return result


def audit_training(folder, settings, schedule):
    expected = settings['updates'] * settings['pairs_per_member_per_update']
    counts = np.zeros((128, 2), dtype=int)
    heldout = {tuple(sorted(s['ids'])) for s in schedule}
    updates = {}
    contexts_a, contexts_b = Counter(), Counter()
    episodes, decisions = 0, 0
    with ExitStack() as files:
        handle = files.enter_context(gzip.open(folder/'training.jsonl.gz', 'rt'))
        reference = None
        if settings.get('reference_run'):
            reference = files.enter_context(gzip.open(Path(settings['reference_run'])/'training.jsonl.gz', 'rt'))
        for line in handle:
            row = json.loads(line)
            expected_reward = learning_reward(row['arm'], row['confirmed_job'], row['departed'], row['won'],
                                               row.get('combat_reward', row['reward']),
                                               settings.get('equalized_failure_reward'))
            assert row['reward'] == float(np.float32(expected_reward))
            if reference is not None:
                original = json.loads(next(reference))
                for key in ('ids', 'focal_id', 'focal_seat', 'changed_seat', 'A_jobs', 'env_seed', 'arm', 'update'):
                    assert row[key] == original[key]
                if row['update'] == 1:
                    assert row['history'] == original['history']
                    assert float(np.float32(row['combat_reward'])) == original['reward']
            episodes += 1
            update = row['update']
            arm_index = int(row['arm'] == 'B')
            counts[row['focal_id'], arm_index] += 1
            if update not in updates:
                updates[update] = np.zeros((128, 2), dtype=int)
            updates[update][row['focal_id'], arm_index] += 1
            assert tuple(sorted(row['ids'])) not in heldout
            assert len(set(row['ids'])) == 6
            assert row['ids'][row['focal_seat']] == row['focal_id']
            key = (update, tuple(row['ids']), row['focal_seat'], row['changed_seat'], row['env_seed'])
            (contexts_b if arm_index else contexts_a)[key] += 1
            job, ready = -1, False
            for action, message, new_job, new_ready in row['history']:
                decisions += 1
                assert not ready
                assert action in (0, 4, 5, 6, 7) and 0 <= message < 4
                if action in (4, 5, 6):
                    job = action - 4
                elif action == 7:
                    assert job >= 0
                    ready = True
                assert new_job == job and new_ready == ready
            assert row['hoped_job'] == job
            assert row['departed'] == ready
            assert row['confirmed_job'] == (job if ready else -1)
            assert 1 <= len(row['history']) <= settings['cap']
            assert ready or len(row['history']) == settings['cap']
            if not ready:
                assert not row['won'] and row['reward'] == 0
        if reference is not None:
            assert next(reference, None) is None
    assert (counts == expected).all()
    assert set(updates) == set(range(1, settings['updates'] + 1))
    assert all((v == settings['pairs_per_member_per_update']).all() for v in updates.values())
    assert contexts_a == contexts_b
    assert np.array_equal(counts, json.loads((folder/'appearances.json').read_text()))
    return dict(episodes=episodes, decisions=decisions, context_pairs=len(contexts_a),
                per_member_A=expected, per_member_B=expected, balanced_every_update=True,
                evaluation_party_overlap=0, training_history_consistent=True,
                learning_reward_transform_matched=True,
                reference_schedules_and_first_update_matched=bool(settings.get('reference_run')))



def validate_evaluation(rows, schedule):
    for row, scenario in zip(rows, schedule):
        for arm in ('A', 'B'):
            outcome = row[arm]
            job, ready = -1, False
            for step in outcome['steps']:
                action = step['action']
                assert not ready and action in (0, 4, 5, 6, 7)
                assert len(step['action_probs']) == 8
                assert abs(sum(step['action_probs']) - 1) < 1e-5
                if action in (4, 5, 6):
                    job = action - 4
                elif action == 7:
                    assert job >= 0
                    ready = True
                assert step['job'] == job and step['ready'] == ready
            actual = scenario['A_jobs'].copy()
            if arm == 'B':
                actual[scenario['changed_seat']] = 1
            actual[scenario['focal_seat']] = job
            assert actual == outcome['final_jobs']
            assert outcome['departed'] == ready
            assert outcome['confirmed_job'] == (job if ready else -1)
            assert outcome['hoped_job'] == job
            assert outcome['turns'] == len(outcome['steps'])
            assert 1 <= outcome['turns'] <= 16
            assert ready or outcome['turns'] == 16


def analyze(folder):
    settings = json.loads((folder/'settings.json').read_text())
    schedule = read_schedule(Path(settings['eval_schedule']))
    original = json.loads((folder/'summary.json').read_text())
    result = dict(training=audit_training(folder, settings, schedule), evaluation={})
    for path in sorted(folder.glob('*_sampled.jsonl')) + sorted(folder.glob('*_greedy.jsonl')):
        rows = read_results(path)
        assert [(r['pair'], r['focal_id']) for r in rows] == [(s['pair'], s['focal_id']) for s in schedule]
        validate_evaluation(rows, schedule)
        recomputed = summary(rows)
        assert recomputed == original[path.stem]
        result['evaluation'][path.stem] = dict(**recomputed, role_history=role_metrics(rows))
    result['display_intervention'] = {}
    for mode in ('sampled', 'greedy'):
        normal = read_results(folder/f'final_{mode}.jsonl')
        swapped = read_results(folder/f'final_swapped_{mode}.jsonl')
        result['display_intervention'][mode] = dict(
            pairs=len(normal),
            normal_correct_then_swapped_inverse=sum(
                a['A']['confirmed_job'] == 1 and a['B']['confirmed_job'] == 2
                and b['A']['confirmed_job'] == 2 and b['B']['confirmed_job'] == 1
                for a, b in zip(normal, swapped)),
            both_arms_choices_unchanged=sum(
                all(a[arm]['confirmed_job'] == b[arm]['confirmed_job'] for arm in ('A', 'B'))
                for a, b in zip(normal, swapped)))
        for a, b, scenario in zip(normal, swapped, schedule):
            assert a['A']['steps'] == b['B']['steps']
            assert a['B']['steps'] == b['A']['steps']
            for arm in ('A', 'B'):
                for seat in range(6):
                    if seat != scenario['focal_seat']:
                        assert a[arm]['final_jobs'][seat] == b[arm]['final_jobs'][seat]
    torch.manual_seed(settings['seed'])
    fresh = Policies(width=settings['width'])
    initial = torch.load(folder/'initial.pt', map_location='cpu', weights_only=True)
    assert all(torch.equal(v, initial[k]) for k, v in fresh.state_dict().items())
    if settings.get('reference_run'):
        reference = Path(settings['reference_run'])
        assert sha256(reference/'initial.pt') == settings['reference_initial_sha256']
        assert sha256(reference/'settings.json') == settings['reference_settings_sha256']
        original_initial = torch.load(reference/'initial.pt', map_location='cpu', weights_only=True)
        assert all(torch.equal(v, original_initial[k]) for k, v in initial.items())
    final = torch.load(folder/'final.pt', map_location='cpu', weights_only=True)
    assert all(torch.isfinite(v).all() for v in final.values())
    assert any(not torch.equal(v, initial[k]) for k, v in final.items())
    for name, expected in settings['source_hashes'].items():
        assert sha256(Path(__file__).with_name(name)) == expected
    assert sha256(Path(settings['eval_schedule'])) == settings['eval_schedule_sha256']
    result['validation'] = dict(from_scratch_initial_weights_matched=True, final_weights_finite=True,
                                normal_swapped_real_peer_roles_matched=True, source_hashes_matched=True,
                                eval_schedule_hash_matched=True, evaluation_histories_consistent=True,
                                swapped_trajectories_follow_display=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = analyze(args.run)
    (args.run/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(training=result['training'], validation=result['validation'],
                          display_intervention=result['display_intervention']), indent=2))


if __name__ == '__main__':
    main()
