"""CPU/CUDA runner for Colab width/experience comparisons.

Training starts only through the explicit train command. Completed log chunks
and full optimizer/RNG checkpoints are published with checksummed manifests.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import csv
import gzip
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import tempfile
from time import perf_counter
import uuid

import numpy as np
import torch

from .certified_battle import CertifiedBattle
from .last_picker import sha256
from .ready_battle import Battle
from .ready_policy import Policies
from .train_last_picker import collect, evaluate, read_schedule, training_schedule

MILESTONES = (256, 1024, 4096)
SCHEMA = 'last_picker_scaling_v1'
ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = [
    'nmmo2/raid_nmmo/' + name for name in (
        'scaling_last_picker.py', 'train_last_picker.py', 'last_picker.py',
        'ready_policy.py', 'ready_prep.py', 'ready_partner.py', 'random_ready.py',
        'ready_battle.py', 'certified_battle.py', 'raid_game.py', 'combat_policies.py')
] + ['nmmo2/raid_proto/env.py', 'nmmo2/raid_proto/boss_catalog.py']


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def runtime(device='cpu'):
    versions = {p: importlib.metadata.version(p) for p in ('torch', 'numpy', 'nmmo', 'gymnasium', 'pettingzoo')}
    return dict(python=platform.python_version(), packages=versions,
                platform=platform.platform(), machine=platform.machine(),
                processor=platform.processor(), cpu_count=os.cpu_count(),
                torch_threads=torch.get_num_threads(), device=device, dtype='float32',
                torch_cuda_version=torch.version.cuda,
                cuda_device=torch.cuda.get_device_name(0) if device == 'cuda' else None,
                cuda_capability=list(torch.cuda.get_device_capability(0)) if device == 'cuda' else None)


def provenance():
    def git(*args):
        return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()
    return dict(commit=git('rev-parse', 'HEAD'),
                tracked_dirty=bool(git('status', '--porcelain', '--untracked-files=no')),
                source_hashes={name: sha256(ROOT / name) for name in SOURCE_FILES})


def validate_schedule(schedule):
    if len(schedule) != 1536:
        raise ValueError('The comparison requires the agreed 1,536-pair evaluation schedule')
    counts = Counter()
    parties = set()
    for row in schedule:
        ids, seat, changed = row['ids'], row['focal_seat'], row['changed_seat']
        if (len(ids) != 6 or len(set(ids)) != 6 or any(not 0 <= i < 128 for i in ids)
                or not 0 <= seat < 6 or changed == seat or not 0 <= changed < 6
                or ids[seat] != row['focal_id'] or row['A_jobs'][seat] != -1
                or row['A_jobs'][changed] != 2 or sorted(row['A_jobs']) != [-1, 0, 2, 2, 2, 2]):
            raise ValueError('Invalid A/B last-picker scenario')
        parties.add(tuple(sorted(ids)))
        counts[row['focal_id'], seat] += 1
    if len(parties) != 1536 or any(counts[i, s] != 2 for i in range(128) for s in range(6)):
        raise ValueError('Schedule must have unique parties and two pairs per member per seat')


def configure(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@contextmanager
def isolated_evaluation():
    """Evaluation must not advance any training random stream."""
    py_state, np_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=devices):
        try:
            yield
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)


def snapshot(model, optimizer, rng, appearances, update, settings, battle):
    return dict(schema=SCHEMA, model=model.state_dict(), optimizer=optimizer.state_dict(),
                next_update=update, appearances=appearances.tolist(),
                numpy_generator=rng.bit_generator.state, numpy_global=np.random.get_state(),
                python_random=random.getstate(), torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
                settings=settings, battle_cache=dict(battle._unambiguous))


def restore(state, model, optimizer, rng, battle):
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    rng.bit_generator.state = state['numpy_generator']
    np.random.set_state(state['numpy_global'])
    random.setstate(state['python_random'])
    torch.set_rng_state(state['torch_cpu'])
    if state['torch_cuda']:
        torch.cuda.set_rng_state_all(state['torch_cuda'])
    battle._unambiguous = state['battle_cache']
    return np.array(state['appearances'], dtype=np.int64), state['next_update']


def checked_manifest(folder, *, checkpoint=False):
    manifest = json.loads((folder / 'manifest.json').read_text())
    if manifest.get('schema') != SCHEMA:
        raise ValueError('Unknown commit schema')
    for filename, digest in manifest['files'].items():
        if Path(filename).name != filename or sha256(folder / filename) != digest:
            raise ValueError(f'Corrupt artifact: {folder / filename}')
    if checkpoint and sha256(folder / 'state.pt') != manifest['checkpoint_sha256']:
        raise ValueError(f'Corrupt checkpoint: {folder}')
    return manifest


def publish_folder(staged, destination):
    """The final manifest is the commit marker, including on Drive's FUSE mount."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        checked_manifest(destination)
        if sha256(destination / 'manifest.json') != sha256(staged / 'manifest.json'):
            raise ValueError(f'Refusing to overwrite a different committed range: {destination}')
        return
    pending = destination.parent / ('.pending-' + uuid.uuid4().hex)
    pending.mkdir()
    for item in staged.iterdir():
        if item.name != 'manifest.json':
            shutil.copyfile(item, pending / item.name)
    shutil.copyfile(staged / 'manifest.json', pending / 'manifest.json')
    checked_manifest(pending, checkpoint=(pending / 'state.pt').exists())
    os.replace(pending, destination)


class RunStore:
    """One writer per run. Logs remain immutable; keep bounded resume snapshots."""
    def __init__(self, local, durable=None):
        self.local = Path(local).resolve()
        self.durable = Path(durable).resolve() if durable is not None else None
        if self.local == self.durable:
            raise ValueError('Use a separate local working directory and durable directory')
        self.local.mkdir(parents=True, exist_ok=True)

    def initialize(self, settings, schedule_path):
        # A fresh /content session restores the durable run, not a fresh model.
        if self.durable and (self.durable / 'settings.json').exists() and not (self.local / 'settings.json').exists():
            self.pull()
        if (self.local / 'settings.json').exists():
            saved = json.loads((self.local / 'settings.json').read_text())
            for key in ('schema', 'seed', 'width', 'pairs_per_member', 'milestones', 'checkpoint_every',
                        'eval_schedule_sha256', 'reward_mode', 'battle_mode', 'source_hashes', 'backend'):
                if saved[key] != settings[key]:
                    raise ValueError(f'Resume configuration differs: {key}')
            # Python/package/backend differences may change exact continuation.
            for key in ('python', 'packages', 'device', 'dtype', 'torch_threads',
                        'torch_cuda_version', 'cuda_device', 'cuda_capability'):
                if saved['runtime'][key] != settings['runtime'][key]:
                    raise ValueError(f'Resume runtime differs: {key}')
            return saved
        write_json(self.local / 'settings.json', settings)
        shutil.copyfile(schedule_path, self.local / 'schedule.jsonl')
        return settings

    def pull(self):
        for filename in ('settings.json', 'schedule.jsonl'):
            shutil.copyfile(self.durable / filename, self.local / filename)
        # Copy only complete, checked ranges. Interrupted pending folders are ignored.
        for folder in sorted((self.durable / 'chunks').glob('u*')):
            checked_manifest(folder)
            publish_folder(folder, self.local / 'chunks' / folder.name)
        if (self.durable / 'evaluations').exists():
            for folder in sorted((self.durable / 'evaluations').glob('u*')):
                checked_manifest(folder)
                publish_folder(folder, self.local / 'evaluations' / folder.name)

    def scan(self):
        previous_end, previous_hash = 0, None
        valid = []
        for folder in sorted((self.local / 'chunks').glob('u*')):
            manifest = checked_manifest(folder)
            if manifest['settings_sha256'] != sha256(self.local / 'settings.json'):
                raise ValueError(f'Run settings changed at {folder}')
            if manifest['start_update'] != previous_end or manifest['previous_manifest'] != previous_hash:
                raise ValueError(f'Broken or duplicate update chain at {folder}')
            previous_end = manifest['end_update']
            previous_hash = sha256(folder / 'manifest.json')
            valid.append((folder, manifest))
        return valid

    def latest(self):
        ranges = self.scan()
        if not ranges:
            return None
        folder, manifest = ranges[-1]
        # Do not silently roll back committed logs to an older state.
        checked_manifest(folder, checkpoint=True)
        state = torch.load(folder / 'state.pt', map_location='cpu', weights_only=False)
        if state['next_update'] != manifest['end_update'] or state['settings'] != json.loads((self.local/'settings.json').read_text()):
            raise ValueError('Checkpoint and committed range disagree')
        return state

    def commit(self, state, outcomes, stats):
        ranges = self.scan()
        start = ranges[-1][1]['end_update'] if ranges else 0
        end = state['next_update']
        if stats and ([s['update'] for s in stats] != list(range(start + 1, end + 1))):
            raise ValueError('Each committed update must appear exactly once')
        stage = Path(tempfile.mkdtemp(prefix='.commit-', dir=self.local))
        with (stage / 'trials.jsonl.gz').open('wb') as raw:
            with gzip.GzipFile(fileobj=raw, filename='', mode='wb', mtime=0) as zipped:
                with io.TextIOWrapper(zipped, encoding='utf-8') as text:
                    for row in outcomes:
                        text.write(json.dumps(row, separators=(',', ':')) + '\n')
        write_json(stage / 'stats.json', stats)
        torch.save(state, stage / 'state.pt')
        manifest = dict(schema=SCHEMA, start_update=start, end_update=end,
                        previous_manifest=sha256(ranges[-1][0] / 'manifest.json') if ranges else None,
                        settings_sha256=sha256(self.local / 'settings.json'),
                        files={name: sha256(stage / name) for name in ('trials.jsonl.gz', 'stats.json')},
                        checkpoint_sha256=sha256(stage / 'state.pt'))
        write_json(stage / 'manifest.json', manifest)
        name = f'u{end:06d}'
        publish_folder(stage, self.local / 'chunks' / name)
        if self.durable:
            self.durable.mkdir(parents=True, exist_ok=True)
            for filename in ('settings.json', 'schedule.jsonl'):
                target = self.durable / filename
                if target.exists() and sha256(target) != sha256(self.local / filename):
                    raise ValueError(f'Durable metadata differs: {target}')
                shutil.copyfile(self.local / filename, target)
            publish_folder(stage, self.durable / 'chunks' / name)
        shutil.rmtree(stage)  # Only our disposable staging directory.
        return manifest

    def prune_checkpoints(self, milestones):
        """Only generated redundant restart states; never logs or milestone states."""
        ranges = self.scan()
        keep = {p.name for p, _ in ranges[-2:]}
        keep.update(f'u{u:06d}' for u in (0, *milestones))
        for folder, _ in ranges:
            if folder.name not in keep:
                for root in (self.local, self.durable):
                    if root is not None:
                        (root / 'chunks' / folder.name / 'state.pt').unlink(missing_ok=True)

    def sync_pending(self):
        """A process may have committed locally just before a Drive interruption."""
        if self.durable:
            self.durable.mkdir(parents=True, exist_ok=True)
            for filename in ('settings.json', 'schedule.jsonl'):
                target = self.durable / filename
                if target.exists() and sha256(target) != sha256(self.local / filename):
                    raise ValueError('Local and Drive run metadata conflict')
                shutil.copyfile(self.local / filename, target)
            for folder, _ in self.scan():
                publish_folder(folder, self.durable / 'chunks' / folder.name)
            for folder in sorted((self.local / 'evaluations').glob('u*')):
                publish_folder(folder, self.durable / 'evaluations' / folder.name)


def member_distribution(result):
    scores = [m['correct'] for m in result['by_member']]
    return dict(correct_pairs_histogram=dict(sorted(Counter(scores).items())),
                median_correct_pairs=float(np.median(scores)),
                members_with_any_correct=sum(x > 0 for x in scores),
                members_with_all_correct=sum(m['correct'] == m['pairs'] for m in result['by_member']))


def evaluate_state(store, state, schedule, battle):
    update = state['next_update']
    name = f'u{update:06d}'
    destination = store.local / 'evaluations' / name
    if destination.exists():
        checked_manifest(destination)
        return
    stage = Path(tempfile.mkdtemp(prefix='.eval-', dir=store.local))
    with isolated_evaluation():
        device = state['settings']['device']
        model = Policies(width=state['settings']['width']).to(device)
        model.load_state_dict(state['model'])
        model.eval().requires_grad_(False)
        result = evaluate(model, schedule, battle, stage, 'eval', include_swapped=True)
        for mode_name, mode in result.items():
            mode.update(member_distribution(mode))
            preferences = {arm: {'first_job': Counter(), 'last_hoped_job': Counter(),
                                 'confirmed_job': Counter()} for arm in ('A', 'B')}
            with (stage / (mode_name + '.jsonl')).open() as handle:
                for line in handle:
                    pair = json.loads(line)
                    for arm in ('A', 'B'):
                        outcome = pair[arm]
                        first = next((s['job'] for s in outcome['steps'] if s['job'] >= 0), -1)
                        preferences[arm]['first_job'][first] += 1
                        preferences[arm]['last_hoped_job'][outcome['hoped_job']] += 1
                        preferences[arm]['confirmed_job'][outcome['confirmed_job']] += 1
            mode['preferences'] = preferences
        write_json(stage / 'summary.json', result)
        write_json(stage / 'evaluation.json', dict(update=update,
            judgments_per_member=update * 2 * state['settings']['pairs_per_member'],
            width=model.width, seed=state['settings']['seed'], reward_mode='combat',
            checkpoint_sha256=sha256(store.local / 'chunks' / name / 'state.pt'),
            eval_schedule_sha256=state['settings']['eval_schedule_sha256'], runtime=runtime(device)))
    write_json(stage / 'manifest.json', dict(schema=SCHEMA,
        files={p.name: sha256(p) for p in stage.iterdir()}))
    publish_folder(stage, destination)
    if store.durable:
        publish_folder(stage, store.durable / 'evaluations' / name)
    shutil.rmtree(stage)


def stats_for(update, outcomes, loss, timing):
    stats = dict(update=update, loss=loss, **timing)
    for arm, correct in (('A', 1), ('B', 2)):
        group = [r for r in outcomes if r['arm'] == arm]
        initial = [next((s[2] for s in r['history'] if s[2] >= 0), -1) for r in group]
        stats.update({arm + '_trials': len(group),
                      arm + '_correct': sum(r['confirmed_job'] == correct for r in group),
                      arm + '_confirmed': sum(r['departed'] for r in group),
                      arm + '_first_D': sum(x == 2 for x in initial),
                      arm + '_ever_D': sum(any(s[2] == 2 for s in r['history']) for r in group),
                      arm + '_mean_combat_reward': float(np.mean([r['combat_reward'] for r in group]))})
    return stats


def run_training(store, settings, schedule_path, stop_update, *, evaluations=True, battle_factory=None):
    configure(settings['seed'])
    schedule = read_schedule(schedule_path)
    validate_schedule(schedule)
    settings = store.initialize(settings, schedule_path)
    if sha256(store.local / 'schedule.jsonl') != settings['eval_schedule_sha256']:
        raise ValueError('Saved evaluation schedule is corrupt')
    store.sync_pending()
    excluded = {tuple(sorted(s['ids'])) for s in schedule}
    device = settings['device']
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable; select a GPU runtime and CUDA Torch build')
    model = Policies(width=settings['width']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    rng = np.random.default_rng(settings['seed'])
    appearances = np.zeros((128, 2), dtype=np.int64)
    update = 0
    battle = battle_factory() if battle_factory else (CertifiedBattle() if settings['battle_mode']=='certified' else Battle())
    try:
        state = store.latest()
        if state:
            appearances, update = restore(state, model, optimizer, rng, battle)
        else:
            state = snapshot(model, optimizer, rng, appearances, 0, settings, battle)
            store.commit(state, [], [])
        if stop_update < update:
            raise ValueError('Requested stop is earlier than the latest committed update')
        # Finish an interrupted milestone evaluation before advancing training.
        if evaluations:
            for folder, manifest in store.scan():
                if manifest['end_update'] in settings['milestones']:
                    saved = torch.load(folder / 'state.pt', map_location='cpu', weights_only=False)
                    evaluate_state(store, saved, schedule, battle)
        pending_outcomes, pending_stats = [], []
        while update < stop_update:
            began = perf_counter()
            scenarios = training_schedule(rng, update, settings['pairs_per_member'], excluded)
            timing = dict(scenario_generation_s=perf_counter() - began)
            began = perf_counter()
            records, rewards, outcomes = collect(model, scenarios, battle, timings=timing)
            timing['collect_total_s'] = perf_counter() - began
            began = perf_counter()
            loss = model.update(records, rewards, optimizer)
            if device == 'cuda':
                torch.cuda.synchronize()
            timing['ppo_update_s'] = perf_counter() - began
            update += 1
            for row in outcomes:
                row['update'] = update
                appearances[row['focal_id'], int(row['arm']=='B')] += 1
            if not (appearances == update * settings['pairs_per_member']).all():
                raise ValueError('Every person must have equal A/B decision counts')
            pending_outcomes.extend(outcomes)
            stats = stats_for(update, outcomes, loss, timing)
            pending_stats.append(stats)
            print(json.dumps(dict(update=update, judgments_per_member=2*update*settings['pairs_per_member'],
                                  width=model.width, seed=settings['seed'], device=device, **timing)), flush=True)
            if update % settings['checkpoint_every'] == 0 or update in settings['milestones'] or update == stop_update:
                began = perf_counter()
                state = snapshot(model, optimizer, rng, appearances, update, settings, battle)
                store.commit(state, pending_outcomes, pending_stats)
                print(json.dumps(dict(committed_update=update, save_and_sync_s=perf_counter()-began)), flush=True)
                pending_outcomes, pending_stats = [], []
                if evaluations and update in settings['milestones']:
                    evaluate_state(store, state, schedule, battle)
                store.prune_checkpoints(settings['milestones'])
        return state
    finally:
        battle.close()


def settings_for(args):
    configure(args.seed)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable: use a GPU runtime and the cu126 installer option')
    return dict(schema=SCHEMA, seed=args.seed, width=args.width,
                parameters_per_member=28685 if args.width == 64 else 106317,
                pairs_per_member=4, milestones=list(MILESTONES), checkpoint_every=args.checkpoint_every,
                eval_schedule_sha256=sha256(args.schedule), reward_mode='combat',
                reward='boss HP fraction removed + win bonus 1; no departure 0',
                battle_mode=args.battle, device=args.device,
                backend=args.device + '-float32-tf32off-single-thread',
                cap=16, lock_ready=True, ally_embedding=16, roster=128,
                optimizer=dict(name='Adam', lr=3e-4, epochs=3),
                runtime=runtime(args.device), **provenance())


def profile(args):
    """A disposable calibration, never appended to a scientific training series."""
    if args.out.exists() or args.out.with_suffix('.csv').exists():
        raise FileExistsError(args.out)
    configure(args.seed)
    schedule = read_schedule(args.schedule)
    validate_schedule(schedule)
    excluded = {tuple(sorted(s['ids'])) for s in schedule}
    report = dict(runtime=runtime(), provenance=provenance(), seed=args.seed,
                  eval_schedule_sha256=sha256(args.schedule), results=[])
    for device, width in ((d, w) for d in args.devices for w in (64, 128)):
        if device == 'cuda' and not torch.cuda.is_available():
            report['results'].append(dict(device=device, width=width, skipped='CUDA unavailable'))
            continue
        configure(args.seed)
        model = Policies(width=width).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        if device == 'cuda':
            torch.cuda.synchronize()
        rng = np.random.default_rng(args.seed)
        battle = CertifiedBattle() if args.battle == 'certified' else Battle()
        samples = []
        appearances = np.zeros((128, 2), dtype=np.int64)
        try:
            with tempfile.TemporaryDirectory() as temp:
                for update in range(args.updates):
                    update_start = perf_counter()
                    start = perf_counter()
                    scenarios = training_schedule(rng, update, 4, excluded)
                    timing = dict(scenario_generation_s=perf_counter()-start)
                    records, rewards, outcomes = collect(model, scenarios, battle, timings=timing)
                    if device == 'cuda':
                        torch.cuda.synchronize()
                    start = perf_counter()
                    loss = model.update(records, rewards, optimizer)
                    if device == 'cuda':
                        torch.cuda.synchronize()
                    timing['ppo_update_s'] = perf_counter()-start
                    start = perf_counter()
                    with gzip.open(Path(temp)/'log.gz', 'wt') as handle:
                        for row in outcomes:
                            handle.write(json.dumps(row)+'\n')
                    timing['log_gzip_s'] = perf_counter()-start
                    samples.append(dict(update=update+1, loss=loss, timings=timing,
                                        update_wall_s=perf_counter()-update_start,
                                        log_gzip_bytes=(Path(temp)/'log.gz').stat().st_size))
                start = perf_counter()
                state = snapshot(model, optimizer, rng, appearances, args.updates, dict(width=width), battle)
                torch.save(state, Path(temp)/'state.pt')
                state_bytes = (Path(temp)/'state.pt').stat().st_size
                save_s = perf_counter()-start
            totals = {key: float(np.mean([s['timings'][key] for s in samples])) for key in samples[0]['timings']}
            per_update = float(np.mean([s['update_wall_s'] for s in samples]))
            estimate = {str(u): dict(training_hours=per_update*u/3600,
                         trial_log_gib=sum(s['log_gzip_bytes'] for s in samples)/len(samples)*u/2**30)
                        for u in MILESTONES}
            report['results'].append(dict(width=width, device=device, runtime=runtime(device),
                parameters_per_member=model.members[0].numel(),
                samples=samples, mean_stage_seconds=totals, checkpoint_bytes=state_bytes,
                mean_update_wall_s=per_update,
                checkpoint_local_save_s=save_s, estimates=estimate,
                limitation='Cold-start extrapolation, excludes full evaluation and Drive copy time; cache warms up.'))
        except RuntimeError as error:
            report['results'].append(dict(width=width, device=device, failed=str(error)))
        finally:
            battle.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(args.out)
    write_json(args.out, report)
    with args.out.with_suffix('.csv').open('w', newline='') as handle:
        fields = ['device', 'width', 'status', 'update_seconds', 'scenario_generation_s',
                  'prep_observation_s', 'prep_inference_s', 'prep_advance_trace_s',
                  'battle_s', 'ppo_update_s', 'log_gzip_s', 'checkpoint_local_save_s']
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for row in report['results']:
            timing = row.get('mean_stage_seconds', {})
            writer.writerow(dict(device=row['device'], width=row['width'],
                                 status=row.get('skipped', row.get('failed', 'completed')),
                                 update_seconds=row.get('mean_update_wall_s', ''),
                                 checkpoint_local_save_s=row.get('checkpoint_local_save_s', ''), **timing))
    print(json.dumps(report, indent=2))


def report_runs(args):
    records = []
    for root in args.runs:
        settings = json.loads((root / 'settings.json').read_text())
        for folder in sorted((root / 'evaluations').glob('u*')):
            checked_manifest(folder)
            meta = json.loads((folder/'evaluation.json').read_text())
            results = json.loads((folder/'summary.json').read_text())
            for mode, summary in results.items():
                n = summary['pairs']
                records.append(dict(run=str(root), width=settings['width'], seed=settings['seed'],
                    update=meta['update'], judgments_per_member=meta['judgments_per_member'], mode=mode,
                    pairs=n, paired_correct_percent=100*summary['correct']/n,
                    inverse_percent=100*summary['inverse']/n,
                    unconfirmed_percent=100*summary['unconfirmed_either']/n,
                    A_clear_percent=100*summary['A']['wins']/n, B_clear_percent=100*summary['B']['wins']/n,
                    delta_H_pp=summary['delta_H_pp'], **member_distribution(summary)))
    if not records:
        raise ValueError('No completed milestone evaluations; incomplete evaluations are not results')
    # Comparing widths on different schedules/backends is not the designed contrast.
    configs = [json.loads((root/'settings.json').read_text()) for root in args.runs]
    for key in ('eval_schedule_sha256', 'reward_mode', 'battle_mode', 'pairs_per_member', 'backend', 'source_hashes'):
        if len({json.dumps(c[key], sort_keys=True) for c in configs}) != 1:
            raise ValueError(f'Cannot compare runs with different {key}')
    if any(c['runtime']['packages'] != configs[0]['runtime']['packages']
           or c['runtime']['python'] != configs[0]['runtime']['python'] for c in configs):
        raise ValueError('Cannot compare runs with different Python/package versions')
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / 'comparison.json', records)
    with (args.out / 'comparison.csv').open('w', newline='') as handle:
        fields = [k for k in records[0] if k != 'correct_pairs_histogram']
        writer = csv.DictWriter(handle, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    lines = ['# Colab scaling results', '',
             'Each row is one independently trained roster. Missing milestones are omitted.', '',
             '| Width | Seed | Judgments/person | Mode | A=H and B=D (%) | Members any/all correct | Median correct pairs |',
             '|---:|---:|---:|---|---:|---:|---:|']
    for r in records:
        lines.append(f"| {r['width']} | {r['seed']} | {r['judgments_per_member']} | {r['mode']} | "
                     f"{r['paired_correct_percent']:.3f} | {r['members_with_any_correct']}/{r['members_with_all_correct']} | "
                     f"{r['median_correct_pairs']:.1f} |")
    (args.out / 'RESULTS.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('train', 'profile'):
        p = commands.add_parser(command)
        p.add_argument('--schedule', type=Path, required=True)
        p.add_argument('--seed', type=int, choices=(41, 42, 43), default=41)
        p.add_argument('--battle', choices=('certified', 'native'), default='certified')
        p.add_argument('--out', type=Path, required=True)
        if command == 'train':
            p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
            p.add_argument('--width', type=int, choices=(64, 128), default=64)
            p.add_argument('--checkpoint-every', type=int, default=16)
            p.add_argument('--stop-update', type=int, default=256,
                           help='Total target update, not additional updates; maximum 4096')
            p.add_argument('--durable', type=Path, help='Separate Google Drive run directory')
        else:
            p.add_argument('--updates', type=int, default=1)
            p.add_argument('--devices', nargs='+', choices=('cpu', 'cuda'), default=['cpu', 'cuda'])
    e = commands.add_parser('evaluate')
    e.add_argument('--out', type=Path, required=True, help='Existing local run directory')
    e.add_argument('--update', type=int, choices=MILESTONES, required=True)
    e.add_argument('--durable', type=Path)
    r = commands.add_parser('report')
    r.add_argument('runs', type=Path, nargs='+')
    r.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'train':
        if not 1 <= args.stop_update <= 4096 or args.checkpoint_every < 1:
            parser.error('Positive update target <=4096 and checkpoint interval required')
        run_training(RunStore(args.out, args.durable), settings_for(args), args.schedule, args.stop_update)
    elif args.command == 'profile':
        if args.updates < 1:
            parser.error('Positive profiling update count required')
        profile(args)
    elif args.command == 'evaluate':
        configure(0)
        store = RunStore(args.out, args.durable)
        if not (store.local/'settings.json').exists() and store.durable:
            store.pull()
        settings = json.loads((store.local/'settings.json').read_text())
        current = dict(runtime=runtime(settings['device']), **provenance())
        if settings['source_hashes'] != current['source_hashes'] or settings['runtime']['packages'] != current['runtime']['packages']:
            parser.error('Use the original code and package versions for evaluation')
        folder = store.local/'chunks'/f'u{args.update:06d}'
        if sha256(store.local/'schedule.jsonl') != settings['eval_schedule_sha256']:
            parser.error('Evaluation schedule differs from training metadata')
        checked_manifest(folder, checkpoint=True)
        state = torch.load(folder/'state.pt', map_location='cpu', weights_only=False)
        battle = CertifiedBattle() if settings['battle_mode']=='certified' else Battle()
        try:
            evaluate_state(store, state, read_schedule(store.local/'schedule.jsonl'), battle)
            store.sync_pending()
        finally:
            battle.close()
    else:
        report_runs(args)


if __name__ == '__main__':
    main()
