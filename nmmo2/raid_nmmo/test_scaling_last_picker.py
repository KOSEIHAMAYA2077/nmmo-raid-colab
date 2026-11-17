"""Small synthetic checks, not research training or a Colab benchmark."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import random
import tempfile
import hashlib
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from .last_picker import make_schedule
from .ready_policy import Policies
from .scaling_last_picker import (
    RunStore, checked_manifest, configure, evaluate_state, isolated_evaluation,
    run_training, settings_for, validate_schedule,
)


class ToyBattle:
    """No NMMO battles; repeatable aggregate returns exercise actual PPO code."""
    def __init__(self):
        self._unambiguous = {}

    def evaluate(self, jobs, seed):
        won = jobs.count(0) == 1 and jobs.count(1) == 1
        return won, 2.0 if won else jobs.count(2) / 10

    def close(self):
        pass


def nested_equal(test, a, b):
    if torch.is_tensor(a):
        test.assertTrue(torch.equal(a, b))
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        test.assertEqual(a.keys(), b.keys())
        for key in a:
            nested_equal(test, a[key], b[key])
    elif isinstance(a, (list, tuple)):
        test.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            nested_equal(test, x, y)
    else:
        test.assertEqual(a, b)


class ScalingTest(unittest.TestCase):
    def setUp(self):
        configure(41)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.schedule = self.root / 'schedule.jsonl'
        self.scenarios = make_schedule(2026111601, 2, set())
        self.schedule.write_text(''.join(json.dumps(s)+'\n' for s in self.scenarios))
        args = SimpleNamespace(seed=41, width=64, checkpoint_every=1,
                               schedule=self.schedule, battle='certified', device='cpu')
        self.settings = settings_for(args)
        # Only the recurrent math uses width6 to keep these tests small.
        self.settings['width'] = 6

    def train(self, path, stop, durable=None):
        with redirect_stdout(io.StringIO()):
            return run_training(RunStore(path, durable), copy.deepcopy(self.settings),
                                self.schedule, stop, evaluations=False, battle_factory=ToyBattle)

    def test_both_width_counts_shapes_and_independence(self):
        for width, expected in ((64, 28685), (128, 106317)):
            model = Policies(width)
            self.assertEqual(model.members[0].numel(), expected)
            self.assertEqual(sum(p.numel() for p in model.parameters()), 128*expected)
            ids = torch.tensor([[0], [1]])
            action, message, logp, value, hidden = model.step(
                ids, torch.zeros(2, 1, 25), torch.zeros(2, 1, 5, 11),
                torch.zeros(2, 1, width), torch.ones(2, 1, 8, dtype=torch.bool))
            self.assertEqual(hidden.shape, (2, 1, width))
            self.assertEqual(action.shape, (2, 1))
            self.assertEqual(message.shape, logp.shape)
            self.assertEqual(value.shape, (2, 1))
            before = model.members[127].detach().clone()
            optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
            (-logp.sum() + value.sum()).backward()
            optimizer.step()
            self.assertTrue(torch.equal(before, model.members[127]))
            self.assertIsNone(model.members[127].grad)

    def test_resume_on_fresh_local_folder_is_exact(self):
        base = self.root
        reports = []
        for width, count in ((64, 28685), (128, 106317)):
            with self.subTest(width=width):
                self.root = base / f'width{width}'
                self.root.mkdir()
                self.settings['width'] = width
                self.settings['parameters_per_member'] = count
                self._check_resume_width()
                reports.append(self.resume_evidence)
        self.resume_evidence = dict(widths=[64, 128], by_width=reports)

    def _check_resume_width(self):
        baseline = self.train(self.root/'continuous', 2)
        self.train(self.root/'first_session', 1, self.root/'drive')
        # Incomplete writes must not become restart points.
        (self.root/'drive'/'chunks'/'.pending-interrupted').mkdir()
        resumed = self.train(self.root/'second_session', 2, self.root/'drive')
        for key in ('model', 'optimizer', 'appearances', 'numpy_generator',
                    'numpy_global', 'python_random', 'torch_cpu', 'torch_cuda'):
            nested_equal(self, baseline[key], resumed[key])
        nested_equal(self, baseline['next_update'], resumed['next_update'])
        for name in ('u000001', 'u000002'):
            a = self.root/'continuous'/'chunks'/name/'trials.jsonl.gz'
            b = self.root/'second_session'/'chunks'/name/'trials.jsonl.gz'
            self.assertEqual(a.read_bytes(), b.read_bytes())
            stats_a = json.loads(a.with_name('stats.json').read_text())
            stats_b = json.loads(b.with_name('stats.json').read_text())
            self.assertEqual([{k:v for k,v in s.items() if not k.endswith('_s')} for s in stats_a],
                             [{k:v for k,v in s.items() if not k.endswith('_s')} for s in stats_b])
        # Draw the next unconsumed update from both saved generators as well.
        from .train_last_picker import training_schedule
        next_draws = []
        for state in (baseline, resumed):
            rng = np.random.default_rng()
            rng.bit_generator.state = state['numpy_generator']
            next_draws.append(training_schedule(rng, 2, 4, {tuple(sorted(s['ids'])) for s in self.scenarios}))
        self.assertEqual(next_draws[0], next_draws[1])
        self.resume_evidence = dict(
            fixture=f"width{self.settings['width']}, 128 independent policies, synthetic deterministic combat, two updates",
            continuous_vs_save_after_one_update=True,
            next_scenario_schedule_equal=True, next_scenarios=len(next_draws[0]),
            all_128_model_tensors_equal=True, full_adam_state_equal=True,
            appearances_equal=True, all_saved_rng_states_equal=True,
            compressed_operation_logs_byte_equal=True, loss_and_non_timing_stats_equal=True,
            timing_values_excluded=True,
            log_sha256={name:hashlib.sha256((self.root/'continuous'/'chunks'/name/'trials.jsonl.gz').read_bytes()).hexdigest()
                        for name in ('u000001','u000002')})

    def test_evaluation_does_not_change_next_update(self):
        baseline = self.train(self.root/'baseline', 2)
        state = self.train(self.root/'with_eval', 1)
        with redirect_stdout(io.StringIO()):
            evaluate_state(RunStore(self.root/'with_eval'), state, self.scenarios[:2], ToyBattle())
        resumed = self.train(self.root/'with_eval', 2)
        for key in ('model', 'optimizer', 'torch_cpu', 'numpy_generator'):
            nested_equal(self, baseline[key], resumed[key])
        result = json.loads((self.root/'with_eval'/'evaluations'/'u000001'/'summary.json').read_text())
        self.assertEqual(len(result), 4)
        self.assertTrue(all('correct_pairs_histogram' in x and 'preferences' in x for x in result.values()))

    def test_corrupt_logs_and_changed_settings_are_rejected(self):
        self.train(self.root/'run', 1)
        changed = copy.deepcopy(self.settings)
        changed['width'] = 128
        with self.assertRaisesRegex(ValueError, 'width'):
            RunStore(self.root/'run').initialize(changed, self.schedule)
        path = self.root/'run'/'chunks'/'u000001'/'trials.jsonl.gz'
        path.write_bytes(path.read_bytes()+b'corrupt')
        with self.assertRaisesRegex(ValueError, 'Corrupt artifact'):
            RunStore(self.root/'run').latest()

    def test_retention_preserves_logs_and_milestones(self):
        self.train(self.root/'run', 3)
        store = RunStore(self.root/'run')
        self.assertFalse((store.local/'chunks'/'u000001'/'state.pt').exists())
        self.assertTrue((store.local/'chunks'/'u000000'/'state.pt').exists())
        for folder, _ in store.scan():
            checked_manifest(folder)
            self.assertTrue((folder/'trials.jsonl.gz').exists())
        self.assertEqual(store.latest()['next_update'], 3)

    def test_isolation_and_schedule_validation(self):
        validate_schedule(self.scenarios)
        with self.assertRaises(ValueError):
            validate_schedule(self.scenarios[:-1])
        a, b, c = torch.get_rng_state(), np.random.get_state(), random.getstate()
        with isolated_evaluation():
            torch.rand(13)
            np.random.random(7)
            random.random()
        nested_equal(self, a, torch.get_rng_state())
        nested_equal(self, b, np.random.get_state())
        nested_equal(self, c, random.getstate())

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA device unavailable on this host')
    def test_cuda_collection_update_and_evaluation(self):
        from .train_last_picker import collect
        from .last_picker import run_arm
        from .ready_policy import MemberBatch
        configure(41)
        for width in (64, 128):
            model = Policies(width).cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
            rows = self.scenarios[:2]
            records, rewards, _ = collect(model, rows, ToyBattle(), timings={})
            self.assertTrue(rewards.is_cuda)
            self.assertTrue(all(r['x_self'].is_cuda and r['active'].is_cuda for r in records))
            self.assertTrue(np.isfinite(model.update(records, rewards, optimizer)))
            batch = MemberBatch(torch.tensor([[s['focal_id']] for s in rows], device='cuda'),
                                model.members, model.shapes)
            result = run_arm(batch, rows, torch.rand(16, 2, len(rows), 1), width, 'A', False)
            self.assertEqual(len(result), 2)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA device unavailable on this host')
    def test_cuda_resume_state_is_exact_on_same_device(self):
        from .scaling_last_picker import runtime
        self.settings['device'] = 'cuda'
        self.settings['backend'] = 'cuda-float32-tf32off-single-thread'
        self.settings['runtime'] = runtime('cuda')
        continuous = self.train(self.root/'continuous_cuda', 2)
        self.train(self.root/'resumed_cuda', 1)
        resumed = self.train(self.root/'resumed_cuda', 2)
        for key in ('model', 'optimizer', 'appearances', 'numpy_generator', 'torch_cpu', 'torch_cuda'):
            nested_equal(self, continuous[key], resumed[key])


if __name__ == '__main__':
    unittest.main()
