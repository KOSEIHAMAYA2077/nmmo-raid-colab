"""Require identical trained weights and complete trajectories before switching engines."""
import csv
from contextlib import nullcontext
import gzip
import json
from pathlib import Path

import torch

from .last_picker import sha256


def main():
    root = Path('runs/2026-11-17_equalized_acceleration_check')
    results = {}
    adopted = (root/'replacement.json').exists()
    for seed in (41, 42, 43):
        native = Path(f'runs/2026-11-17_last_picker_equalized_s{seed}')
        replay = native if adopted else Path(f'runs/2026-11-17_equalized_replay_s{seed}')
        old_state = torch.load(root/f'seed{seed}_update128.pt', map_location='cpu', weights_only=True)
        new_state = torch.load(replay/'update128.pt', map_location='cpu', weights_only=True)
        assert old_state.keys() == new_state.keys()
        assert all(torch.equal(v, new_state[k]) for k, v in old_state.items())
        native_csv = root/f'native_loss_prefix_s{seed}.csv' if adopted else native/'training.csv'
        with native_csv.open() as handle:
            old_rows = list(csv.DictReader(handle))
        with (replay/'training.csv').open() as handle:
            new_rows = list(csv.DictReader(handle))
        assert len(new_rows) >= 132
        assert old_rows[:128] == new_rows[:128]
        count = 128*1024
        temporary = root/f'native_prefix_s{seed}.jsonl.gz.tmp'
        native_log = root/f'native_prefix_s{seed}.jsonl.gz' if adopted else native/'training.jsonl.gz'
        with gzip.open(native_log, 'rt') as old, gzip.open(replay/'training.jsonl.gz', 'rt') as new:
            with (nullcontext(None) if adopted else gzip.open(temporary, 'wt')) as output:
                for i in range(count):
                    left, right = next(old), next(new)
                    assert json.loads(left) == json.loads(right), (seed, i)
                    if output is not None:
                        output.write(left)
        if not adopted:
            temporary.replace(root/f'native_prefix_s{seed}.jsonl.gz')
        results[seed] = dict(update=128, matched_episodes=count, all_member_tensors_equal=True,
                             all_training_rows_equal=True, all_episode_fields_equal=True,
                             native_checkpoint_sha256=sha256(root/f'seed{seed}_update128.pt'),
                             replay_checkpoint_sha256=sha256(replay/'update128.pt'))
        print(seed, results[seed], flush=True)
    (root/'replay_verification.json').write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
