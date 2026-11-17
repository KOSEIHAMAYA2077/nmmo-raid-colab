"""Paired frozen evaluation of one unassigned player and five committed teammates."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .random_ready import draw
from .ready_battle import Battle
from .ready_partner import sample_logits
from .ready_policy import MemberBatch, Policies
from .ready_prep import Prep


ROLE_NAMES = {-1: "unconfirmed", 0: "T", 1: "H", 2: "D"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_schedule(seed, repeats, excluded_parties):
    """Every member is focal twice per seat when repeats=2."""
    rng = np.random.default_rng(seed)
    schedule = []
    used = set(excluded_parties)
    for focal_id in range(128):
        candidates = np.delete(np.arange(128), focal_id)
        for focal_seat in range(6):
            for _ in range(repeats):
                while True:
                    ids = rng.choice(candidates, 5, replace=False).tolist()
                    ids.insert(focal_seat, focal_id)
                    party = tuple(sorted(ids))
                    if party not in used:
                        used.add(party)
                        break
                other_seats = [i for i in range(6) if i != focal_seat]
                tank, changed = map(int, rng.choice(other_seats, 2, replace=False))
                jobs = [2] * 6
                jobs[focal_seat] = -1
                jobs[tank] = 0
                schedule.append(dict(pair=len(schedule), ids=ids, focal_id=focal_id,
                                     focal_seat=focal_seat, changed_seat=changed,
                                     A_jobs=jobs, env_seed=int(rng.integers(0, 2**31)),
                                     action_seed=int(rng.integers(0, 2**63 - 1))))
    return schedule


def prepare(scenarios, arm):
    prep = Prep(len(scenarios), lock_ready=True)
    prep.jobs = torch.tensor([s["A_jobs"] for s in scenarios])
    if arm == "B":
        for i, scenario in enumerate(scenarios):
            prep.jobs[i, scenario["changed_seat"]] = 1
    prep.ready = prep.jobs >= 0
    return prep


@torch.no_grad()
def run_arm(batch, scenarios, draws, width, arm, greedy, *, swap_role_display=False):
    device = getattr(batch, 'device', torch.device('cpu'))
    prep = prepare(scenarios, arm)
    seats = torch.tensor([s["focal_seat"] for s in scenarios])
    rows = torch.arange(len(scenarios))
    hidden = torch.zeros(batch.rows, batch.columns, width, device=device)
    traces = [[] for _ in scenarios]
    for turn, (action_u, message_u) in enumerate(draws):
        own, allies, mask = prep.observations()
        visible_allies = allies[rows, seats].unsqueeze(1).clone()
        if swap_role_display:
            peer = torch.tensor([s["changed_seat"] - int(s["changed_seat"] > s["focal_seat"])
                                 for s in scenarios])
            old = visible_allies[rows, 0, peer].clone()
            visible_allies[rows, 0, peer, 1] = old[:, 2]
            visible_allies[rows, 0, peer, 2] = old[:, 1]
        logits, messages, _, hidden = batch.forward(
            own[rows, seats].unsqueeze(1).to(device), visible_allies.to(device), hidden,
            mask[rows, seats].unsqueeze(1).to(device))
        if greedy:
            choices, symbols = logits.argmax(-1), messages.argmax(-1)
        else:
            choices = sample_logits(logits, action_u.to(device))
            symbols = sample_logits(messages, message_u.to(device))
        actions = torch.zeros_like(prep.jobs)
        comms = torch.zeros_like(prep.jobs)
        actions[rows, seats] = choices.squeeze(1).cpu()
        comms[rows, seats] = symbols.squeeze(1).cpu()
        active = prep.active.clone()
        prep.advance(actions, comms)
        for i, seat in enumerate(seats.tolist()):
            if active[i]:
                traces[i].append(dict(turn=turn + 1, job=int(prep.jobs[i, seat]),
                                      ready=bool(prep.ready[i, seat]), action=int(actions[i, seat]),
                                      message=int(comms[i, seat]),
                                      action_probs=logits[i, 0].softmax(-1).tolist()))
        if not prep.active.any():
            break
    outcomes = []
    for i, seat in enumerate(seats.tolist()):
        departed = bool(prep.departed[i])
        confirmed = int(prep.jobs[i, seat]) if departed else -1
        outcomes.append(dict(confirmed_job=confirmed, hoped_job=int(prep.jobs[i, seat]),
                             departed=departed, turns=int(prep.turns[i]),
                             final_jobs=prep.jobs[i].tolist(), steps=traces[i]))
    return outcomes


def summary(rows):
    n = len(rows)
    a = [r["A"]["confirmed_job"] for r in rows]
    b = [r["B"]["confirmed_job"] for r in rows]
    delta = np.array([int(x == 1) - int(y == 1) for x, y in zip(a, b)])
    by_member = []
    for member in sorted({r["focal_id"] for r in rows}):
        indices = [i for i, r in enumerate(rows) if r["focal_id"] == member]
        by_member.append(dict(focal_id=member, pairs=len(indices),
                              A_H=sum(a[i] == 1 for i in indices), B_H=sum(b[i] == 1 for i in indices),
                              correct=sum(a[i] == 1 and b[i] == 2 for i in indices),
                              inverse=sum(a[i] == 2 and b[i] == 1 for i in indices),
                              unconfirmed_either=sum(a[i] < 0 or b[i] < 0 for i in indices),
                              delta_H_pp=float(delta[indices].mean() * 100)))
    member_deltas = np.array([r["delta_H_pp"] for r in by_member])
    # Resample focal members, retaining their entire matched set of scenarios.
    rng = np.random.default_rng(20261116)
    bootstrap = member_deltas[rng.integers(0, len(by_member), (5000, len(by_member)))].mean(1)
    result = dict(pairs=n, A_H=sum(x == 1 for x in a), B_H=sum(x == 1 for x in b),
                  delta_H_pp=float(delta.mean() * 100),
                  delta_H_member_bootstrap_95_interval_pp=np.quantile(bootstrap, [.025, .975]).tolist(),
                  correct=sum(x == 1 and y == 2 for x, y in zip(a, b)),
                  inverse=sum(x == 2 and y == 1 for x, y in zip(a, b)),
                  same_confirmed=sum(x == y and x >= 0 for x, y in zip(a, b)),
                  unconfirmed_either=sum(x < 0 or y < 0 for x, y in zip(a, b)),
                  joint_confirmed_jobs=dict(sorted(Counter(
                      f'{ROLE_NAMES[x]}/{ROLE_NAMES[y]}' for x, y in zip(a, b)).items())),
                  by_member=by_member)
    for arm in ("A", "B"):
        result[arm] = dict(confirmed=sum(r[arm]["departed"] for r in rows),
                           never_assigned=sum(r[arm]["hoped_job"] < 0 for r in rows),
                           wins=sum(r[arm]["won"] for r in rows))
    return result


def training_parties(folders):
    seen = set()
    for folder in folders:
        settings = json.loads((folder / "settings.json").read_text())
        rng = np.random.default_rng(settings["seed"])
        for _ in range(settings["updates"]):
            parties, _ = draw(rng, settings["batch"])
            seen.update(tuple(sorted(p)) for p in parties)
    return seen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026111601)
    parser.add_argument("--repeats", type=int, default=2, help="Scenarios per member per seat")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    checkpoints = []
    for folder in args.runs:
        cfg = json.loads((folder / "settings.json").read_text())
        if not cfg.get("lock_ready"):
            parser.error("This protocol requires irreversible-ready checkpoints")
        checkpoints.append(dict(folder=folder, cfg=cfg, hash=sha256(folder / "final.pt")))
    schedule = make_schedule(args.seed, args.repeats, training_parties(args.runs))
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / "schedule.jsonl").open("w") as handle:
        for scenario in schedule:
            print(json.dumps(scenario), file=handle)
    settings = dict(protocol="last_picker_frozen_v1", cap=16, schedule_seed=args.seed,
                    repeats_per_member_per_seat=args.repeats, pairs=len(schedule),
                    teammate_messages="constant 0; absent at turn 0, delivered from turn 1",
                    hidden_state="zero at start", modes=["sampled", "greedy"],
                    source_checkpoint_hashes={str(r['folder']): r['hash'] for r in checkpoints},
                    torch_version=torch.__version__, baseline_code_commit="b8ae7a277cb118a3b13b3bc9dc36b70d83b7e0e1",
                    source_hashes={name: sha256(Path(__file__).with_name(name)) for name in
                                   ("last_picker.py", "ready_prep.py", "ready_policy.py",
                                    "ready_battle.py", "ready_partner.py", "random_ready.py")})
    (args.out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    battle = Battle()
    results = {}
    try:
        for checkpoint in checkpoints:
            cfg, folder = checkpoint["cfg"], checkpoint["folder"]
            model = Policies(width=64)
            model.load_state_dict(torch.load(folder / "final.pt", map_location="cpu", weights_only=True))
            model.eval()
            model.requires_grad_(False)
            before = {key: value.clone() for key, value in model.state_dict().items()}
            for greedy in (False, True):
                mode = "greedy" if greedy else "sampled"
                name = f"seed{cfg['seed']}_{mode}"
                rows = []
                with (args.out / (name + ".jsonl")).open("w") as handle:
                    for start in range(0, len(schedule), 64):
                        scenarios = schedule[start:start + 64]
                        ids = torch.tensor([[s["focal_id"]] for s in scenarios])
                        batch = MemberBatch(ids, model.members, model.shapes)
                        draws = torch.stack([torch.rand(16, 2, 1, generator=torch.Generator().manual_seed(s['action_seed']))
                                             for s in scenarios], dim=2)
                        arms = {arm: run_arm(batch, scenarios, draws, 64, arm, greedy) for arm in ("A", "B")}
                        for i, scenario in enumerate(scenarios):
                            row = dict(pair=scenario["pair"], focal_id=scenario["focal_id"],
                                       A=arms["A"][i], B=arms["B"][i])
                            for arm in ("A", "B"):
                                outcome = row[arm]
                                outcome["won"], outcome["team_reward"] = (
                                    battle.evaluate(outcome["final_jobs"], scenario["env_seed"])
                                    if outcome["departed"] else (False, 0.0))
                            rows.append(row)
                            print(json.dumps(row), file=handle)
                results[name] = dict(training=cfg, **summary(rows))
                print(name, {key: value for key, value in results[name].items()
                             if key not in ("by_member", "training")}, flush=True)
            assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
            assert sha256(folder / "final.pt") == checkpoint["hash"]
        (args.out / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    finally:
        battle.close()


if __name__ == "__main__":
    main()
