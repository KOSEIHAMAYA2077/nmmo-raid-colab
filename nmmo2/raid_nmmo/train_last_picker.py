"""Audit rewards, then train balanced last-picker IPPO from random initialization."""
import argparse
import csv
import gzip
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .last_picker import make_schedule, prepare, run_arm, sha256, summary
from .ready_battle import Battle
from .certified_battle import CertifiedBattle
from .ready_policy import MemberBatch, Policies


def read_schedule(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def reward_audit(schedule, out):
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    battle = Battle()
    try:
        for scenario in schedule:
            for arm in ('A', 'B'):
                for chosen in (0, 1, 2):
                    jobs = scenario['A_jobs'].copy()
                    if arm == 'B':
                        jobs[scenario['changed_seat']] = 1
                    jobs[scenario['focal_seat']] = chosen
                    won, reward = battle.evaluate(jobs, scenario['env_seed'])
                    rows.append(dict(pair=scenario['pair'], arm=arm, chosen_job=chosen,
                                     env_seed=scenario['env_seed'], jobs=jobs, won=won, reward=reward))
    finally:
        battle.close()
    with (out/'trials.jsonl').open('w') as handle:
        for row in rows:
            print(json.dumps(row), file=handle)
    results = {}
    for arm in ('A', 'B'):
        for chosen in (0, 1, 2):
            group = [r for r in rows if r['arm'] == arm and r['chosen_job'] == chosen]
            rewards = [r['reward'] for r in group]
            results[f'{arm}_{chosen}'] = dict(trials=len(group), wins=sum(r['won'] for r in group),
                                              mean_reward=float(np.mean(rewards)),
                                              min_reward=min(rewards), max_reward=max(rewards))
    (out/'summary.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results), flush=True)


def training_schedule(rng, update, pairs_per_member, excluded):
    scenarios = []
    for focal in range(128):
        for repetition in range(pairs_per_member):
            seat = (update*pairs_per_member + repetition + focal) % 6
            while True:
                party = rng.choice(np.delete(np.arange(128), focal), 5, replace=False).tolist()
                party.insert(seat, focal)
                if tuple(sorted(party)) not in excluded:
                    break
            tank, changed = map(int, rng.choice([i for i in range(6) if i != seat], 2, replace=False))
            jobs = [2]*6
            jobs[seat], jobs[tank] = -1, 0
            scenarios.append(dict(ids=party, focal_id=focal, focal_seat=seat, changed_seat=changed,
                                  A_jobs=jobs, env_seed=int(rng.integers(0, 2**31))))
    return scenarios


def learning_reward(arm, chosen, departed, won, combat_reward, equalized_failure_reward=None):
    """The diagnostic changes only two failed, departed outcomes."""
    if not departed:
        return 0.0
    corrected_failure = (arm == 'A' and chosen == 2) or (arm == 'B' and chosen == 1)
    if equalized_failure_reward is not None and not won and corrected_failure:
        return float(equalized_failure_reward)
    return float(combat_reward)


@torch.no_grad()
def collect(model, scenarios, battle, *, equalized_failure_reward=None, timings=None):
    device = next(model.parameters()).device
    if timings is not None:
        for name in ('prep_observation_s', 'prep_inference_s', 'prep_advance_trace_s', 'battle_s'):
            timings.setdefault(name, 0.0)
    def sync():
        if timings is not None and device.type == 'cuda':
            torch.cuda.synchronize(device)
    count = len(scenarios)
    all_scenarios = scenarios + scenarios
    prep = prepare(all_scenarios, 'A')
    for i, scenario in enumerate(scenarios):
        prep.jobs[count+i, scenario['changed_seat']] = 1
    seats = torch.tensor([s['focal_seat'] for s in all_scenarios])
    rows = torch.arange(len(all_scenarios))
    ids = torch.tensor([[s['focal_id']] for s in all_scenarios], device=device)
    hidden = torch.zeros(len(all_scenarios), 1, model.width, device=device)
    records, histories = [], [[] for _ in all_scenarios]
    while prep.active.any():
        sync()
        started = perf_counter()
        own, peers, mask = prep.observations()
        x = own[rows, seats].unsqueeze(1).to(device)
        allies = peers[rows, seats].unsqueeze(1).to(device)
        masks = mask[rows, seats].unsqueeze(1).to(device)
        active = prep.active.clone()
        sync()
        if timings is not None:
            timings['prep_observation_s'] = timings.get('prep_observation_s', 0) + perf_counter() - started
        started = perf_counter()
        action, message, logp, _, hidden = model.step(ids, x, allies, hidden, masks)
        sync()
        if timings is not None:
            timings['prep_inference_s'] = timings.get('prep_inference_s', 0) + perf_counter() - started
        started = perf_counter()
        records.append(dict(ids=ids, x_self=x, x_ally=allies, mask=masks, active=active.to(device),
                            action=action, msg=message, logp=logp))
        actions, messages = torch.zeros_like(prep.jobs), torch.zeros_like(prep.jobs)
        actions[rows, seats], messages[rows, seats] = action[:, 0].cpu(), message[:, 0].cpu()
        prep.advance(actions, messages)
        for i in active.nonzero().flatten().tolist():
            histories[i].append([int(actions[i, seats[i]]), int(messages[i, seats[i]]),
                                 int(prep.jobs[i, seats[i]]), bool(prep.ready[i, seats[i]])])
        sync()
        if timings is not None:
            timings['prep_advance_trace_s'] = timings.get('prep_advance_trace_s', 0) + perf_counter() - started
    rewards = torch.zeros(len(all_scenarios))
    outcomes = []
    for i, scenario in enumerate(all_scenarios):
        departed = bool(prep.departed[i])
        won, combat_reward = False, 0.0
        chosen = int(prep.jobs[i, seats[i]]) if departed else -1
        arm = 'A' if i < count else 'B'
        if departed:
            started = perf_counter()
            won, combat_reward = battle.evaluate(prep.jobs[i].tolist(), scenario['env_seed'])
            if timings is not None:
                timings['battle_s'] = timings.get('battle_s', 0) + perf_counter() - started
        rewards[i] = learning_reward(arm, chosen, departed, won, combat_reward, equalized_failure_reward)
        outcomes.append(dict(**scenario, arm=arm, confirmed_job=chosen,
                             hoped_job=int(prep.jobs[i, seats[i]]), departed=departed,
                             won=won, combat_reward=float(combat_reward),
                             reward=float(rewards[i]), history=histories[i]))
    return records, rewards.to(device), outcomes


@torch.no_grad()
def evaluate(model, schedule, battle, out, label, *, include_swapped):
    device = next(model.parameters()).device
    results = {}
    for swapped in ([False, True] if include_swapped else [False]):
        for greedy in (False, True):
            name = f'{label}_'+('swapped_' if swapped else '')+('greedy' if greedy else 'sampled')
            rows = []
            with (out/(name+'.jsonl')).open('w') as handle:
                for start in range(0, len(schedule), 64):
                    scenarios = schedule[start:start+64]
                    ids = torch.tensor([[s['focal_id']] for s in scenarios], device=device)
                    batch = MemberBatch(ids, model.members, model.shapes)
                    draws = torch.stack([torch.rand(16, 2, 1, generator=torch.Generator().manual_seed(s['action_seed']))
                                         for s in scenarios], dim=2)
                    arms = {arm: run_arm(batch, scenarios, draws, model.width, arm, greedy,
                                         swap_role_display=swapped) for arm in ('A', 'B')}
                    for i, scenario in enumerate(scenarios):
                        row = dict(pair=scenario['pair'], focal_id=scenario['focal_id'],
                                   A=arms['A'][i], B=arms['B'][i])
                        for arm in ('A', 'B'):
                            item = row[arm]
                            item['won'], item['team_reward'] = (battle.evaluate(item['final_jobs'], scenario['env_seed'])
                                                                 if item['departed'] else (False, 0.0))
                        rows.append(row)
                        print(json.dumps(row), file=handle)
            results[name] = summary(rows)
            print(name, {k:v for k,v in results[name].items() if k not in ('by_member','joint_confirmed_jobs')}, flush=True)
    return results


def train(args):
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = Policies(width=args.width)
    reference = None
    if args.reference_run is not None:
        reference = json.loads((args.reference_run / 'settings.json').read_text())
        expected = dict(seed=args.seed, updates=args.updates,
                        pairs_per_member_per_update=args.pairs_per_member, width=args.width, cap=16, lock_ready=True)
        if any(reference.get(k) != v for k, v in expected.items()):
            raise ValueError('Reference training settings must match the diagnostic budget and seed')
        original_initial = torch.load(args.reference_run / 'initial.pt', map_location='cpu', weights_only=True)
        if not all(torch.equal(v, original_initial[k]) for k, v in model.state_dict().items()):
            raise ValueError('Fresh initialization differs from the reference weights')
        if sha256(args.schedule) != reference['eval_schedule_sha256']:
            raise ValueError('Reference evaluation schedule must match')
        for name in ('last_picker.py', 'ready_policy.py', 'ready_prep.py', 'ready_battle.py'):
            if sha256(Path(__file__).with_name(name)) != reference['source_hashes'][name]:
                raise ValueError(f'Reference dependency changed: {name}')
    equalized = None
    if args.equalize_failure_rewards:
        audit = json.loads(args.failure_reward_audit.read_text())
        if audit['A_2']['wins'] or audit['B_1']['wins']:
            raise ValueError('The two calibrated conditions must be failed battles')
        equalized = (audit['A_2']['mean_reward'] + audit['B_1']['mean_reward']) / 2
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    schedule = read_schedule(args.schedule)
    excluded = {tuple(sorted(s['ids'])) for s in schedule}
    settings = dict(seed=args.seed, updates=args.updates, pairs_per_member_per_update=args.pairs_per_member,
                    A_per_member=args.updates*args.pairs_per_member, B_per_member=args.updates*args.pairs_per_member,
                    width=args.width, parameters_per_member=model.members[0].numel(), from_scratch=True,
                    cap=16, lock_ready=True, gamma=1, reward='boss HP fraction removed + win bonus 1',
                    source_hashes={name:sha256(Path(__file__).with_name(name)) for name in
                                   ('train_last_picker.py','last_picker.py','ready_policy.py','ready_prep.py','ready_battle.py','certified_battle.py')},
                    eval_schedule_sha256=sha256(args.schedule), eval_schedule=str(args.schedule),
                    equalized_failure_reward=equalized,
                    reference_run=str(args.reference_run) if reference is not None else None,
                    reference_initial_sha256=sha256(args.reference_run / 'initial.pt') if reference is not None else None,
                    reference_settings_sha256=sha256(args.reference_run / 'settings.json') if reference is not None else None,
                    reward_mode='equalized_failures' if equalized is not None else 'combat',
                    certified_battle=args.certified_battle,
                    failure_reward_audit_sha256=sha256(args.failure_reward_audit) if equalized is not None else None)
    (args.out/'settings.json').write_text(json.dumps(settings, indent=2)+'\n')
    torch.save(model.state_dict(), args.out/'initial.pt')
    appearances = np.zeros((128,2), dtype=int)
    battle = CertifiedBattle() if args.certified_battle else Battle()
    results = {}
    try:
        with torch.random.fork_rng():
            results.update(evaluate(model,schedule,battle,args.out,'initial',include_swapped=False))
        with gzip.open(args.out/'training.jsonl.gz','wt') as logs, (args.out/'training.csv').open('w') as csv_file:
            writer = None
            for update in range(args.updates):
                scenarios = training_schedule(rng,update,args.pairs_per_member,excluded)
                records,rewards,outcomes = collect(model,scenarios,battle, equalized_failure_reward=equalized)
                loss = model.update(records,rewards,optimizer)
                for row in outcomes:
                    row['update'] = update+1
                    appearances[row['focal_id'],int(row['arm']=='B')] += 1
                    print(json.dumps(row),file=logs)
                stats = dict(update=update+1,A_per_member=int(appearances[0,0]),B_per_member=int(appearances[0,1]),
                             loss=loss)
                for arm,correct in [('A',1),('B',2)]:
                    group = [r for r in outcomes if r['arm']==arm]
                    stats.update({arm+'_confirmed':sum(r['departed'] for r in group),
                                  arm+'_correct':sum(r['confirmed_job']==correct for r in group),
                                  arm+'_wins':sum(r['won'] for r in group),
                                  arm+'_mean_reward':float(np.mean([r['reward'] for r in group])),
                                  arm+'_mean_combat_reward':float(np.mean([r['combat_reward'] for r in group]))})
                if writer is None:
                    writer=csv.DictWriter(csv_file,fieldnames=stats);writer.writeheader()
                writer.writerow(stats);csv_file.flush()
                if (update+1)%16==0:
                    print(stats,flush=True)
                if (update+1)%64==0:
                    torch.save(model.state_dict(),args.out/'latest.pt')
                if args.certified_battle and update+1 == 128:
                    torch.save(model.state_dict(),args.out/'update128.pt')
            assert (appearances[:,0]==args.updates*args.pairs_per_member).all()
            assert np.array_equal(appearances[:,0],appearances[:,1])
        torch.save(model.state_dict(),args.out/'final.pt')
        (args.out/'appearances.json').write_text(json.dumps(appearances.tolist())+'\n')
        model.eval();model.requires_grad_(False)
        final_hash=sha256(args.out/'final.pt')
        results.update(evaluate(model,schedule,battle,args.out,'final',include_swapped=True))
        assert sha256(args.out/'final.pt')==final_hash
        (args.out/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    finally:
        battle.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    audit=sub.add_parser('audit')
    audit.add_argument('--schedule',type=Path,required=True)
    audit.add_argument('--out',type=Path,required=True)
    training=sub.add_parser('train')
    training.add_argument('--schedule',type=Path,required=True)
    training.add_argument('--out',type=Path,required=True)
    training.add_argument('--seed',type=int,default=41)
    training.add_argument('--width',type=int,choices=(64,128),default=64)
    training.add_argument('--updates',type=int,default=256)
    training.add_argument('--pairs-per-member',type=int,default=4)
    training.add_argument('--reference-run', type=Path)
    training.add_argument('--certified-battle', action='store_true')
    training.add_argument('--equalize-failure-rewards', action='store_true')
    training.add_argument('--failure-reward-audit', type=Path,
                          default=Path('runs/2026-11-16_last_picker_reward_audit/summary.json'))
    args=parser.parse_args()
    torch.set_num_threads(1)
    if args.command=='audit':
        reward_audit(read_schedule(args.schedule),args.out)
    else:
        if args.updates<1 or args.pairs_per_member<1:
            parser.error('Positive updates and pairs-per-member are required')
        train(args)


if __name__=='__main__':
    main()
