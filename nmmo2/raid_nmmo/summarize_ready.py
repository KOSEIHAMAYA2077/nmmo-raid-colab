"""Summarize random-ready pilot logs without loading policies or running raids."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean


def evaluation(path):
    stats = Counter(dict.fromkeys((
        "games", "departures", "wins", "previously_seen_parties", "never_fully_assigned",
        "initial_support_shortage", "originally_missing_filled_once", "both_present_once",
        "both_present_at_end", "both_present_at_departure", "shortage_then_win",
        "departed_with_other_support_lost",
    ), 0))
    for line in path.read_text().splitlines():
        row = json.loads(line)
        stats["games"] += 1
        stats["departures"] += int(row["departed"])
        stats["wins"] += int(row["won"])
        stats["previously_seen_parties"] += int(row["party_seen_in_training"])
        jobs = row["jobs"]
        first = next((i for i, roles in enumerate(jobs) if -1 not in roles), None)
        if first is None:
            stats["never_fully_assigned"] += 1
            continue
        if {0, 1}.issubset(jobs[first]):
            continue
        stats["initial_support_shortage"] += 1
        missing = {0, 1}.difference(jobs[first])
        subsequent = jobs[first + 1:]
        if any(missing.issubset(roles) for roles in subsequent):
            stats["originally_missing_filled_once"] += 1
        if any({0, 1}.issubset(roles) for roles in subsequent):
            stats["both_present_once"] += 1
        if {0, 1}.issubset(jobs[-1]):
            stats["both_present_at_end"] += 1
            if row["departed"]:
                stats["both_present_at_departure"] += 1
                stats["shortage_then_win"] += int(row["won"])
        elif row["departed"] and missing.issubset(jobs[-1]):
            stats["departed_with_other_support_lost"] += 1
    return dict(sorted(stats.items()))


def summarize(folder):
    settings = json.loads((folder / "settings.json").read_text())
    appear = settings["appearances"]
    result = {"exposure": {"min": min(appear), "mean": mean(appear), "max": max(appear)}}
    with (folder / "training.csv").open() as f:
        rows = list(csv.DictReader(f))
    train = {}
    # Equal halves for comparing the early and late trajectory at different budgets.
    for label, block in (("first_half", rows[:len(rows)//2]), ("last_half", rows[len(rows)//2:])):
        train[label] = {key: sum(int(r[key]) for r in block) for key in
                        ("games", "departures", "wins", "shortages", "filled_at_departure", "filled_and_won")}
    result["training"] = train
    result["evaluation"] = {p.stem: evaluation(p) for p in sorted(folder.glob("*.jsonl"))}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    print(json.dumps(summarize(parser.parse_args().folder), indent=2))
