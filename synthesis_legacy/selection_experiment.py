"""Compare selection policies on the same eligible preference candidates and released family quotas."""
import argparse
import hashlib
import json
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REASONS = ("factual_error", "incomplete", "off_target")
AXES = ("task", "profile", "reason")
POLICIES = ("A_random", "B_labels", "C_length_labels")
DEFAULT_SEED = 20261104
DEFAULT_REPEATS = 50


@dataclass(frozen=True)
class Pair:
    cid: str
    family: str
    task: str
    profile: str
    reason: str
    direction: int  # -1, 0, +1 when chosen has fewer, the same, or more characters

    @property
    def shorter(self):
        return self.direction < 0


def read_inputs(dataset):
    ledger_bytes = (dataset / "audit" / "candidates.jsonl").read_bytes()
    ledger = [json.loads(line) for line in ledger_bytes.splitlines()]
    # generate.py assigns candidate C00001, C00002, ... at generation time; JSONL line
    # order and the shuffled export_line are not used as candidate tie-breakers.
    def sequence(row):
        cid = row["candidate_id"]
        if not cid.startswith("C") or not cid[1:].isdigit():
            raise ValueError(f"Invalid generation ID: {cid}")
        return int(cid[1:])

    ledger.sort(key=sequence)
    if [sequence(row) for row in ledger] != list(range(1, len(ledger) + 1)):
        raise ValueError("Generation IDs missing or duplicated")
    pref = [r for r in ledger if r["split"] == "preference"]
    pool, released, quotas = [], {}, Counter()
    export_bytes = (dataset / "preference.jsonl").read_bytes()
    exported = [json.loads(line) for line in export_bytes.splitlines()]
    for r in pref:
        if r["decision"] not in ("selected", "quota_surplus"):
            continue
        delta = len(r["chosen"]) - len(r["rejected"])
        pair = Pair(r["candidate_id"], r["family"], r["task"], r["profile"], r["reason"],
                    (delta > 0) - (delta < 0))
        if pair.shorter != r["chosen_shorter"] or pair.reason not in REASONS:
            raise ValueError(f"Ledger label differs from recomputed text properties: {pair.cid}")
        pool.append(pair)
        if r["decision"] == "selected":
            line = r["export_line"]
            if line in released or not 1 <= line <= len(exported):
                raise ValueError(f"Invalid export pointer: {pair.cid}")
            if exported[line - 1] != {k: r[k] for k in ("prompt", "chosen", "rejected")}:
                raise ValueError(f"Export disagrees with ledger: {pair.cid}")
            released[line] = pair.cid
            quotas[pair.family] += 1
    if set(released) != set(range(1, 501)) or len(exported) != 500 or set(quotas) != {p.family for p in pool}:
        raise ValueError("Incomplete preference release or family quota")
    inputs = {"ledger_rows": len(ledger), "preference_candidates": len(pref), "eligible_preference_rows": len(pool),
              "released_preference_rows": len(exported), "ledger_sha256": hashlib.sha256(ledger_bytes).hexdigest(),
              "export_sha256": hashlib.sha256(export_bytes).hexdigest()}
    return pool, quotas, set(released.values()), inputs


def by_family(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row.family].append(row)
    return groups


def choose(groups, quotas, policy):
    selected = []
    for family, original in groups.items():
        target = quotas[family]
        if policy == "A_random":
            selected.extend(original[:target])
            continue
        pool = original.copy()
        tasks, profiles, reasons, short_groups = (Counter() for _ in range(4))

        def priority(row):
            labels = tasks[row.task] + profiles[row.profile] + reasons[row.reason]
            first = short_groups[row.shorter] if policy == "C_length_labels" else 0
            return first, labels

        for _ in range(target):
            # min() returns the earliest index in the shared order when priorities tie.
            index = min(range(len(pool)), key=lambda j: priority(pool[j]))
            row = pool.pop(index)
            selected.append(row)
            tasks[row.task] += 1
            profiles[row.profile] += 1
            reasons[row.reason] += 1
            short_groups[row.shorter] += 1
    if Counter(row.family for row in selected) != quotas:
        raise ValueError("Selection failed to match the release quotas")
    return selected


def ids_hash(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode("ascii")).hexdigest()


def metrics(rows, axis_levels):
    n = len(rows)
    sizes = Counter(p.direction for p in rows)
    long_score = (sizes[1] + 0.5 * sizes[0]) / n
    values = {"short_count": sizes[-1], "equal_count": sizes[0], "long_count": sizes[1],
              "short_rate": sizes[-1] / n, "equal_rate": sizes[0] / n, "long_rate": sizes[1] / n,
              "long_heuristic": long_score, "majority_direction_heuristic": max(long_score, 1 - long_score)}
    for reason in REASONS:
        matched = [r for r in rows if r.reason == reason]
        if not matched:
            raise ValueError(f"No observations for {reason}")
        values[f"count_{reason}"] = len(matched)
        values[f"short_rate_{reason}"] = sum(p.shorter for p in matched) / len(matched)
    groups = by_family(rows)
    for axis in AXES:
        levels = axis_levels[axis]
        score = 0.0
        for family_rows in groups.values():
            size = len(family_rows)
            counts = Counter(getattr(p, axis) for p in family_rows)
            tv = sum(abs(counts[v] / size - 1 / len(levels)) for v in levels) / 2
            score += size / n * tv
        values[f"local_tv_{axis}"] = score
    values["local_tv_avg"] = sum(values[f"local_tv_{a}"] for a in AXES) / len(AXES)
    return values


def describe(values):
    return {"mean": statistics.mean(values), "sd": statistics.stdev(values),
            "min": min(values), "max": max(values)}


def run(dataset, seed, repeats):
    if repeats < 30:
        raise ValueError("Use at least 30 paired repetitions")
    pool, quotas, release, inputs = read_inputs(dataset)
    categories = {axis: sorted({getattr(p, axis) for p in pool}) for axis in AXES}
    original_groups = by_family(pool)
    original_rows = choose(original_groups, quotas, "C_length_labels")
    recovered = {r.cid for r in original_rows}
    if recovered != release:
        raise ValueError(f"Unshuffled policy C misses {len(release - recovered)} release IDs")
    supply = {f: {"quota": quotas[f], "eligible": len(group),
                  "short": sum(p.shorter for p in group),
                  "equal": sum(p.direction == 0 for p in group),
                  "long": sum(p.direction > 0 for p in group)}
              for f, group in sorted(original_groups.items())}
    trials = []
    for iteration in range(repeats):
        trial_seed = seed + iteration
        shuffled = pool.copy()
        random.Random(trial_seed).shuffle(shuffled)
        groups = by_family(shuffled)
        observations = {}
        for policy in POLICIES:
            rows = choose(groups, quotas, policy)
            observations[policy] = {"metrics": metrics(rows, categories),
                                    "sorted_ids_sha256": ids_hash(p.cid for p in rows)}
        trials.append({"iteration": iteration, "seed": trial_seed, "policies": observations})

    reference_metrics = metrics(original_rows, categories)
    summary = {p: {key: describe([trial["policies"][p]["metrics"][key] for trial in trials])
                   for key in reference_metrics} for p in POLICIES}
    fractions = [k for k in reference_metrics if "count" not in k]
    paired = {}
    for treatment, control in ((POLICIES[1], POLICIES[0]), (POLICIES[2], POLICIES[0]), (POLICIES[2], POLICIES[1])):
        pair_name = f"{treatment}_minus_{control}"
        paired[pair_name] = {}
        for key in fractions:
            diffs = [100 * (trial["policies"][treatment]["metrics"][key] - trial["policies"][control]["metrics"][key])
                     for trial in trials]
            paired[pair_name][key] = describe(diffs)
    return {
        "design": {"seed": seed, "repeats": repeats,
                   "eligible_pool": "preference rows whose decision is selected or quota_surplus; fixed inputs for all trials",
                   "quotas": "Released number of rows in each family (verified against the export), identical under A/B/C",
                   "pairing": "One global permutation per repetition, same ordered candidates and quotas for all policies. Random(seed + iteration), starting from numeric candidate-ID order.",
                   "policies": {"A_random": "First q candidates in each family; uniform without replacement after shuffling.",
                                "B_labels": "Greedy minimum sum of already-selected counts for candidate task, profile, reason inside family; tie goes to earliest shared position.",
                                "C_length_labels": "Greedy minimum (count in candidate chosen-shorter boolean group, sum of three label counts). Equal joins not-shorter. Same tie-break as B."}},
        "definitions": {
            "length": "Python len on entire Unicode response, including code, whitespace and punctuation; no tokenizer.",
            "short_rate_REASON": "Strictly shorter chosen inside that reason / all selected examples of the same reason.",
            "long_heuristic": "Expected accuracy of always choosing the longer response; tied lengths score 0.5. Mathematically (long_count + 0.5 * equal_count) / N.",
            "majority_direction_heuristic": "The larger of long_heuristic and 1 - long_heuristic, computed per selected set; best single global length direction in hindsight, not a trained or held-out model.",
            "local_tv_AXIS": "Family-quota-weighted mean of 0.5 * sum_c |observed fraction in family - 1/K| over the K eligible-pool categories of that axis. Smaller is closer to uniform.",
            "local_tv_avg": "Simple mean of the three family-weighted TV values. Fractions and TV are reported on a 0..1 scale.",
            "summary": "Mean, sample SD (ddof=1), observed min/max over the permutations of this one fixed eligible pool. Counts are numbers of pairs.",
            "paired_differences_pp": "Per-permutation treatment minus control in percentage points (100 times the difference of fractions/TV). A SD of zero means a fixed result across these repeats.",
            "limits": "Describes selection composition and tie-break sensitivity on one fixed synthetic pool. It measures neither training outcomes nor learning outcomes."},
        "inputs": inputs, "axis_categories": categories, "family_supply_and_quota": supply,
        "original_order_verification": {"published_ids_matched": len(recovered & release), "missing": len(release - recovered),
                                        "unexpected": len(recovered - release), "sorted_ids_sha256": ids_hash(release),
                                        "selected_metrics": reference_metrics},
        "policies_summary": summary, "paired_differences_pp": paired, "trials": trials}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset")
    parser.add_argument("--out", type=Path, default=ROOT / "selection_experiment.json")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    args = parser.parse_args()
    report = run(args.dataset, args.seed, args.repeats)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    print(f"Published IDs matched: {report['original_order_verification']['published_ids_matched']}/500; "
          f"eligible: {report['inputs']['eligible_preference_rows']}; paired trials: {args.repeats}.")
    for policy, summary in report["policies_summary"].items():
        print(f"{policy}: chosen shorter {100 * summary['short_rate']['mean']:.2f}%, "
              f"always longer {100 * summary['long_heuristic']['mean']:.2f}%, "
              f"within-family TV {summary['local_tv_avg']['mean']:.4f}")


if __name__ == "__main__":
    main()
