"""Describe Lag-1 transitions and test whole-sequence exchangeability.

Inference shuffles event order separately within each learner. This tests the
global null of exchangeable order conditional on each learner's event counts.
It does not assume that overlapping adjacent pairs are independent.
Cell-wise BH/BY results are exploratory localization under this global null,
not validated local Markov-transition hypothesis tests under partial alternatives.
Run: python src/analysis.py
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import false_discovery_control

from generate_data import BEHAVIORS, DATA_DIR, positive_int


def sequences_from_dataframe(df):
    """Validate and sort each session; never concatenate different learners."""
    required = {"student_id", "event_index", "behavior"}
    if not required.issubset(df.columns) or df.empty:
        raise ValueError("Non-empty data with student_id, event_index, behavior is required.")
    if df[list(required)].isna().any().any():
        raise ValueError("Required fields cannot be missing.")
    if df.duplicated(["student_id", "event_index"]).any():
        raise ValueError("Duplicate event indices within a learner.")
    unknown = set(df["behavior"]) - set(BEHAVIORS)
    if unknown:
        raise ValueError(f"Unknown behavior codes: {sorted(unknown)}")
    if not pd.api.types.is_numeric_dtype(df["event_index"]):
        raise ValueError("event_index must be numeric.")
    mapping = {behavior: i for i, behavior in enumerate(BEHAVIORS)}
    sequences, student_ids = [], []
    for student_id, group in df.groupby("student_id", sort=True):
        group = group.sort_values("event_index")
        if not np.array_equal(group["event_index"].to_numpy(), np.arange(len(group))):
            raise ValueError("This demo expects contiguous indices starting at zero per learner.")
        sequences.append(group["behavior"].map(mapping).to_numpy(dtype=int))
        student_ids.append(student_id)
    return student_ids, sequences


def count_transitions(sequences):
    k = len(BEHAVIORS)
    per_student = np.zeros((len(sequences), k, k), dtype=np.int64)
    for i, sequence in enumerate(sequences):
        pairs = sequence[:-1] * k + sequence[1:]
        per_student[i] = np.bincount(pairs, minlength=k * k).reshape(k, k)
    return per_student


def transition_probabilities(counts):
    """No outgoing observations means unknown (NaN), not probability zero."""
    totals = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, totals, out=np.full(counts.shape, np.nan, dtype=float),
                     where=totals > 0)


def learner_bootstrap_intervals(per_student, n_bootstrap=1999, seed=44, confidence=.95):
    """Pointwise percentile intervals for pooled transition probabilities.

    Resample whole learners, preserving their complete transition counts.
    Learners are assumed independent. These intervals are not simultaneous
    confidence bands or cell-level significance tests.
    """
    if n_bootstrap < 2 or not 0 < confidence < 1:
        raise ValueError("Use at least two bootstrap draws and confidence in (0, 1).")
    if per_student.ndim != 3 or len(per_student) < 2:
        raise ValueError("At least two learner count matrices are required for bootstrap intervals.")
    if not np.isfinite(per_student).all() or (per_student < 0).any():
        raise ValueError("Learner counts must be finite and non-negative.")
    rng = np.random.default_rng(seed)
    samples = np.full((n_bootstrap, *per_student.shape[1:]), np.nan)
    for b in range(n_bootstrap):
        selected = rng.integers(0, len(per_student), size=len(per_student))
        samples[b] = transition_probabilities(per_student[selected].sum(axis=0))
    lower = np.full(per_student.shape[1:], np.nan)
    upper = lower.copy()
    valid_draws = np.isfinite(samples).sum(axis=0)
    tail = (1 - confidence) / 2
    for i, j in np.ndindex(lower.shape):
        finite = samples[:, i, j][np.isfinite(samples[:, i, j])]
        if len(finite):
            lower[i, j], upper[i, j] = np.quantile(finite, [tail, 1 - tail])
    return lower, upper, valid_draws


def exchangeable_expected_counts(sequences):
    """Exact expected adjacencies when each learner's event order is uniform.

For n events with category counts c: E[N_ij] = c_i*(c_j - I[i=j])/n.
This preserves each learner's frequencies rather than pooling learners first.
"""
    k = len(BEHAVIORS)
    expected = np.zeros((k, k), dtype=float)
    for sequence in sequences:
        counts = np.bincount(sequence, minlength=k)
        expected += (np.outer(counts, counts) - np.diag(counts)) / len(sequence)
    return expected


def permutation_analysis(sequences, n_permutations=4999, seed=43):
    """Calibrate statistics by reconstructing transitions after every shuffle."""
    if n_permutations < 2:
        raise ValueError("At least two permutations are required.")
    observed = count_transitions(sequences).sum(axis=0)
    if observed.sum() == 0:
        raise ValueError("No Lag-1 transitions; sessions must contain at least two events.")
    expected = exchangeable_expected_counts(sequences)
    valid = expected > 0
    difference = observed - expected
    statistic = np.sum(difference[valid] ** 2 / expected[valid])
    rng = np.random.default_rng(seed)
    draws = np.empty((n_permutations, *observed.shape), dtype=np.int64)
    for i in range(n_permutations):
        shuffled = [rng.permutation(sequence) for sequence in sequences]
        draws[i] = count_transitions(shuffled).sum(axis=0)
    null_difference = draws - expected
    null_statistic = (null_difference[:, valid] ** 2 / expected[valid]).sum(axis=1)
    # The plus-one calculation prevents zero Monte Carlo p-values.
    overall_p = (1 + np.count_nonzero(null_statistic >= statistic - 1e-12)) / (n_permutations + 1)
    cell_p = (1 + (np.abs(null_difference) >= np.abs(difference) - 1e-12).sum(axis=0)) / (n_permutations + 1)
    cell_p[~valid] = 1.0
    bh = false_discovery_control(cell_p.ravel(), method="bh").reshape(observed.shape)
    by = false_discovery_control(cell_p.ravel(), method="by").reshape(observed.shape)
    null_sd = draws.std(axis=0, ddof=1)
    permutation_z = np.divide(difference, null_sd,
                              out=np.full(observed.shape, np.nan), where=null_sd > 0)
    return {"counts": observed, "expected": expected, "statistic": float(statistic),
            "overall_p": float(overall_p), "cell_p": cell_p, "bh": bh, "by": by,
            "permutation_z": permutation_z}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["markov", "iid"], default="markov")
    parser.add_argument("--permutations", type=positive_int, default=4999)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--bootstrap", type=positive_int, default=1999)
    parser.add_argument("--bootstrap-seed", type=int, default=44)
    args = parser.parse_args()
    if min(args.seed, args.bootstrap_seed) < 0 or min(args.permutations, args.bootstrap) < 2:
        parser.error("Use non-negative seeds and at least two permutation/bootstrap draws.")
    filename = "synthetic_sequences.csv" if args.dataset == "markov" else "iid_sequences.csv"
    student_ids, sequences = sequences_from_dataframe(pd.read_csv(DATA_DIR / filename))
    per_student = count_transitions(sequences)
    counts = per_student.sum(axis=0)
    expected_total = sum(max(len(sequence) - 1, 0) for sequence in sequences)
    if int(counts.sum()) != expected_total:
        raise RuntimeError("Transition total failed the session-boundary check.")
    result = permutation_analysis(sequences, args.permutations, args.seed)
    probabilities = transition_probabilities(counts)
    if len(per_student) >= 2:
        lower, upper, valid_draws = learner_bootstrap_intervals(
            per_student, args.bootstrap, args.bootstrap_seed)
        bootstrap_status = "computed; pointwise percentile intervals"
    else:
        lower = upper = np.full(probabilities.shape, np.nan)
        valid_draws = np.zeros(probabilities.shape, dtype=int)
        bootstrap_status = "not estimable with fewer than two independent learners"
    output = DATA_DIR / f"analysis_{args.dataset}"
    output.mkdir(exist_ok=True)
    for name, matrix in (("transition_counts", counts),
                         ("transition_probabilities", probabilities),
                         ("shuffle_expected_counts", result["expected"])):
        pd.DataFrame(matrix, index=BEHAVIORS, columns=BEHAVIORS).rename_axis(
            "from_behavior").to_csv(output / f"{name}.csv")

    cells = []
    for i, current in enumerate(BEHAVIORS):
        for j, following in enumerate(BEHAVIORS):
            cells.append({"from_behavior": current, "to_behavior": following,
                          "count": int(counts[i, j]), "probability": probabilities[i, j],
                          "ci_95_lower": lower[i, j], "ci_95_upper": upper[i, j],
                          "valid_bootstrap_draws": int(valid_draws[i, j]),
                          "expected_under_shuffle": result["expected"][i, j],
                          "difference_from_shuffle": counts[i, j] - result["expected"][i, j],
                          "permutation_z_descriptive": result["permutation_z"][i, j],
                          "p_shuffle_exploratory": result["cell_p"][i, j],
                          "p_bh_exploratory": result["bh"][i, j],
                          "p_by_exploratory": result["by"][i, j],
                          "selected_exploratory_bh": bool(result["bh"][i, j] <= .05),
                          "selected_exploratory_by": bool(result["by"][i, j] <= .05)})
    pd.DataFrame(cells).to_csv(output / "transition_diagnostics.csv", index=False)
    pd.DataFrame([{"student_id": student_id, "from_behavior": BEHAVIORS[i],
                   "to_behavior": BEHAVIORS[j], "count": int(per_student[s, i, j])}
                  for s, student_id in enumerate(student_ids)
                  for i in range(len(BEHAVIORS)) for j in range(len(BEHAVIORS))]
                 ).to_csv(output / "per_student_transition_counts.csv", index=False)
    summary = {
        "dataset": args.dataset, "n_students": len(sequences),
        "n_records": sum(map(len, sequences)), "n_transitions": expected_total,
        "permutation_seed": args.seed, "n_permutations": args.permutations,
        "overall_statistic": result["statistic"], "overall_permutation_p": result["overall_p"],
        "minimum_monte_carlo_p": 1 / (args.permutations + 1),
        "bootstrap_seed": args.bootstrap_seed, "n_bootstrap": args.bootstrap,
        "bootstrap_status": bootstrap_status,
        "interval_scope": "pointwise 95% learner-bootstrap percentile intervals; not simultaneous",
        "source_dataset_sha256": hashlib.sha256((DATA_DIR / filename).read_bytes()).hexdigest(),
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "simulation_metadata_sha256": hashlib.sha256(
            (DATA_DIR / "simulation_metadata.json").read_bytes()).hexdigest(),
        "null_hypothesis": "exchangeable event order within each learner conditional on category counts",
        "interpretation": "global order-dependence test; cell localization is exploratory",
        "limitations": [
            "Event-order exchangeability under the null is required.",
            "Real learning trends, task phases and multiple sessions require a redesigned null model.",
            "BH/BY do not establish cell-specific FDR control under partial temporal alternatives.",
            "Both excess and deficit transitions are tested; inspect the signed difference.",
        ],
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
    }
    (output / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"{args.dataset}: {len(sequences)} learners; {expected_total} within-learner transitions.")
    print(f"Global exchangeability permutation p = {result['overall_p']:.6f}")
    print("BH/BY cell results saved as exploratory diagnostics, not cognitive findings.")


if __name__ == "__main__":
    main()
