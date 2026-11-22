"""Check recovery of generating probabilities, separately from significance.

Finite samples produce estimation error. Tied maximum probabilities are kept
as sets; uniform rows do not provide an informative top-transition target.
Run: python src/validate.py
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from generate_data import (BEHAVIORS, DATA_DIR, TRUE_TRANSITION,
                           generate_dataframe, positive_int, validate_transition)
from analysis import count_transitions, sequences_from_dataframe, transition_probabilities


def recovery_metrics(truth, estimate):
    if truth.shape != estimate.shape:
        raise ValueError("Truth and estimate must have the same shape.")
    observed = np.isfinite(estimate)
    if not observed.any():
        raise ValueError("No estimated probabilities are available.")
    difference = estimate[observed] - truth[observed]
    correlation = None
    if np.std(truth[observed]) > 1e-12 and np.std(estimate[observed]) > 1e-12:
        correlation = float(np.corrcoef(truth[observed], estimate[observed])[0, 1])
    return {"mean_absolute_error": float(np.abs(difference).mean()),
            "root_mean_squared_error": float(np.sqrt(np.mean(difference ** 2))),
            "maximum_absolute_error": float(np.abs(difference).max()),
            "finite_cell_count": int(observed.sum()),
            "total_cell_count": int(truth.size), "correlation_secondary": correlation}


def top_transition_diagnostics(truth, estimate):
    records = []
    for i, behavior in enumerate(BEHAVIORS):
        true_set = np.flatnonzero(np.isclose(truth[i], truth[i].max(), rtol=0, atol=1e-12))
        estimable = np.isfinite(estimate[i]).all()
        estimated_set = (np.flatnonzero(np.isclose(estimate[i], estimate[i].max(),
                                                 rtol=0, atol=1e-12))
                         if estimable else np.array([], dtype=int))
        uniform = len(true_set) == len(BEHAVIORS)
        records.append({
            "from_behavior": behavior,
            "true_maximum_set": " | ".join(BEHAVIORS[j] for j in true_set),
            "estimated_maximum_set": " | ".join(BEHAVIORS[j] for j in estimated_set),
            "informative_row": not uniform,
            "estimated_maxima_all_in_true_set": (
                bool(set(estimated_set).issubset(set(true_set)))
                if estimable and not uniform else None),
            "interpretation": ("uniform truth: top-1 is uninformative" if uniform else
                               "unobserved row" if not estimable else "tie-aware descriptive check"),
        })
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=positive_int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be non-negative.")
    analysis_summary = json.loads((DATA_DIR / "analysis_markov" / "analysis_summary.json"
                                   ).read_text(encoding="utf-8"))
    if analysis_summary.get("analysis_source_sha256") != hashlib.sha256(
            Path(__file__).with_name("analysis.py").read_bytes()).hexdigest():
        raise ValueError("Analysis source changed; rerun src/analysis.py.")
    for filename, key in (("synthetic_sequences.csv", "source_dataset_sha256"),
                          ("simulation_metadata.json", "simulation_metadata_sha256")):
        current_hash = hashlib.sha256((DATA_DIR / filename).read_bytes()).hexdigest()
        if analysis_summary.get(key) != current_hash:
            raise ValueError("Analysis is stale or missing provenance; rerun src/analysis.py.")
    truth_frame = pd.read_csv(DATA_DIR / "true_transition_matrix.csv", index_col=0)
    estimate_frame = pd.read_csv(DATA_DIR / "analysis_markov" / "transition_probabilities.csv",
                                 index_col=0)
    for frame in (truth_frame, estimate_frame):
        if set(frame.index) != set(BEHAVIORS) or set(frame.columns) != set(BEHAVIORS):
            raise ValueError("Matrix labels do not match the behavior codebook.")
    truth = validate_transition(truth_frame.loc[list(BEHAVIORS), list(BEHAVIORS)].to_numpy())
    estimate = estimate_frame.loc[list(BEHAVIORS), list(BEHAVIORS)].to_numpy()
    if not np.allclose(truth, TRUE_TRANSITION, rtol=0, atol=1e-12):
        raise ValueError("Saved truth differs from the generator; regenerate the dataset first.")
    metrics = recovery_metrics(truth, estimate)
    metadata = json.loads((DATA_DIR / "simulation_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("generator_sha256") != hashlib.sha256(
            Path(__file__).with_name("generate_data.py").read_bytes()).hexdigest():
        raise ValueError("Generator source changed; regenerate and reanalyze the data.")
    replicate_errors = []
    seeds = np.random.SeedSequence(args.seed).spawn(args.replicates)
    for i, seed in enumerate(seeds):
        df = generate_dataframe(metadata["n_students"], metadata["n_events"], seed)
        _, sequences = sequences_from_dataframe(df)
        counts = count_transitions(sequences).sum(axis=0)
        recovered = transition_probabilities(counts)
        replicate_errors.append({"replicate": i, **recovery_metrics(truth, recovered)})
    errors = pd.DataFrame(replicate_errors)
    metrics.update({
        "replicate_count": args.replicates, "replicate_root_seed": args.seed,
        "replicate_mae_median": float(errors["mean_absolute_error"].median()),
        "replicate_mae_quantile_025": float(errors["mean_absolute_error"].quantile(.025)),
        "replicate_mae_quantile_975": float(errors["mean_absolute_error"].quantile(.975)),
        "interpretation": "finite-sample recovery diagnostic; no automatic pass/fail threshold",
        "scope": "recovery within the specified synthetic Markov model only",
    })
    output = DATA_DIR / "validation"
    output.mkdir(exist_ok=True)
    (output / "recovery_summary.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    errors.to_csv(output / "replicate_recovery_errors.csv", index=False)
    top_transition_diagnostics(truth, estimate).to_csv(output / "top_transition_checks.csv",
                                                     index=False)
    print(f"MAE = {metrics['mean_absolute_error']:.6f}; RMSE = {metrics['root_mean_squared_error']:.6f}")
    print(f"Across {args.replicates} replicates: median MAE = {metrics['replicate_mae_median']:.6f}")
    print("Uniform rows and tied maxima are handled explicitly; recovery is not a cognition test.")


if __name__ == "__main__":
    main()
