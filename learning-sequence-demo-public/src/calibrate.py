"""Measure global-test rejection rates under specified synthetic scenarios.

Calibration is simulation evidence for these scenarios, not a universal
guarantee of error control. Cell-specific FDR is not evaluated here.
"""

import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import beta

from generate_data import DATA_DIR, generate_dataframe, positive_int
from analysis import permutation_analysis, sequences_from_dataframe


def binomial_interval(successes: int, trials: int, confidence: float = .95):
    """Exact Clopper-Pearson interval for a binomial rejection probability."""
    if not 0 <= successes <= trials or trials < 1 or not 0 < confidence < 1:
        raise ValueError("Invalid binomial interval arguments.")
    tail = (1 - confidence) / 2
    lower = 0.0 if successes == 0 else float(beta.ppf(tail, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(1 - tail, successes + 1, trials - successes))
    return lower, upper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=positive_int, default=200)
    parser.add_argument("--permutations", type=positive_int, default=499)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--alpha", type=float, default=.05)
    args = parser.parse_args()
    if args.seed < 0 or args.permutations < 2 or not 0 < args.alpha < 1:
        parser.error("Use a non-negative seed, at least two permutations, and alpha in (0, 1).")
    metadata_path = DATA_DIR / "simulation_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    output = DATA_DIR / "calibration"
    output.mkdir(exist_ok=True)
    records, summaries = [], []
    cases = ("iid", "heterogeneous_iid", "markov")
    for case_index, model in enumerate(cases):
        case_records = []
        for replicate in range(args.replicates):
            # Independent streams for data and random permutations; case/replicate
            # identifiers make streams invariant to loop ordering.
            data_seed = np.random.SeedSequence([args.seed, case_index, replicate, 0])
            permutation_seed = np.random.SeedSequence([args.seed, case_index, replicate, 1])
            data = generate_dataframe(metadata["n_students"], metadata["n_events"], data_seed, model)
            _, sequences = sequences_from_dataframe(data)
            result = permutation_analysis(sequences, args.permutations, permutation_seed)
            record = {"scenario": model, "replicate": replicate,
                      "global_p": result["overall_p"],
                      "rejected": bool(result["overall_p"] <= args.alpha)}
            records.append(record)
            case_records.append(record)
            if (replicate + 1) % 25 == 0 or replicate + 1 == args.replicates:
                print(f"Calibration {model}: {replicate + 1}/{args.replicates}", flush=True)
        successes = sum(record["rejected"] for record in case_records)
        lower, upper = binomial_interval(successes, args.replicates)
        summaries.append({"scenario": model,
                          "quantity": "power" if model == "markov" else "false_positive_rate",
                          "replicates": args.replicates, "rejections": successes,
                          "rejection_rate": successes / args.replicates,
                          "ci_95_lower": lower, "ci_95_upper": upper})
        pd.DataFrame(records).to_csv(output / "replicate_tests.csv", index=False)
        pd.DataFrame(summaries).to_csv(output / "rejection_rates.csv", index=False)
    settings = {
        "root_seed": args.seed, "replicates_per_scenario": args.replicates,
        "permutations_per_test": args.permutations, "alpha": args.alpha,
        "n_students": metadata["n_students"], "n_events": metadata["n_events"],
        "simulation_metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "scenarios": {
            "iid": "independent events with the stationary category probabilities",
            "heterogeneous_iid": "learner probabilities drawn from Dirichlet(20 * stationary probabilities); events independent within each learner",
            "markov": "the specified homogeneous first-order Markov alternative",
        },
        "intervals": "pointwise 95% Clopper-Pearson intervals across independent simulations",
        "scope": "global exchangeability test only; no cell-specific FDR or bootstrap-coverage claim",
        "interpretation": "Rates have Monte Carlo uncertainty; compatibility with alpha is not proof of exact calibration.",
    }
    (output / "calibration_settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
