"""Generate synthetic coded events, not raw eye-tracking or real learner data.

The first-order Markov probabilities are simulation assumptions, not empirical
findings. The IID baseline matches stationary category probabilities in expectation.
Run: python src/generate_data.py
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
BEHAVIORS = ("Gaze_A", "Gaze_B", "Error_Op", "Look_Back", "Correct_Op")

# Rows = current event; columns = next event, both in BEHAVIORS order.
# These probabilities are chosen for a method demonstration.
TRUE_TRANSITION = np.array([
    [0.20, 0.20, 0.20, 0.20, 0.20],
    [0.20, 0.20, 0.20, 0.20, 0.20],
    [0.10, 0.10, 0.10, 0.50, 0.20],
    [0.10, 0.10, 0.10, 0.10, 0.60],
    [0.40, 0.40, 0.05, 0.05, 0.10],
], dtype=float)


def validate_transition(matrix):
    """Reject incorrect probabilities instead of silently normalizing them."""
    matrix = np.asarray(matrix, dtype=float)
    k = len(BEHAVIORS)
    if matrix.shape != (k, k):
        raise ValueError(f"Transition matrix must have shape {(k, k)}.")
    if not np.isfinite(matrix).all() or (matrix < 0).any() or (matrix > 1).any():
        raise ValueError("Transition probabilities must be finite and in [0, 1].")
    if not np.allclose(matrix.sum(axis=1), 1, rtol=0, atol=1e-12):
        raise ValueError("Every transition-matrix row must sum to 1.")
    return matrix


def stationary_distribution(matrix: np.ndarray) -> np.ndarray:
    """Find the unique stationary distribution, including periodic chains.

    Solving the balance equations avoids an arbitrary burn-in period. Reject
    matrices without a unique stationary distribution instead of selecting one.
    """
    matrix = validate_transition(matrix)
    k = len(matrix)
    equations = np.vstack((matrix.T - np.eye(k), np.ones(k)))
    target = np.r_[np.zeros(k), 1.0]
    probabilities, _, rank, _ = np.linalg.lstsq(equations, target, rcond=None)
    if rank != k or probabilities.min() < -1e-12:
        raise ValueError("A unique non-negative stationary distribution is required.")
    # Remove floating-point roundoff only; the transition matrix is never normalized.
    probabilities = np.maximum(probabilities, 0)
    probabilities /= probabilities.sum()
    if not np.allclose(probabilities @ matrix, probabilities, rtol=0, atol=1e-12):
        raise ValueError("Stationary balance equations failed validation.")
    return probabilities


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return value


def generate_dataframe(n_students=50, n_events=30, seed=42, model="markov"):
    """Generate one session per simulated learner using a local RNG."""
    if model not in {"markov", "iid", "heterogeneous_iid"}:
        raise ValueError("model must be 'markov', 'iid', or 'heterogeneous_iid'.")
    for name, value in (("n_students", n_students), ("n_events", n_events)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    matrix = validate_transition(TRUE_TRANSITION)
    initial_probabilities = stationary_distribution(matrix)
    rng = np.random.default_rng(seed)
    k = len(BEHAVIORS)
    records = []
    for student_id in range(n_students):
        learner_probabilities = (rng.dirichlet(20 * initial_probabilities)
                                 if model == "heterogeneous_iid" else initial_probabilities)
        # Integer codes avoid repeated string lookups in the transition matrix.
        states = np.empty(n_events, dtype=int)
        states[0] = rng.choice(k, p=learner_probabilities)
        for event_index in range(1, n_events):
            probabilities = (matrix[states[event_index - 1]]
                             if model == "markov" else learner_probabilities)
            states[event_index] = rng.choice(k, p=probabilities)
        for event_index, state in enumerate(states):
            records.append({
                "student_id": student_id,
                "event_index": event_index,
                "behavior": BEHAVIORS[state],
                "is_synthetic": True,
                "simulation_model": model,
            })
    return pd.DataFrame.from_records(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--students", type=positive_int, default=50)
    parser.add_argument("--events", type=positive_int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be non-negative.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "figures").mkdir(exist_ok=True)
    seeds = np.random.SeedSequence(args.seed).spawn(2)
    filenames = {"markov": "synthetic_sequences.csv", "iid": "iid_sequences.csv"}
    for model, seed in zip(filenames, seeds):
        df = generate_dataframe(args.students, args.events, seed, model)
        df.to_csv(DATA_DIR / filenames[model], index=False)
        print(f"{model}: {args.students} simulated learners, {len(df)} coded events, "
              f"{args.students * (args.events - 1)} within-learner Lag-1 transitions.")

    pd.DataFrame(TRUE_TRANSITION, index=BEHAVIORS, columns=BEHAVIORS).rename_axis(
        "from_behavior").to_csv(DATA_DIR / "true_transition_matrix.csv")
    initial_probabilities = stationary_distribution(TRUE_TRANSITION)
    pd.DataFrame(np.tile(initial_probabilities, (len(BEHAVIORS), 1)),
                 index=BEHAVIORS, columns=BEHAVIORS).rename_axis(
        "from_behavior").to_csv(DATA_DIR / "iid_transition_matrix.csv")

    metadata = {
        "data_status": "fully synthetic; no real learner records",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "n_students": args.students, "n_events": args.events,
        "root_seed": args.seed,
        "random_streams": {model: {"spawn_key": list(seed.spawn_key)}
                           for model, seed in zip(filenames, seeds)},
        "rng": "numpy.random.default_rng / PCG64",
        "initial_probabilities": initial_probabilities.tolist(),
        "initialization": "stationary distribution for both datasets",
        "baseline_matching": "same expected marginal category probabilities; independent streams",
        "behavior_order": list(BEHAVIORS),
        "transition_matrix": TRUE_TRANSITION.tolist(),
        "code_definitions": {
            "Gaze_A": "hypothetical gaze event at region A",
            "Gaze_B": "hypothetical gaze event at region B",
            "Error_Op": "hypothetical incorrect operation event",
            "Look_Back": "hypothetical return-look event, assigned by the simulator",
            "Correct_Op": "hypothetical correct operation event",
        },
        "coding_assumption": "one mutually exclusive coded event per index",
        "limitations": [
            "Not raw gaze samples, fixation detection, or gaze duration data.",
            "No timestamps, simultaneous channels, cognitive labels or learning outcomes.",
            "One homogeneous, time-invariant Markov matrix; no learner differences.",
            "IID and Markov marginal probabilities agree in expectation, not exactly in each sample.",
        ],
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__},
    }
    (DATA_DIR / "simulation_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved synthetic datasets and generating probabilities to {DATA_DIR}")


if __name__ == "__main__":
    main()
