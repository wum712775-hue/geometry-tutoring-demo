"""Predict the next synthetic Error_Op event using learner-separated folds.

This is an engineering benchmark on coded synthetic events. It does not predict
confusion, cognitive load, or real VR behavior. The generating model is first-order,
so longer history has no population-level information beyond the current state.
"""

import argparse
import hashlib
import json

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from generate_data import BEHAVIORS, DATA_DIR, TRUE_TRANSITION, positive_int
from analysis import sequences_from_dataframe


def make_prediction_examples(df: pd.DataFrame, window: int = 5):
    """Features use only the window ending immediately BEFORE the target event."""
    if window < 2:
        raise ValueError("The observation window must contain at least two events.")
    learner_ids, sequences = sequences_from_dataframe(df)
    k = len(BEHAVIORS)
    error_code = BEHAVIORS.index("Error_Op")
    features, labels, groups, rows = [], [], [], []
    for learner_id, sequence in zip(learner_ids, sequences):
        for target_index in range(window, len(sequence)):
            history = sequence[target_index - window:target_index]
            last = np.eye(k)[history[-1]]
            frequencies = np.bincount(history, minlength=k) / window
            pairs = history[:-1] * k + history[1:]
            transitions = np.bincount(pairs, minlength=k * k) / (window - 1)
            features.append(np.r_[last, frequencies, transitions])
            labels.append(int(sequence[target_index] == error_code))
            groups.append(learner_id)
            rows.append({"student_id": learner_id, "target_event_index": target_index,
                         "last_observed_code": int(history[-1]),
                         "observed_until_index": target_index - 1})
    if not features:
        raise ValueError("No prediction examples: increase sequence length or reduce window size.")
    return np.asarray(features), np.asarray(labels), np.asarray(groups), pd.DataFrame(rows)


def metrics(y, probabilities):
    return {"roc_auc": float(roc_auc_score(y, probabilities)),
            "average_precision": float(average_precision_score(y, probabilities)),
            "brier_score": float(brier_score_loss(y, probabilities)),
            "log_loss": float(log_loss(y, probabilities, labels=[0, 1]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=positive_int, default=5)
    parser.add_argument("--folds", type=positive_int, default=5)
    args = parser.parse_args()
    path = DATA_DIR / "synthetic_sequences.csv"
    X, y, groups, rows = make_prediction_examples(pd.read_csv(path), args.window)
    if len(np.unique(y)) != 2 or not 2 <= args.folds <= len(np.unique(groups)):
        parser.error("Use both outcome classes and 2 <= folds <= number of learners.")
    k = len(BEHAVIORS)
    feature_sets = {"frequency_only": slice(k, 2 * k),
                    "last_event_only": slice(0, k),
                    "last_event_plus_sequence": slice(0, X.shape[1])}
    names = ["training_prevalence", *feature_sets]
    predictions = {name: np.full(len(y), np.nan) for name in names}
    folds = np.full(len(y), -1, dtype=int)
    split_records = []
    fold_metrics = []
    for fold, (train, test) in enumerate(GroupKFold(n_splits=args.folds).split(X, y, groups)):
        train_ids, test_ids = set(groups[train]), set(groups[test])
        if train_ids & test_ids:
            raise RuntimeError("Learner leakage detected.")
        if len(np.unique(y[train])) != 2 or len(np.unique(y[test])) != 2:
            raise ValueError("A fold lacks an outcome class; use more data or fewer folds.")
        folds[test] = fold
        split_records.append({"fold": fold,
                              "training_learners": sorted(map(int, train_ids)),
                              "test_learners": sorted(map(int, test_ids))})
        baseline = DummyClassifier(strategy="prior").fit(X[train, :1], y[train])
        predictions["training_prevalence"][test] = baseline.predict_proba(X[test, :1])[:, 1]
        for name, columns in feature_sets.items():
            # Preprocessing is fitted inside each training fold. No tuning or
            # threshold selection uses the held-out learners.
            model = make_pipeline(StandardScaler(),
                                  LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000))
            model.fit(X[train, columns], y[train])
            predictions[name][test] = model.predict_proba(X[test, columns])[:, 1]
        for name in names:
            fold_metrics.append({"fold": fold, "model": name,
                                 **metrics(y[test], predictions[name][test])})
    if (folds < 0).any() or any(not np.isfinite(p).all() for p in predictions.values()):
        raise RuntimeError("Incomplete out-of-fold predictions.")
    output = DATA_DIR / "prediction"
    output.mkdir(exist_ok=True)
    rows["target_error"] = y
    rows["fold"] = folds
    for name, probabilities in predictions.items():
        rows[name] = probabilities
    # This uses known simulation parameters and is a reference, not a trained model.
    oracle = TRUE_TRANSITION[rows["last_observed_code"].to_numpy(), BEHAVIORS.index("Error_Op")]
    rows["generating_probability_reference"] = oracle
    summary = [{"model": name, **metrics(y, probabilities)}
               for name, probabilities in predictions.items()]
    summary.append({"model": "generating_probability_reference", **metrics(y, oracle)})
    rows.to_csv(output / "out_of_fold_predictions.csv", index=False)
    pd.DataFrame(summary).to_csv(output / "prediction_metrics.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(output / "fold_metrics.csv", index=False)
    (output / "fold_assignments.json").write_text(json.dumps(split_records, indent=2) + "\n", encoding="utf-8")
    settings = {
        "target": "whether the immediately next synthetic event is Error_Op",
        "window_events": args.window, "folds": args.folds,
        "n_examples": len(y), "n_learners": len(np.unique(groups)),
        "outcome_prevalence": float(y.mean()), "hyperparameter_C": 1.0,
        "source_dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sklearn_version": sklearn.__version__,
        "estimand": "pooled out-of-fold performance for unseen simulated learners",
        "limitations": [
            "Within-learner windows overlap; event-level metric uncertainty is not estimated.",
            "No hyperparameter tuning, real-data generalization, or cognitive-state validation.",
            "The known first-order model makes last-event information sufficient in the population.",
            "The generating-probability reference accesses simulation truth and is not deployable.",
        ],
    }
    (output / "prediction_settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
