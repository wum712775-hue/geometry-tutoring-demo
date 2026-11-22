"""Evaluate prediction of recorded GeoDial teacher highlight flags from learner-turn prefixes."""

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import re

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline


DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data/visual_tutor/geodial_results.json"
SPLITS = ("train", "validation", "test")
WINDOWS = {"latest": 1, "prefix3": 3}


def read_split(path):
    raw = path.read_bytes()
    dialogues = json.loads(raw)
    if not isinstance(dialogues, list) or not dialogues:
        raise ValueError(f"Expected a nonempty GeoDial JSON array in {path.name}.")
    return dialogues, hashlib.sha256(raw).hexdigest()


def check_splits(splits):
    for left, right in itertools.combinations(splits, 2):
        for key in ("qid", "group_id"):
            overlap = {str(row[key]) for row in splits[left]} & {str(row[key]) for row in splits[right]}
            if overlap:
                raise ValueError(f"{left}/{right} overlap on {key} ({len(overlap)} IDs).")


def text_only(text):
    return re.sub(r"\s+", " ", text).strip()


def extract(dialogues):
    """Only learner messages strictly before the target teacher turn are features."""
    rows = []
    cold_start = 0
    entire_turns = 0
    for dialogue in dialogues:
        learner_turns = []
        for turn in dialogue["conversation"]:
            role = turn["role"].lower()
            if role == "student":
                learner_turns.append(text_only(turn["turn"]))
            elif role == "teacher":
                flag = turn["draw_on_image"]
                if type(flag) is not str or flag not in {"0", "1"}:
                    raise ValueError(f"Unexpected draw_on_image flag: {flag!r}")
                entire_turns += 1
                if not learner_turns:
                    cold_start += 1
                    continue
                rows.append({
                    "learner_turns": tuple(learner_turns[-3:]),
                    "highlight": int(flag),
                    "group": str(dialogue["group_id"]),
                })
            else:
                raise ValueError(f"Unexpected conversation role: {role!r}")
    return rows, {"cold_start_excluded": cold_start, "all_teacher_turns": entire_turns}


def corpus(rows, window):
    separator = " RAWTURNDIVIDER "
    return [separator.join(row["learner_turns"][-window:]) for row in rows]


def metrics(labels, predictions):
    return {
        "auroc": float(roc_auc_score(labels, predictions)),
        "average_precision": float(average_precision_score(labels, predictions)),
        "brier": float(brier_score_loss(labels, predictions)),
    }


def paired_brier_interval(rows, labels, latest, prefix, draws=2000):
    """Resample test problem groups for a paired Brier difference, holding models fixed."""
    difference = (prefix - labels) ** 2 - (latest - labels) ** 2
    grouped = {}
    for row, delta in zip(rows, difference):
        group = grouped.setdefault(row["group"], [0.0, 0])
        group[0] += delta
        group[1] += 1
    groups = np.array(list(grouped.values()))
    rng = np.random.default_rng(20261120)
    sampled = groups[rng.integers(len(groups), size=(draws, len(groups)))].sum(axis=1)
    interval = np.quantile(sampled[:, 0] / sampled[:, 1], [0.025, 0.975])
    return {
        "estimate": float(difference.mean()),
        "interval_95": interval.tolist(),
        "resamples": draws,
        "resampling_unit": "test problem group (group_id)",
    }


def run(paths):
    loaded = {split: read_split(paths[split]) for split in SPLITS}
    dialogues = {split: loaded[split][0] for split in SPLITS}
    check_splits(dialogues)
    extracted = {split: extract(dialogues[split]) for split in SPLITS}
    rows = {split: extracted[split][0] for split in SPLITS}
    labels = {split: np.array([row["highlight"] for row in rows[split]]) for split in SPLITS}
    prevalence = float(labels["train"].mean())

    predictions = {
        split: {"prevalence": np.full(len(rows[split]), prevalence)}
        for split in ("validation", "test")
    }
    for model, window in WINDOWS.items():
        estimator = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=25000, sublinear_tf=True),
            LogisticRegression(C=1.0, max_iter=1500, random_state=0),
        )
        estimator.fit(corpus(rows["train"], window), labels["train"])
        for split in predictions:
            predictions[split][model] = estimator.predict_proba(corpus(rows[split], window))[:, 1]

    counts = {}
    for split in SPLITS:
        flags = Counter(labels[split])
        counts[split] = {
            "dialogues": len(dialogues[split]),
            "problem_groups": len({str(row["group_id"]) for row in dialogues[split]}),
            "decision_points": len(rows[split]),
            "recorded_highlights": int(flags[1]),
            **extracted[split][1],
        }
    test = predictions["test"]
    return {
        "dataset": "SankalanPC/Geo2Dial",
        "source": "https://huggingface.co/datasets/SankalanPC/Geo2Dial",
        "dataset_commit": "ba900ab",
        "target": "draw_on_image on the current teacher turn after any prior learner text",
        "feature_scope": "One versus up to three prior student messages. No teacher text, answers, images, or current annotations.",
        "sha256": {split: loaded[split][1] for split in SPLITS},
        "split_overlap": {"qid": 0, "group_id": 0},
        "counts": counts,
        "validation": {model: metrics(labels["validation"], y) for model, y in predictions["validation"].items()},
        "test": {model: metrics(labels["test"], y) for model, y in test.items()},
        "test_brier_delta_prefix_minus_latest": paired_brier_interval(
            rows["test"], labels["test"], test["latest"], test["prefix3"]),
        "versions": {"numpy": np.__version__, "scikit_learn": sklearn.__version__},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for split in SPLITS:
        parser.add_argument("--" + split, type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    report = run({split: getattr(args, split) for split in SPLITS})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": report["counts"], "test": report["test"],
                      "delta": report["test_brier_delta_prefix_minus_latest"]}, indent=2))


if __name__ == "__main__":
    main()
