"""Replay scripted decision points and export auditable results and an offline viewer."""
import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "data/tutoring/scenarios.json"
EVENTS = {"plan", "fail", "pass", "revise", "help"}
ACTIONS = {"wait", "probe", "retest"}
POLICIES = ("fixed_question", "latest_event", "failure_count", "sequence")
POLICY_LABELS = {
    "fixed_question": "Fixed question",
    "latest_event": "Latest event",
    "failure_count": "Unordered failure count",
    "sequence": "Sequence-aware",
}


def decide(events, policy):
    """Return an action and a reason using only the observed event prefix."""
    if policy == "fixed_question":
        return "probe", "Offer a question at every decision point."
    latest = events[-1] if events else None
    if policy == "failure_count":
        if events.count("fail") >= 2 or "help" in events:
            return "probe", "The prefix contains two failures or a help request."
        return "wait", "The count threshold was not reached."
    if policy not in {"latest_event", "sequence"}:
        raise ValueError(f"Unknown policy: {policy}")
    if latest == "help":
        return "probe", "The learner just asked for help."
    if latest == "revise":
        return "retest", "The latest revision needs a fresh check."
    if latest == "pass":
        return "wait", "The latest checkpoint passed."
    if policy == "latest_event":
        return "wait", "This event alone does not trigger support."
    if latest == "fail":
        since_revision_or_pass = 0
        for event in reversed(events):
            if event in {"revise", "pass"}:
                break
            since_revision_or_pass += event == "fail"
        if since_revision_or_pass >= 2:
            return "probe", "Two or more failures followed the latest revision or pass."
    return "wait", "Allow independent work while evidence is limited."


def load_cases(path):
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not cases:
        raise ValueError("At least one scenario is required.")
    known_ids = set()
    for case in cases:
        case_id = case["id"]
        if case_id in known_ids:
            raise ValueError(f"Duplicate scenario: {case_id}")
        known_ids.add(case_id)
        if not case["events"] or set(case["events"]) - EVENTS:
            raise ValueError(f"Invalid events in {case_id}")
        if not case["acceptable"] or set(case["acceptable"]) - ACTIONS:
            raise ValueError(f"Invalid rubric actions in {case_id}")
    return cases


def replay(cases):
    decisions = []
    for case in cases:
        for step in range(1, len(case["events"]) + 1):
            prefix = case["events"][:step]
            final = step == len(case["events"])
            for policy in POLICIES:
                action, reason = decide(prefix, policy)
                message = {
                    "wait": "Continue independent work.",
                    "retest": "Rerun the relevant checkpoint on the revised version.",
                    "probe": case["probe"],
                }[action]
                decisions.append({
                    "case": case["id"], "domain": case["domain"], "step": step,
                    "prefix": " → ".join(prefix), "policy": policy, "action": action,
                    "reason": reason, "message": message,
                    "acceptable_actions": "|".join(case["acceptable"]) if final else "",
                    "match": int(action in case["acceptable"]) if final else "",
                })
    return decisions


def summarise(rows):
    summary = []
    for domain in ("All", *sorted({row["domain"] for row in rows})):
        for policy in POLICIES:
            scored = [row for row in rows if row["policy"] == policy
                      and row["match"] != "" and (domain == "All" or row["domain"] == domain)]
            summary.append({
                "domain": domain, "policy": policy, "matched": sum(row["match"] for row in scored),
                "cases": len(scored), "questions": sum(row["action"] == "probe" for row in scored),
            })
    return summary


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_viewer(path, cases, rows, summary):
    template = (ROOT / "src/tutoring_viewer.html").read_text(encoding="utf-8")
    payload = json.dumps({"cases": cases, "rows": rows, "summary": summary,
                          "labels": POLICY_LABELS}, ensure_ascii=True).replace("<", "\\u003c")
    path.write_text(template.replace("DEMO_PAYLOAD", payload), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=Path, default=SCENARIOS)
    parser.add_argument("--out", type=Path, default=ROOT / "data/tutoring")
    parser.add_argument("--viewer", type=Path, default=ROOT / "SCRIPTED_REPLAY.html")
    args = parser.parse_args()
    cases = load_cases(args.scenarios)
    rows = replay(cases)
    summary = summarise(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "decisions.csv", rows)
    write_csv(args.out / "summary.csv", summary)
    write_viewer(args.viewer, cases, rows, summary)
    for row in summary:
        if row["domain"] == "All":
            print(f"{row['policy']}: {row['matched']}/{row['cases']} rubric matches")


if __name__ == "__main__":
    main()
