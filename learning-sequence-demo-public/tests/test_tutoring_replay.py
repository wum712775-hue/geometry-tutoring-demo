import unittest

from src.tutoring_replay import POLICIES, SCENARIOS, decide, load_cases, replay, summarise


class TutoringReplayTests(unittest.TestCase):
    def test_order_adds_information_beyond_counts_and_latest_event(self):
        repeated = ["revise", "fail", "fail"]
        first_after_revision = ["fail", "revise", "fail"]
        for policy in ("latest_event", "failure_count"):
            self.assertEqual(decide(repeated, policy), decide(first_after_revision, policy))
        self.assertEqual(decide(repeated, "sequence")[0], "probe")
        self.assertEqual(decide(first_after_revision, "sequence")[0], "wait")

    def test_revisions_and_passes_reset_failure_evidence(self):
        self.assertEqual(decide(["fail", "fail", "revise"], "sequence")[0], "retest")
        self.assertEqual(decide(["fail", "fail", "pass", "fail"], "sequence")[0], "wait")
        self.assertEqual(decide(["fail", "revise", "help"], "sequence")[0], "probe")

    def test_scoring_uses_only_final_decisions_and_exposes_context_error(self):
        cases = load_cases(SCENARIOS)
        rows = replay(cases)
        self.assertEqual(sum(row["match"] != "" for row in rows), len(cases) * len(POLICIES))
        for row in (row for row in summarise(rows) if row["domain"] == "All"):
            self.assertEqual(row["cases"], len(cases))
        context_case = [row for row in rows if row["case"] == "P10"
                        and row["policy"] == "sequence" and row["match"] != ""]
        self.assertEqual(context_case[0]["match"], 0)
