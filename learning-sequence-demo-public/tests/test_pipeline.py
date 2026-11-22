"""Regression checks for session boundaries, null expectations and recovery."""

import itertools
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from generate_data import (BEHAVIORS, TRUE_TRANSITION, generate_dataframe,
                           stationary_distribution, validate_transition)
from analysis import (count_transitions, exchangeable_expected_counts,
                      sequences_from_dataframe, transition_probabilities,
                      permutation_analysis, learner_bootstrap_intervals)
from validate import top_transition_diagnostics, recovery_metrics
from calibrate import binomial_interval
from predict_next_event import make_prediction_examples


class PipelineChecks(unittest.TestCase):
    def test_session_boundaries_and_sorting(self):
        # Two sessions have exactly two transitions, not three.
        df = pd.DataFrame({"student_id": [1, 0, 1, 0], "event_index": [1, 1, 0, 0],
                           "behavior": ["Correct_Op", "Gaze_B", "Error_Op", "Gaze_A"]})
        _, sequences = sequences_from_dataframe(df)
        counts = count_transitions(sequences).sum(axis=0)
        self.assertEqual(int(counts.sum()), 2)
        self.assertEqual(counts[0, 1], 1)
        self.assertEqual(counts[2, 4], 1)
        self.assertEqual(counts[1, 2], 0)

    def test_conditional_expectation_against_exhaustive_orders(self):
        # Exhaust all distinct orders to check the shuffle expectation formula.
        orders = [np.array(order) for order in set(itertools.permutations([0, 0, 1, 2]))]
        exhaustive_mean = count_transitions(orders).mean(axis=0)
        formula_mean = exchangeable_expected_counts([np.array([0, 0, 1, 2])])
        np.testing.assert_allclose(formula_mean, exhaustive_mean, rtol=0, atol=1e-12)

    def test_unknown_rows_remain_missing(self):
        counts = np.zeros((5, 5), dtype=int)
        counts[0, 1] = 3
        probabilities = transition_probabilities(counts)
        self.assertEqual(probabilities[0, 1], 1)
        self.assertTrue(np.isnan(probabilities[1:]).all())

    def test_uniform_and_tied_truth(self):
        estimate = TRUE_TRANSITION.copy()
        estimate[0] = [.18, .24, .19, .20, .19]
        estimate[4] = [.39, .41, .05, .05, .10]
        checks = top_transition_diagnostics(TRUE_TRANSITION, estimate).set_index("from_behavior")
        self.assertFalse(checks.loc["Gaze_A", "informative_row"])
        self.assertTrue(checks.loc["Correct_Op", "estimated_maxima_all_in_true_set"])
        self.assertIn("Gaze_A", checks.loc["Correct_Op", "true_maximum_set"])
        self.assertIn("Gaze_B", checks.loc["Correct_Op", "true_maximum_set"])

    def test_invalid_matrix_is_rejected(self):
        invalid = TRUE_TRANSITION.copy()
        invalid[0, 0] = .30
        with self.assertRaises(ValueError):
            validate_transition(invalid)

    def test_duplicate_event_is_rejected(self):
        df = pd.DataFrame({"student_id": [0, 0], "event_index": [0, 0],
                           "behavior": ["Gaze_A", "Gaze_B"]})
        with self.assertRaises(ValueError):
            sequences_from_dataframe(df)

    def test_seed_and_default_transition_total(self):
        a = generate_dataframe(seed=42)
        b = generate_dataframe(seed=42)
        pd.testing.assert_frame_equal(a, b)
        _, sequences = sequences_from_dataframe(a)
        self.assertEqual(len(a), 1500)
        self.assertEqual(int(count_transitions(sequences).sum()), 1450)

    def test_monte_carlo_p_values_and_singletons(self):
        sequences = [np.array([0, 1, 0, 1]), np.array([2, 3, 2, 3])]
        result = permutation_analysis(sequences, n_permutations=99, seed=123)
        self.assertGreaterEqual(result["overall_p"], .01)
        self.assertTrue(((result["cell_p"] >= .01) & (result["cell_p"] <= 1)).all())
        with self.assertRaises(ValueError):
            permutation_analysis([np.array([0]), np.array([1])], n_permutations=2)

    def test_stationary_distribution_known_solutions(self):
        probabilities = np.array([.10, .20, .30, .15, .25])
        iid_matrix = np.tile(probabilities, (5, 1))
        np.testing.assert_allclose(stationary_distribution(iid_matrix), probabilities,
                                   atol=1e-12, rtol=0)
        # A deterministic cycle is periodic but has a unique stationary measure.
        cycle = np.roll(np.eye(5), 1, axis=1)
        np.testing.assert_allclose(stationary_distribution(cycle), np.full(5, .2),
                                   atol=1e-12, rtol=0)
        stationary = stationary_distribution(TRUE_TRANSITION)
        np.testing.assert_allclose(stationary @ TRUE_TRANSITION, stationary,
                                   atol=1e-12, rtol=0)

    def test_nonunique_stationary_distribution_is_rejected(self):
        with self.assertRaises(ValueError):
            stationary_distribution(np.eye(5))

    def test_bootstrap_resamples_whole_learners_and_preserves_missing_rows(self):
        # Cluster resampling must include the two extreme learner-only outcomes.
        # Resampling 101 individual transitions would give a very different range.
        per_learner = np.zeros((2, 5, 5), dtype=int)
        per_learner[0, 0, 0] = 100
        per_learner[1, 0, 1] = 1
        lower, upper, valid = learner_bootstrap_intervals(per_learner, 500, seed=7)
        np.testing.assert_array_equal(lower[0, :2], [0, 0])
        np.testing.assert_array_equal(upper[0, :2], [1, 1])
        self.assertTrue(np.isnan(lower[1:]).all())
        self.assertTrue(np.isnan(upper[1:]).all())
        np.testing.assert_array_equal(valid[0], np.full(5, 500))
        self.assertTrue((valid[1:] == 0).all())
        repeated = learner_bootstrap_intervals(per_learner, 500, seed=7)
        for original, rerun in zip((lower, upper, valid), repeated):
            np.testing.assert_array_equal(original, rerun)

    def test_bootstrap_requires_independent_units(self):
        with self.assertRaises(ValueError):
            learner_bootstrap_intervals(np.ones((1, 5, 5)), 10)

    def test_binomial_intervals_include_boundary_uncertainty(self):
        low, high = binomial_interval(0, 10)
        self.assertEqual(low, 0)
        self.assertAlmostEqual(high, 1 - .025 ** .1)
        low, high = binomial_interval(10, 10)
        self.assertAlmostEqual(low, .025 ** .1)
        self.assertEqual(high, 1)

    def test_prediction_uses_only_available_history(self):
        df = generate_dataframe(n_students=2, n_events=8, seed=9)
        X, y, groups, rows = make_prediction_examples(df, window=3)
        changed = df.copy()
        first_target = (changed.student_id == 0) & (changed.event_index == 3)
        changed.loc[first_target, "behavior"] = "Gaze_A" if y[0] else "Error_Op"
        changed.loc[(changed.student_id == 0) & (changed.event_index > 3), "behavior"] = "Look_Back"
        changed_X, changed_y, _, _ = make_prediction_examples(changed, window=3)
        np.testing.assert_array_equal(X[0], changed_X[0])
        self.assertNotEqual(y[0], changed_y[0])
        np.testing.assert_array_equal(X[groups == 1], changed_X[groups == 1])
        self.assertEqual(len(X), 2 * (8 - 3))
        self.assertTrue((rows.observed_until_index < rows.target_event_index).all())
        self.assertEqual(X.shape[1], 35)

    def test_prediction_requires_an_observed_history_and_future_target(self):
        df = generate_dataframe(n_students=2, n_events=3, seed=9)
        with self.assertRaises(ValueError):
            make_prediction_examples(df, window=3)

    def test_constant_recovery_correlation_is_undefined(self):
        result = recovery_metrics(np.full((5, 5), .2), np.full((5, 5), .2))
        self.assertEqual(result["mean_absolute_error"], 0)
        self.assertIsNone(result["correlation_secondary"])

    def test_unknown_codes_and_noncontiguous_indices_are_rejected(self):
        valid = pd.DataFrame({"student_id": [0, 0], "event_index": [0, 1],
                              "behavior": ["Gaze_A", "Gaze_B"]})
        for column, value in (("behavior", "Unknown"), ("event_index", 3)):
            with self.subTest(column=column):
                invalid = valid.copy()
                invalid.loc[1, column] = value
                with self.assertRaises(ValueError):
                    sequences_from_dataframe(invalid)


if __name__ == "__main__":
    unittest.main()
