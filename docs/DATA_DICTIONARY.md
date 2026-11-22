# Data dictionary

All included event records and derived numbers are synthetic. No participant demographics or private logs are present.

## Event tables

The files data/synthetic_sequences.csv and data/iid_sequences.csv each include:

| Column | Meaning |
| --- | --- |
| student_id | Integer identifier for a simulated learner with one session |
| event_index | Zero-based event position within the learner, without gaps |
| behavior | One of five mutually exclusive hypothetical codes |
| is_synthetic | Always True |
| simulation_model | markov or iid |

Gaze_A and Gaze_B mean hypothetical events at regions A and B; Error_Op means hypothetical incorrect operation; Look_Back means a hypothetical return look; Correct_Op means hypothetical correct operation. The simulator directly assigns these labels. They are **not** gaze fixations extracted from samples; neither raw gaze positions nor timestamps, hand signals, or verified actions exist in these tables.

## Other outputs

| Path | Description |
| --- | --- |
| data/true_transition_matrix.csv | Specified Markov conditional probabilities, rows current events and columns next events |
| data/iid_transition_matrix.csv | Stationary category probabilities repeated in every row |
| data/simulation_metadata.json | RNG streams, seed, distribution, code definitions, assumptions, versions, generator source hash |
| data/analysis_markov/ and data/analysis_iid/ | Counts, pooled conditional probabilities, shuffle expectations, global summaries, exploratory cell diagnostics |
| data/validation/ | Recovery metrics, repeated recovery errors, maxima/tie checks |
| data/calibration/ | Global-test rejection rates and replicate p-values |
| data/prediction/ | Out-of-fold predictions, learner-fold assignments, model metrics |
| figures/ | Five PNG/SVG pairs and a local HTML event explorer |

Rows without outgoing transitions have undefined estimated probabilities. The network's BY-selected cell results are exploratory; learner-bootstrap probability intervals are pointwise; calibration intervals describe simulation rejection-rate uncertainty.
