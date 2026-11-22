# Run and explain the pipeline

Open a terminal in the extracted folder containing README.md and run_pipeline.py. Python 3.11 or later is required; the included full results were generated with Python 3.12.14 and the pinned dependencies.

## Replay the tutoring scenarios

Open [the offline viewer](../SCRIPTED_REPLAY.html) in a browser. To rebuild it and the decision tables, run `python src/tutoring_replay.py` from the project root. This step uses only the Python standard library. The synthetic methods workflow below uses the pinned scientific dependencies.

## 1. Prepare an environment

~~~bash
python --version
python -m venv .venv
~~~

Activate the environment: on macOS/Linux use source .venv/bin/activate; on Windows Command Prompt use .venv\Scripts\activate.bat; on Windows PowerShell use .venv\Scripts\Activate.ps1. If PowerShell refuses local activation, use Command Prompt. On systems where python is called python3, substitute python3 in the commands.

~~~bash
python -m pip install -r requirements.txt
~~~

**Why:** pinned package versions and a local environment make the supplied numerical run easier to reproduce.

## 2. Generate events

~~~bash
python src/generate_data.py
~~~

**What:** writes two 1,500-row synthetic CSVs, the known Markov matrix, an IID reference matrix, and metadata to data/. Each learner has positions 0–29.

**Why:** known generating probabilities make it possible to measure recovery. The IID dataset tests a distinct no-order scenario with the same expected category frequencies.

## 3. Count transitions and test event order

~~~bash
python src/analysis.py
python src/analysis.py --dataset iid
~~~

**What:** each dataset contributes exactly 1,450 within-learner adjacent pairs. Outputs in data/analysis_markov/ and data/analysis_iid/ include probability matrices, shuffle-based global p-values, exploratory cell diagnostics, and whole-learner bootstrap intervals. Default budgets are 4,999 permutations and 1,999 bootstrap draws per dataset.

**Why:** the global test asks whether event ordering contains information conditional on each learner's category frequencies. One IID dataset can still have a small p-value by chance.

## 4. Check recovery and repeated test behavior

~~~bash
python src/validate.py
python src/calibrate.py
~~~

**What:** compares truth with the included estimate and 100 independent recoveries. Then tests 200 new datasets in each of three scenarios, with 499 permutations per global test. Inspect data/validation/recovery_summary.json and data/calibration/rejection_rates.csv. The calibration step can take several minutes.

**Why:** matrix recovery and repeated null rejection are distinct checks. If validation reports stale data, regenerate and rerun the analysis before validation.

## 5. Fit predictive baselines and rebuild figures

~~~bash
python src/predict_next_event.py
python src/plot_results.py
~~~

**What:** GroupKFold holds out complete simulated learners for next-event prediction. Plotting creates five PNG/SVG pairs and an offline event explorer. Extract figures/event_explorer.html and open it locally in a browser to inspect one learner and its selected event prefix.

**Why:** plotting shows the generating assumptions and estimates side by side; prediction compares time-available features against simple baselines.

## 6. Run regression checks

~~~bash
python -m unittest discover -s tests -v
~~~

The 17 checks cover boundaries, exhaustive null expectations, deterministic generation, missing probability rows, stationary initialization, whole-learner bootstrap, future-event exclusion, and edge cases. The final line should report OK.

Run all six stages and refresh the tutoring replay with `python run_pipeline.py`. The `--quick` option uses reduced budgets and overwrites synthetic study outputs. Rerun the full command before quoting the included [synthetic results](SYNTHETIC_METHODS_STUDY.md).

## Explain the work during an interview

“I specified a known Markov model, generated privacy-safe coded sequences, and tested a within-learner global event-order null. I checked whether the estimated probabilities recovered their generating values over repeated simulations, then compared next-event baselines on held-out learners. The figures show the assumptions and error. It has not processed recorded eye tracking or validated cognitive indicators.”

Prepare to answer: Why are there 1,450 transitions? Why shuffle within learners? What does p = 0.0002 mean at 4,999 permutations? Why are network edges exploratory? Why can older events add no information beyond the current event under this first-order model? What exact new data and labels would make a VR learning study possible? Describe accurately which implementation and verification steps you personally performed, including any assistance you used.
