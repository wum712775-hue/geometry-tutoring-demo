# Teacher control during live visual tutoring

## Research question

When should a geometry tutoring system request teacher review before showing a draft visual hint? Compare reviewing all visual drafts with requesting review when a mathematical check fails or the learner’s recent evidence is ambiguous or favours waiting.

The [2026 teacher-control study](https://arxiv.org/abs/2605.10672) examined the stage at which teachers could steer or edit mathematical visual materials. The proposed follow-on studies routing during a learner’s live problem-solving process. [SketchMind](https://doi.org/10.1145/3830398.3830500) motivates capturing actions on visual objects as inspectable traces. [GeoDial](https://arxiv.org/abs/2606.12419) motivates verifying the objects selected for a visual hint.

## What the prototype implements

- A rendered geometric triangle with semantic angle and segment IDs. A draft specifies a fact, exactly two objects, and a disclosure level. A verifier checks the objects against the two relations supported here and detects an explicit calculation or numeric answer in this restricted format. The teacher can change which relation is shown, ask a question, hold the draft or release a checked draft.
- A matched pair with the same wrong numeric answer, ending event and event counts. The order of the two angle selections changes which observation is most recent. A second pair also matches the current answer and current selected angles; it swaps whether the wrong value came before or after the correct selection.
- An offline comparison with the same proposal and verifier in both conditions. Selective routing escalates incorrect drawings, unsupported disclosures, unparsed learner statements near the decision, and situations in which the ordered rule favours waiting or asking. The authored suite has 14 draft opportunities, including three intentionally bad drafts in different cases. Review-all requests 14 reviews. Selective routing requests 12 reviews and releases 2 automatically under this setup. Timing labels express author judgment. In the app, manual release can override a timing recommendation for a valid diagram, and an override is recorded in the export.
- The authored suite intentionally includes learner statements the ordered rule does not interpret. It can suggest a hint even though the learner just asked to work independently. The routing rule escalates those cases to a human. The narrow geometry verifier assumes correct authored givens and has no vision or free-text model.

## Public-data timing pilot

The public [GeoDial dataset](https://huggingface.co/datasets/SankalanPC/Geo2Dial) contains experienced teachers’ contributions elicited in conversations with simulated students, as described in [the preprint](https://arxiv.org/abs/2606.12419). The outcome here is the released `draw_on_image` flag, which records whether the teacher drew on a turn. The flag does not indicate whether drawing was the best intervention or whether the target was correct.

- Unit of prediction is a teacher turn with at least one previously observed learner message. An opening teacher turn with no preceding learner text is excluded.
- Training, validation and test use the official file splits. The implementation verifies that both question and problem group IDs are disjoint across the three splits.
- The comparison uses TF-IDF unigram/bigram logistic regression with the latest learner message or up to three previous learner messages. Both versions use the same model settings, training split and test rows. No teacher text, final answers, tutor labels, images, or future learner messages enter the features. A constant model uses the training prevalence. Concatenation preserves the chronology in the input text, but this is not an order-sensitive sequence model beyond local text n-grams. The comparison measures the information in a larger context window.
- The official test file has 400 dialogues grouped into 299 problem groups. Of 2,935 eligible teacher turns, 1,416 record a highlight.
- Test AUROC is 0.627 for one message and 0.648 for up to three. Average precision is 0.592 versus 0.623, and Brier score is 0.239 versus 0.233. The paired Brier change is −0.0055 (95% fixed-model interval from 2,000 resamples of test problem groups, −0.0101 to −0.0005). This interval concerns variation in test groups. It does not account for retraining variability.

The results are saved in [geodial_results.json](../data/visual_tutor/geodial_results.json) with the full-precision metrics, split counts and SHA-256 values. The project includes no third-party raw data or teacher identifiers.

## Reproduce the pilot

1. Download the public [release at ba900ab](https://huggingface.co/datasets/SankalanPC/Geo2Dial/tree/ba900ab) and retain `train_set.json`, `val_set.json`, and `test_set.json`.
2. From the project root, install the pinned dependencies, run the evaluation and rebuild the offline site.

~~~bash
python -m pip install -r requirements.txt
python src/geodial_timing.py --train train_set.json --validation val_set.json --test test_set.json
python src/build_visual_tutor.py
~~~

## Proposed teacher study

Run a counterbalanced comparison with teachers reviewing matched sets of fresh tutoring situations. Freeze the proposal generator, checks and proposed hint content across workflows. Have independent reviewers rate timing, entities, and disclosure before seeing the workflow decisions. Record time spent reviewing per opportunity, inappropriate hints released, useful hints withheld, corrections, and override reasons. Evaluate group-level uncertainty and examine both missed risks and excess escalations. In a subsequent learner study, assess whether participants can solve a differently oriented triangle without hints. The present software check and public-data pilot are preliminary work for that study.
