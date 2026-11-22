# When should the teacher step in?

**Meiying Wu · Geometry tutoring prototype · November 2026**

Open [**START_HERE.html**](START_HERE.html) in a browser after extracting the folder. This is a self-contained offline geometry tutoring prototype. A teacher compares two workflows: review every suggested visual hint, or review drafts flagged by the learner’s process or by a diagram check.

## A one-minute walkthrough

1. **Learner A** writes 140° and ends with the wrong angle pair. Under **Review flagged drafts only**, the checked base-angle highlight is eligible for display. **Run selected workflow** shows what the learner would see.
2. **Learner B** writes the same 140° and ends by marking the correct pair. The same mathematically valid draft is sent for teacher review. Use **Withhold hint** to model letting the learner finish the correction.
3. Return to Learner A. Select **Audit a highlight on the wrong angle**. The draft marks a real but inappropriate angle. The release button becomes unavailable. Change the draft to **Start with the equal sides** and inspect the revised hint.

Use **Try it yourself** to write a value and mark a pair, and **Export decision log** to inspect the observed prefix, proposal, checks and teacher decisions. The application uses a single local teacher/learner preview. All live decisions are deterministic.

## Evidence in the demo

| Evaluation | Included evidence |
| --- | --- |
| Review workflow | Both workflows use identical checks. Review-all requests 14 reviews. Flagged review requests 12 and routes two drafts to automatic preview on 14 authored cases. |
| Turn timing on public GeoDial data | On 400 held-out dialogues (2,935 eligible teacher turns), AUROC is 0.627 for one prior learner message and 0.648 for up to three. The target is the recorded teacher highlight flag. |
| Geometry checks | A structured highlight has two named objects, a geometric fact and an explicit disclosure level. The check detects nonexistent objects, objects inconsistent with the selected fact, and drafts that supply the calculation or answer. |

The challenge cases and reference decisions were authored with the prototype. The GeoDial comparison uses human teacher contributions with simulated learners in the published collection setting, and the application’s deterministic rules are separate from that classifier. Neither evaluation measures teacher workload or learner transfer. The experiment to test those is described in [the study protocol](docs/TEACHER_REVIEW_STUDY.md).

## Research and reproduction

- Rebuild the self-contained page with `python src/build_visual_tutor.py`.
- Run `node --test tests/test_visual_tutor.js` for geometry, review routing, and offline interface checks. Run `python -m unittest discover -s tests` for the sequence pipeline checks.
- See [the study protocol](docs/TEACHER_REVIEW_STUDY.md) to reproduce the public-data pilot from the release files and compare SHA-256 hashes.

Earlier work remains available as [live programming evidence](LIVE_LAB.html), [scripted tutoring replay](SCRIPTED_REPLAY.html) and the [synthetic methods study](docs/SYNTHETIC_METHODS_STUDY.md). The latter includes event generation, learner-safe sequence analysis, permutation calibration, and learner-held-out prediction.
