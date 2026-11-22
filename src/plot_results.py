"""Generate reproducible scientific figures and a standalone event explorer."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd

from generate_data import BEHAVIORS, DATA_DIR, PROJECT_ROOT


FIGURES = PROJECT_ROOT / "figures"
LABELS = ["Gaze A", "Gaze B", "Error", "Look back", "Correct"]
COLORS = ["#2563eb", "#0891b2", "#c94b46", "#9b6acb", "#168568"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none", "savefig.facecolor": "white"})


def save_figure(fig, stem):
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def read_matrix(path):
    frame = pd.read_csv(path, index_col=0)
    return frame.loc[list(BEHAVIORS), list(BEHAVIORS)].to_numpy()


def heatmaps():
    truth = read_matrix(DATA_DIR / "true_transition_matrix.csv")
    estimate = read_matrix(DATA_DIR / "analysis_markov/transition_probabilities.csv")
    error = estimate - truth
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    for axis, matrix, title, diverging in zip(
            axes, [truth, estimate, error],
            ["Generating probabilities", "Estimated probabilities", "Estimation error"],
            [False, False, True]):
        limit = max(.05, float(np.nanmax(np.abs(error))))
        image = axis.imshow(matrix, cmap="RdBu_r" if diverging else "Blues",
                            vmin=-limit if diverging else 0,
                            vmax=limit if diverging else 1)
        axis.set(xticks=range(5), yticks=range(5), xticklabels=LABELS,
                 yticklabels=LABELS, title=title, xlabel="Next event")
        axis.tick_params(axis="x", rotation=35)
        for i, j in np.ndindex(matrix.shape):
            value = matrix[i, j]
            label = "NA" if not np.isfinite(value) else f"{value:+.2f}" if diverging else f"{value:.2f}"
            bright = abs(value) > .65 * limit if diverging else value > .55
            axis.text(j, i, label, ha="center", va="center", color="white" if bright else "#152238", fontsize=10)
        fig.colorbar(image, ax=axis, shrink=.76, label="Estimate minus generating value" if diverging else "Conditional probability")
    axes[0].set_ylabel("Current event")
    fig.suptitle("Can the pipeline recover known transition probabilities?", fontsize=16, fontweight="bold")
    save_figure(fig, "transition_heatmaps")


def transition_network():
    diagnostics = pd.read_csv(DATA_DIR / "analysis_markov/transition_diagnostics.csv")
    data = pd.read_csv(DATA_DIR / "synthetic_sequences.csv")
    frequency = data["behavior"].value_counts().reindex(BEHAVIORS, fill_value=0)
    sizes = 24000 * frequency.to_numpy() / len(data)
    angles = np.pi / 2 + np.arange(5) * 2 * np.pi / 5
    positions = np.c_[np.cos(angles), np.sin(angles)]
    selected = diagnostics[(diagnostics["p_by_exploratory"] <= .05)
                           & (diagnostics["difference_from_shuffle"] > 0)]
    fig, axis = plt.subplots(figsize=(8.7, 7.4))
    for _, edge in selected.iterrows():
        i, j = BEHAVIORS.index(edge["from_behavior"]), BEHAVIORS.index(edge["to_behavior"])
        start, end = positions[i], positions[j]
        if i == j:
            start, end = start + [-.17, .22], end + [.17, .22]
            radius, shrink_a, shrink_b = -2.4, 0, 0
            label_position = positions[i] + [0, .56]
        else:
            radius = .15
            shrink_a, shrink_b = np.sqrt(sizes[i]) / 2 + 4, np.sqrt(sizes[j]) / 2 + 5
            vector = end - start
            label_position = (start + end) / 2 + .085 * np.array([vector[1], -vector[0]])
        arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18,
                                connectionstyle=f"arc3,rad={radius}", shrinkA=shrink_a,
                                shrinkB=shrink_b, linewidth=1.0 + 5 * edge["probability"],
                                color="#466478", alpha=.82, zorder=1)
        axis.add_patch(arrow)
        axis.text(*label_position, f"{edge['probability']:.2f}", ha="center", va="center",
                  bbox={"boxstyle": "round,pad=.18", "fc": "white", "ec": "none"}, zorder=5)
    axis.scatter(positions[:, 0], positions[:, 1], s=sizes, c=COLORS,
                 edgecolors="white", linewidths=2, zorder=3)
    for i, label in enumerate(LABELS):
        axis.text(*positions[i], f"{label}\nn = {frequency.iloc[i]}", ha="center", va="center",
                  color="white", fontsize=10, fontweight="bold", zorder=4)
    axis.set(xlim=(-1.55, 1.55), ylim=(-1.47, 1.50), aspect="equal")
    axis.axis("off")
    axis.set_title("Exploratory transition network", fontsize=17, fontweight="bold", pad=14)
    fig.text(.5, .025,
             "Positive shuffle deviations with BY-adjusted exploratory p <= 0.05.\n"
             "Edge labels: estimated probabilities. Node area: event frequency.\n"
             "Synthetic events; selected edges are not causal or cognitive claims.",
             ha="center", va="bottom", fontsize=10, color="#465366")
    fig.subplots_adjust(bottom=.16)
    save_figure(fig, "transition_network")


def recovery_distribution():
    frame = pd.read_csv(DATA_DIR / "validation/replicate_recovery_errors.csv")
    summary = json.loads((DATA_DIR / "validation/recovery_summary.json").read_text())
    fig, axis = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    axis.hist(frame["mean_absolute_error"], bins=15, color="#3079ad", edgecolor="white")
    axis.axvline(summary["mean_absolute_error"], color="#c94b46", linewidth=2,
                  label=f"Included dataset: MAE = {summary['mean_absolute_error']:.4f}")
    axis.set(title=f"Probability recovery across {len(frame)} independent simulations",
             xlabel="Mean absolute error across estimable transition probabilities", ylabel="Simulation count")
    axis.legend(frameon=False)
    save_figure(fig, "recovery_distribution")


def calibration_figure():
    rates = pd.read_csv(DATA_DIR / "calibration/rejection_rates.csv")
    settings = json.loads((DATA_DIR / "calibration/calibration_settings.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.5, 1]})
    for axis, subset, labels, title in (
            (axes[0], rates.iloc[:2], ["IID", "Heterogeneous IID"], "Null-model rejection rates"),
            (axes[1], rates.iloc[2:], ["Specified Markov model"], "Power under this alternative")):
        values = subset["rejection_rate"].to_numpy()
        errors = np.vstack((values - subset["ci_95_lower"], subset["ci_95_upper"] - values))
        axis.errorbar(values, np.arange(len(subset)), xerr=errors, fmt="o", markersize=8,
                       capsize=5, color="#256b94")
        axis.set(yticks=np.arange(len(subset)), yticklabels=labels, title=title,
                 xlabel="Rejection probability", ylim=(-.65, len(subset) - .35))
        if len(subset) > 1:
            axis.axvline(settings["alpha"], color="#c94b46", linestyle="--", label="Nominal alpha = 0.05")
            axis.set_xlim(0, max(.12, subset["ci_95_upper"].max() * 1.2))
            axis.legend(frameon=False, loc="lower right", fontsize=9)
        else:
            axis.set_xlim(.9, 1.005)
        axis.grid(axis="x", alpha=.15)
    fig.suptitle(f"{settings['replicates_per_scenario']} simulations per scenario; pointwise 95% binomial intervals",
                 fontsize=13, fontweight="bold")
    save_figure(fig, "global_test_calibration")


def prediction_figure():
    metrics = pd.read_csv(DATA_DIR / "prediction/prediction_metrics.csv")
    labels = ["Training prevalence", "Window frequencies", "Last event", "Last event + sequence", "Generating reference"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    for axis, metric, title in ((axes[0], "roc_auc", "ROC AUC (higher is better)"),
                                (axes[1], "brier_score", "Brier score (lower is better)")):
        bars = axis.barh(np.arange(len(metrics)), metrics[metric], color=["#b2becb", "#6b9eb8", "#397f9f", "#195b80", "#a980bc"])
        axis.set(yticks=np.arange(len(metrics)), yticklabels=labels, title=title)
        axis.invert_yaxis()
        axis.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
        axis.set_xlim(0, 1 if metric == "roc_auc" else max(.16, metrics[metric].max() * 1.2))
        axis.grid(axis="x", alpha=.15)
    fig.suptitle("Next-event error prediction on held-out simulated learners", fontsize=14, fontweight="bold")
    save_figure(fig, "prediction_baselines")


def interactive_explorer():
    data = pd.read_csv(DATA_DIR / "synthetic_sequences.csv")
    payload = {str(student): group.sort_values("event_index")["behavior"].tolist()
               for student, group in data.groupby("student_id")}
    html = '''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Synthetic Event Explorer</title>
<style>body{font:16px system-ui;background:#f3f6fa;color:#18283b;margin:0;padding:32px}main{max-width:1080px;margin:auto;background:white;padding:28px;border-radius:16px}h1{margin-top:0}label{display:inline-block;margin:12px 24px 12px 0}table{border-collapse:collapse;margin:24px 0}th,td{padding:13px;text-align:center;border:2px solid white}th{background:#f1f4f8}.event{display:inline-block;color:white;padding:7px 10px;margin:3px;border-radius:5px;font-size:13px}.muted{color:#536478}input{vertical-align:middle}select{padding:7px}</style>
<main><h1>Synthetic Event Explorer</h1><p>Inspect how a learner's transition estimates change as more events are included. These are synthetic coded events, not raw eye-tracking samples.</p>
<label>Learner <select id="learner"></select></label><label>Observed prefix <input id="prefix" type="range" min="2"><strong id="length"></strong></label>
<p id="status" class="muted"></p><div id="timeline"></div><table id="matrix"></table><p class="muted">Rows: current event. Columns: next event. NA means no outgoing observations. Colors encode conditional probability from 0 to 1. Each learner has one session; changing the prefix does not join different learners.</p></main>
<script>
const sequences=__DATA__, codes=__CODES__, colors=__COLORS__;
const learner=document.getElementById('learner'), prefix=document.getElementById('prefix');
Object.keys(sequences).forEach(id=>{const option=document.createElement('option');option.value=id;option.textContent='Learner '+id;learner.appendChild(option)});
function draw(){const sequence=sequences[learner.value].slice(0,Number(prefix.value));document.getElementById('length').textContent=' '+sequence.length+' events';
document.getElementById('status').textContent=(sequence.length-1)+' within-learner transitions. Hover over an event to see its index.';
const timeline=document.getElementById('timeline');timeline.replaceChildren();sequence.forEach((event,i)=>{const tag=document.createElement('span');tag.className='event';tag.style.background=colors[codes.indexOf(event)];tag.textContent=event;tag.title='Event index '+i;timeline.appendChild(tag)});
const counts=Array.from({length:5},()=>Array(5).fill(0));for(let i=0;i<sequence.length-1;i++)counts[codes.indexOf(sequence[i])][codes.indexOf(sequence[i+1])]++;
const table=document.getElementById('matrix');table.replaceChildren();const head=document.createElement('tr');['Current / next',...codes].forEach(label=>{const th=document.createElement('th');th.textContent=label;head.appendChild(th)});table.appendChild(head);
counts.forEach((row,i)=>{const tr=document.createElement('tr'),label=document.createElement('th');label.textContent=codes[i];tr.appendChild(label);const total=row.reduce((a,b)=>a+b,0);row.forEach(count=>{const td=document.createElement('td');td.textContent=total?(count/total).toFixed(2):'NA';td.title=count+' observed transitions';td.style.background=total?'rgba(37,99,170,'+(0.04+0.75*count/total)+')':'#eeeeee';if(total&&count/total>.6)td.style.color='white';tr.appendChild(td)});table.appendChild(tr)});}
function selectLearner(){prefix.max=sequences[learner.value].length;prefix.value=prefix.max;draw()}
learner.addEventListener('change',selectLearner);prefix.addEventListener('input',draw);selectLearner();
</script></html>'''
    html = html.replace("__DATA__", json.dumps(payload)).replace("__CODES__", json.dumps(list(BEHAVIORS))).replace("__COLORS__", json.dumps(COLORS))
    (FIGURES / "event_explorer.html").write_text(html, encoding="utf-8")


def main():
    FIGURES.mkdir(exist_ok=True)
    heatmaps()
    transition_network()
    recovery_distribution()
    calibration_figure()
    prediction_figure()
    interactive_explorer()
    print(f"Saved five figure pairs (PNG/SVG) and the offline explorer to {FIGURES}")


if __name__ == "__main__":
    main()
