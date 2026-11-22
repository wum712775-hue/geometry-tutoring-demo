"use strict";

(() => {
  const {challenge, geodial} = JSON.parse(document.getElementById("project-data").textContent);
  const tutor = VisualTutor;
  const byId = id => document.getElementById(id);
  const node = (tag, text, className) => {
    const element = document.createElement(tag);
    element.textContent = text;
    if (className) element.className = className;
    return element;
  };

  const cases = new Map(challenge.cases.map(row => [row.id, row]));
  const live = {id: "live", problem: "abc", events: []};
  const teacherLog = [];
  let selected;
  let step;
  let reviewed;
  let workflowStatus = '';

  const currentProblem = () => challenge.problems[selected.problem];
  const currentPrefix = () => selected.events.slice(0, step);
  const label = tutor.actionLabels;

  function diagramPoints(problem) {
    const [left, right] = tutor.geometry(problem).base;
    const half = Math.tan(problem.apex_degrees * Math.PI / 360);
    const rotation = problem.rotation * Math.PI / 180;
    const positions = {
      [problem.apex]: [0, -0.5],
      [left]: [-half, 0.5],
      [right]: [half, 0.5],
    };
    const rotated = Object.fromEntries(Object.entries(positions).map(([id, [x, y]]) => [
      id, [x * Math.cos(rotation) - y * Math.sin(rotation), x * Math.sin(rotation) + y * Math.cos(rotation)],
    ]));
    const xs = Object.values(rotated).map(point => point[0]);
    const ys = Object.values(rotated).map(point => point[1]);
    const midX = (Math.min(...xs) + Math.max(...xs)) / 2;
    const midY = (Math.min(...ys) + Math.max(...ys)) / 2;
    const scale = Math.min(246 / (Math.max(...xs) - Math.min(...xs)),
      140 / (Math.max(...ys) - Math.min(...ys)));
    return Object.fromEntries(Object.entries(rotated).map(([id, [x, y]]) =>
      [id, [180 + (x - midX) * scale, 104 + (y - midY) * scale]]));
  }

  function drawDiagram(canvas, problem, targets, color, caption) {
    const ctx = canvas.getContext("2d");
    const graph = tutor.geometry(problem);
    const points = diagramPoints(problem);
    const vertices = problem.vertices;
    const center = [0, 1].map(axis => vertices.reduce((sum, id) => sum + points[id][axis], 0) / 3);
    const line = (from, to, width, ink) => {
      ctx.beginPath();
      ctx.moveTo(...from);
      ctx.lineTo(...to);
      ctx.lineWidth = width;
      ctx.strokeStyle = ink;
      ctx.stroke();
    };

    ctx.setTransform(2, 0, 0, 2, 0, 0);
    ctx.clearRect(0, 0, 360, 208);
    ctx.lineCap = "round";
    for (let index = 0; index < vertices.length; index++) {
      const a = vertices[index];
      const b = vertices[(index + 1) % vertices.length];
      const edge = "side:" + [a, b].sort().join("");
      const ink = targets.includes(edge) ? color : "#476359";
      line(points[a], points[b], targets.includes(edge) ? 4 : 2, ink);
      if (graph.facts.given_equal_sides.includes(edge)) {
        const [x1, y1] = points[a];
        const [x2, y2] = points[b];
        const x = (x1 + x2) / 2;
        const y = (y1 + y2) / 2;
        const length = Math.hypot(x2 - x1, y2 - y1);
        const dx = -5 * (y2 - y1) / length;
        const dy = 5 * (x2 - x1) / length;
        line([x - dx, y - dy], [x + dx, y + dy], 2, ink);
      }
    }

    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (const vertex of vertices) {
      const [x, y] = points[vertex];
      const neighbors = vertices.filter(id => id !== vertex).map(id => points[id]);
      const [first, second] = neighbors.map(([nx, ny]) => Math.atan2(ny - y, nx - x));
      const span = (second - first + 3 * Math.PI) % (2 * Math.PI) - Math.PI;
      const bisector = first + span / 2;
      const marked = targets.includes("angle:" + vertex);
      if (marked) {
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.arc(x, y, 23, first, first + span, span < 0);
        ctx.closePath();
        ctx.fillStyle = color + "32";
        ctx.fill();
      }
      ctx.beginPath();
      ctx.arc(x, y, 19, first, first + span, span < 0);
      ctx.strokeStyle = marked ? color : "#82988d";
      ctx.lineWidth = marked ? 3.5 : 1.4;
      ctx.stroke();
      const dx = x - center[0];
      const dy = y - center[1];
      const offset = 15 / Math.hypot(dx, dy);
      ctx.font = "600 16px system-ui";
      ctx.fillStyle = marked ? color : "#152f2b";
      ctx.fillText(vertex, x + dx * offset, y + dy * offset);
      if (vertex === problem.apex || vertex === problem.target) {
        ctx.font = "600 14px system-ui";
        const value = vertex === problem.apex ? problem.apex_degrees + "°" : "?";
        ctx.fillText(value, x + 32 * Math.cos(bisector), y + 32 * Math.sin(bisector));
      }
    }
    canvas.setAttribute("aria-label", tutor.prompt(problem) + (caption ? " " + caption : ""));
  }

  function reflectSelect(id) {
    const select = byId(id);
    const option = Array.from(select.options ?? select.children).find(item => item.value === select.value);
    byId(id + "-display").textContent = option?.textContent ?? "";
  }

  function selectCase(id) {
    selected = id === live.id ? live : cases.get(id);
    step = selected.events.length;
    reviewed = null;
    byId("draft").value = "angles";
    reflectSelect("draft");
    workflowStatus = "";
    byId("input-message").textContent = "";
    render();
  }

  function advance(delta) {
    step += delta;
    reviewed = null;
    workflowStatus = "";
    render();
  }

  function addLive(event) {
    live.events.push(event);
    step = live.events.length;
    reviewed = null;
    workflowStatus = "";
    byId("input-message").textContent = "";
    render();
  }

  function renderTimeline(prefix, recommendation) {
    const timeline = prefix.map((event, index) => {
      const cited = recommendation.evidence.includes(index + 1);
      const li = node("li", "", cited ? "used" : "");
      li.append(node("span", String(index + 1), "step-number"), node("span", tutor.describe(event)));
      if (cited) li.setAttribute("aria-label", "Evidence step " + (index + 1) + ". " + tutor.describe(event));
      return li;
    });
    byId("timeline").replaceChildren(...timeline);
  }

  function renderPolicy(key, decision) {
    const card = node("article", "", "policy" + (key === "trace" ? " active" : ""));
    const header = node("div", "", "policy-title");
    header.append(node("strong", tutor.policyLabels[key]), node("span", label[decision.action], "action"));
    card.append(header, node("p", decision.reason));
    if (decision.evidence.length) {
      const references = [...new Set(decision.evidence)].sort((a, b) => a - b);
      card.append(node("div", "Evidence steps " + references.join(", "), "small"));
    }
    return card;
  }

  function currentDraft() {
    return tutor.draft(currentProblem(), byId("draft").value);
  }

  function renderDraft(problem) {
    const proposal = currentDraft();
    const checked = tutor.audit(problem, proposal);
    const targets = proposal.diagram.entities;
    const entities = targets.map(tutor.entityLabel).join(" and ");
    drawDiagram(byId("draft-figure"), problem, targets, "#5e3ab8", "Draft targets " + entities + ".");
    byId("draft-caption").textContent = tutor.hint(problem, proposal);
    byId("audit").className = checked.allowed ? "audit" : "audit blocked";
    const status = checked.allowed
      ? [node("p", "Targets " + entities + ". The objects support the selected fact. The final answer and calculation are withheld.")]
      : [node("p", "Blocked from learner preview."), ...checked.issues.map(issue => node("p", issue))];
    byId("audit").replaceChildren(...status);
    byId("release").disabled = !checked.allowed;
    const mode = byId("workflow").value;
    const route = tutor.routeReview(problem, currentPrefix(), proposal, mode);
    byId("routing").replaceChildren(node("p", route.path === "review" ? "Teacher review requested" : "Eligible for automatic release"),
      node("p", route.reason), ...(workflowStatus ? [node("p", workflowStatus)] : []));
  }

  function renderOutcome(problem) {
    const slot = byId("learner-message");
    if (!reviewed) {
      slot.replaceChildren(node("p", "No tutor message sent at this step.", "small"));
      return;
    }
    const action = reviewed.teacherAction;
    const text = action === "wait" ? "No hint sent. The learner can continue working." :
      action === "question" ? "Which angles are opposite the two equal sides?" :
      tutor.hint(problem, reviewed.proposal);
    const message = node("p", text);
    if (action === "highlight") {
      const figure = node("div", "", "figure draft-figure");
      const canvas = document.createElement("canvas");
      canvas.width = 720;
      canvas.height = 416;
      canvas.setAttribute("role", "img");
      figure.append(canvas);
      slot.replaceChildren(figure, message);
      drawDiagram(canvas, problem, reviewed.proposal.diagram.entities, "#5e3ab8", text);
    } else {
      slot.replaceChildren(message);
    }
  }

  function render() {
    const problem = currentProblem();
    const events = currentPrefix();
    const decisions = Object.fromEntries(Object.keys(tutor.policyLabels).map(key =>
      [key, tutor.decide(problem, events, key)]));
    const latestAnswer = events.findLast(event => event.type === "answer");
    const latestPair = events.findLast(event => event.type === "pair");
    const selectedAngles = latestPair?.vertices.map(vertex => "angle:" + vertex) ?? [];
    const saved = latestAnswer ? latestAnswer.degrees + "°" : "—";
    byId("problem").textContent = tutor.prompt(problem);
    byId("saved").replaceChildren(node("span", "Saved answer "), node("strong", saved, "saved"));
    drawDiagram(byId("learner-figure"), problem, selectedAngles, "#9c4c2c",
      latestPair ? tutor.describe(latestPair) + "." : "No angles marked by the learner.");
    for (const button of byId("scenarios").children) {
      button.setAttribute("aria-pressed", String(button.value === selected.id));
    }
    byId("live").hidden = selected !== live;
    byId("replay-controls").hidden = selected === live;
    byId("step-count").textContent = "Step " + step + " of " + selected.events.length;
    byId("previous").disabled = step === 0;
    byId("next").disabled = step === selected.events.length;
    renderTimeline(events, decisions.trace);
    byId("policies").replaceChildren(...Object.entries(decisions).map(([key, decision]) =>
      renderPolicy(key, decision)));
    renderDraft(problem);
    renderOutcome(problem);
  }

  function record(action, decidedBy) {
    const problem = currentProblem();
    const prefix = currentPrefix();
    const proposal = action === "highlight" ? currentDraft() : {action};
    const validation = tutor.audit(problem, proposal);
    if (!validation.allowed) return;
    const recommendation = tutor.decide(problem, prefix, "trace");
    reviewed = {
      case: selected.id, problem: selected.problem, step, decidedBy, workflow: byId("workflow").value,
      sequence: teacherLog.length + 1, prefix, recommendation, teacherAction: action,
      actionOverride: action !== recommendation.action, proposal, validation,
    };
    teacherLog.push(reviewed);
    renderOutcome(problem);
  }

  function runWorkflow() {
    const problem = currentProblem();
    const mode = byId("workflow").value;
    const proposal = currentDraft();
    const route = tutor.routeReview(problem, currentPrefix(), proposal, mode);
    if (route.path === "auto") {
      record("highlight", "workflow");
      workflowStatus = "The learner preview shows the released hint.";
    } else {
      reviewed = null;
      renderOutcome(problem);
      teacherLog.push({case: selected.id, problem: selected.problem, prefix: currentPrefix(), step, decidedBy: "workflow",
        workflow: mode, sequence: teacherLog.length + 1, teacherAction: null, proposal, route});
      workflowStatus = "Draft queued for the teacher. Wait, revise it, or send a question.";
    }
    renderDraft(problem);
  }

  function exportLog() {
    const problem = currentProblem();
    const prefix = currentPrefix();
    const draft = currentDraft();
    const report = {
      schema: "geometry-tutor-session-v1", origin: selected === live ? "interactive" : "authored replay",
      case: selected.id, problem, currentPrefix: prefix,
      policyDecisions: Object.fromEntries(Object.keys(tutor.policyLabels).map(key =>
        [key, tutor.decide(problem, prefix, key)])),
      inspectedDraft: {draft, route: tutor.routeReview(problem, prefix, draft, byId("workflow").value)},
      reviewEvents: teacherLog,
    };
    const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], {type: "application/json"}));
    const link = document.createElement("a");
    link.href = url;
    link.download = "visual-tutor-session.json";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function tableRow(title, values) {
    const row = document.createElement("tr");
    const header = node("th", title);
    header.scope = "row";
    row.append(header, ...values.map(value => node("td", String(value))));
    return row;
  }

  function showEvaluation() {
    const test = geodial.counts.test;
    const scopes = "Held out by problem group: " + test.decision_points.toLocaleString("en-US") +
      " eligible teacher turns in " + test.dialogues + " test dialogues across " +
      test.problem_groups + " problem groups. Training uses the official training split.";
    byId("geodial-scope").textContent = scopes;
    const timingLabels = {prevalence: "Training prevalence", latest: "Latest learner message",
      prefix3: "Up to three learner messages"};
    byId("geodial-metrics").replaceChildren(...Object.entries(timingLabels).map(([model, title]) => {
      const scores = geodial.test[model];
      return tableRow(title, [scores.auroc.toFixed(3), scores.average_precision.toFixed(3), scores.brier.toFixed(3)]);
    }));
    const delta = geodial.test_brier_delta_prefix_minus_latest;
    byId("geodial-effect").textContent = "Paired Brier difference (three messages minus one, lower is better): " +
      delta.estimate.toFixed(4) + ". 95% problem-group bootstrap interval [" +
      delta.interval_95.map(value => value.toFixed(4)).join(", ") + "].";
    const review = tutor.evaluateReview(challenge);
    byId("review-metrics").replaceChildren(...Object.entries(review).map(([mode, result]) =>
      tableRow(tutor.reviewModes[mode], [result.reviews, result.automatic, result.automaticOutsideRubric])));
    const evaluation = tutor.evaluate(challenge);
    byId("challenge-metrics").replaceChildren(...Object.entries(evaluation.totals).map(([key, result]) =>
      tableRow(tutor.policyLabels[key], [result.agrees + "/" + evaluation.detail.length, result.unwanted, result.missed])));
  }

  const scenarioLabels = ["Learner A", "Learner B", "One wrong attempt"];
  [...challenge.featured, live.id].forEach((id, index) => {
    const button = node("button", scenarioLabels[index] ?? "Try it yourself", "segment");
    button.value = id;
    if (id !== live.id) button.setAttribute("aria-label", scenarioLabels[index] + " — " + cases.get(id).title);
    button.addEventListener("click", () => selectCase(id));
    byId("scenarios").append(button);
  });
  for (const [value, title] of Object.entries(tutor.draftLabels)) {
    const option = node("option", title);
    option.value = value;
    byId("draft").append(option);
  }
  const vertices = challenge.problems[live.problem].vertices;
  for (const [a, b] of [[0, 1], [0, 2], [1, 2]]) {
    const option = node("option", "∠" + vertices[a] + " and ∠" + vertices[b]);
    option.value = vertices[a] + vertices[b];
    byId("live-pair").append(option);
  }
  for (const row of challenge.cases) {
    const option = node("option", row.title);
    option.value = row.id;
    byId("challenge-case").append(option);
  }
  byId("challenge-case").addEventListener("change", event => {
    reflectSelect("challenge-case");
    const row = cases.get(event.target.value);
    byId("challenge-reference").textContent = row
      ? "Authored reference. " + row.acceptable.map(action => label[action]).join(" or ") + ". " + row.rationale
      : "";
    if (row) selectCase(row.id);
  });
  byId("previous").addEventListener("click", () => advance(-1));
  byId("next").addEventListener("click", () => advance(1));
  for (const id of ["draft", "workflow"]) {
    byId(id).addEventListener("change", () => {
      reflectSelect(id);
      workflowStatus = "";
      renderDraft(currentProblem());
    });
  }
  byId("apply-workflow").addEventListener("click", runWorkflow);
  byId("release").addEventListener("click", () => record("highlight", "teacher"));
  byId("send-question").addEventListener("click", () => record("question", "teacher"));
  byId("withhold").addEventListener("click", () => record("wait", "teacher"));
  byId("export").addEventListener("click", exportLog);
  byId("save-answer").addEventListener("click", () => {
    const degrees = byId("live-answer").valueAsNumber;
    if (!Number.isFinite(degrees) || degrees <= 0 || degrees >= 180) {
      byId("input-message").textContent = "Enter an angle greater than 0° and less than 180°.";
      return;
    }
    addLive({type: "answer", degrees});
    byId("live-answer").value = "";
  });
  byId("live-pair").addEventListener("change", () => reflectSelect("live-pair"));
  byId("save-pair").addEventListener("click", () =>
    addLive({type: "pair", vertices: byId("live-pair").value.split("")}));
  byId("ask-for-help").addEventListener("click", () => addLive({type: "help", kind: "question"}));
  byId("ask-for-picture").addEventListener("click", () => addLive({type: "help", kind: "highlight"}));
  byId("clear-trace").addEventListener("click", () => {
    live.events = [];
    selectCase(live.id);
  });

  for (const id of ["workflow", "live-pair", "challenge-case"]) reflectSelect(id);
  showEvaluation();
  selectCase(challenge.featured[0]);
})();
