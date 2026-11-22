"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const tutor = require("../src/visual_tutor_core");
const challenge = require("../data/visual_tutor/challenge.json");

const byCase = id => challenge.cases.find(row => row.id === id);
const problem = challenge.problems.abc;

test("same saved answer and event counts can call for different supports", () => {
  const first = byCase("persistent");
  const second = byCase("repair");
  const bag = events => events.map(event => JSON.stringify(event)).sort();
  assert.deepEqual(bag(first.events), bag(second.events));
  assert.equal(tutor.decide(problem, first.events, "answer").action, "highlight");
  assert.equal(tutor.decide(problem, second.events, "answer").action, "highlight");
  assert.equal(tutor.decide(problem, first.events, "trace").action, "highlight");
  assert.equal(tutor.decide(problem, second.events, "trace").action, "wait");
});

test("order still changes the result when both final responses match", () => {
  const current = challenge.problems.lmn;
  const first = byCase("after-repair");
  const second = byCase("before-repair");
  assert.deepEqual(tutor.decide(current, first.events, "snapshot").action,
    tutor.decide(current, second.events, "snapshot").action);
  assert.equal(tutor.decide(current, first.events, "trace").action, "question");
  assert.equal(tutor.decide(current, second.events, "trace").action, "wait");
});

test("the verifier distinguishes entity existence, fact support and disclosure", () => {
  const valid = tutor.audit(problem, tutor.draft(problem, "angles"));
  assert.deepEqual([valid.entities, valid.relation, valid.disclosure, valid.allowed], [true, true, true, true]);
  const wrong = tutor.audit(problem, tutor.draft(problem, "wrong"));
  assert.deepEqual([wrong.entities, wrong.relation, wrong.disclosure, wrong.allowed], [true, false, true, false]);
  const unknown = tutor.audit(problem, tutor.draft(problem, "missing"));
  assert.equal(unknown.entities, false);
  assert.equal(unknown.allowed, false);
  for (const variant of ["operation", "answer"]) {
    const disclosure = tutor.audit(problem, tutor.draft(problem, variant));
    assert.deepEqual([disclosure.relation, disclosure.disclosure, disclosure.allowed], [true, false, false]);
  }
  const extra = {...tutor.draft(problem, "angles"), uncheckedMessage: "70°"};
  assert.equal(tutor.audit(problem, extra).shape, false);
});

test("angle identities remain tied to the problem after rotation and relabeling", () => {
  const rotated = challenge.problems.pqr;
  assert.deepEqual(tutor.draft(rotated, "angles").diagram.entities, ["angle:P", "angle:R"]);
  assert.equal(tutor.audit(rotated, tutor.draft(rotated, "sides")).allowed, true);
  assert.equal(tutor.audit(rotated, tutor.draft(problem, "angles")).entities, false);
});

test("only flagged routing automates a checked and timely draft", () => {
  const good = tutor.draft(problem, "angles");
  const wrong = tutor.draft(problem, "wrong");
  assert.equal(tutor.routeReview(problem, byCase("persistent").events, good, "all").path, "review");
  assert.equal(tutor.routeReview(problem, byCase("persistent").events, good, "flagged").path, "auto");
  assert.equal(tutor.routeReview(problem, byCase("repair").events, good, "flagged").path, "review");
  assert.equal(tutor.routeReview(problem, byCase("persistent").events, wrong, "flagged").path, "review");
  const verbal = byCase("verbal-repair");
  const current = challenge.problems[verbal.problem];
  const verified = tutor.routeReview(current, verbal.events, tutor.draft(current, "angles"), "flagged");
  assert.match(verified.reason, /interpretation/);
  assert.deepEqual(tutor.evaluateReview(challenge).flagged,
    {reviews: 12, automatic: 2, automaticOutsideRubric: 0});
});

test("offline app can run both workflows, withhold and export", async () => {
  const canvas = new Proxy({}, {get(target, key) { return target[key] ?? (() => {}); }});
  class Element {
    constructor() {
      this.children = [];
      this.attrs = {};
      this.handlers = {};
      this.textContent = "";
      this.value = "";
    }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children = children; }
    setAttribute(name, value) { this.attrs[name] = value; }
    addEventListener(name, callback) { this.handlers[name] = callback; }
    getContext() { return canvas; }
    get valueAsNumber() { return this.value === "" ? NaN : Number(this.value); }
    click() { if (!this.disabled) this.handlers.click?.({target: this}); }
    change(value) { this.value = value; this.handlers.change?.({target: this}); }
  }
  const elements = new Map();
  const get = id => {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id);
  };
  const read = value => value.textContent + value.children.map(read).join(" ");
  const page = fs.readFileSync(path.join(__dirname, "../START_HERE.html"), "utf-8");
  assert.match(page, /When should the teacher step in/);
  const scripts = [...page.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)].map(match => match[1]);
  get("project-data").textContent = scripts[0];
  get("workflow").value = "flagged";
  let download;
  const context = vm.createContext({
    document: {getElementById: get, createElement: () => new Element()},
    Blob,
    URL: {createObjectURL: data => (download = data, "blob:test"), revokeObjectURL() {}},
    setTimeout: callback => callback(),
  });
  for (const script of scripts.slice(1)) vm.runInContext(script, context);

  assert.match(read(get("routing")), /Eligible for automatic release/);
  get("apply-workflow").click();
  assert.match(read(get("learner-message")), /marked angles/);
  get("workflow").change("all");
  get("apply-workflow").click();
  assert.match(read(get("routing")), /queued for the teacher/);
  assert.match(read(get("learner-message")), /No tutor message/);
  get("workflow").change("flagged");
  get("scenarios").children[1].click();
  assert.match(read(get("saved")), /140°/);
  assert.match(read(get("routing")), /supports waiting/);
  get("withhold").click();
  assert.match(read(get("learner-message")), /can continue working/);
  get("previous").click();
  assert.equal(read(get("step-count")), "Step 3 of 4");
  get("scenarios").children[0].click();
  get("draft").change("wrong");
  assert.match(read(get("audit")), /do not support/);
  assert.equal(get("release").disabled, true);
  get("draft").change("sides");
  get("release").click();
  assert.match(read(get("learner-message")), /marked sides/);
  get("scenarios").children[3].click();
  get("live-answer").value = "140";
  get("save-answer").click();
  get("live-pair").value = "AB";
  get("save-pair").click();
  assert.match(read(get("routing")), /Eligible for automatic release/);
  get("export").click();

  const report = JSON.parse(await download.text());
  assert.equal(report.origin, "interactive");
  assert.equal(report.currentPrefix.length, 2);
  assert.ok(report.reviewEvents.some(event => event.teacherAction === "wait"));
  assert.ok(report.reviewEvents.some(event => event.route?.path === "review"));
  assert.equal(report.reviewEvents.filter(event => event.decidedBy === "workflow" &&
    event.teacherAction === "highlight").length, 1);
});
