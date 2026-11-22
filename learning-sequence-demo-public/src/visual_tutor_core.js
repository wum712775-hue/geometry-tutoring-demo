"use strict";

const VisualTutor = (() => {
  const policyLabels = {
    answer: "Answer only",
    snapshot: "Current snapshot",
    trace: "Ordered evidence",
  };
  const actionLabels = {
    wait: "Wait",
    question: "Ask a question",
    highlight: "Highlight the base angles",
  };
  const draftLabels = {
    angles: "Highlight base angles",
    sides: "Start with the equal sides",
    relation: "Highlight and state the equality",
    wrong: "Audit a highlight on the wrong angle",
    missing: "Audit a highlight on an unknown point",
    operation: "Audit a draft that gives the calculation",
    answer: "Audit a draft that gives the answer",
  };

  const side = (a, b) => "side:" + [a, b].sort().join("");
  const angle = vertex => "angle:" + vertex;
  const sameSet = (left, right) => left.length === right.length &&
    new Set(left).size === left.length && left.every(item => right.includes(item));

  function geometry(problem) {
    const base = problem.vertices.filter(vertex => vertex !== problem.apex);
    const facts = {
      equal_base_angles: base.map(angle),
      given_equal_sides: base.map(vertex => side(problem.apex, vertex)),
    };
    const entities = [
      ...problem.vertices.map(angle),
      ...facts.given_equal_sides,
      side(...base),
    ];
    return {base, facts, entities, answer: (180 - problem.apex_degrees) / 2};
  }

  function entityLabel(id) {
    const [kind, label] = id.split(":");
    return kind === "angle" ? "∠" + label : label;
  }

  function prompt(problem) {
    const [left, right] = geometry(problem).facts.given_equal_sides.map(entityLabel);
    return left + " = " + right + " and ∠" + problem.apex + " = " +
      problem.apex_degrees + "°. Find ∠" + problem.target + ".";
  }

  function observe(problem, events) {
    const graph = geometry(problem);
    return events.map((event, index) => {
      const row = {...event, index};
      if (row.type === "answer") row.correct = row.degrees === graph.answer;
      if (row.type === "pair") row.correct = sameSet(row.vertices, graph.base);
      return row;
    });
  }

  function describe(event) {
    switch (event.type) {
      case "answer": return "Writes " + event.degrees + "°";
      case "pair": return "Marks ∠" + event.vertices[0] + " and ∠" + event.vertices[1] + " as equal";
      case "review": return "Reviews the draft";
      case "help": return event.kind === "highlight" ? "Asks for a visual hint" : "Asks for help";
      case "statement": return "“" + event.text + "”";
      default: throw new Error("Unknown event type: " + event.type);
    }
  }

  function decision(action, reason, evidence = []) {
    return {action, reason, evidence: evidence.filter(Boolean).map(event => event.index + 1)};
  }

  function decide(problem, events, policy) {
    const seen = observe(problem, events);
    const answer = seen.findLast(event => event.type === "answer");
    const pair = seen.findLast(event => event.type === "pair");
    const latestAction = seen.findLast(event => ["help", "answer", "pair"].includes(event.type));
    const wrong = answer && !answer.correct;
    const wrongPair = pair && !pair.correct;

    if (policy === "answer") {
      return wrong
        ? decision("highlight", "The saved value is incorrect.", [answer])
        : decision("wait", "The saved value does not trigger a hint.", [answer]);
    }
    if (policy === "snapshot") {
      if (latestAction?.type === "help") {
        return decision("question", "Respond to the help request.", [latestAction]);
      }
      if (wrong && wrongPair) {
        return decision("highlight", "The saved value and the angle pair are both incorrect.", [answer, pair]);
      }
      if (wrong && !pair) {
        return decision("question", "Clarify how the learner obtained this value.", [answer]);
      }
      return decision("wait", "The current answers give no clear trigger for a highlight.", [answer, pair]);
    }
    if (policy !== "trace") throw new Error("Unknown policy: " + policy);
    if (latestAction?.type === "help") {
      return latestAction.kind === "highlight" || (wrong && wrongPair)
        ? decision("highlight", "The learner requested support and a diagram can address it.", [latestAction, pair])
        : decision("question", "Ask for the learner’s reasoning in response to the request.", [latestAction]);
    }
    if (wrong && wrongPair) {
      return decision("highlight", "The latest marked pair is wrong and the saved value remains incorrect.", [answer, pair]);
    }
    if (wrongPair && answer?.correct && pair.index > answer.index) {
      return decision("question", "A wrong pair followed a correct value. Check the connection.", [answer, pair]);
    }
    if (pair?.correct && wrong) {
      return pair.index > answer.index
        ? decision("wait", "The learner corrected the angle pair after writing the wrong value.", [answer, pair])
        : decision("question", "The learner wrote a wrong value after selecting the right pair.", [pair, answer]);
    }
    if (answer?.correct) {
      return decision("wait", "The latest value matches the problem.", [answer]);
    }
    if (wrongPair) {
      return decision("question", "Ask the learner how they identified the equal angles.", [pair]);
    }
    const wrongValues = seen.filter(event => event.type === "answer" && !event.correct);
    if (wrongValues.length >= 2) {
      return decision("question", "Repeated attempts warrant a question. The trace gives no angle pair.", wrongValues.slice(-2));
    }
    return decision("wait", "Allow time for another attempt while the evidence is limited.", [answer, pair]);
  }

  function draft(problem, variant) {
    if (!Object.hasOwn(draftLabels, variant)) throw new Error("Unknown draft: " + variant);
    const graph = geometry(problem);
    const fact = variant === "sides" ? "given_equal_sides" : "equal_base_angles";
    let entities = [...graph.facts[fact]];
    if (variant === "wrong") entities = [angle(problem.apex), angle(graph.base[0])];
    if (variant === "missing") entities = [angle("Z"), angle(graph.base[0])];
    const disclosure = ["relation", "operation", "answer"].includes(variant) ? variant : "question";
    return {action: "highlight", diagram: {fact, entities, disclosure}};
  }

  function exactKeys(value, keys) {
    return value && typeof value === "object" && !Array.isArray(value) &&
      Object.keys(value).length === keys.length && keys.every(key => Object.hasOwn(value, key));
  }

  function audit(problem, proposal) {
    const action = proposal?.action;
    if (["wait", "question"].includes(action) && exactKeys(proposal, ["action"])) {
      return {shape: true, entities: null, relation: null, disclosure: null, allowed: true, issues: []};
    }
    const spec = proposal?.diagram;
    const graph = geometry(problem);
    const shape = action === "highlight" && exactKeys(proposal, ["action", "diagram"]) &&
      exactKeys(spec, ["fact", "entities", "disclosure"]) && Object.hasOwn(graph.facts, spec.fact) &&
      Array.isArray(spec.entities) && spec.entities.length === 2 &&
      spec.entities.every(id => typeof id === "string") &&
      ["question", "relation", "operation", "answer"].includes(spec.disclosure);
    if (!shape) {
      return {shape: false, entities: null, relation: null, disclosure: null, allowed: false,
        issues: ["The proposal does not match the structured visual-hint format."]};
    }
    const entities = spec.entities.every(id => graph.entities.includes(id));
    const relation = entities && sameSet(spec.entities, graph.facts[spec.fact]);
    const disclosure = ["question", "relation"].includes(spec.disclosure);
    const issues = [];
    if (!entities) issues.push("The draft refers to an object outside the diagram.");
    else if (!relation) issues.push("The highlighted objects do not support the stated geometric fact.");
    if (!disclosure) {
      issues.push(spec.disclosure === "answer" ? "The draft reveals the final value." : "The draft reveals the calculation.");
    }
    return {shape: true, entities, relation, disclosure, allowed: relation && disclosure, issues};
  }

  function hint(problem, proposal) {
    const spec = proposal.diagram;
    const selected = spec.entities.map(entityLabel).join(" and ");
    if (spec.disclosure === "answer") return "∠" + problem.target + " = " + geometry(problem).answer + "°.";
    if (spec.disclosure === "operation") return "Calculate (180 − " + problem.apex_degrees + ") ÷ 2.";
    if (spec.disclosure === "relation") return selected + " have equal measures. How does that help you continue?";
    if (spec.fact === "given_equal_sides") return "Follow the marked sides to their opposite angles. What do you notice?";
    return "Look at the marked angles. How are they related to the equal sides?";
  }

  const reviewModes = {
    all: "Review every draft",
    flagged: "Review flagged drafts only",
  };

  function routeReview(problem, events, proposal, mode) {
    if (!Object.hasOwn(reviewModes, mode)) throw new Error("Unknown review mode: " + mode);
    const checked = audit(problem, proposal);
    if (!checked.allowed || proposal.action !== "highlight") {
      return {path: "review", reason: "The draft must be corrected before it can be shown.", checked};
    }
    if (mode === "all") {
      return {path: "review", reason: "This workflow routes every draft to a teacher.", checked};
    }
    const recommendation = decide(problem, events, "trace");
    const coded = events.findLastIndex(event => ["answer", "pair", "help"].includes(event.type));
    const pendingNote = events.slice(coded + 1).some(event => event.type === "statement");
    if (pendingNote) {
      return {path: "review", reason: "A learner statement needs human interpretation.", checked};
    }
    if (recommendation.action !== "highlight") {
      return {path: "review", reason: "The observed trace supports waiting or asking a question.", checked};
    }
    return {path: "auto", reason: "The trace supports a hint and the diagram passes its checks.", checked};
  }

  function evaluateReview(set) {
    return Object.fromEntries(Object.keys(reviewModes).map(mode => {
      const totals = {reviews: 0, automatic: 0, automaticOutsideRubric: 0};
      for (const row of set.cases) {
        const problem = set.problems[row.problem];
        const proposal = draft(problem, row.review_draft ?? "angles");
        const route = routeReview(problem, row.events, proposal, mode);
        if (route.path === "review") totals.reviews++;
        else {
          totals.automatic++;
          if (!row.acceptable.includes("highlight")) totals.automaticOutsideRubric++;
        }
      }
      return [mode, totals];
    }));
  }

  function score(row, action) {
    return {
      agrees: row.acceptable.includes(action),
      unwanted: action === "highlight" && !row.acceptable.includes("highlight"),
      missed: action !== "highlight" && row.acceptable.length === 1 && row.acceptable[0] === "highlight",
    };
  }

  function evaluate(set) {
    const detail = set.cases.map(row => {
      const problem = set.problems[row.problem];
      const choices = Object.fromEntries(Object.keys(policyLabels).map(policy => {
        const {action} = decide(problem, row.events, policy);
        return [policy, {action, ...score(row, action)}];
      }));
      return {id: row.id, acceptable: row.acceptable, choices};
    });
    const totals = Object.fromEntries(Object.keys(policyLabels).map(policy => [
      policy, detail.reduce((sum, row) => {
        const choice = row.choices[policy];
        return {agrees: sum.agrees + Number(choice.agrees), unwanted: sum.unwanted + Number(choice.unwanted),
          missed: sum.missed + Number(choice.missed)};
      }, {agrees: 0, unwanted: 0, missed: 0}),
    ]));
    return {detail, totals};
  }

  return {policyLabels, actionLabels, draftLabels, geometry, entityLabel, prompt, observe, describe,
    decide, draft, audit, hint, score, evaluate, reviewModes, routeReview, evaluateReview};
})();

if (typeof module !== "undefined" && module.exports) module.exports = VisualTutor;
