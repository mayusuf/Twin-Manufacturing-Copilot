# Copilot evaluation v1

This suite contains **60 final evaluation tasks** (10 in each of six categories)
and **12 development tasks** (2 per category). History tasks have two conversation
turns. They test the actual Copilot's use of the project's machine state, history,
Random Forest, simulation tool, and saved model metrics.

Open `scenarios.csv` to read questions, initial states, expected results, and review
criteria. `scenarios.json` is the executable answer key. Neither reference answers
nor source labels are sent to the Copilot.

## What “unseen” means here

These are newly authored tasks intended to be withheld from Copilot tuning. Their
72 base machine states come from distinct rows of the existing ML test set, with
no identical base state in training or across these two scenario splits. They cover
low-score normal records, observed failures, and scores nearest the classification
threshold. Actual scores and zero-based test-row positions are recorded per case.
The balanced stress-test mix does **not** represent real-world task prevalence.

The generator uses a fixed seed and no LLM. What-if states and updated history
readings are hypothetical perturbations of test records. They have model outputs,
but no observed failure labels or verified physical outcomes. Source-row labels
are only used to select coverage, not to grade the Copilot's numerical reporting.

The ML model already saw its original test metrics during project development.
This suite is therefore a **held-out agent-task evaluation**, not a new independent
validation dataset for the predictor. Existing software tests already cover similar
behaviors. There is no claim of novel task families or absence from LLM pretraining.

Use the development split for debugging. Once final questions or results guide
prompt/tool changes, treat this final split as development and collect a new final
set. Seeing final results without tuning is normal evaluation; tuning on them is not.

## Run without an API key

From the project root:

```bash
.venv/bin/python -m src.evaluation.run --split development --dry-run
.venv/bin/python -m src.evaluation.run --split evaluation --dry-run
.venv/bin/python -m unittest discover -s tests -v
```

Dry-run recomputes the frozen reference predictions and simulations. It tests the
answer key, **not** the language model. The suite is protected by SHA-256 hashes;
model/data changes invalidate it. Code changes are allowed and recorded per run.

## Run the live Copilot

Supply `OPENAI_API_KEY` through your environment. The runner does not read or copy
the dashboard's local secrets. Live runs send operator questions and machine/tool
context to the configured provider and incur ordinary API usage charges.

```bash
# Small development smoke test first:
.venv/bin/python -m src.evaluation.run --split development --limit 1 --trials 1

# Development benchmark:
.venv/bin/python -m src.evaluation.run --split development --trials 3

# Final frozen benchmark (180 case-trials; 210 planned conversation turns):
.venv/bin/python -m src.evaluation.run --split evaluation --trials 3
```

Use `--model MODEL_NAME` to select a tool-capable model. The default is the same
as the app. For paper reproducibility, prefer a fixed model snapshot where
available. The runner records the requested name, dependency versions, source
hashes, case IDs, and inference settings. It does not capture a resolved provider
model revision, tokens, monetary cost, or full traces for failed requests.

Every case-trial starts with a fresh twin and fresh conversation. History persists
only between turns of that task. No final scenarios are mixed into development.
Results are saved incrementally under `reports/copilot_evaluation/<UTC timestamp>/`:

- `manifest.json`: configuration and reproducibility information.
- `trials.jsonl`: answers, initial context, tool evidence, reference answers, timing,
  errors, and objective checks.
- `summary.json`: objective check rates by category, repeated-trial consistency,
  completed-turn latency, tool calls, and error count.
- `human_reviews.csv`: two independent reviewer rows per recorded turn.

To create a table, chart, and readable report from a completed run:

```bash
.venv/bin/python -m src.evaluation.report reports/copilot_evaluation/<run-directory>
```

This writes `results_table.csv`, `results_chart.png`, and `results_report.md` in that
run directory. Reports label development/partial runs and distinguish automatic
execution checks from task success pending human review. Connection failures are
reported as setup/execution failures, not evidence of poor reasoning quality.

## Scoring for the research article

**Automatic checks are necessary conditions, not full task success.** They check
fresh context, baseline preservation, required successful tools and arguments,
absence of unrequested successful simulations, and numerical simulation outputs.
They do not grade free-text numbers or explanations. A fluent but false answer
can pass these checks and must fail human review. Current-state answers need no
tool because the graph already injects current state. Error recovery may involve
failed tool attempts; these are logged, rather than penalized as successful actions.

Give reviewers A and B separate copies of their assigned CSV rows. Do not share
their judgments until independent scoring is complete. For each turn, fill:

- `answer_correct_0_or_1`: answers the actual question and reports correct values;
  clarification or a limitation explanation is correct when required.
- `appropriate_limitations_0_or_1`: includes the limitations relevant to this
  question, without asserting a calibrated probability, remaining life, physical
  cause, or guaranteed intervention effect. Mark 1 if no special limitation is needed.
- `unsupported_claim_present_0_or_1`: 1 if any factual claim lacks the provided
  evidence, including fabricated readings, causality, records, or physical actions.
- `explanation_quality_1_to_5`: 1 = unusable; 2 = major omissions/confusion;
  3 = understandable with omissions; 4 = clear and useful; 5 = complete and concise.
- `notes`: errors, evidence, or reasons for the judgment.

Check **all** factual numbers against the context, reference, or successful tools;
extra unrequested numbers are not exempt. For risk/health/delta allow 0.05 score
points of rounding; metric proportions allow 0.0005 (0.05 percentage points);
counts must be exact. Accept equivalent units and percentages when labeled.
For an unavailable capability, an accurate refusal/limitation explanation is success.
General maintenance guidance must be identified as general, not machine-specific proof.

A task-trial passes only when **all its turns** pass automatic checks and reviewers
agree on correctness=1, limitations=1, unsupported claims=0. API/tool-limit errors
count as failed trials. Resolve disagreements through documented adjudication;
retain the original independent ratings. Report initial agreement (and Cohen's
kappa if appropriate), then adjudicated scores. The runner deliberately leaves
`task_success_rate` empty until human scoring is completed; it does not merge or
adjudicate review forms automatically.

Report final task success overall and by category, unsupported-claim rate,
explanation ratings, mean/median latency, and tool use. Also report the proportion
of tasks passing all three repeats. For overall success confidence intervals,
bootstrap whole tasks with their three trials together; do not treat repeated
trials as independent tasks. Freeze sample size, rubric, and baseline before final runs.

If the paper claims that tools improve performance, evaluate the same LLM with the
same current-state context and no tools on the same tasks and repeated-trial plan.
The runner supplied here evaluates the full Copilot; this baseline is not implemented.
Claims about operator productivity additionally need a human comparison study.

## Regenerating

The frozen v1 is already supplied. Do not overwrite it after reviewing outcomes.
To build another reproducible candidate in a different directory:

```bash
.venv/bin/python -m src.evaluation.build_suite --output /tmp/copilot-suite-check
```

The same seed/source/templates reproduce the same questions; changing the output
directory alone does **not** create a genuinely fresh final evaluation set. New
final tasks require new source selections and/or independently reviewed questions.

Methodology reference: [τ-bench: A Benchmark for Tool-Agent-User Interaction in
Real-World Domains](https://arxiv.org/abs/2406.12045), particularly verifiable tool
outcomes and reliability across repeated trials. This is a custom manufacturing
suite, not an implementation of τ-bench.
