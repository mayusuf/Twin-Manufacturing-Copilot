# Copilot research results — pending live evaluation

This is a presentation template, **not a table of measured results**.

Planned experiment: 60 held-out tasks, six categories, three independent trials
per task (180 task-trials; 210 conversation turns including history warm-ups).
Default model: the same model configured in the Copilot implementation.

| Task category | Tasks | Planned task-trials | Automatic tool/state checks | Human-evaluated task success |
| --- | ---: | ---: | --- | --- |
| Current-state explanation | 10 | 30 | Not run | Pending |
| What-if simulation | 10 | 30 | Not run | Pending |
| History and follow-ups | 10 | 30 | Not run | Pending |
| Model performance questions | 10 | 30 | Not run | Pending |
| Ambiguous or invalid requests | 10 | 30 | Not run | Pending |
| Unsupported requests | 10 | 30 | Not run | Pending |
| **Overall** | **60** | **180** | **Not run** | **Pending** |

After live trials, `src.evaluation.report` generates a measured table and chart
from the recorded results. Automatic checks cover tools, inputs, state freshness,
and baseline preservation. Reviewers must additionally grade factual correctness,
appropriate limitations, unsupported claims, and explanation quality.

For the article, report task success overall and by category, the proportion of
tasks passing all three trials, uncertainty, unsupported-claim rate, response
time, and tool use. Task success remains pending until human review is complete.

The passing software tests and offline reference checks establish implementation
and answer-key correctness; they must not be presented as live Copilot accuracy.
