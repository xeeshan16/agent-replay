# Evaluation data

## `regressions.jsonl`: seeded regressions for the support-desk agent

64 cases: every agent policy in `agent_replay.reference.support_desk.POLICIES` (the `v1`
baseline, six seeded regressions and one benign change) on eight support tasks. Each line is
one case:

| Field | Meaning |
|-------|---------|
| `id` | `<policy>/<task>` |
| `task` | the customer message |
| `policy` | the candidate policy replayed against a `v1` recording of the task |
| `label` | `regression` (behaves worse), `benign` (behaves differently but acceptably) or `no_effect` (the policy change does not apply to this task) |
| `category` | `skipped_lookup`, `missing_retry`, `argument_drift`, `extra_loop`, `unsafe_write`, `unsupported_claim`, `extra_read` or `none` |
| `decision_step` | the candidate's model step that made the decisive choice (changed cases only) |
| `note` | why the case has its label |

Labels were written from what each policy is defined to do (see the policy docstrings) and the
synthetic order data, not by running any detector. Counts: 18 regression, 5 benign, 41
no_effect (8 of them the baseline replayed against itself).

Run it with `agent-replay eval --out-dir evals/results`. The report gives precision and recall
of each detector (the diff alone, each scoring layer alone, and the layered verdict), recall per
regression category, first-divergence localisation against `decision_step`, and the judge's
spend in the layered configuration against an always-on judge.

**Provenance and licence.** Entirely synthetic: the tasks, orders and policies were written for
this repository, contain no real customer data, and are released under the repository's MIT
licence.

**Limits.** One small agent with deterministic fake models: the eval measures whether the
detectors catch the regression types it seeds and stay quiet on the controls, not how a real
model's run-to-run variance affects them. The judge in this eval is a deterministic stand-in
that applies the rubric's grounding rules; a real model judge would need its own agreement
study before its numbers mean anything.
