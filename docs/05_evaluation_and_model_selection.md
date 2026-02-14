# Phase 5 — Evaluation & Model Selection

> **Goal of this tutorial:** understand what "the new model is better" actually has to mean, why a strict comparison rule prevents quiet regressions, how the MLflow Registry turns "the production model" from a convention into a queryable fact, and what guardrails real production systems layer on top of the simple "F1 went up → ship it" rule we built here.

---

## 1. The problem this phase solves

Phase 4 trains many models and tracks every run in MLflow. By the end of training you might have:

- 12 logreg runs across hyperparameter combinations
- 8 random forest runs
- 3 gradient boosting runs (you added it as the exercise)
- ...and one of them is the current production model from last week.

The question is no longer *"can we train a model?"* — it's:

> **"Which run, if any, deserves to replace the model we are serving right now?"**

This is **model selection** in production: not picking the best of a one-time bake-off, but maintaining a *running* answer to the question every time a new training round completes.

## 2. The decision rule we built

```python
if candidate.metric > current_production.metric:
    promote(candidate)
else:
    keep(current_production)
```

Three properties of this rule are worth pulling apart:

### 2.1 Strict greater-than

Equal metrics get rejected. Why? Because changing production has cost:

- A registry version transition is auditable; people will ask "why did we promote v3?"
- A new model means new latency, memory, and feature-importance characteristics — even if the offline metric is identical
- Rollback complexity grows

The asymmetry is **the incumbent has the burden of staying, the candidate has the burden of beating.** Equal isn't beating.

### 2.2 One metric

We compare on F1. Not because F1 is "the right metric" — it's because **picking one rule prevents a thousand small wars about whether v17 is "actually better."** Every team that has tried "promote if better on any of these 5 metrics" has ended up promoting things that are worse on the metric that mattered most.

(Phase 6's CI will let you change the metric via a config; the *rule* stays one-metric.)

### 2.3 No model knows it's "better"

The model object itself doesn't decide. **The decision is data, written to a JSON file, with the inputs that produced it.** That JSON is committable, reviewable, alertable. If you want to override the decision, you commit the override — visible in git, attributable to a person.

## 3. Why F1 (and not the others)

A quick recap of the metric landscape from docs/04, focused on this decision:

| Metric    | Why we wouldn't gate on it                                                |
| --------- | ------------------------------------------------------------------------- |
| Accuracy  | A "do nothing" model gets 83% on this dataset — uninformative for churn   |
| Precision | Maximizing precision rewards being conservative — misses real churners    |
| Recall    | Maximizing recall rewards being liberal — buries retention team in alarms |
| ROC-AUC   | Threshold-independent but inflated on imbalanced data                     |
| PR-AUC    | Honest on imbalance — but threshold-independent: doesn't reflect *deployed* quality |
| **F1**    | Combines precision and recall at the threshold actually deployed           |

PR-AUC is the better *modeling* metric (curve quality across all thresholds). **F1 is the better *deployment* metric** because production runs at *one* threshold, and F1 measures quality at that threshold.

In practice you'd often use both: PR-AUC as a guardrail ("don't regress more than 2%"), F1 as the primary gate. Phase 6 will let you wire that.

## 4. MLflow Registry: aliases over stages

We use **aliases**, not the older **stages** API. Both exist in MLflow 2.x:

| API                          | Idea                                              | Example                                                                |
| ---------------------------- | ------------------------------------------------- | ---------------------------------------------------------------------- |
| Stages (deprecated)          | Each version sits in one of {None, Staging, Production, Archived} | `client.transition_model_version_stage(name, v, "Production")`        |
| **Aliases (we use this)**    | Mutable named pointers to versions                | `client.set_registered_model_alias(name, "production", version=3)`     |

Why aliases:

1. **Multiple aliases per version.** A version can be `production` *and* `canary` *and* `shadow`. Stages forced one-of-these-four.
2. **Decoupled deployment targets.** Want a `production-eu` and `production-us` aliases pointing at different versions? Trivial. With stages, you'd add tags everywhere.
3. **Reading is one call.** `get_model_version_by_alias("churn-model", "production")` returns the version. With stages: `get_latest_versions(name, stages=["Production"])` returned a *list* (always of length 1 if used right) — every call site had to handle the list case.
4. **MLflow is moving here.** Stages are explicitly deprecated for new code.

You'll see stages in older codebases. Not wrong — just dated.

## 5. The bootstrap problem

The first run is special: there's no current production to compare against. We handle this with a `bootstrap` decision: register and promote unconditionally on the first call.

This is fine for a portfolio project. In real systems the first model is **always** a hand-blessed one — you don't "bootstrap" production via an automated pipeline because the rest of the company hasn't seen the model yet. The standard pattern:

1. A human trains the first model and hand-validates.
2. A human registers it and sets the `production` alias.
3. *Then* automation can take over with the compare-then-promote rule.

We collapsed steps 1–2 into the bootstrap branch for clarity. The tutorial mentions it as a code shortcut.

## 6. Asymmetric error costs (a churn example)

Choosing F1 implicitly weights false positives and false negatives equally — but in real churn use cases they aren't equal:

| Action                               | Cost per occurrence                                     |
| ------------------------------------ | ------------------------------------------------------- |
| **False negative** (missed churner)  | One lost customer × LTV (e.g. €500–€2,000)              |
| **False positive** (false alarm)     | One retention email/discount (e.g. €5–€50)              |

A 30:1 cost asymmetry is common. Optimizing F1 (which weights them 1:1) underweights catching churners. The right knob: **threshold tuning**.

### Threshold tuning (preview)

The model outputs a *probability*. The default threshold is 0.5: predict churn if `P >= 0.5`. Lowering the threshold to 0.3 means more positives flagged → higher recall, lower precision. The cost ratio above suggests a much lower threshold than 0.5.

We didn't tune the threshold here — the focus of Phase 5 is the *promotion decision*, not the operating point. Phase 6 will likely tune it as a CI step that searches the threshold maximizing expected business value.

If you want to do it now: load the production model, run on the test set, sweep thresholds 0.1 to 0.9, compute precision/recall at each, pick the threshold maximizing `precision * cost_fp / (precision * cost_fp + (1 - recall) * cost_fn)`.

## 7. Guardrails real systems add

The compare-then-promote rule is the *minimum* viable promotion logic. Real production systems layer on top of it. None of this is built here; all of it is on the "Phase 6+" or "next portfolio project" list.

### 7.1 Margin thresholds

> "Promote only if the new model beats current by at least 2 absolute F1 points."

Prevents promoting a marginal improvement that's likely noise. The 2-point figure is itself a hyperparameter; pick it from the historical run-to-run variance of your metric.

### 7.2 Multi-metric guards

> "Promote if F1 went up AND PR-AUC didn't go down by more than 1% AND latency didn't go up."

Catches the case where you optimized F1 by sacrificing recall, or by adding a feature that doubled inference latency.

### 7.3 Soak periods (canary)

> "Run as canary on 5% of traffic for 3 days, only flip to 100% if live metrics still beat current."

Offline metrics on a static test set are an estimate of production quality — *not* production quality itself. The canary tests the estimate against reality. This is the most expensive but also the most catching guardrail.

### 7.4 Statistical significance

> "Don't promote unless the F1 difference is significant at p<0.05."

Avoids ping-ponging production back and forth between equally-good models. McNemar's test on the held-out predictions is the standard tool.

### 7.5 Traffic / sample minima

> "Don't trust metrics computed on test sets smaller than N rows."

Small test sets produce metrics with huge confidence intervals; "F1 went from 0.40 to 0.45" might be entirely within the bootstrap noise on a 200-row set.

### What we built vs the full picture

```
What Phase 5 built:                    What real production layers on:
┌─────────────────────────┐            ┌──────────────────────────────────┐
│ if candidate > current: │            │  margin threshold                │
│     promote             │   <─── +   │  multi-metric guards             │
│ else:                   │            │  soak / canary period            │
│     reject              │            │  statistical significance        │
└─────────────────────────┘            │  minimum sample size             │
                                       │  human approval for big changes  │
                                       └──────────────────────────────────┘
```

The simple rule is the foundation; the layers above are organizational decisions, not algorithmic ones. Build the simple rule first, then add the layers as your team decides the rule is wrong in specific ways.

## 8. Real-world analogy

> Promoting a model is like firing a senior engineer.
>
> The current production model is doing a job. Maybe not perfectly, but the company has built *around* it: monitoring is calibrated to its quirks, the retention team's playbook assumes its threshold, downstream systems expect its output distribution. Replacing it has institutional cost beyond "the new candidate scored 0.01 higher on a bake-off."
>
> A reasonable hiring committee for the new candidate doesn't ask "is candidate marginally better than incumbent?" They ask "is the *gain* big enough to justify the *transition cost*?" That's exactly what margin thresholds, soak periods, and significance tests are: making the transition cost legible to the decision rule.
>
> Strict-greater alone is the bottom-floor: the candidate must at least be better at the one job we measured. Anything beyond that is a more sophisticated hiring decision.

## 9. How this maps to your existing experience

- **Blue/green deployments.** A canary/soak period for ML models is the same idea as a blue/green deploy: run new version on a fraction of traffic, compare to baseline, flip when confident. The difference is what "compare to baseline" means — for a Go service, it's error rate and latency; for an ML model, it's F1 / precision / recall on outcomes that may take days to materialize.
- **Feature flag percent rollout.** Aliases let you do this: `production = v3` for 100%, or `production = v3, canary = v4` for split traffic. The serving layer reads both and routes accordingly.
- **CI gates on regressions.** "Don't merge if test coverage went down" is a margin guardrail. ML adds the wrinkle that the metric is noisy and statistical significance becomes important.
- **Schema versioning.** Mature API versioning (semver, deprecation windows) is what mature MLOps teams adopt for feature schemas — a breaking change to `feature_v3` is a major bump that downstream consumers explicitly opt into.

## 10. Exercises

1. **Watch a reject.** Open `src/training/pipeline.py`. Crank `RandomForestClassifier(max_depth=10)` down to `max_depth=2`. Run `dvc repro`. Look at `metrics/promotion_decision.json`. What decision did you get? Why?
2. **Force a promote.** With pipeline back to normal: open `src/training/train.py`, change `random_state=42` to `random_state=43`. Run `dvc repro`. Did training change? What happened to the promotion?
3. **Implement a margin threshold.** Edit `decide_promotion` so the candidate must beat current by ≥0.005 F1, not just any improvement. Update tests. What's the right place to make this configurable — CLI arg, env var, params.yaml?
4. **What if the metric we care about isn't logged?** What does `decide_promotion` do if `metric="balanced_accuracy"` but training didn't log that metric? Trace the code path. Is the failure loud enough? If not, fix it.
5. **The bootstrap problem in production.** You join a team with a production model deployed manually six months ago, with no MLflow registry. Write the steps to migrate that model into the registry as v1 *without* re-training. (Hint: `mlflow.register_model` accepts a local artifact URI, not just `runs:/...`.)

## 11. How to explain this in an interview

**60-second pitch:**

> "Phase 5 codifies model selection. The decision rule is simple: the new run only replaces production if it strictly beats the current production model on F1. The MLflow Registry holds the production version with an alias pointer; the evaluate DVC stage reads training results, runs the comparison, writes a verdict JSON to git, and on a positive decision moves the alias to a newly-registered version. The simple rule is the foundation; production systems layer margin thresholds, soak periods, and multi-metric guards on top. I picked F1 over accuracy because of class imbalance, and over PR-AUC because PR-AUC is threshold-independent — F1 measures quality at the operating point you actually deploy."

**If they ask "why aliases over stages?":**

> "Aliases are mutable named pointers — `production`, `canary`, `shadow` — multiple per version. Stages were a fixed enum where each version had to be in one of four states. Aliases compose: a single version can be production *and* shadow at the same time. Stages were also deprecated in MLflow 2.x for new code; aliases are the path forward."

**If they ask "what if a metric is noisy and the rule promotes a worse model?":**

> "That's the whole point of the guardrails on top of the simple rule. The smallest one is a margin threshold — require the gain to exceed historical run-to-run variance, often 2 standard deviations. The right one is a canary period — small fraction of traffic on the new version, compare *live* metrics, flip when statistically significant. Both layer on the simple rule without changing it."

## 12. Common mistakes

- **Promoting on accuracy.** Especially on imbalanced data — you'll promote a model that's silently worse at the actual task. Always promote on a metric that survives the imbalance.
- **Promoting "if any metric improved."** Lets you optimize one number while regressing the others. Pick *the* metric and stick to it.
- **Not committing the verdict.** A promotion decision that lives only in MLflow's UI isn't auditable in a pull request. We commit the JSON.
- **Letting `apply` default to True everywhere.** Dry-run mode is essential for CI preview. We separated `decide_promotion(apply=False)` from `apply=True` so a CI job can show the verdict in a PR comment without actually promoting.
- **Coupling promotion to retraining.** They're separate decisions. You may want to retrain weekly but only promote when something materially changed. Our DVC graph keeps them as separate stages so you can `dvc repro train` without `dvc repro evaluate`.
- **Trusting the test set forever.** As the data distribution drifts, the test set's relevance erodes. Phase 8's drift detection handles this — but the lesson is: the test set is a snapshot, not the truth.

## 13. What changes at scale

- **Promotion approval workflows.** A "promote" verdict triggers a Slack message that requires human ACK before the alias actually moves. This is where automation deliberately stops short of fully autonomous shipping.
- **Multi-environment registries.** Separate `production` aliases for `eu`, `us`, `apac` — each with its own promotion cadence and rollback policy.
- **A/B promotion.** Instead of replace-the-incumbent, run candidate and incumbent side-by-side on different traffic slices and promote the winner of a real-world metric (revenue, retention) — not an offline proxy.
- **Promotion ledger.** Every promotion decision becomes a row in an audit log: who/what proposed, who approved, what offline metrics, what live metrics post-promotion. Becomes important once regulators or partner teams ask "why is *this* model deciding *this* thing?".
- **Shadow traffic.** Set a `shadow` alias on a candidate, route real-time inference traffic to *both* incumbent and shadow, log the disagreements. Compare predictions on live data without ever exposing the candidate's predictions to users. The quietest, safest pre-promotion test.

---

## ✅ Phase 5 done. What's next.

Phase 6 will:
- Add **GitHub Actions** workflows that automate the whole DAG: code change → re-run pipeline → check the verdict → block PR if "reject" or fail-loud.
- Implement the three retraining triggers: code change (PR), data change (cron), scheduled (weekly cron).
- Use the `metrics/promotion_decision.json` file as the gating signal CI reads.
- Teach: ML CI/CD vs normal CI/CD, why ML automation needs different timing (training is minutes, not seconds), and what's safe vs unsafe to automate.

Wait for the user's go-ahead.
