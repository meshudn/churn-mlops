# Phase 8 — Drift Monitoring & the Retraining Loop

> **Goal of this tutorial:** understand the two kinds of drift that silently kill ML systems, what the four standard statistical tests actually measure, why "just retrain on new data" isn't enough for *concept* drift, and how the monitor stage closes the loop that turns a one-shot pipeline into a *system*.

---

## 1. Two kinds of drift, one source of bugs

A model that scored F1=0.42 on the test set last Tuesday is solving an idealized problem: predict churn for customers *whose distribution looks like the training data*. Two ways the world breaks that assumption:

| Kind                | What changes                                                | Example                                                    |
| ------------------- | ----------------------------------------------------------- | ---------------------------------------------------------- |
| **Data drift**      | The input feature distribution `P(X)` shifts                | Marketing pushes more customers onto fiber-optic plans     |
| **Concept drift**   | The relationship `P(y \| X)` itself changes                 | A competitor offers free month-1 → even Two-year customers churn now |

Crucially:

- **Data drift can happen without quality dropping.** If the model learned good general patterns, a shift in the input mix may not hurt much.
- **Concept drift is invisible to input monitoring.** Inputs look fine; predictions are silently wrong.
- **Most production ML failures are concept drift, not data drift.** Inputs are usually monitored upstream. The relationship to outcomes is what nobody checks.

That asymmetry shapes what we monitor and how we respond.

## 2. What we built

```
src/monitoring/
├── __init__.py
└── drift.py            # compute_drift() + CLI; uses Evidently AI's DataDriftPreset

dvc.yaml :: monitor stage
  - deps:  data/features/churn_features.csv, src/monitoring
  - outs:  metrics/drift_report.html  (cached, gitignored — Evidently's full visual report)
  - metrics: metrics/drift_report.json (cache: false — small summary, in git)
```

The CLI compares two feature batches:

- **Reference**: `data/features/churn_features.csv` (the training-time features).
- **Current**: an on-the-fly synthetic batch with a deliberate distribution shift (boosted fiber-optic share + raised charges) — stands in for "recent inference inputs" until you have a real production stream.

Output of the first run:

```
drifted_columns: 2 of 24
drift_share:     0.083
dataset_drifted: false (threshold = 0.5)
```

Two columns were flagged: `monthly_charges` and `monthly_charges_per_service`. Notably, the *cause* — `internet_service` — sat just under its drift threshold (Jensen-Shannon 0.0999 vs 0.1). The drift signal **amplified through derived features**: the boosted fiber-optic mix didn't quite trip the categorical test, but the higher charges that came with it did.

That's a real lesson. Engineered features are how you discover drift the raw distribution barely shows.

## 3. The statistical tests Evidently picks for you

`DataDriftPreset` chooses a test per column based on type and cardinality:

| Method                          | When picked                                       | What it computes                                          | Drift means                          |
| ------------------------------- | ------------------------------------------------- | --------------------------------------------------------- | ------------------------------------ |
| **Kolmogorov-Smirnov** (K-S)    | Numeric columns, smaller sample                   | p-value of the null "two samples come from the same dist" | p-value < threshold                  |
| **Wasserstein distance**         | Numeric columns, larger sample                    | Earth-mover's distance between the two distributions      | distance > threshold                 |
| **Z-test / Chi-square**          | Categorical, low cardinality                      | p-value on category proportions                           | p-value < threshold                  |
| **Jensen-Shannon distance**      | Categorical, higher cardinality                   | symmetric divergence between distributions                | distance > threshold                 |
| **PSI** (Population Stability Index) | Banking / risk industry standard               | weighted log-ratio across bins                            | PSI > 0.25 (rough)                   |

Notice the polarity flip: p-value tests say "drift = small," distance tests say "drift = large." The drift module's parser handles both (`_is_drifted_for_method`).

**Which one is "right"?** Each has tradeoffs:

- K-S is the textbook choice but fights small sample sizes (it's sensitive on small data).
- Wasserstein is robust and intuitive ("how much mass needs to move?") but doesn't give a probabilistic interpretation.
- PSI is what banks use because regulators understand it; it's coarse but stable.
- Jensen-Shannon is symmetric (good) but bounded in [0, ln(2)] which can hide extreme drift in the tail.

Evidently picks reasonable defaults. Real production teams pick *one* test per column type and stick to it for comparability across runs, not "let the library choose."

## 4. Why "just retrain" doesn't always fix concept drift

The naive remediation:

```
drift detected → retrain on the latest data → ship new model
```

This works for **data drift**: the relationships are still valid, the input mix is just different, more recent data better represents the current mix. New model usually wins.

For **concept drift** it can fail in two specific ways:

1. **The new data doesn't have ground truth yet.** Churn is observed 30 days later. By the time you have labels, the world has moved on again. Retraining on labels from "last quarter's world" doesn't fix today's drift.
2. **The new relationship is fundamentally different.** The competitor offering free month-1 didn't just shift the distribution — it changed *what predicts churn*. Tenure used to be a strong signal; now it isn't. Retraining will eventually catch up, but the model will be wrong in the interim, and may need different features or an entirely different model class.

So the closed loop is **not** "drift → retrain → ship":

```
                   ┌─────── retrain & evaluate ──────┐
                   │                                 │
   drift detected ─┤                                 ▼
                   │                            promotion?
                   │                              ↓ no
                   ├─────── alert humans ───────► investigate
                   │
                   └─────── log + dashboard ────► trends
```

Drift triggers *investigation*, which leads to one of: retrain, redesign features, change the model class, or accept that this is a new regime that needs a different approach. Most of the time the answer is retrain; the system is built so the other answers are reachable.

## 5. What we monitor (and what we should monitor next)

Three things in production ML:

| What                  | What it catches                          | Phase 8 status         |
| --------------------- | ---------------------------------------- | ---------------------- |
| **Input drift**       | Data drift on the feature inputs         | ✅ Built               |
| **Prediction drift**  | Drift in the *output* distribution       | ⏸ Not yet              |
| **Performance drift** | Direct quality measurement on labels     | ⏸ Not yet (no labels in production) |

Why prediction drift matters: even if inputs look normal, the model's outputs may be shifting (e.g. far more "predicted churn = True" than usual). That can be the early signal of either data or concept drift, observable *before* labels arrive. It's also the easiest to add: just save predictions and run the same drift detector on the prediction column.

Why performance drift is the gold standard: if you have labels, you can directly measure whether F1 / PR-AUC dropped. But labels arrive late (30 days for churn). Performance drift is the slowest, most certain signal.

## 6. The closed loop, end-to-end

This is the diagram for the *system* the project built — every phase plays a role:

```
                            ┌─────────────────────────────────────────────┐
                            │                                             │
                            ▼                                             │
       ┌─────────┐    ┌─────────┐    ┌──────────┐    ┌──────────┐        │
   ┌──►│ Ingest  │───►│Features │───►│  Train   │───►│ Evaluate │        │
   │   │ (P2)    │    │  (P3)   │    │  (P4)    │    │  (P5)    │        │
   │   └─────────┘    └─────────┘    └──────────┘    └────┬─────┘        │
   │                                                      │ promote       │
   │                                                      │ alias         │
   │                                                      ▼               │
   │                                                ┌──────────┐         │
   │                                                │  Serve   │         │
   │                                                │  (P7)    │         │
   │                                                └────┬─────┘         │
   │                                                     │ predictions    │
   │                                                     ▼               │
   │                                                ┌──────────┐         │
   │   triggers retrain                              │ Monitor  │         │
   │     ▲      ▲                                    │  (P8)    │         │
   │     │      │                                    └────┬─────┘         │
   │     │      │ 3. scheduled (P6 cron)                  │ drift signal   │
   │     │      │                                         │                │
   └─ 1. code change (P6)               2. data change (P6) ───────────────┘
        ▲
        │
        └── developer commits / merges to main
```

What makes this a **system** and not a pipeline:

- **The arrow back from Monitor to Ingest.** Without that, it's a one-way recipe.
- **Three retraining triggers** (Phase 6) close the loop on three failure modes.
- **The promotion rule** (Phase 5) prevents the loop from blindly shipping worse models.
- **The registry** (Phase 4 + Phase 7) decouples model deploys from container deploys.
- **Schemas at every boundary** (Phase 2 + Phase 3) catch shape regressions before they pollute downstream.

Take any one of those out and the system degrades quietly. Together they make a system that *can run unattended* — which is the only kind of ML system that matters in production.

## 7. Drift response policies

When drift fires, three policies are common:

### 7.1 Alert-only (we'd do this first)

Drift alert goes to Slack / email / PagerDuty. Humans investigate. Retraining is manual.

- **Pro**: safest. No model surprises. Catches concept drift that auto-retraining would mishandle.
- **Con**: slow. Drift may already be hurting users by the time someone investigates.
- **When**: early days of an ML system, before you trust the loop.

### 7.2 Auto-retrain on alert

Drift alert short-circuits the next scheduled cron. Pipeline runs immediately. Promotion still goes through Phase 5's strict-greater rule.

- **Pro**: fast response to data drift.
- **Con**: can cycle if the new training data is the *cause* of drift (e.g. a data quality bug).
- **When**: mature system with high confidence in the data pipeline.

### 7.3 Auto-retrain + canary

Drift alert → train → register as `canary` alias (not `production`) → route 5% of traffic → measure live metrics for N hours → flip alias if good.

- **Pro**: catches the failure mode where offline metrics look fine but live behavior is wrong.
- **Con**: needs a routing layer that supports alias-based traffic split.
- **When**: real-time inference at meaningful traffic; LinkedIn / Netflix / Uber level.

This project is at level (1). The hooks for (2) are there: the scheduled retrain workflow can read `metrics/drift_report.json` and short-circuit the cron. The hooks for (3) need a routing layer we deliberately didn't build.

## 8. How this maps to your existing experience

- **APM tools (Datadog, New Relic).** Drift detection on inputs is conceptually a *probability-distribution-aware metric* on features. Same dashboards, same alert rules, just with statistical-distance metrics instead of latency percentiles.
- **AWS CloudWatch alarms with anomaly detection.** Statistical anomaly detection on a single metric is a 1D version of what `DataDriftPreset` does on dozens of columns.
- **Logging + queryable observability.** `metrics/drift_report.json` in git is your audit log; the HTML report is your detailed view. In a real system both end up in a dashboard tool (Grafana, Looker) keyed on time.
- **n8n cron + conditional branch.** "If drift > X, fire retrain workflow" is the same structure as an n8n IF node downstream of a cron node.

## 9. Real-world analogy

> A model in production is like a **bridge inspector** trained ten years ago.
>
> The inspector knows how bridges built in the 2010s fail: certain rust patterns, certain cracking signatures, common defects in the materials of that era. They work great for ten years.
>
> Then a manufacturer changes the steel alloy. The new bridges (data drift) look different — the inspector's checklist needs adjusting, but their *expertise* still applies after a refresher (retraining).
>
> Then a new failure mode emerges: a kind of fatigue crack that didn't exist before because of a new traffic pattern (concept drift). The inspector's checklist isn't *wrong* — it's *incomplete*. Refreshing on more recent inspections won't help; the failure mode just hasn't been logged yet. The fix isn't retraining; the fix is figuring out the new pattern.
>
> Drift detection tells the bridge owner that *something* changed. Investigating — sometimes by an expert, sometimes by adding new data, sometimes by replacing the inspector entirely — is what closes the loop.

## 10. Exercises

1. **Make the drift trigger fail hard.** In `metrics/drift_report.json`, what would push `dataset_drifted` to `true`? Edit `generate_drifted_current` to also boost `Contract = "Month-to-month"` to 90% of records. Re-run `dvc repro monitor`. Did `dataset_drifted` flip?
2. **Compare to a stale model.** Imagine production has been on the same model for 6 months. Open the MLflow UI, find the production run's metrics. Now look at `metrics/train_metrics.json` from the most recent training. Is the new model better? If not, what would you investigate?
3. **Wire drift into CI.** Open `.github/workflows/retrain.yml`. Add a step on the scheduled trigger that reads `metrics/drift_report.json`; if `drift_share > 0.3`, mark the run as "fast-tracked" (e.g. via a workflow output or a tag).
4. **Add prediction drift.** Modify `src/deployment/api.py` to log every prediction (csv append). Add a `predict_drift` stage in DVC that compares recent predictions to training-time predictions. What's the right reference distribution — last week's predictions, or training-time?
5. **Catch concept drift offline.** You don't have live labels. Sketch how you'd detect concept drift using only the data you have. (Hint: shadow predictions on labeled historical data, or back-test the current model on a hold-out time window from the original training period.)

## 11. How to explain this in an interview

**60-second pitch:**

> "Phase 8 closes the loop. The monitor stage uses Evidently AI's DataDriftPreset to compare a current feature batch to the training reference, picks an appropriate statistical test per column — Wasserstein for numeric, Jensen-Shannon for categorical, K-S/Z-test for smaller samples — and writes a JSON summary the CI cron reads. The interesting bit was that my deliberately-shifted current data only barely tripped the categorical test on `internet_service`, but the *derived* `monthly_charges` feature drifted clearly because the fiber boost raised charges. Real lesson about engineered features amplifying drift signals. The whole system now has three retraining triggers (code, data, scheduled), strict-greater promotion gating, and drift alerting — at this scale I'd send drift to Slack rather than auto-retraining, because most production failures are concept drift and auto-retraining can't fix that."

**If they ask "how do you tell data drift from concept drift?":**

> "Input drift detectors only see inputs — they catch data drift by definition, they're blind to concept drift. The way to detect concept drift is *prediction* drift on outputs and, ultimately, *performance* drift once labels arrive. Performance drift is the gold standard because it's a direct measurement, but it's slow — for churn, you wait 30 days for labels. Input drift is the fastest signal but the least specific."

**If they ask "what's wrong with auto-retraining on every drift alert?":**

> "Three things. One: if the data drift is *caused* by a data-quality bug, auto-retraining poisons the model with bad data. Two: if it's concept drift, retraining on labels from the old regime doesn't help and you're shipping a still-wrong model with new metadata. Three: in real systems retraining is non-zero cost — compute, latency on the model registry, ops attention — so you want a smart trigger, not a hair-trigger. The right pattern is alert-then-investigate at first, then add automation as you learn the failure modes."

## 12. Common mistakes

- **Monitoring inputs only.** Misses concept drift entirely. Add prediction drift (output distribution) and performance drift (when labels exist).
- **One drift threshold for all columns.** Different columns have different "natural" variation. Per-column thresholds calibrated from historical noise are the next-step refinement.
- **Treating drift as a bug.** It's a *signal*. The bug might be a data quality issue, a real-world change, a marketing campaign, or actual concept drift. Investigating is the work.
- **Auto-retraining the moment drift fires.** See above. Layer alerting first.
- **No reference dataset versioning.** Drift is "compared to *what*?" — the reference matters. We use `data/features/churn_features.csv` which is DVC-tracked, so the reference is reproducible. Floating-reference drift detection is meaningless after a few runs.
- **Storing only the latest report.** History is the value — you want to see drift *trends*. Real systems append to a time-series store and graph the drift_share over time.
- **Drift tests on tiny windows.** A 50-row "current" sample produces wildly noisy drift estimates. Keep windows realistic (100s-1000s of rows minimum).

## 13. What changes at scale

- **A real prediction stream.** The "current" batch is a sliding window of inference inputs from a Kafka topic / log table, not a synthetic generator. The monitor stage runs on a cadence (e.g. hourly).
- **Time-series drift store.** Drift metrics over time end up in Prometheus + Grafana, or InfluxDB, or any TSDB. You want to graph drift_share for the last 90 days, not just see the latest.
- **Per-segment drift.** "Drift across all customers" is too coarse. You'd compute drift per cohort (geography, plan tier, signup month) — more sensitive, more diagnostic.
- **Multivariate drift.** Univariate drift misses correlations. Tools like `evidently`'s embedding-based drift, or PCA-projection drift, or autoencoder-based anomaly detection catch shifts the per-column tests miss.
- **Causal hooks.** "What changed upstream that explains this drift?" You'd connect drift events to deployment timelines (your team's and other teams') so a drift spike correlates with the marketing campaign launched the same morning.
- **Action playbooks.** Drift alerts route to a runbook: "if X column drifts, suspect Y; if drift_share > threshold and prediction drift > threshold, do Z." Codifying this into the alert is what makes drift response sustainable for an on-call rotation.

---

## ✅ Phase 8 done. The system is complete.

Eight phases, ~30 commits, ~3 months of (faked-by-design) wall-clock time later, the project is end-to-end:

| Layer            | Phase | What                                                                  |
| ---------------- | ----- | --------------------------------------------------------------------- |
| **Foundation**   | 1     | Python 3.12, uv, src layout, ML-aware gitignore                       |
| **Data**         | 2     | Pandera schemas, raw-vs-processed, validation on read AND write       |
| **Reproducibility** | 3  | DVC pipeline, content-addressed deps, CSV-as-data-versioned-artifact  |
| **Modeling**     | 4     | sklearn Pipeline (encoder fitted-in), MLflow tracking, multiple metrics |
| **Selection**    | 5     | Strict-greater promotion rule, alias-based registry, decision-as-data |
| **Automation**   | 6     | GitHub Actions, three triggers, PR metric-delta gate                  |
| **Serving**      | 7     | FastAPI, Docker (multi-stage, non-root), alias as deploy unit         |
| **Closing the loop** | 8 | Evidently drift, monitor stage, the retraining-loop diagram          |

What this project deliberately *does not* have, and why:

- **No GPU / deep learning.** Tabular wins on tabular. Different portfolio project.
- **No remote MLflow server.** Local file-store works, the migration is a config change.
- **No real production deploy.** That's a different stage of work and would obscure the lessons.
- **No A/B framework or canary router.** Built the hooks (aliases, threshold env var); didn't build the routing infra.
- **No drift response automation beyond alert-only.** Discussed at length in §7 why that's the right choice for a single-developer system.

The thing this project is meant to demonstrate: **understanding what an ML system *is*, not just how to call `model.fit()`**. Every phase corresponds to a concrete failure mode that lives in production ML, and a concrete defense against it.
