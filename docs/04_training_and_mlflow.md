# Phase 4 — Training & Experiment Tracking with MLflow

> **Goal of this tutorial:** understand why offline accuracy is the most misleading number in ML, what each metric actually tells you, how MLflow turns "I trained a model" into "I trained model run *4cbba60c* with these params, these metrics, these artifacts, and you can reproduce it tomorrow," and why the result we got — logreg beating RF on F1 despite RF having higher accuracy — is the canonical teaching example for imbalanced classification.

---

## 1. The accuracy trap

Our test set has ~83% non-churners and ~17% churners. A model that ignores all the inputs and **always predicts "no churn"** would score:

- accuracy = **83%**
- precision (on the positive class) = undefined (0/0)
- recall = **0%**
- F1 = 0
- PR-AUC ≈ baseline (~17%)

83% accuracy. Looks fine in a slide. Captures literally zero churners. Costs the company every retention opportunity.

This is why **"my model is X% accurate"** is the most dangerous number in ML. On any imbalanced problem, it tells you almost nothing about whether the model is useful.

### What our two models actually did

This commit's `metrics/train_metrics.json` (you can `cat` it):

| Model           | accuracy | precision | recall | F1     | PR-AUC | ROC-AUC |
| --------------- | -------- | --------- | ------ | ------ | ------ | ------- |
| logistic regression  | 0.643    | 0.287     | 0.761  | **0.416** | **0.389**  | 0.759   |
| random forest        | 0.828    | 0.458     | 0.164  | 0.242  | 0.350  | 0.707   |

**Random forest has higher accuracy. Logistic regression is the better model.**

Why? RF is being conservative. It only flags 16% of the actual churners (recall = 0.16) — it misses most of them. When it does flag, it's right ~46% of the time (precision = 0.46). LogReg flags more aggressively: catches 76% of churners (recall = 0.76) at the cost of more false positives (precision = 0.29). For churn — where missing a churner means losing revenue and chasing a false positive means a cheap retention email — that's the right tradeoff.

This is why we look at **F1** (harmonic mean of precision and recall) and **PR-AUC** (precision-recall area, threshold-independent). Both rank logreg above RF. Both ignore the fact that RF is "more accurate."

## 2. The metric zoo, briefly

You will be asked about these in interviews. Here's the version that actually clicks.

For a binary classifier with positive class = "will churn":

| Term       | What it answers                                                         | When you care most                                |
| ---------- | ----------------------------------------------------------------------- | ------------------------------------------------- |
| Accuracy   | Of all predictions, what fraction were right?                           | Balanced classes; rare in production              |
| Precision  | Of the customers I flagged as churners, what fraction actually churned? | False positives are expensive (e.g. spam filters) |
| Recall     | Of the customers who *did* churn, what fraction did I flag?             | False negatives are expensive (e.g. cancer screening, churn) |
| F1         | Harmonic mean of precision and recall                                   | Single number when you care about both            |
| ROC-AUC    | How well the model *ranks* positives above negatives, all thresholds    | Threshold-independent quality of probability scores |
| PR-AUC     | Like ROC-AUC but precision–recall, more honest on imbalance             | **The right "summary number" on imbalanced data** |

**ROC-AUC vs PR-AUC** is a frequent interview gotcha. ROC-AUC averages over all thresholds, but on heavily imbalanced data the "easy" thresholds where almost everything is negative dominate the area, and you get inflated scores. PR-AUC focuses on the regime where positives matter and is the more diagnostic metric for rare-positive problems like churn.

> If you remember one thing: **never quote accuracy alone on imbalanced data.** Always quote F1 or PR-AUC together with the class balance.

## 3. Why we used `class_weight="balanced"`

The two models in `pipeline.py` both pass `class_weight="balanced"`. This tells the loss function to weight a misclassified positive ~5× more than a misclassified negative (because positives are ~17% of the data, so weight = 1 / class_freq).

Alternatives:

- **Threshold tuning.** Train normally, then pick a probability threshold that achieves the precision/recall mix you want. We'll do this in Phase 5.
- **Resampling.** Oversample positives (SMOTE) or undersample negatives. Adds complexity and can introduce its own biases.
- **Custom loss / focal loss.** Useful for deep nets; overkill for sklearn.

`class_weight="balanced"` is the cheapest reasonable starting point. If it's not enough, we'd layer threshold tuning on top.

## 4. What MLflow actually is

MLflow is **four loosely-coupled components** wearing one logo:

```
┌────────────────────────────────────────────────────────────────────┐
│  MLflow                                                             │
├──────────────┬──────────────┬───────────────┬──────────────────────┤
│ Tracking     │ Models       │ Registry      │ Projects (we skip)   │
│ (params,     │ (uniform     │ (versioned    │ (CLI to run an       │
│  metrics,    │  serialization│  model        │  experiment from a  │
│  artifacts)  │  format)     │  store with   │  GitHub URL)         │
│              │              │  stages)      │                      │
└──────────────┴──────────────┴───────────────┴──────────────────────┘
```

- **Tracking** — what we used in `train_one()`. `mlflow.log_param`, `mlflow.log_metric`, `mlflow.log_dict`, `mlflow.sklearn.log_model`. Each `with mlflow.start_run()` is one experiment row.
- **Models** — the standardized format `mlflow.sklearn.log_model` writes. A directory with `MLmodel` (metadata), `model.pkl`, `requirements.txt`, and an `input_example.json`. Any framework that knows MLflow can load it.
- **Registry** — a versioned store on top of Models with stages (`None` → `Staging` → `Production` → `Archived`). Phase 5 will promote the best run's model into the registry.
- **Projects** — a way to package a training run as a reproducible CLI. We don't need it because **DVC already does this for us** — `dvc repro train` is our reproducible run command.

### The MLflow tracking store

Default backend: a local `mlruns/` directory with one folder per experiment, one folder per run. Each run holds: params (text files), metrics (CSV), tags (text files), and artifacts (the directory MLflow's UI calls "Artifacts").

You can run `uv run mlflow ui --backend-store-uri ./mlruns` and visit http://localhost:5000 to browse, compare, and download artifacts. Try it.

For multi-developer teams: switch to a remote store (Postgres + S3) by setting `MLFLOW_TRACKING_URI=postgresql://...`. The code doesn't change.

## 5. What we logged per run (and why)

Look at the `train_one()` function. Each run gets:

| Logged                        | Why                                                                                  |
| ----------------------------- | ------------------------------------------------------------------------------------ |
| `model_kind`                  | Lets you filter the experiment table by model family later                           |
| `random_state`, `n_train`, `n_test` | Reproducibility metadata — necessary to rerun                                  |
| `test_positive_rate`          | Detects accidental data drift between runs (different ingest seed, different mix)    |
| All 6 metrics                 | Comparing on a single number is a footgun; six metrics show the *shape* of quality   |
| `feature_importances.json`    | Explainability — what is the model actually leaning on?                              |
| The fitted Pipeline           | The whole point — serving needs *this* object, with *these* fitted encoders         |
| `input_example`               | MLflow infers the model signature from this; serving uses the signature for validation |

The Pipeline artifact is the highest-leverage piece. Because the encoder is *inside* the Pipeline, when Phase 7's FastAPI service does `mlflow.sklearn.load_model(...)` and calls `.predict()`, it uses the **same** OneHotEncoder instance that fit on training data. Train/serve skew dies on day one.

## 6. Why two metrics files, not one

We log metrics to two places:

1. **MLflow** — the long log of every run, with full artifacts. Browseable in the UI.
2. **`metrics/train_metrics.json`** (DVC-tracked, `cache: false`, committed to git) — a small summary of the latest run set.

Why both? **They serve different audiences.**

- **MLflow** is for *the data scientist* who wants to compare 50 experiments to find the best one, dig into hyperparameters, look at curves.
- **`train_metrics.json` in git** is for *the CI system* (Phase 6) that needs to gate a merge: "did the F1 on this PR beat main? if not, fail the build." It's also for *the reviewer* who wants to see metric evolution in the diff of a PR without standing up MLflow.

This is the standard pattern in production. The detailed lab notebook lives in MLflow. The shippable scorecard lives in git.

## 7. Hyperparameter tuning (preview)

We didn't tune anything. Both models use sensible defaults. That's deliberate — you should always start with a baseline that has no hyperparameter tuning so you can attribute later improvements to the tuning, not to other changes.

When we tune (likely Phase 5), the natural next steps:

- **Grid search** over `LogisticRegression(C=...)` and `RandomForest(max_depth=..., n_estimators=...)`.
- Each grid cell becomes its own MLflow run — already supported, no extra plumbing.
- For more dimensions: **Optuna** or `sklearn.model_selection.HalvingGridSearchCV`.

The discipline: each experiment is one run, every run is in MLflow, the comparison is one query.

## 8. How this maps to your existing experience

- **Cloudwatch metrics for a Lambda.** MLflow tracking is "Cloudwatch metrics, but per training run." Each run is a small immutable record of what happened.
- **Datadog dashboards.** The MLflow UI is structurally similar — filter by tag, compare runs, look at history.
- **Docker image registry.** The MLflow Model Registry is to models what ECR/GCR is to Docker images: a versioned, deployable artifact store with stages.
- **Go pkg.go.dev for a published library.** `mlflow.sklearn.log_model` writes a self-describing artifact: `MLmodel` is the equivalent of `package.json` — declares deps and entry points so anything can pick the artifact up.

## 9. Real-world analogy

> Training a model without experiment tracking is like running a test suite that prints "✅" to stdout, then closing the terminal.
>
> Sure, the test passed *that time*. But six weeks later you want to know whether the test was actually run on a release candidate, on what code revision, with what dependencies, on what hardware, and whether it produced the same artifact you eventually shipped. Without a recorded run, you have no answers.
>
> MLflow is the **CI report** for ML training. Every run is recorded with its inputs, outputs, environment, and artifacts — so the question "is this model better than last week's?" becomes a database query rather than a conversation.

## 10. Exercises

1. **Open the MLflow UI.** `uv run mlflow ui --backend-store-uri ./mlruns`. Browse to http://localhost:5000. Find the two runs from `dvc repro`. Pick a metric (say PR-AUC) and click the column to sort. Click into one run and look at the artifacts.
2. **Promote logreg as the "current best."** In the MLflow UI, register the logreg run's model as `churn-model`. Set its stage to `Staging`. Phase 5 will codify this promotion programmatically.
3. **Beat the baseline.** Open `src/training/pipeline.py`. Add a `gradient_boosting` kind using `sklearn.ensemble.HistGradientBoostingClassifier`. Re-run `dvc repro train`. Did it beat logreg on F1? On PR-AUC?
4. **Find the best threshold.** For the logreg run, recall = 0.76 at the default 0.5 threshold. Open a Python REPL, load the model, sweep the threshold from 0.1 to 0.9 and compute precision and recall at each. Which threshold gives the best F1? What about the best precision at recall ≥ 0.5?
5. **Spot the silent regression.** Edit `src/training/pipeline.py` to remove `class_weight="balanced"` from the logreg branch. Re-run `dvc repro train`. What changes in `metrics/train_metrics.json`? Which metric reveals the regression most clearly? What does this tell you about which metric a CI gate should use in Phase 6?

## 11. How to explain this in an interview

**60-second pitch:**

> "Phase 4 trains both a logistic regression and a random forest, wraps each in a sklearn Pipeline so the encoder is part of the persisted model, and logs everything to MLflow — params, six metrics, the fitted Pipeline, feature importances. The interesting result is that random forest had higher accuracy but logreg won on F1 and PR-AUC. On imbalanced data accuracy is misleading because a model that always predicts the majority class is already 83% accurate. We use class_weight='balanced' to compensate, and we'll add threshold tuning in Phase 5. Two metrics destinations: full history in MLflow for analysis, a small JSON in git for CI gating."

**If they ask "why MLflow over Weights & Biases / Comet / Neptune?":**

> "MLflow is open-source, runs locally with no infra, and is the de-facto standard. W&B and Comet are SaaS — better UI but a vendor lock-in and another bill. Neptune sits in between. For a portfolio project I want zero infra and full control; MLflow is the obvious pick. The API is similar enough across all four that switching is a day of work."

**If they ask "how do you prevent train/serve skew?":**

> "By making encoding part of the model artifact, not part of the feature pipeline. The sklearn Pipeline contains the fitted ColumnTransformer, so when serving loads the model with `mlflow.sklearn.load_model`, it gets the encoder with the exact same fitted state. Train-time and serve-time encoding are literally the same Python object. The class of bug where serving uses different encoding from training is impossible by construction."

## 12. Common mistakes

- **Optimizing accuracy.** On imbalanced data, this rewards models that ignore the minority class. Always check class balance before picking a metric.
- **One metric only.** Six metrics tell you the *shape* of model quality. One metric collapses that shape into a number that always lies a little.
- **Encoding outside the model.** Discussed at length in docs/03 — leads to train/serve skew. Always inside the Pipeline.
- **Not logging the model artifact.** MLflow runs without artifacts are just "I trained a model with score X." You need the artifact to actually deploy or compare.
- **Treating ROC-AUC as truth on imbalanced data.** It looks high on imbalanced classes even when the model is mediocre. PR-AUC is the more honest summary.
- **Reusing `random_state=None`.** Makes runs un-reproducible. We pin `random_state=42` everywhere a model touches randomness.
- **`mlflow.sklearn.autolog()` on its own.** It logs everything, including a *lot* of noise (every sklearn parameter, every internal call). Combine with explicit `log_param` / `log_metric` for the things that matter.

## 13. What changes at scale

- **Remote tracking store.** Local `mlruns/` is fine for one developer. With a team you'd run `mlflow server` against Postgres + S3, and set `MLFLOW_TRACKING_URI` in everyone's environment.
- **Model registry as the source of truth.** Beyond a hundred models, the Registry replaces ad-hoc "the latest model is at /opt/models/v3.pkl" conventions. Each model has a name, versions, stage transitions with audit trail.
- **Distributed training.** The Pipeline construct stays; the estimator under it changes (XGBoost on Dask, PyTorch with DDP, etc.). MLflow's `log_model` has flavors for most.
- **Hyperparameter tuning at scale.** Optuna or Ray Tune launches hundreds of MLflow runs in parallel. The bottleneck shifts from compute to *organizing* runs — tags and parent–child relationships in MLflow handle this.
- **Lineage.** "Which dataset did this model train on?" needs an answer. Tools like Marquez/OpenLineage track inputs and outputs across pipeline runs, complementing MLflow's per-run metadata.
- **Approval workflows.** Promotion from Staging → Production becomes a tracked human action, often gated by automated checks. Phase 5 will introduce the *automatic* check; production environments add the human one on top.

---

## ✅ Phase 4 done. What's next.

Phase 5 will:
- Codify the promotion rule: a new model only deploys if it beats the current production model on the chosen metric (F1 or PR-AUC).
- Register the winning run's model in the MLflow Model Registry with a `Production` stage.
- Add an `evaluate` stage to `dvc.yaml` that compares the just-trained model against the registered Production model and writes a verdict.
- Teach: model selection strategy, the asymmetry of false-positives vs false-negatives, why automatic promotion is risky and what guardrails are essential.

Wait for the user's go-ahead.
