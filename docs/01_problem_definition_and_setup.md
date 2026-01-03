# Phase 1 — Problem Definition & Setup

> **Goal of this tutorial:** before writing a single line of ML code, understand *what MLOps is*, *what problem we are solving*, and *why we set the project up this particular way*.

---

## 1. What is MLOps, really?

**MLOps = the engineering discipline of running ML systems in production.**

That sounds vague, so here is the sharper version:

> A normal software system has *one* moving part that decides its behavior: **code**.
> A machine-learning system has *three* moving parts: **code, data, and the trained model**.
> MLOps exists because all three can change independently, and any of them can break production.

If you forget every other definition, remember that one. It explains why MLOps tooling looks the way it does.

### The three things that change

| What changes  | Example                                    | What can break                                |
| ------------- | ------------------------------------------ | --------------------------------------------- |
| **Code**      | new feature engineering logic              | feature/training skew                         |
| **Data**      | upstream system changes a column meaning   | model silently degrades                       |
| **Model**     | retraining produces worse model than prod  | regression in user-facing metrics             |

A production ML system needs to track, version, and react to **all three**.

## 2. ML vs MLOps vs DevOps — framed for a backend engineer

You already know DevOps from your Go/AWS Lambda work. Let's translate.

| Concern               | DevOps (your Go service)              | MLOps (this project)                                                |
| --------------------- | ------------------------------------- | ------------------------------------------------------------------- |
| Source of truth       | Git                                   | Git **+ DVC (data) + MLflow (experiments/models)**                  |
| Build artifact        | Docker image                          | Docker image **+ trained model file + feature pipeline**            |
| Test gate             | unit + integration tests pass         | tests pass **+ new model beats current model on offline metrics**   |
| Deploy unit           | service binary                        | service binary **+ model artifact** (versioned together)            |
| Failure mode          | crash, 500s, latency spikes           | silent quality decay (predictions get worse, no error thrown)       |
| Detection             | logs, metrics, traces                 | logs, metrics, traces **+ data drift + prediction drift**           |
| Rollback              | redeploy previous image               | redeploy image **and roll back to previous model in registry**      |
| Trigger to redeploy   | code merge                            | code merge **OR** new data **OR** scheduled retrain                 |

The shaded right-hand cells are precisely what makes MLOps a separate discipline. Everything Phase 2–8 builds is one of those right-hand items.

### Where ML and MLOps differ

- **ML** = *model.fit, model.predict.* The math. The notebook. Done in hours.
- **MLOps** = the rest of the iceberg. Data pipelines, validation, versioning, retraining, serving, monitoring, governance. Done forever.

A useful rule of thumb from Google's "Hidden Technical Debt in Machine Learning Systems" paper:
> The ML model is **5%** of a real ML system. The other 95% is plumbing.

This project is about that 95%.

## 3. The full ML lifecycle (where each phase fits)

```
        ┌────────────────────────────────────────────────────────────────┐
        │                                                                │
        ▼                                                                │
   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌──────────┐    ┌────────┴───────┐
   │  Data   │───▶│ Feature │───▶│ Train + │───▶│ Evaluate │───▶│   Deploy &     │
   │ Ingest  │    │  Eng.   │    │ Tracking│    │ + Select │    │     Serve      │
   └─────────┘    └─────────┘    └─────────┘    └──────────┘    └────────┬───────┘
     Phase 2       Phase 3        Phase 4         Phase 5                │
                                                                         ▼
                                                                  ┌─────────────┐
                                                                  │  Monitor +  │
                                                                  │  Drift Det. │
                                                                  └──────┬──────┘
                                                                         │ Phase 8
                                                                         │
                                              (drift / new data /        │
                                               schedule / code change) ──┘
                                                   ─── Phase 6: CI/CD ───
```

This whole picture **must be a closed loop**, not a one-way pipeline. The arrow from Monitor back to Ingest is what makes it MLOps.

## 4. The use case — customer churn prediction

**Problem:** A subscription business (think SaaS, telecom, streaming) loses customers every month. Acquiring a new customer costs ~5–10× more than retaining an existing one. If we can predict *which customers are about to churn*, we can intervene (discount, outreach, support) before they leave.

### Formal problem definition

- **Task:** Binary classification.
- **Input:** A row per customer with usage, billing, demographic, and tenure features.
- **Output:** `P(customer will churn in next 30 days)`, a probability in `[0, 1]`.
- **Decision:** Above some threshold (tuned to business cost of false positives vs false negatives), trigger a retention action.

### Why churn is a good first MLOps use case

1. **Tabular data.** No GPU pain. `sklearn` handles it. Lets us focus on *the system*, not the model.
2. **Class imbalance is realistic.** Churn rates are usually 2–10% — forces us to think about precision/recall, not raw accuracy. (Phase 5.)
3. **Has a true feedback loop.** 30 days after prediction, you know whether the customer actually churned. That's exactly the data we'll feed back in Phase 8.
4. **Concept drift is real.** A pricing change, a new competitor, a pandemic — any of these silently shifts what "about to churn" means. Perfect for the drift work in Phase 8.
5. **Easy to fake plausible data.** We'll use a public-style synthetic dataset in Phase 2 — no licensing headache.

### Success metric (preliminary)

Final metric will be picked in Phase 5, but plan to optimize for **F1 on the positive (churn) class**, with **PR-AUC** as a tiebreaker. Reasoning: the business cares about *catching churners* (recall) without flooding the retention team with false alarms (precision). We'll revisit.

---

## 5. What we built in Phase 1

```
churn-mlops/
├── .gitignore             # Python + ML-aware ignores (mlruns, .dvc/cache, data, models)
├── pyproject.toml         # uv-managed; pinned Python 3.12; minimal deps for now
├── uv.lock                # exact dep versions — committed for reproducibility
├── README.md              # project vision + 8-phase roadmap
├── data/{raw,processed,external}/
├── src/{ingestion,features,training,evaluation,deployment,monitoring}/
├── pipelines/             # will hold dvc.yaml from Phase 3
├── models/                # local model artifacts (gitignored)
├── docs/                  # this file lives here
├── tests/
└── .github/workflows/     # GitHub Actions definitions land here in Phase 6
```

### Why each piece exists

- **`pyproject.toml` (not `requirements.txt`).** `pyproject.toml` is the modern Python standard (PEP 621). It separates runtime deps (`dependencies`) from dev deps (`[dependency-groups].dev`). It's what every serious Python project uses now.
- **`uv` (not pip / poetry / conda).** uv is a Rust-based package manager: ~10–100× faster than pip, manages Python versions itself, produces a deterministic lockfile. You'll see this stack at modern AI/ML companies.
- **`uv.lock` is committed.** This is the equivalent of `go.sum` or `package-lock.json`. It pins exact versions of every transitive dependency. Without it, "works on my machine" returns with a vengeance — and ML reproducibility dies.
- **The `src/` layout (vs flat layout).** Putting code under `src/` prevents accidental imports from the project root and forces you to install your own package (`pip install -e .` / `uv sync`). It catches packaging bugs before deploy.
- **One subpackage per pipeline stage.** `ingestion`, `features`, `training`, etc. — each is small, replaceable, and matches one stage of the lifecycle. This will map 1:1 to DVC stages in Phase 3.
- **`.gitignore` blacklists `models/` and `data/raw/*`.** Models are large reproducible artifacts; data is huge and sometimes sensitive. Neither belongs in git. The **registry** (MLflow, Phase 4) is the source of truth for models; **DVC** (Phase 3) is the source of truth for data. We just keep `.gitkeep` placeholders so git tracks the empty folders.

## 6. Why this approach (and not the others)

### Why monorepo (everything in one repo)?

For a portfolio project: simplicity wins. At a real company you'd often split:

- A repo for the **training pipeline** (this codebase).
- A separate repo for the **inference service** (deployed on a different cadence).
- A separate repo for **shared feature definitions** (a feature store).

The monorepo is the right starting point. Splitting can come later when teams diverge.

### Why no orchestrator (Prefect / Airflow / Dagster) yet?

You already do orchestration with **n8n** and **AWS Step Functions** — you know what an orchestrator buys you (retries, scheduling, observability, dependency graphs). For an early MLOps project, an orchestrator hides the actual pipeline behind UI and YAML. We'll build the bare pipeline first, *then* see what orchestration adds. This is deliberate.

### Why not start with deep learning / PyTorch?

Two reasons:
1. **Shape of system, not size of model.** Everything you learn here — versioning, tracking, drift, promotion, CI/CD — applies identically to a PyTorch system. The model is swappable. The system isn't.
2. **Tabular wins on tabular.** Gradient-boosted trees and well-tuned logistic regression beat neural nets on most tabular problems. PyTorch would be ceremony without payoff for *this* problem.

A natural next portfolio project is a deep-learning one (e.g. fine-tuning a small LLM, image classification with PyTorch + LoRA). That's where you close the PyTorch gap. For now: don't mix learning goals.

## 7. Tradeoffs we are accepting

| Decision                            | Tradeoff (cost)                                                   | Why we accept it                                          |
| ----------------------------------- | ----------------------------------------------------------------- | --------------------------------------------------------- |
| Local-only first (no cloud)         | Won't catch cloud-specific issues (IAM, regions, network)         | Lower friction; you already know AWS; can lift later      |
| sklearn over XGBoost / LightGBM     | A bit lower out-of-box accuracy                                   | sklearn API is the lingua franca; less to learn at once   |
| Synthetic data                      | Doesn't replicate full real-world messiness                       | Avoids legal/PII issues; reproducible by anyone           |
| MLflow local file backend           | Single user, no team UI server                                    | Zero infra; same API as remote MLflow                     |
| One repo, no service split          | Less realistic of how big teams structure ML                      | Way easier to read end-to-end as a portfolio piece        |

## 8. Real-world analogy

> Setting up an MLOps project is like building a kitchen, not cooking a meal.
>
> If you only care about *one dinner*, you don't need a kitchen — you can cook on a campfire (a notebook). But if you want to cook **every night**, with **different ingredients**, for **different guests**, and have someone able to take over when you're sick — you need a kitchen: labelled ingredient bins (data versioning), recipes written down (code in git), a thermometer on the oven (monitoring), and a way to throw out a bad batch (model rollback).
>
> Phase 1 is buying the kitchen and putting up the shelves. We haven't cooked anything yet.

## 9. How this maps to your existing experience

- **Lambda functions** → the FastAPI inference service we'll build in Phase 7 is conceptually a Lambda handler that loads a model on cold start.
- **n8n workflows** → DVC pipelines (Phase 3) are essentially the same idea: a DAG of stages where each stage has typed inputs/outputs and reruns only when its inputs change.
- **Go interfaces** → in Phase 4 we'll define a `Model` protocol (Python's `typing.Protocol`) so the training and serving code don't depend on a specific algorithm. Same idea, same payoff.
- **GitHub Actions for Go services** → in Phase 6 we'll do the *same thing*, but the test suite includes "did the new model beat the old one?".

## 10. Exercises

1. **Define your own success metric.** If you ran a SaaS where the average customer is worth €500 and a retention discount costs €50, what does the false-positive vs false-negative tradeoff look like? Sketch it. (We'll formalize this in Phase 5.)
2. **Find the three moving parts in a system you've built.** Pick a Go/Lambda service from your past work. What plays the role of "code"? Of "data"? Is there a "model" hiding in there (e.g. a heuristic, a config table, a rules engine)? How would you version each?
3. **Inventory your reproducibility.** Could a teammate clone this repo today and reproduce the exact environment? List anything missing. (Hint: we have `uv.lock`, but Phase 3 will add data versioning.)
4. **Map the lifecycle to a system you know.** Take your favorite RAG system (since you've built one). Where do "data ingestion", "feature engineering", "training", "evaluation", "monitoring" live in a RAG context? (Some of those map weirdly — that's the point.)
5. **Spot the silent failure mode.** Imagine the upstream team renames the column `monthly_charges` to `monthly_amount` without telling you. In a normal Go service, what fails and how loudly? In an ML system trained on `monthly_charges`, what fails and how loudly? Which would you catch first?

## 11. How to explain this in an interview

**60-second pitch:**

> "I built an end-to-end MLOps system around customer churn prediction. The point wasn't the model — sklearn handles it — it was the surrounding production system: data versioning with DVC, experiment tracking with MLflow, a FastAPI inference service, drift monitoring with Evidently, and a CI/CD pipeline that automatically retrains on three different triggers and only promotes a new model if it beats the current production one on offline metrics. I treated it like building a normal backend service, except the artifact you ship is *both* the code and the model, and the system can silently degrade in ways no exception will catch — which is what most of the tooling exists to prevent."

**If they ask "what's the difference between MLOps and DevOps?":**

> "DevOps versions one thing: code. MLOps versions three: code, data, and the trained model. That's why MLOps stacks have extra moving parts — DVC for data, a model registry for models, and drift monitoring because data quality can degrade without anything throwing an error."

**If they ask "why didn't you use Airflow / Kubernetes / SageMaker?":**

> "I deliberately kept the orchestration and infra simple to make the *ML lifecycle itself* visible. Once you understand the lifecycle as plain Python + GitHub Actions, swapping in Airflow or SageMaker is mechanical. I'd rather demonstrate I understand what those tools are *replacing* than hide everything behind a managed service."

## 12. Common mistakes (that we are deliberately avoiding)

- **Starting in a notebook.** Notebooks hide order-of-execution bugs and don't translate to production. We start with packages.
- **Committing data to git.** Eventually breaks: repo bloats, secrets leak, cloning takes forever. We gitignore data and use DVC from Phase 3.
- **Committing models to git.** Same reasons. The MLflow registry is the source of truth from Phase 4 on.
- **Pinning to system Python.** System Python is owned by the OS — if the OS upgrades, your project breaks. We pinned 3.12 via `uv` so the project owns its runtime.
- **No lockfile.** Without `uv.lock`, "I can't reproduce the bug" is a coin flip. We commit it.
- **Conflating ML success with system success.** A model with 99% accuracy that nobody can deploy, monitor, or retrain has zero business value. The system is what matters.

## 13. What changes at scale

This Phase 1 setup is fine for a portfolio project and small teams. At scale (≥10 ML engineers, multi-team):

- **One repo per concern.** Training pipeline, inference service, feature definitions, infra — separate repos with semver-pinned shared libraries. Caused by team boundaries, not by code volume.
- **Monorepo build tools.** If you stay monorepo, you'd add Bazel / Pants / Nx so changes to `src/training/` don't trigger CI for `src/deployment/`.
- **A real package index.** `pyproject.toml` would publish to a private PyPI mirror so `features/` can be reused by adjacent projects.
- **Centralized environment management.** `uv` per-repo becomes a base image + lockfile inheritance to keep CUDA / Python / sklearn versions consistent across hundreds of pipelines.
- **Data contracts.** A schema registry (e.g. Protobuf, Avro, or Great Expectations + a contract repo) — so when the upstream team renames a column, *your CI fails*, not your model.

These are real problems. They are not Phase 1 problems. Don't pre-build infrastructure for a team that doesn't exist yet.

---

## ✅ Phase 1 done. What's next.

Phase 2 will:
- Add a script that ingests a public-style churn dataset.
- Validate the schema and key data-quality assertions on every load.
- Separate raw vs processed storage and explain why.
- Teach: data contracts, why "garbage in, model out" is the entire ML system in one phrase, and how broken data shows up in production.

Wait for the user to say go.
