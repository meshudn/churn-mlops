# churn-mlops

End-to-end **MLOps system for customer churn prediction** — built as a portfolio project to demonstrate the full ML production lifecycle: data → features → training → evaluation → deployment → monitoring → automatic retraining.

This is not a Jupyter notebook. It is a system you can run, retrain, deploy, and monitor.

---

## Why this project exists

Most ML tutorials stop at `model.fit()`. Real ML systems break *after* that — on data drift, on broken pipelines, on models that beat the previous one in offline metrics but tank in production. This project is the antidote: every component you'd build at a real company, in miniature.

## What this demonstrates

Each of the 8 phases targets one production-ML failure mode and the defense against it:

| Failure mode                                                         | Defense                                  | Where           |
| -------------------------------------------------------------------- | ---------------------------------------- | --------------- |
| Upstream silently changes data shape                                 | Pandera schemas on read AND write        | docs/02         |
| "Works on my machine" — irreproducible data                          | DVC content-addressed pipeline           | docs/03         |
| Train/serve skew — encoding drifts between training and inference    | Encoder lives inside the persisted Pipeline | docs/04 + 07 |
| Worse model gets shipped because accuracy went up on imbalanced data | Strict-greater promotion rule on F1      | docs/04 + 05    |
| Refactor regression — code change silently makes the model worse     | PR metric-delta gate in CI               | docs/06         |
| Concept drift — relationship between inputs and target changes       | Drift detection + alert (not auto-retrain) | docs/08       |

If you're reviewing this for a role: read **docs/01** for the lifecycle map, then any 2–3 phase tutorials whose failure mode is closest to what your team faces. Each tutorial has a "How to explain this in an interview" section with a 60-second pitch and the canonical follow-up questions.

## The 8-phase build

| #   | Phase                              | Status         | Tutorial                                           |
| --- | ---------------------------------- | -------------- | -------------------------------------------------- |
| 1   | Problem definition + setup         | ✅ done        | [docs/01](docs/01_problem_definition_and_setup.md) |
| 2   | Data ingestion + validation        | ✅ done        | [docs/02](docs/02_data_ingestion.md)               |
| 3   | Feature engineering + DVC          | ✅ done        | [docs/03](docs/03_features_and_dvc.md)             |
| 4   | Training + MLflow tracking         | ✅ done        | [docs/04](docs/04_training_and_mlflow.md)          |
| 5   | Evaluation + model selection       | ✅ done        | [docs/05](docs/05_evaluation_and_model_selection.md) |
| 6   | CI/CD with GitHub Actions          | ✅ done        | [docs/06](docs/06_cicd.md)                         |
| 7   | FastAPI deployment + Docker        | ✅ done        | [docs/07](docs/07_deployment.md)                   |
| 8   | Drift monitoring + retraining loop | ✅ done        | [docs/08](docs/08_drift_and_retraining.md)         |

## Tech stack

- **Language:** Python 3.12
- **Env / packaging:** `uv`
- **ML:** scikit-learn (start simple — no PyTorch in this project)
- **Data versioning:** DVC
- **Experiment tracking:** MLflow
- **Serving:** FastAPI + Docker
- **CI/CD:** GitHub Actions
- **Drift monitoring:** Evidently AI

No orchestrator (Prefect / Airflow) — kept intentionally out of scope to keep the system understandable. May be added later as an improvement.

## The continuous ML loop

The system supports three retraining triggers (Phase 6 onward):

1. **Code change** — model or feature code modified → retrain
2. **Data change** — new data version → retrain
3. **Scheduled** — weekly cron → retrain

A trained model **never ships blindly**. It must beat the current production model on the holdout set:

```python
if new_model.metric > production_model.metric:
    promote_to_production()
else:
    reject_and_keep_current()
```

## Project structure

```
churn-mlops/
├── data/                   # raw / processed / external (DVC-tracked from Phase 3)
├── src/
│   ├── ingestion/          # load + validate raw data
│   ├── features/           # feature engineering
│   ├── training/           # model training + tuning
│   ├── evaluation/         # metrics, comparison, promotion
│   ├── deployment/         # FastAPI inference service
│   └── monitoring/         # drift detection + retraining triggers
├── pipelines/              # DVC pipeline definitions (dvc.yaml from Phase 3)
├── models/                 # local model artifacts (gitignored — registry is source of truth)
├── docs/                   # one tutorial per phase
├── tests/
└── .github/workflows/      # CI/CD (Phase 6)
```

## Quickstart

```bash
# Install Python 3.12 if you don't have it
uv python install 3.12

# Install dependencies
uv sync

# Run tests
uv run pytest

# Lint
uv run ruff check .
```

## Learning material

Every phase ships with a tutorial under [`docs/`](docs/). Read them in order — they explain not just *what* was built but *why*, what the alternatives were, and how to talk about it in interviews.
