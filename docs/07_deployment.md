# Phase 7 — Deployment with FastAPI and Docker

> **Goal of this tutorial:** understand what "deploying an ML model" actually means in a system that has a registry, why the registry alias is the deploy unit (not the model file), why the persisted Pipeline eliminates a class of bug at the deployment boundary, and where batch vs real-time inference each fit in real ML systems.

---

## 1. What we built

A FastAPI service (`src/deployment/api.py`) that exposes:

| Endpoint              | Method | Purpose                                                      |
| --------------------- | ------ | ------------------------------------------------------------ |
| `/health`             | GET    | Liveness probe: is a model loaded?                           |
| `/info`               | GET    | Currently-served model name, version, run id, threshold      |
| `/predict`            | POST   | Single customer record → churn probability + decision        |
| `/predict/batch`      | POST   | List of records, run as one DataFrame for efficiency         |

Plus a multi-stage `Dockerfile` and a `docker-compose.yml` with **two services** sharing one image:

- `train` runs `dvc repro` once at startup, populating a named volume with a registry whose artifact paths are *container-internal* (`/app/mlruns/...`). Exits when done.
- `api` waits for `train` to finish, mounts the same volume read-only, and serves predictions.

```bash
# First time / after model code changes:
docker compose up --build

# Hit it:
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
     -H 'content-type: application/json' \
     -d @sample.json

# Force a fresh training run (e.g. after pulling new data):
docker compose run --rm train
```

> **Why the train service exists:** MLflow's local file-store backend writes *absolute paths* into each run's `meta.yaml`. If you train on the host and bind-mount `./mlruns:/app/mlruns` into the container, the YAML still contains `/Users/<you>/.../mlruns/...` paths the container can't see. Running training inside the container makes the paths container-internal from creation. In a real deployment you'd skip the train service and point at a remote MLflow tracking server (Postgres + S3); the registry is then HTTP-accessed and paths are no longer baked into local YAML. See §14.

## 2. The deploy unit is the alias, not the model file

This is the most important idea in the phase.

A naive ML deploy looks like:

```
trained model.pkl   ─►   docker build   ─►   ECR push   ─►   k8s rolling update
```

That works, but it ties **model promotion** to **container deploys** — they happen at the same cadence. A small model improvement means a full container build and rollout. Worse, there's no audit trail: which container had which model? You'd parse image tags forever.

The pattern we built decouples them:

```
trained model   ─►   MLflow registry   ─►   alias 'production' = v3
                                                        │
                                                        ▼
                                            container restarts → loads v3
```

The container image stays the same. The **registry alias moves**. Restarting the container is what picks up the new pointer. So:

- **New model approved by Phase 5's evaluate stage?** Promote → alias flips → restart. Zero new build, zero new image.
- **New code (e.g. logging change)?** New container build, redeploy. Model alias unchanged.
- **Both?** Two independent transitions, observed independently.

This is the "decouple deploy from release" pattern from continuous-delivery applied to ML: the model artifact and the serving code have separate release cadences because they have separate failure modes.

## 3. Why no train/serve skew

Phase 3's tutorial promised this; Phase 7 cashes the check.

The fitted sklearn `Pipeline` we logged in Phase 4 contains:

```
Pipeline([
    ("preprocess", ColumnTransformer([
        ("ohe", OneHotEncoder(<fitted vocabulary from training data>), [...]),
        ("scale", StandardScaler(<fitted mean/var from training data>), [...]),
        ("bool_pass", "passthrough", [...]),
    ])),
    ("model", LogisticRegression(<fitted weights>)),
])
```

When the API does `mlflow.sklearn.load_model(...)`, it gets that *exact* object — same fitted vocabulary, same fitted scalers, same fitted weights. The encoder at training time and the encoder at serving time are **literally the same Python instance, deserialized from the same pickle**.

If we'd one-hot-encoded outside the Pipeline (in `featurize.py`, for example), the API would have to *reimplement* the encoding — same column order, same handling of unseen values, same dtype quirks. Get one detail wrong and predictions silently drift from training.

This is what "no translation layer at the API boundary" means: the API receives raw fields, runs `clean()` and `build_features()` (the same functions used at training time), passes the result straight to the loaded Pipeline. Single source of truth, single code path.

## 4. Loading the model at startup vs per-request

Three options and when each is right:

| Strategy                | Cold-start | Per-request latency | Memory | When right                                 |
| ----------------------- | ---------- | ------------------- | ------ | ------------------------------------------ |
| **Startup** (we use)    | seconds    | fast                | constant | Model is small + traffic is steady         |
| Per-request             | none       | seconds             | per-request | Lambda-style, infrequent traffic        |
| LRU cache               | seconds for first request | warm: fast | bounded | Many models, only some hot                |
| Lazy on first request   | none       | first slow          | constant | Long-lived service; cold start tolerated |

Our churn model is ~kilobytes and traffic is steady → load once at startup, reuse. The `lifespan` async context manager in FastAPI is the right place: it runs after the app is built but before the first request, so the model is in `app.state` when `/predict` runs.

### Why lifespan, not module-level

Module-level loading runs at *import time*, which means it runs in tests too — wherever you import `api`, the model loads. The fix in test_api.py would be to mock or patch before import; pulling load into `lifespan` lets tests use a fresh tmp registry instead.

## 5. Why one uvicorn worker + replicas

The Dockerfile launches `uvicorn` with no `--workers` flag → single worker process → single in-memory model.

Two reasons not to crank up workers per process:

1. **Memory.** Each worker gets its own copy of the model. With a 1 KB sklearn model that's irrelevant; with a 10 GB transformer, four workers becomes 40 GB.
2. **Concurrency model.** Multiple workers in one process complicate logging, profiling, restarts. Multiple containers each with one worker is the simpler operational model — Kubernetes already does the orchestration.

Scaling out is "more replicas," not "more workers per replica." This is the same lesson Go services learned a decade ago — small, single-purpose processes scale better than fat multi-tenant ones.

(For very low-throughput services where the cold-start cost dominates, you'd flip to multiple workers per container so a single replica can serve more concurrent requests. Default is one — change it deliberately.)

## 6. Batch vs real-time inference

Three patterns, all common in production:

| Pattern                                         | When it fits                                                             |
| ----------------------------------------------- | ------------------------------------------------------------------------ |
| **Real-time** (HTTP per request)                | Need decision in milliseconds (fraud check, rec on page load)            |
| **Micro-batch** (many requests, one DataFrame)  | API receives bursts; group within a few-ms window                        |
| **Offline batch** (cron over the whole table)   | Daily/weekly reports, marketing campaigns, anything not user-facing      |

We built (1) and (2). Phase 8 will use (3) — drift detection over the recent feature batch.

The crucial detail in (2): `_predict_records` runs a single `predict_proba` call over a multi-row DataFrame. sklearn vectorizes; per-row prediction would be N times slower. Real serving stacks layer this with a request batcher (group requests for ~5 ms, then run as one batch) — a common 5-10x throughput win.

## 7. Why a non-root container user

The Dockerfile creates `appuser` (uid 10001) and runs the service as that user. Three reasons:

1. **Defense in depth.** A vulnerability in the service or any dependency that gives RCE is bounded by what `appuser` can do, not what root can do.
2. **Kubernetes security policies.** Most prod clusters block containers running as root via `PodSecurityPolicy` / `SecurityContext` rules. Building root-by-default is a known-bad anti-pattern.
3. **Filesystem mistakes.** Files written by the container land owned by `appuser`, not root, which makes mount-point cleanup sane.

uid 10001 is high enough not to collide with system uids on the host but low enough to be representable everywhere. Some teams pick 65532 (Google distroless convention) — either is fine.

## 8. The settings layer (pydantic-settings)

`src/deployment/settings.py` reads four knobs from the environment with the prefix `CHURN_API_`:

```
CHURN_API_MODEL_NAME           # default: "churn-model"
CHURN_API_MODEL_ALIAS          # default: "production"
CHURN_API_MLFLOW_TRACKING_URI  # default: file://./mlruns
CHURN_API_THRESHOLD            # default: 0.5
```

This is the *same* image in dev / staging / prod. You don't bake env-specific config into the image; you set env vars at deploy time. Twelve-factor app pattern, applied to ML serving.

Threshold is the most useful one: as Phase 5's tutorial discussed, the right threshold is a business decision (cost-of-FP vs cost-of-FN), and you may want different thresholds in staging (more aggressive, catches more) and prod (calibrated). One env var change, no code change.

## 9. How this maps to your existing experience

- **AWS Lambda.** A FastAPI handler is conceptually a Lambda function: receive event, do work, return result. The two big differences for ML: Lambda's per-invocation cold start kills you for big models, and Lambda's 15-min limit is fine for inference but bad for training. We picked a long-lived container.
- **Express.js + middleware.** FastAPI's dependency injection (route handlers receive validated pydantic models) is structurally identical to Express's route handlers receiving parsed bodies — pydantic just adds validation that mirrors the Telco schema.
- **Go HTTP server.** Single uvicorn worker + replicas is the same pattern as a Go service in a container — the GOMAXPROCS knob is one-per-container.
- **Twelve-factor.** Env-driven config + stateless service + ephemeral filesystem + scale by adding processes — all of those map directly to what we built.

## 10. Real-world analogy

> A model registry is a **library catalog**. A serving container is a **librarian**.
>
> The catalog (registry) holds every book (model version) ever published, with metadata: author, edition, location. Each librarian (serving instance) checks the catalog at the start of their shift to find out which edition is the "currently recommended" one (the `production` alias), grabs that book off the shelf, and reads from it for their whole shift.
>
> Promoting a new edition is editing the catalog ("the recommended edition is now v3"). Existing librarians don't know yet — they're still reading from v2. Restarting a librarian is what makes them re-check the catalog and pick up v3. Older books stay on the shelf for audit / rollback.
>
> The container (librarian) is generic. The book (model) is what changes. Conflating them — putting the book inside the librarian — means hiring a new librarian every time you publish a new edition. Most ML systems make that mistake. The registry pattern is the fix.

## 11. Exercises

1. **Watch a model swap without a redeploy.** Start the service. In another terminal, deliberately worse-train a model, then *manually* set the `production` alias to it via the MLflow UI. Without restarting the service, hit `/predict`. Did the prediction change? (It shouldn't — the loaded model is in memory.) Now restart with `docker compose restart api`. Does it change now?
2. **Trace the latency.** Time a `/predict` call: `time curl -X POST .../predict -d @sample.json`. Where does the time go? Print timings inside `_predict_records` for: pydantic validation, `clean()`, `build_features()`, `predict_proba()`. Which is the bottleneck on a single record? On a batch of 100?
3. **Add a `/reload` endpoint.** Sometimes you want to swap models without a container restart (canary tests, debugging). Add a `POST /reload` that re-runs `load_production_model()` and replaces `app.state.model`. What goes wrong if you call it during in-flight requests? (Hint: race conditions in async code.)
4. **Hit the size limit.** Send a `/predict/batch` with 10,000 records. What happens? Where's the bottleneck — pydantic, pandas, sklearn, JSON serialization? Add a request-size limit + a response-size limit.
5. **Make the threshold per-request.** The default threshold (0.5) is the operational threshold. Add an optional `threshold` query param to `/predict` so callers can experiment. Should it override `CHURN_API_THRESHOLD` per request, or be additive ("show me the prediction at *this* threshold and also at the default")?

## 12. How to explain this in an interview

**60-second pitch:**

> "Phase 7 deploys the model as a FastAPI service in Docker. The key idea is that the deploy unit is the registry alias — `production` — not the model artifact. The container loads whatever the alias points to at startup; promoting a new model in Phase 5 just moves the alias, and the next container restart picks it up. No image rebuild needed for model swaps. The persisted sklearn Pipeline includes its fitted encoder, so preprocessing at serving time is literally the same Python objects as at training time — eliminates train/serve skew by construction. The container runs as a non-root user with one uvicorn worker; scale out via replicas, not workers-per-process."

**If they ask "why not a Lambda?":**

> "Cold start dominates for anything heavier than tiny sklearn models, and Lambda's 15-minute timeout limits some workflows. For real-time inference at meaningful throughput, a long-lived container is the standard. Lambda still makes sense for pure batch — a daily scoring job over a table fits Lambda's 15-min window cleanly. We'd use both: container for real-time, Lambda for the batch slice."

**If they ask "how do you avoid train/serve skew?":**

> "By making encoding part of the model artifact. The sklearn Pipeline at training time is the same Pipeline served at inference, deserialized from the same pickle. There's no translation layer at the API boundary — the API takes raw Telco fields and the same `clean()` + `build_features()` functions used in training transform them. Single source of truth for preprocessing."

## 13. Common mistakes

- **Embedding the model in the container.** Couples model deploys to image deploys. The registry pattern decouples them — model promotion happens via alias flip, not image build.
- **Per-request model loading.** Adds seconds to every call. Load at startup, reuse the in-memory object.
- **One worker process serving many models.** Sounds DRY, isn't. Memory grows linearly with models; debugging gets hard. One model per container; scale via replicas.
- **Re-implementing encoding in the API.** The classic train/serve skew bug. Use the persisted Pipeline.
- **Running as root.** Common, dangerous, blocked by most production clusters. uid 10000+ is the standard.
- **Hardcoding `localhost:5000` for MLflow.** Fine in dev, wrong in prod. Use env vars (we did).
- **Forgetting to set a healthcheck.** Without `model_loaded`-aware health check, an orchestrator routes traffic to unconfigured containers and you get 503 storms.
- **Multiple uvicorn workers when the model is large.** Quadruples memory for no real gain on most ML workloads.
- **Bind-mounting a host-trained `mlruns/` into a container.** MLflow's local file-store records absolute host paths in `meta.yaml`. The container can't resolve them and starts in `degraded` mode. Either train inside the container (the train service in our compose file does this) or use a remote tracking server.

## 14. What changes at scale

- **Remote registry.** Local file store → an MLflow tracking server backed by Postgres + S3 (or the Databricks/SageMaker managed equivalent). The container's tracking URI changes; nothing else does.
- **Auto-scaling on load.** Kubernetes HPA on QPS or CPU; serverless (Cloud Run, Lambda) for spiky traffic. The single-worker-per-container choice plays well with both.
- **Request batcher.** A small async layer that groups requests within a 5-10 ms window before calling the model. Often 5-10x throughput at small extra latency.
- **Shadow / canary deployment.** Two aliases (`production`, `canary`) and a routing layer that sends 1% of traffic to canary, compares predictions, decides whether to promote. The container code doesn't change — the routing does.
- **Multi-region.** Each region has its own registry replica (or reads from a central one with CDN-cached artifacts). Aliases are region-scoped (`production-eu`, `production-us`) so canaries can be region-by-region.
- **Per-request feature lookup.** A feature store (Feast, Tecton) provides features at request time so the API only needs the customer ID, not the full record. We didn't build this — `customerID` is a passthrough field — but the door is open.
- **Lower-latency model formats.** ONNX, TensorRT, or torch-script for cases where pure Python sklearn isn't fast enough. The registry can hold the same model in multiple flavors; the serving code picks the right one.

---

## ✅ Phase 7 done. What's next.

Phase 8 will:
- Use **Evidently AI** to measure data drift between training data and recent inference inputs.
- Add a `monitor` stage that produces an HTML drift report and a JSON drift summary.
- Define a drift-triggered retrain rule: if drift exceeds threshold X, the next scheduled trigger gets fast-tracked.
- Teach: data drift vs concept drift, why models silently degrade in production, what statistical tests catch each kind, and how the monitor → retrain feedback loop closes the system.

Wait for the user's go-ahead. (After Phase 8 the system is end-to-end complete.)
