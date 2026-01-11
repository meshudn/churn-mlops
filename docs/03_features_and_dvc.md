# Phase 3 — Feature Engineering & Data Versioning (DVC)

> **Goal of this tutorial:** understand why ML reproducibility requires more than git, what DVC actually does (and doesn't), how to think about a data pipeline as a DAG instead of a script, and why we deliberately *don't* encode features here.

---

## 1. The reproducibility triangle

Software reproducibility means: clone the repo, run the command, get the same result. For a regular Go service that's almost free — code + dependency lock is enough.

For an ML system, the inputs to "the same result" are:

```
        ┌──────────┐
        │   Code   │   ← git
        └──────────┘
              ▲
              │
              │
   ┌──────────┴──────────┐
   │                     │
┌──┴──┐               ┌──┴──┐
│ Env │ ← uv.lock     │Data │ ← DVC (this phase)
└─────┘               └─────┘
```

If any one of these three drifts, your "reproducible" experiment isn't. Phase 1 nailed code (git) and env (uv.lock). Phase 3 nails data.

> **Mental model:** DVC is "git for data, but stored where git would choke."

## 2. Why git can't track data

You *could* `git add data/raw/churn.csv`. Three things go wrong fast:

1. **Repo size explodes.** Git stores every revision of every file, forever. A 100 MB CSV with weekly updates becomes a 5 GB repo by next year.
2. **Cloning becomes painful.** Every contributor pulls every historical version of the data, even if they only need the latest.
3. **Diffs are useless.** `git diff` on a 50,000-row CSV is unreadable noise; it tells you nothing about *what changed in the data*.
4. **Merge conflicts are catastrophic.** Two branches that both regenerated the data end up with conflicting binary blobs git cannot 3-way merge.

Git is optimized for code, which is small, line-oriented, and composable. Data is the opposite of all three.

## 3. What DVC actually does

DVC stores **content-hashes** of large files in git, and stores the actual files in a separate **cache** (`.dvc/cache/`) that's git-ignored.

When the data changes, the *hash* in git changes — a tiny, mergeable, diff-able event. The actual bytes live in the cache, optionally pushed to a remote (S3, GCS, Azure, SSH, etc.).

```
git tracks:           DVC cache holds:                 DVC remote (optional):
├── dvc.lock          ├── (hash) churn.csv             s3://my-bucket/cache/...
│   md5: ca16...      ├── (hash) churn_features.csv
└── dvc.yaml          └── (hash) ...
```

`dvc push` / `dvc pull` move bytes between cache and remote, exactly like `git push` / `git pull` move objects between local and remote git.

### What we did in Phase 3

1. **`dvc init`** — creates `.dvc/`, `.dvcignore`, and a default local cache.
2. **`dvc.yaml`** — declares the pipeline as a DAG of stages (`ingest` → `featurize`).
3. **`dvc repro`** — runs the DAG, only re-running stages whose inputs have changed (content-based, not timestamp-based).
4. **`dvc.lock`** — pinned hashes for every input and output, committed to git.

Run `uv run dvc dag` to print the current graph:

```
+--------+
| ingest |
+--------+
     *
     *
+-----------+
| featurize |
+-----------+
```

Each box is a stage. Arrows are data dependencies. Each stage has a `cmd`, a list of `deps` (input files / dirs), and a list of `outs` (files DVC will track in the cache).

## 4. Stages, content-addressing, and "smart re-runs"

The crucial DVC behavior — and the reason it's better than `make` for ML pipelines:

- DVC computes an MD5 of every dep file (or directory, recursively).
- It stores those hashes in `dvc.lock`.
- On `dvc repro`, if a dep's current hash matches the locked hash, the stage is **skipped**.

What this gets you:

- Edit `src/features/build.py` → only `featurize` re-runs (not `ingest`).
- Touch `src/ingestion/synthetic.py` → only `ingest` re-runs (which cascades into `featurize` since its dep changed).
- `git checkout` an old commit → `dvc repro` brings the data back to that commit's state by reading hashes from that commit's `dvc.lock`.

Compare to a `make`-based pipeline that uses file timestamps: a `touch` re-runs the whole thing. DVC won't.

### `dvc.lock` as the snapshot of "what the data was"

`dvc.lock` is the most important file in the entire phase. It's a YAML map of `(stage, dep_path) → md5_hash`. With it committed:

> **Any commit on this branch implies a single, recoverable data state.** Different team members can `git checkout`, `dvc pull`, and have byte-identical CSVs.

This is what real reproducibility looks like.

## 5. DVC vs the alternatives

| Tool                   | Sweet spot                                        | Why we picked / didn't                                |
| ---------------------- | ------------------------------------------------- | ------------------------------------------------------ |
| **DVC**                | code + data co-versioning, lightweight, runs anywhere | ✅ Picked: minimal infra, plays with git, declarative pipelines |
| Git-LFS                | one-off large binaries (images, PDFs) in git      | ❌ Stores *every* version forever in remote, no pipeline concept |
| S3 + manifest.json     | data in S3, paths in git, hand-written            | ⚠️ Works at small scale but you reinvent DVC poorly  |
| lakeFS                 | git-style branching for warehouse data            | ❌ Heavyweight; great for whole-warehouse use cases   |
| Pachyderm              | k8s-native data pipelines with versioning         | ❌ Way too much infra for a portfolio project         |
| Feature stores (Feast) | online + offline feature serving for many models  | ⏳ Not yet — single model, single dataset right now   |

The trap most teams fall into: "we'll just use S3 and a naming convention." It works for two months. By month six you have data files named `churn_v3_FINAL_actually_use_this_one.csv`. Adopt DVC (or equivalent) on day one.

## 6. Why encoding lives in training, not in features

The hardest design decision in Phase 3 is *what* to put in the feature CSV. Three reasonable answers:

| Option                                                | Pros                                            | Cons                                                  |
| ----------------------------------------------------- | ----------------------------------------------- | ----------------------------------------------------- |
| Raw types (what we did)                               | Reusable across models; serving-side encoding is identical to training-side | More work in the training Pipeline                    |
| One-hot encoded                                       | Training is "just `model.fit(X, y)`"            | **Train/serve skew waiting to happen** (see below)    |
| Pre-scaled, target-encoded, balanced                  | Training is trivial; experiments are fast      | Feature CSV is now coupled to *one* model's needs     |

We picked option 1 — and the reason matters.

### The train/serve skew problem

Imagine you one-hot encode `payment_method` in the feature stage. Training sees:

```
payment_method_Electronic_check, payment_method_Mailed_check, ...
```

Now serving time. A single customer arrives over an HTTP call:

```json
{ "payment_method": "Electronic check" }
```

Whoever wrote the inference service must replicate the **exact same** one-hot logic — same column order, same handling of unseen values — in a totally different code path. Get one detail wrong (column order, unseen-value default, dtype) and predictions silently drift from training.

This is **train/serve skew**. It's the #1 source of "the model worked great offline but is bad in production" failures.

### How we avoid it

Encoding lives **inside** the sklearn `Pipeline` we'll build in Phase 4:

```python
pipe = Pipeline([
    ("encode", ColumnTransformer([
        ("ohe", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ("scale", StandardScaler(), NUMERIC_COLS),
    ])),
    ("model", LogisticRegression()),
])
```

When we `joblib.dump(pipe)` and load it at serving time, the **same** `OneHotEncoder` (with the same fitted vocabulary) does encoding. Train-time and serve-time encoding are *literally the same Python objects*. Skew becomes impossible by construction.

So: feature CSV holds the "honest" types. The model artifact holds the encoder. They version separately.

## 7. Feature drift (preview for Phase 8)

Now that we have a feature dataset, we can talk about *drift*. Drift comes in two flavors:

- **Data drift:** The distribution of an input feature changes. E.g. last quarter `Fiber optic` was 44% of customers; this quarter it's 71% because marketing pushed it. Model still works mathematically, but it's now operating in a regime it hasn't seen much of in training.
- **Concept drift:** The *relationship* between features and target changes. E.g. churn used to correlate with `Month-to-month` contract; after a competitor launches with a free month, even `Two year` customers start churning when their contract expires.

Phase 8 will use **Evidently AI** to compute drift metrics on the features we just built, and trigger retraining when drift exceeds thresholds. The features we picked here are the ones that will be *monitored* there. That's why we picked features with business meaning: a drift alert on `services_count` is interpretable; an alert on `pca_component_3` is not.

## 8. How this maps to your existing experience

- **n8n DAG.** `dvc.yaml` is structurally identical to an n8n workflow: nodes (stages) with typed inputs (deps) and outputs (outs). The crucial difference is content-addressing — n8n re-runs every node every time; DVC re-runs only nodes whose deps changed.
- **AWS S3 versioning + Lambda.** `dvc push` / `dvc pull` is essentially `aws s3 sync` between a local cache and a remote bucket, with content-hashing layered on top to dedupe identical files across versions.
- **Go interfaces.** The composition pattern in `CHURN_FEATURES_SCHEMA = CHURN_PROCESSED_SCHEMA.add_columns({...})` is the same idea as embedding a struct in Go: the inner type's contract is preserved, the outer type adds capabilities.
- **`uv.lock` for code.** `dvc.lock` is exactly that, but for data. Both pin a hash of an artifact so future runs reproduce the exact bytes.

## 9. Real-world analogy

> If git is the **filing cabinet** for your company's documents, DVC is the **warehouse** for your inventory.
>
> The filing cabinet (git) is fast, small, and every employee gets a complete copy. That's wrong for inventory — pallets of inventory don't fit in filing cabinets. Instead, the filing cabinet stores *receipts*: "Pallet #4F-892 of churn-data, 50,000 rows, last updated 2026-01-09, hash ca160b1d…". The actual pallet lives in the warehouse (DVC cache). When a colleague needs that pallet they pull it down by receipt.
>
> The DVC pipeline is the **recipe** that says how today's pallet was made: "take yesterday's processed pallet, run the featurize procedure, produce today's features pallet." If anyone ever asks "how did we make this number?" the recipe + the pinned receipts answers it.

## 10. Exercises

1. **Trigger a smart re-run.** Open `src/features/build.py` and change `_TENURE_BIN_EDGES` to `[-0.5, 6, 18, 36, 72]`. Run `uv run dvc repro`. Which stages re-ran? Which didn't? Why? (Look at `dvc.yaml` deps.)
2. **Walk the lockfile.** Open `dvc.lock`. Find the md5 of `data/processed/churn.csv`. Now `cat data/processed/churn.csv | md5sum` (or `md5 -q` on macOS). Do they match? What does this tell you about reproducibility?
3. **Break a stage on purpose.** Run `echo " " >> data/raw/churn.csv` to add a stray space. Run `uv run dvc status`. What does DVC report? Now run `dvc repro`. What does DVC do, and why?
4. **Time-travel.** `git log --oneline` to find a commit before the featurize stage existed. `git checkout <that hash>`. Now run `uv run dvc repro` and `ls data/features/`. Does the features file exist? `git checkout main` and check again.
5. **Why the schema lives in the deps.** Look at the featurize stage's `deps` in `dvc.yaml`. We listed `src/features` (the whole directory), not just `src/features/build.py`. What goes wrong if you list only `build.py` and then change `src/features/schema.py`? Try it.

## 11. How to explain this in an interview

**60-second pitch:**

> "Phase 3 was about the reproducibility triangle. Code is in git, env is in `uv.lock`, but data is too big and too binary for git. I added DVC, which stores content hashes of every dataset in git and the actual bytes in a separate cache. Then I declared the pipeline in `dvc.yaml` as a DAG: `ingest → featurize`. Each stage has typed inputs and outputs and only re-runs when its content-addressed deps change. The key design decision was *not* encoding categoricals here — that lives in the training-time sklearn Pipeline so train-time and serve-time encoding are literally the same fitted objects. That eliminates train/serve skew."

**If they ask "what's the difference between DVC and git-LFS?":**

> "Git-LFS is a single-purpose extension to git for large binaries — every version is stored on the LFS remote, and the file shows up in the working tree as a pointer. DVC is a workflow tool: it has *pipelines*, *stages*, content-addressed dependencies, and a separate cache that's not bound to git's object format. You'd use git-LFS for occasional big assets in a normal repo. You'd use DVC for an ML repo where data and code evolve together and need the same time-travel guarantees."

**If they ask "why content-addressed and not timestamp-based?":**

> "Two reasons. One: timestamps lie — `git checkout` and `git pull` change them in ways that don't reflect content changes. Two: distributed teams. If I regenerate `churn.csv` on my machine and you regenerate it on yours from the same code and seed, the timestamps differ but the *content* is identical. Timestamp-based pipelines would re-run downstream stages unnecessarily. Content-based pipelines correctly notice nothing changed."

## 12. Common mistakes

- **`dvc add data/raw/churn.csv` and *also* listing it as a stage `outs`.** Pick one. `dvc add` is for tracked artifacts not produced by a pipeline. Pipeline outputs are tracked automatically.
- **Forgetting to commit `dvc.lock`.** Without it, your "reproducible pipeline" can't actually be reproduced — there's no record of which input hashes produced which outputs.
- **Listing only one file as a dep instead of a directory.** A change to `src/features/schema.py` won't trigger re-runs if the dep is just `src/features/build.py`. Listing the directory is safer at this scale.
- **Encoding in feature stage.** Sets you up for train/serve skew the day you build the inference service.
- **Hand-editing data files DVC tracks.** The cache will get out of sync with the file. Use `dvc unprotect` first if you really need to edit a tracked file by hand.
- **Pushing the cache to a public remote that already holds another project's data.** The dedupe is by content hash, so unrelated projects in the same remote work fine, but the access controls are usually wrong. Use one remote per project until you have a real story for multi-tenant caching.

## 13. What changes at scale

- **Remote cache becomes mandatory.** Single-developer projects use the local cache. Once two engineers collaborate, the cache moves to S3/GCS/Azure (`dvc remote add -d origin s3://bucket/path`).
- **Pipeline runners.** `dvc repro` runs locally. At scale you wrap it in a CI job (Phase 6) and / or a real orchestrator (Airflow, Prefect, Dagster) where each DVC stage becomes a task.
- **Feature stores.** Once you have multiple models sharing features, a feature CSV per pipeline doesn't scale. Tools like **Feast** or **Tecton** centralize features, give you online + offline serving paths, and (importantly) prevent two teams from defining `tenure_bucket` differently.
- **Data contracts as artifacts.** The schema files (`schema.py`) get extracted into a separately-versioned package the producing teams depend on, like Protobuf definitions across services.
- **Lineage and audits.** DVC tells you the input hash → output hash relationship. At scale you also need to know *who* triggered the pipeline, *when*, and *with what params* — that's where MLflow (Phase 4) and a metadata store (Marquez, OpenLineage) come in.

---

## ✅ Phase 3 done. What's next.

Phase 4 will:
- Train multiple sklearn models (Logistic Regression, Random Forest) on `churn_features.csv`.
- Wrap each in a `Pipeline` that does the encoding inline (so no train/serve skew).
- Track every run in **MLflow** — params, metrics, the model artifact itself.
- Add a `train` stage to `dvc.yaml` that depends on the features and produces an MLflow run id + a model artifact.
- Teach: experiment tracking, why offline accuracy isn't enough, and what the model registry will hold.

Wait for the user's go-ahead.
