# Phase 2 — Data Ingestion & Validation

> **Goal of this tutorial:** understand why ML systems live or die by their data layer, what a *data contract* is, how schema validation prevents silent production failures, and why we split raw from processed storage instead of cleaning data inline.

---

## 1. Why data pipelines matter (more than the model)

A famous saying in ML is **"garbage in, model out."** The slightly more useful version:

> An ML model is a probabilistic function trained on the *statistical shape* of its training data. Change that shape — even a little — and the model's outputs degrade. Often without anything throwing an error.

This is fundamentally different from a backend service.

A typical bug in a Go service:
- The service crashes, logs a stack trace, alarms fire, oncall pages.
- You see the failure within minutes.

A typical bug in an ML data pipeline:
- An upstream team renames `monthly_charges` → `monthly_amount` and adds a backward-compatible alias.
- Your pipeline keeps loading "successfully" — except `monthly_charges` is now `NaN` for everyone.
- Your model imputes 0 for missing values.
- Predictions silently skew toward "won't churn" (low charges = stable customer in this dataset).
- You notice three weeks later when the retention team asks why everyone is suddenly green.

There is no exception. There is no log line. Your CI passed. **The system is degrading on you and won't tell you.**

This is why the very first thing we built in Phase 2 is not "a CSV loader." It is a **data contract** that fails the pipeline the instant the data shape changes.

## 2. The four ways data breaks (and we now catch all four)

| Failure mode                          | Example                                                | What our pipeline does now           |
| ------------------------------------- | ------------------------------------------------------ | ------------------------------------ |
| Extra column appears                  | upstream adds `marketing_segment`                      | `strict=True` → schema fails         |
| Column disappears                     | upstream drops `tenure`                                | required column missing → fails      |
| Categorical gains a new value         | new payment method `"Apple Pay"` shows up              | `Check.isin([...])` → fails per-row  |
| Numeric drifts out of range           | `MonthlyCharges = -99` (sign flip bug upstream)        | `Check.greater_than(0)` → fails      |

Each of these has a corresponding test in `tests/test_schema.py`. Run `uv run pytest -v` to see them.

## 3. The "TotalCharges" story (real-world teaching moment)

In the actual public Telco Customer Churn dataset, the column `TotalCharges` is delivered as a **string**, with **empty strings** for customers whose `tenure` is 0. Here's what happens to a typical "I just want to fit a model" first pass:

```python
df = pd.read_csv("Telco-Customer-Churn.csv")
df["TotalCharges"].astype(float)   # ValueError: could not convert string ''
```

So the engineer writes:

```python
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].mean())
```

**That is a bug that will not throw.** The mean is ~2,283. A new customer (tenure 0) gets imputed at 2,283. Now the model thinks new customers and 4-year customers look similar in cumulative billing. Quality goes down, no exception is raised, no test fails.

What we did instead:
1. The **raw schema** *requires* `TotalCharges` to be a string. We honor what upstream sends.
2. The **cleaner** explicitly handles the case: blank → 0.0 (faithful: they truly haven't been billed).
3. The **processed schema** then enforces that `total_charges >= 0` and is a float.

The bug-prone behavior is now impossible — by construction, not by hope.

## 4. What we built

```
src/ingestion/
├── schema.py        # Pandera DataFrameSchema for raw input — the contract
├── synthetic.py     # generate_churn_data(n_rows, seed) — keeps repo self-contained
├── loader.py        # load_raw / write_raw with validation on both sides
├── clean.py         # clean(raw) -> processed, plus CHURN_PROCESSED_SCHEMA
└── cli.py           # `python -m src.ingestion.cli` — the single entry point

tests/
├── test_schema.py    # rejects extra cols, missing cols, bad categoricals, bad numerics
└── test_pipeline.py  # end-to-end: generate -> CSV -> load -> clean -> validate
```

### What `uv run python -m src.ingestion.cli` does

1. Generates 2000 synthetic rows (or reads `--source path/to/csv`).
2. Validates against `CHURN_RAW_SCHEMA`.
3. Writes `data/raw/churn.csv` (gitignored).
4. Cleans → processed: types coerced, columns renamed to `snake_case`, blanks handled.
5. Validates against `CHURN_PROCESSED_SCHEMA`.
6. Writes `data/processed/churn.csv` (gitignored).

The CLI is the contract Phase 3's DVC pipeline will call. We don't write a separate orchestrator yet — DVC just runs this CLI.

## 5. Why this approach (and not the obvious alternatives)

### Raw vs processed — why two stages?

The tempting thing is to read the CSV and clean it in one pass:

```python
df = pd.read_csv(path)
df = clean(df)
df.to_csv("clean.csv")
```

What you lose:
- **Re-cleanability.** Discovered a bug in `clean()` six months later? With the two-stage split, you re-run the cleaner over historical raw data. With one stage, the bug is baked into your "clean" files and you have to re-ingest *every upstream pull* — possibly impossible if upstream doesn't keep history.
- **Auditability.** Regulators / ops want to know "what did we receive vs what did we use." Keeping raw separate gives you that for free.
- **Independent versioning.** Phase 3 will version raw and processed separately in DVC. Different change cadences (raw changes when upstream changes; processed changes when our cleaning logic changes) deserve different version histories.

### Validation on **both** read and write — why?

`load_raw` validates because we don't trust the file system or whoever wrote it.
`write_raw` validates because we don't trust ourselves.

This is the same reason a database has constraints even though "the application validates everything." Two layers of defense costs almost nothing and catches a class of bugs that pure reading-validation misses (e.g. a future synthetic generator whose probabilities accidentally produce an out-of-range `MonthlyCharges`).

### Why Pandera (not Great Expectations, not Pydantic, not bare assertions)?

| Tool                | Sweet spot                                         | Why we picked / didn't                     |
| ------------------- | -------------------------------------------------- | ------------------------------------------ |
| **Pandera**         | DataFrame schemas, types + value checks            | ✅ Picked: declarative, plays with pandas, schema is itself a value you can pass around |
| Great Expectations  | Heavyweight expectation suites + data docs         | ❌ Overkill for a single-team project; better fit when you need a separate data-quality team |
| Pydantic            | Per-row models                                     | ❌ Slow on millions of rows (validates per dict); we'd lose vectorization |
| `assert df.col.min() >= 0` | Quick sanity in a notebook                  | ❌ Not composable, no error reporting, dies quietly |
| dbt tests           | Warehouse-side validation                          | ❌ Different layer; useful in addition to, not instead of |

In real production stacks, Pandera + dbt tests + occasional Great Expectations layers live happily together. Each guards a different boundary.

## 6. Tradeoffs we are accepting

| Decision                                  | Cost                                                           | Why we accept it                                            |
| ----------------------------------------- | -------------------------------------------------------------- | ----------------------------------------------------------- |
| Synthetic data by default                 | Slightly less convincing as portfolio piece                    | Project always runs; real data is a one-flag swap           |
| `strict=True` on schema                   | Adding a column upstream breaks our pipeline immediately       | **That's the point** — it forces a deliberate update        |
| Two schemas (raw + processed)             | More code to keep in sync                                      | Versions and audits each side independently                 |
| CLI argparse, not click                   | Less ergonomic flag handling                                   | One fewer dependency; argparse is fine for ~5 flags         |
| TotalCharges blank → 0                    | We're fabricating a "true" value                               | 0 is faithful (no billing happened); other choices distort  |
| `customerID` dropped during cleaning      | Loses ability to re-join after prediction                      | We'll re-introduce as a passthrough in Phase 7 (serving)    |

## 7. Real-world analogy

> Data ingestion is the **loading dock** of an ML system.
>
> Things arrive on trucks (the upstream system). Some shipments are damaged, some have wrong labels, some have items the warehouse never ordered. A bad receiving process accepts everything, and you find out three months later when half your inventory is wrong and nobody can trace which truck delivered it.
>
> A *good* receiving process:
> 1. Inspects every shipment against a packing manifest (the **schema**).
> 2. Refuses anything that doesn't match — politely sends it back to the supplier with a specific complaint.
> 3. Logs every accepted shipment with a date and a checksum (we'll do this with **DVC** in Phase 3).
> 4. Only after acceptance does the shipment go to processing (the **cleaner**).
>
> What we built in Phase 2 is the loading dock. It is small. It is unglamorous. It will save your job.

## 8. How this maps to your existing experience

- **AWS Lambda + S3 events.** A Lambda triggered by `s3:ObjectCreated` is the same idea as ingestion: an event arrives, you validate the payload, you reject or accept, you write to a downstream destination. Our `load_raw → validate → write_raw` is the synchronous version of that pattern.
- **n8n DAG with a "Validate JSON" node.** Pandera is the DataFrame equivalent: a typed checkpoint between two stages of a workflow.
- **Go struct tags + json.Unmarshal.** Pandera schemas play the same role for tabular data: declarative type info that turns into runtime validation. The difference is that struct tags validate one record; Pandera validates a whole DataFrame at once with vectorized checks (much faster than per-row).

## 9. Exercises

1. **Break the schema on purpose.** Edit `data/raw/churn.csv`, change one row's `Contract` value to `"Lifetime"`, then run `uv run python -c "from src.ingestion.loader import load_raw; load_raw('data/raw/churn.csv')"`. Read the error. What does it tell you, and where would that error end up in production (logs, alerts, an ignored stderr line)?
2. **Add a new check.** The dataset has no rule that `MonthlyCharges < 130`. Add one to `CHURN_RAW_SCHEMA`. Run the pipeline. What happens? Now remove it. What's the tradeoff between strict bounds and brittle pipelines?
3. **Spot the mean-imputation bug yourself.** Open a Python REPL and reproduce: load raw, fill `TotalCharges` blanks with the mean, train any sklearn model on tenure + TotalCharges → churn, look at the predictions for `tenure=0` rows. Compare to imputing 0. Which would you trust?
4. **Real data.** Find the actual Telco Customer Churn CSV (Kaggle / IBM mirrors). Run `python -m src.ingestion.cli --source <that file>`. Does it pass schema validation? If not, what does the error tell you about the difference between our synthetic generator and reality?
5. **Two schemas, one source.** Why do raw and processed schemas use different field names (`SeniorCitizen` vs `senior_citizen`)? What problem does that prevent? (Hint: think about a junior dev importing the wrong schema.)

## 10. How to explain this in an interview

**60-second pitch:**

> "Phase 2 isn't really about reading CSVs — it's about catching the failure modes that don't throw exceptions. I model the raw upstream data as a Pandera schema with type and value constraints, run it on every read and write, and keep raw separate from processed so cleaning bugs are recoverable. The most concrete payoff: in the public Telco dataset, `TotalCharges` is delivered as a string with blanks for new customers. The naive fix — `errors='coerce' + fillna(mean)` — silently corrupts predictions for new customers. With explicit raw and processed contracts, that bug is impossible to land."

**If they ask "why not Great Expectations / dbt tests?":**

> "Different layers. Pandera guards the application boundary — the moment data enters the Python process. dbt tests guard the warehouse boundary. GE is an entire framework for cross-team data quality with its own UI and stores. For a single-service ML pipeline I'd start with Pandera, add dbt tests if there's a warehouse upstream, and only reach for GE when there's a dedicated data-quality team."

**If they ask "what happens if upstream silently changes a column meaning?":**

> "Schema validation can't catch a *semantic* change — same name, same type, different meaning. That's a separate class of bug. We mitigate it with monitoring (Phase 8 — drift detection on input feature distributions) and with data contracts that get version-bumped on the producer side. The schema catches the easy 80%; drift detection catches the hard 20%."

## 11. Common mistakes (we are deliberately avoiding)

- **`errors='coerce'` everywhere.** Pandas' polite-failure default silently turns errors into NaNs. We use `errors='raise'` so a bad value blows up the pipeline immediately.
- **Cleaning during ingestion.** Tempting because it's fewer lines. Costs you the ability to re-run cleaning over old raw data when the cleaning logic changes.
- **Validating only on read.** Forgets that *we ourselves* are also a producer of data. Round-trip bugs in cleaners are the most common culprit.
- **Using `try / except: pass` to swallow validation errors.** This kind of comment shows up in real codebases. It is how silent failures become *invisible* failures.
- **`pd.read_csv(...)` with no `dtype`.** Pandas guesses, and the guess depends on the data. A new batch with all-numeric strings in a usually-categorical column changes the inferred dtype. We pin types.
- **Trusting the file extension.** `.csv` doesn't mean valid CSV. We validate after parsing, not before opening.

## 12. What changes at scale

This Phase 2 is fine for small/medium datasets running on one machine. At scale:

- **Streaming ingestion.** Replace CSV-on-disk with a Kafka topic (raw events) and Iceberg/Delta tables (processed). The schema validation moves into a streaming consumer, often using a schema registry (Confluent / Apicurio) so producers and consumers share the type info.
- **Distributed validation.** Pandera switches backends to PySpark or Polars. The schema definitions stay almost identical; the engine changes.
- **Sampling for cost control.** At billions of rows, validating every record is expensive. Common pattern: full validation on a sampled %, plus cheap Bloom filter / cardinality checks on the full set.
- **Data contracts as a first-class artifact.** The schema lives in its own repo, is versioned with semver, and producer teams break the contract by bumping major versions — exactly like a public API.
- **Quarantine, not crash.** Instead of failing the whole pipeline on one bad row, route the bad rows to a quarantine table and surface them on a dashboard for triage.

These are real engineering problems but Phase 2 problems for *another* project. Don't pre-build streaming for a CSV pipeline.

---

## ✅ Phase 2 done. What's next.

Phase 3 will:
- Initialize **DVC** to version the raw and processed data alongside code.
- Wire the ingestion CLI into a `dvc.yaml` pipeline stage so it only re-runs when its inputs change.
- Add the first feature engineering step (`src/features/`) with its own DVC stage.
- Teach: data versioning, why "reproducibility = code + data + env", and what feature drift starts to look like.

Wait for the user's go-ahead.
