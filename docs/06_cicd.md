# Phase 6 — CI/CD for ML with GitHub Actions

> **Goal of this tutorial:** understand what ML CI/CD has to do that normal-software CI/CD doesn't, why the three retraining triggers (code / data / scheduled) all collapse cleanly into one GitHub Actions workflow, what's safe to automate and what should always require a human in the loop, and how the dry-run-on-PR pattern avoids the most common ML CI footgun (mutating production by accident).

---

## 1. ML CI/CD vs DevOps CI/CD

For a Go service, "CI passes" means:

> Code compiles. Tests pass. Linter is happy. → Safe to merge / deploy.

For an ML system, the same sentence has at least four extra meanings hiding inside:

> Code compiles. **Pipeline reproduces.** Tests pass. Linter is happy. **Trained model isn't worse than the current production model.** **Data still satisfies the schema.** → Safe to merge / deploy.

The four bolded conditions don't exist for plain backend services. They exist for ML systems because the artifact you ship isn't *just* code; it's a tuple `(code, data, model)` and any of them can silently regress without the others changing.

| Concern                              | DevOps CI                         | ML CI                                                 |
| ------------------------------------ | --------------------------------- | ------------------------------------------------------- |
| What "passing" means                 | Tests pass                        | Tests pass + pipeline reproduces + metrics don't regress |
| Time to "passing"                    | Seconds                           | Minutes (training is slow)                              |
| What can silently regress            | Code (caught by tests)            | Code, data shape, data distribution, model quality       |
| What gates a merge                   | Test pass / coverage              | Tests + metric delta (per docs/05's promotion rule)      |
| What gates a deploy                  | Tests + manual approval           | Tests + metric delta + (often) canary + soak             |
| Can it run on every PR?              | Yes, cheaply                      | Maybe — training cost vs PR frequency                    |

Notably, **ML CI training time** is the new constraint. A 5-minute training run on every PR is fine; a 5-hour run is not, and you'd skip the training step on PRs and rely on the scheduled trigger instead. Phase 6's workflow stays under that ceiling because our model is small.

## 2. The three retraining triggers

The project README promised three triggers. All three live in the same workflow file (`.github/workflows/retrain.yml`) because GitHub Actions lets you stack multiple `on:` events on a single workflow.

```yaml
on:
  push:
    paths: [...code paths...]      # trigger 1: code change
  pull_request:
    paths: [...same paths...]      # trigger 1: code change (PR variant)
  push:
    paths: ['dvc.lock']            # trigger 2: data change
  schedule:
    - cron: '0 6 * * 1'            # trigger 3: scheduled
  workflow_dispatch:               # manual escape hatch
```

(Simplified — the real file collapses the duplicate `push` patterns into a single `paths:` list.)

### What each trigger catches

| Trigger          | The bug it prevents                                                            |
| ---------------- | ------------------------------------------------------------------------------ |
| **Code change**  | "I refactored feature engineering and forgot to retrain."                      |
| **Data change**  | "Upstream dataset got 50k new rows; nobody re-ran the model on them."          |
| **Scheduled**    | "The model has been the same for 6 months; data has drifted, but nobody pushed." |

The scheduled trigger is the safety net: even if no code or data change ever fires, the model gets re-trained periodically against the latest data. It's how you catch *concept* drift (the data-target relationship changing) which neither code nor data triggers can detect on their own.

### Why `dvc.lock` is the data-change signal

DVC's lockfile is a content-hash registry of every dataset in the pipeline. When data changes, the lockfile changes; when data doesn't change, the lockfile doesn't change. So `paths: ['dvc.lock']` is a precise data-change trigger that doesn't fire on noise. (Without DVC, you'd have to invent your own hash-and-track system or accept noisier triggers like "any PR touching `data/`.")

## 3. The dry-run-on-PR pattern

The workflow has two distinct modes:

```yaml
- name: Reproduce pipeline (PR — train only, no registry mutation)
  if: github.event_name == 'pull_request'
  run: uv run dvc repro train

- name: Reproduce full pipeline (push / schedule / manual)
  if: github.event_name != 'pull_request'
  run: uv run dvc repro
```

On a PR, we stop *before* the `evaluate` stage. Why?

- The MLflow registry on the GitHub-hosted runner is **ephemeral**. Anything we register or alias-flip vanishes when the runner is destroyed.
- More importantly: **a PR shouldn't move the production alias on `main`.** If the runner's registry persisted, an experimental PR could promote a model that hadn't been merged yet — exactly the train/serve coupling problem we built the registry to avoid.

The PR's job is to *report* the metric delta. The `main`'s job is to *act* on it (when merged, the push trigger runs the full pipeline against the persistent (in our case still local) registry).

This is the same pattern as `terraform plan` on a PR vs `terraform apply` on a merge: PRs preview, merges apply.

## 4. The metric delta gate

`scripts/check_metrics_regression.py` is the actual gate. On a PR it:

1. Loads the PR's `metrics/train_metrics.json` (just produced by `dvc repro train`).
2. Reads the same path from `origin/main` via `git show`.
3. Picks the best run on the chosen metric (default F1) from each side.
4. Computes the delta.
5. Writes a markdown summary to `metrics/_delta.md`.
6. **Exits non-zero if the regression exceeds the tolerance.** That's the merge block.

The default tolerance is `0.005` F1. If you regress F1 by more than 0.5 percentage points, CI fails. You can pass `--tolerance` and `--metric` explicitly to gate on PR-AUC or a different threshold.

This is the **margin threshold** guardrail from docs/05 §7.1, applied to PRs instead of to the registry. The registry-side rule is "must strictly beat current production"; the PR-side rule is "must not regress current `main`'s baseline by more than tolerance." Different polarity, same idea.

### Why a sticky PR comment

The comment script (`actions/github-script` step in the workflow) finds an existing comment by an HTML marker and *updates* it on subsequent runs, instead of creating a new one. Without this, every push to a PR would spawn a new bot comment — quickly turning the PR thread into noise. The HTML marker is a hidden divider any tool (or human) can use to find and update the same comment, no third-party action required.

## 5. What we deliberately did NOT automate

- **Promotion to production.** Even on `main`, the workflow runs `dvc repro` which calls `evaluate`, which moves the production alias if the candidate beat current. That's automatic — *because the strict-greater rule is conservative*. In a real production environment you'd add: a Slack alert, a hold for human ack, and a canary period. None of that is here.
- **Deployment of the new model to the inference service.** Phase 7 will build that service. Its deployment will be a separate workflow that reads from the registry. The two events ("we have a new production model" and "the inference service is now running it") should always be separate, observable transitions — not one atomic step.
- **Rollback.** No automatic rollback yet. If a freshly-promoted model performs badly post-deploy, you currently have to hand-flip the `production` alias to the previous version. A real system has a "last known good" stickiness and an automatic rollback when live metrics tank.
- **Drift response.** Phase 8 will detect drift in the served model. The *response* to drift (retrain immediately? alert humans?) is a policy decision; the safe default is "alert, let humans decide," not "auto-retrain."

The pattern: **automation does the boring, deterministic work; humans handle the irreversible, judgment-required transitions.**

## 6. Why GitHub Actions (and not the alternatives)

| Tool                | What it adds                                          | Why we didn't                                            |
| ------------------- | ----------------------------------------------------- | -------------------------------------------------------- |
| **GitHub Actions**  | Triggers built into the same git host as the code     | ✅ Picked: zero infra, three triggers via one config      |
| CircleCI / Travis   | Similar feature set, separate vendor                  | ❌ Extra account / billing / integrations                |
| Jenkins             | Full CI server, plugin for everything                 | ❌ Heavyweight; you maintain the server                  |
| Argo Workflows      | k8s-native pipeline runner                            | ❌ Requires a k8s cluster; overkill for this project     |
| GitLab CI           | Same idea as Actions but for GitLab repos             | ❌ Repo is on GitHub                                     |
| Buildkite           | Self-hosted runners, nice UI                          | ❌ Self-hosting cost                                     |

For ML specifically: **Actions is "good enough" up to fairly large teams.** The day you need fan-out across 100 GPUs for hyperparameter search, you'd add Argo (still triggered from Actions) — but the Actions workflow stays as the orchestration entry point.

## 7. Real-world analogy

> CI is your **smoke detector**. CD is your **fire department**.
>
> A smoke detector is automatic and noisy: any whiff of trouble and it goes off. False positives are cheap. Skipping the detector to save the noise is how houses burn down.
>
> The fire department is *not* automatic. Even if the smoke detector is screaming, a human dispatcher decides what to send (an engine? a ladder? an ambulance? all three?), and humans on the trucks make the actual rescue calls. False positives there are *expensive* — wasted dispatches, blocked traffic, alarm fatigue.
>
> ML systems are houses with multiple kinds of fire (data drift, concept drift, train/serve skew, broken pipelines). The CI workflow we built is the smoke detector — cheap, noisy, runs on every change. Promotion and deployment are the fire department — they're triggered by CI signals but the actual *do something to production* part should always involve a human at this scale.

## 8. How this maps to your existing experience

- **Lambda + EventBridge cron.** GitHub Actions' `schedule:` trigger is the same idea: a cron expression that fires a job. EventBridge → Lambda is "fires AWS work"; Actions → workflow is "fires CI work."
- **Branch protection rules.** GitHub's "require status checks before merging" + a CI workflow named `ci` is the merge gate. We have the same shape: `ci.yml` for fast checks, `retrain.yml` for the heavy gate. You'd configure both as required status checks on `main`.
- **n8n cron node.** Identical to the `schedule:` trigger. The difference is what the run does — n8n runs JS / API calls; Actions runs shell commands.
- **Terraform plan / apply.** The dry-run-on-PR / apply-on-main split here is exactly Terraform's pattern. PRs preview, merges enact.

## 9. Exercises

1. **Watch CI fail on a regression.** Make a branch. Edit `src/training/pipeline.py` to set `RandomForestClassifier(max_depth=2, n_estimators=20)`. Open a PR. What does the metric-delta comment say? Does CI block the merge? (If you don't want to actually open a PR, run `uv run python scripts/check_metrics_regression.py --base-ref main` locally after the change.)
2. **Audit the path filters.** Open `.github/workflows/retrain.yml`. The `paths:` filter excludes `docs/**`. What about `tests/**`? If you change a test, does retrain fire? Should it? Argue both sides.
3. **Make the metric configurable per-PR.** Right now the workflow always gates on F1. Add a way for a PR to opt into PR-AUC gating with a label (`gate:pr_auc`). Hint: `if: contains(github.event.pull_request.labels.*.name, 'gate:pr_auc')` plus a script flag.
4. **Add a Slack alert on `reject` decisions.** Modify `retrain.yml` to send a Slack message when the push-to-main pipeline produces a `reject` decision. Why is `reject` more interesting than `promote` for an alert?
5. **Trace a scheduled trigger.** Open the GitHub Actions tab on the repo. When the next Monday 06:00 UTC fires, the workflow should run with no PR context. Walk through the steps: which conditional branches execute? Which artifacts are produced? Which are skipped?

## 10. How to explain this in an interview

**60-second pitch:**

> "Phase 6 wires up CI/CD with two GitHub Actions workflows. `ci.yml` is the fast path — ruff and pytest on every push and PR. `retrain.yml` is the heavy path: it fires on three triggers — code change, data change (via dvc.lock), and weekly cron — and reproduces the full pipeline. On PRs it stops before evaluate so the registry isn't mutated by an unmerged candidate, and a Python script compares the PR's metrics against main, posts a sticky bot comment with the delta, and fails CI if the regression exceeds 0.005 F1. On pushes to main it runs end-to-end, including the promote step. The pattern is Terraform-style: PRs preview, merges enact."

**If they ask "why three triggers, not just one?":**

> "Each catches a different class of bug. Code change catches refactor regressions. Data change catches forgotten retrains when upstream data updates. Schedule catches concept drift — the slow kind where neither code nor data changes but the world has moved on. They overlap, that's fine; they're cheap to run, and the cost of *missing* a retrain is always higher than running an extra one."

**If they ask "what's the difference between ML CI and regular CI?":**

> "The artifact you ship. Regular CI gates on tests passing — code is the only moving part. ML CI gates on tests, pipeline reproducibility, *and* metric quality — because the artifact is `(code, data, model)` and any of them can silently regress without the others changing. So ML CI runs the actual training and compares the resulting model to the previous one, which makes the runs slower, the failure modes richer, and the gating logic more nuanced."

## 11. Common mistakes

- **Mutating the registry from PR runs.** The most common ML CI footgun. A passing CI on a PR shouldn't change anything in production registry. We explicitly stop before `evaluate` on PRs.
- **Gating on accuracy.** Same trap as docs/05. Configure your CI to gate on F1 / PR-AUC for imbalanced problems. Default tolerance keyed off historical run-to-run variance, not a guess.
- **No metric tolerance.** A strict "must improve" gate blocks every refactor PR that doesn't change the model. A regression *tolerance* (don't regress more than X) keeps refactors mergeable while catching real degradation.
- **Posting a new comment per run.** Quickly turns a long PR into a comment-spam tunnel. Use a marker + update pattern.
- **No cancel-in-progress.** Stale builds waste runner minutes and block the queue. The `concurrency: cancel-in-progress: true` block prevents that.
- **Storing secrets in `env:`.** Secrets go in repo / org secrets and are referenced as `${{ secrets.NAME }}`. Direct env values land in the workflow file in plaintext.
- **Running the schedule trigger on a fork.** Forks inherit workflows but shouldn't run scheduled jobs at the fork owner's expense. The default GitHub behavior handles this; just be aware of it.

## 12. What changes at scale

- **Self-hosted runners.** GitHub-hosted runners are ~7 GB RAM, 2 cores. Real ML runs need GPUs and bigger CPU. Self-host on EC2 / GKE with the runner registered to the repo. The workflows themselves don't change — just `runs-on:` switches to a label your runners advertise.
- **Per-PR data subsampling.** Full retraining on every PR doesn't scale. Common pattern: PRs run training on a 10% sample for fast feedback; main runs the full thing.
- **Distributed orchestration.** Beyond a couple of stages, GitHub Actions becomes awkward as the orchestrator. Argo Workflows / Kubeflow / Prefect take over for the data-pipeline orchestration; Actions stays as the trigger source and the merge-gate.
- **Multi-environment promotion.** Separate workflows for `staging` and `production` aliases. Promotion to staging is automatic on merge; promotion staging→production requires human approval (a `workflow_dispatch` with environment protection).
- **Cost tracking.** Each retrain costs runner time + (eventually) GPU time. Tag artifacts with cost estimates; budget alerts when monthly retrain spend exceeds threshold. Doesn't seem important for a portfolio project; becomes a line item in any real org.
- **Audit log.** Every push that produces a registry change becomes an entry in a separate audit table — what model was promoted, which commit, which CI run, on whose authority. Compliance teams will ask.

---

## ✅ Phase 6 done. What's next.

Phase 7 will:
- Build a **FastAPI inference service** that loads the model tagged `production` from the MLflow registry and serves predictions over HTTP.
- Dockerize the service so it can run anywhere a container does.
- Add a smoke test that hits the API end-to-end.
- Teach: batch vs real-time inference, why the persisted Pipeline (with its fitted encoder) is what makes serving safe, model loading at startup vs per-request, and how the production alias becomes the deploy unit.

Wait for the user's go-ahead.
