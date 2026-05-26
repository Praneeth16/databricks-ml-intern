# Skill: kaggle-tabular-classification

Ranked playbook for Kaggle tabular AUC/log-loss/accuracy competitions.
Derived from Playground S6E5 (F1 Pit Stops) experience. Read end-to-end
before writing any model code.

## Phase 0 — Confirm the basics (every competition, no shortcuts)

1. **Read `sample_submission.csv` header.** The non-id column IS the target.
2. **Train-test column diff:** target must be in train, absent from test.
3. **Print target distribution + dtype** before training. If task says
   "binary classification" and your target has 17 unique values, you're
   looking at the wrong column.
4. **Detect the validation scheme that matches LB:**
   - Look for a temporal column (Year, Date, Season, Order). If present
     and test rows have only the most-recent value(s), the LB is
     time-split. Use a time-based hold-out as PRIMARY val.
   - If no temporal axis, use 5-fold StratifiedGroupKFold with a leakage-
     safe group key (group must be hierarchically above any feature you
     engineer).
   - On 2 consecutive LB regressions: STOP, your CV is broken. Switch
     validation scheme.

## Phase 1 — Strong baseline (1 job, ~5 min)

- XGBoost with sensible defaults. Time-split or stratified k-fold per
  Phase 0. Submit if val is plausible. This is your reference LB anchor.
- Common starting params (tweak per data):
  ```python
  XGBClassifier(
      max_depth=10, learning_rate=0.02, n_estimators=1500,
      subsample=0.8, colsample_bytree=0.8,
      min_child_weight=8, gamma=0.5,
      reg_alpha=0.001, reg_lambda=5.0,
      early_stopping_rounds=50,
      eval_metric="auc",
      tree_method="hist",
  )
  ```
- Submit this. Now you know LB-vs-val gap; use it to estimate future LB
  from future val.

## Phase 2 — Race-context / cross-entity features (HIGHEST EXPECTED LIFT)

**Rule of thumb:** if your data has multiple entities competing in the
same event (drivers per race, players per game, sellers per category),
*single-entity features leave the biggest signal on the table*. Compute
cross-entity features grouped by (event_id, time).

For Playground S6E5 specifically: group by (Race, Year, LapNumber) and
within each group compute, for every row:
- `pits_this_lap_in_race` — how many drivers pitted on the same lap.
- `driver_ahead_pitted_last_lap` — binary, did the car immediately ahead
  pit on lap N-1.
- `driver_behind_pitted_last_lap` — same for behind.
- `tyrelife_rank_in_lap` — rank of TyreLife within the lap (1 = oldest tyre).
- `lap_time_rank_in_lap` — rank of LapTime.
- `pit_pressure_3lap` — cumulative pits in the last 3 laps for the race.

For other tabular tasks: the same pattern — within-group ranks, leads,
lags, cumulative counts — applied to whichever (entity, event) split
defines your domain (player×match, seller×day, etc).

Implementation note: sort once by (event_id, time), then chain pandas
groupby-shift / groupby-cumsum / groupby-rank. ~60 LOC, runs in <30s on
500k rows.

**Expected lift on val:** +0.004 to +0.010 AUC. Highest single-job lift
in this playbook. Reference: [F1 pit-stop deep-learning paper, Frontiers
2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12626961/) reports +0.02
on real FastF1 data; Playground synthetic data may yield less.

## Phase 3 — Within-entity sequence features (second-highest lift)

For each (entity, sub-event) sequence, compute *trend* and *acceleration*
features, not just levels.

Concrete for S6E5: group by (Race, Year, Driver, Stint), order by
LapNumber, compute:
- `lap_in_stint` — `cumcount()`.
- `laptime_slope_last3` — rolling OLS slope of LapTime over last 3 laps.
- `laptime_delta_vs_stint_mean` — `LapTime - expanding_mean(LapTime)`.
- `degradation_accel` — `Cumulative_Degradation.diff()`.
- `tyrelife_X_stintprogress` — `TyreLife * lap_in_stint`.

All causal (past laps only) → no leakage by construction.

**Expected lift on val:** +0.002 to +0.005.

## Phase 4 — Many-seed retrain on 100% data (insurance)

After phases 2+3 lock in the best single model and your val scheme
matches LB direction:
- Refit the winning model with 5-10 different seeds on the FULL train.
- Average the test-prediction probabilities.
- Always +0.001 to +0.002 on LB. Cheap insurance.

Source: [NVIDIA Grandmasters Playbook §7](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/).

## Phase 5 — Hill-climb blend over diverse OOF preds

After single-model is squeezed dry:
1. Train OOF preds for 3-4 DIVERSE models: XGBoost, LightGBM, CatBoost
   (native cat-handling = real diversity), and one MLP (different
   inductive bias). Split into 3 jobs to respect the 12-min cap.
2. Hill-climb weights against OOF AUC:
   ```python
   from scipy.optimize import minimize
   from sklearn.metrics import roc_auc_score
   def neg_auc(w, oofs, y):
       w = np.abs(w); w /= w.sum()
       return -roc_auc_score(y, (w[:, None] * oofs).sum(0))
   best = minimize(neg_auc, np.ones(len(oofs))/len(oofs),
                   args=(np.array(oofs), y), method="Nelder-Mead")
   ```
3. Apply same weights to test predictions.

**Expected lift on val:** +0.002 to +0.005 on top of phase 2+3.

Two warnings:
- If one model's OOF AUC is significantly lower (Δ > 0.005 below best),
  the optimizer will give it weight ~0. Don't waste compute retraining it.
- If your CV is overfitting (LB regresses on each submission), blending
  averages the same overfit signal. Fix CV first, blend second.

Reference: [Matt-OP hillclimbers](https://github.com/Matt-OP/hillclimbers),
[S5E12 1st place writeup](https://www.kaggle.com/competitions/playground-series-s5e12/writeups/1st-place-solution-hill-climbing-ridge-ensembl).

## Phase 6 — Pseudo-labeling (one round, only if base model is strong)

If OOF AUC ≥ 0.94 and CV-LB alignment is good:
1. Get model's predictions on the test set.
2. Keep only rows where `pred > 0.97 or pred < 0.03` (typically ~10-20%
   of test).
3. Concat those rows + their pseudo-labels onto train.
4. Refit the same model.
5. Predict the FULL test (including pseudo-labeled rows).
6. Submit.

Always validate the pseudo-labeled model on a held-out fold BEFORE
submitting — bad pseudo labels can drop val.

**Expected lift on val:** +0.001 to +0.004. Genuinely uncertain on
synthetic Playground data (test distribution ≈ train).

References: [Deotte pseudo-labeling QDA 0.969](https://www.kaggle.com/code/cdeotte/pseudo-labeling-qda-0-969),
[Regularized pseudo-labeling arXiv 2302.14013](https://arxiv.org/pdf/2302.14013).

## Phase 7 — Skipped on purpose

- **SMOTE / class-weight tweaks** — AUC is rank-only; doesn't move it.
- **Isotonic / Platt calibration** — same; rank-preserving.
- **Two-level stacking with LR meta-learner** — redundant w/ hill climb
  for ≤4 base models; revisit only with 5+ diverse bases.
- **NN-only solutions** — GBDTs dominate tabular AUC by themselves.
  Use NNs only as a diversity ingredient in the blend.

## Anti-patterns observed on Playground S6E5

1. **Wrong target column.** Trained iters predicting `PitStop` instead
   of `PitNextLap`. CV looked great, LB ~0.46. Fix = Phase 0 step 1.
2. **Lag features without time-split val.** Lag of LapTime/Position
   leaks the target by construction at the stint boundary. CV inflated
   to 0.948+, LB regressed by 0.001 each submission.
3. **3-model × 5-fold × Optuna in one job.** 86-min runtime,
   serverless-CPU timed out. Always one model per job at 12-min budget.
4. **CatBoost native cat-handling at default depth.** 12 min budget
   blown at 2 of 5 folds. Use depth ≤ 6 and iterations ≤ 500 if you
   must run CatBoost on CPU. Or move to a CPU-GPU pool.

## Submission discipline (also in core system prompt)

- Most Kaggle Playground competitions = 5 submissions / 24h.
- Only submit when val improves by ≥ 0.001 over current best LB-mapped val.
- "Would-submit" line before each: `(prior_val, new_val, delta, reason)`.
- Never two consecutive submissions whose only diff is hyperparameters.
- Print `READY FOR SUBMIT: <path> | expected LB ~Y based on val Z` as the
  last line of the training job. User submits manually.

## Per-iteration job template

Every training job should:
1. `mlflow.set_experiment("/Shared/ml-intern/<comp_slug>")` with workspace-
   dir collision fallback cascade.
2. Wrap user script with the stdout-tee prelude so `runs/get-output`
   carries the tail.
3. Save submission CSV to `/Volumes/<cat>/<schema>/<vol>/<comp_slug>/
   submission_iter<N>_<method>.csv`.
4. Save OOF predictions CSV alongside for later blending.
5. End with `READY FOR SUBMIT: <path> | expected LB ~Y based on val Z`.

## Tomorrow's submission budget calculator

Given:
- Current LB anchor `LB_v1`.
- Current val anchor `val_v1`.
- val-LB gap `gap = LB_v1 - val_v1`.

For new iter with val `val_new`:
- Estimated LB = `val_new + gap`.
- Submit if estimated LB > `LB_best + 0.001`.
- Hold if estimated LB ≤ `LB_best`.
