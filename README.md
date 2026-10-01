# Kaggle Playground S6E9: Predicting EV Purchases

Solution for [Kaggle Playground Series S6E9](https://www.kaggle.com/competitions/playground-series-s6e9), September 2026.

**Task:** binary classification of `Will_Buy_EV` (Yes/No) on tabular data. **Metric:** ROC-AUC.
**Final result:** equal-weight rank blend of 8 gradient-boosting models, **OOF ROC-AUC 0.94561** (5-fold stratified CV).

> Built with AI-assisted development (Claude). All experiments, validation choices and results are documented in [`experiments.md`](experiments.md).

## Data

- 668,665 training rows, 286,571 test rows, 13 features (7 numeric, 6 categorical/binary), no missing values.
- Positive class share: 17.5%.
- The data is synthetic, generated from the *EV Adoption Behavior and Range Anxiety* dataset.

## Key findings from EDA

- `Environmental_Concern_Level` is the strongest single feature (AUC ≈ 0.84 on its own): the purchase rate grows from 0.6% at level 1 to 51.8% at level 5.
- `Subsidy_Available`: 0.6% buyers without a subsidy vs 27.5% with one.
- **Numeric features carry value-specific, non-monotonic signal**, likely an artifact of synthetic generation. Target-encoding income as a category gives AUC 0.711 vs 0.670 for raw income.

This led to the main modeling decision: **treat every feature, including numeric ones, as categorical.**

## Approach

| Step | What | OOF ROC-AUC |
| --- | --- | --- |
| Baseline | LightGBM on 13 raw features | 0.94180 |
| Feature engineering | count encoding, ratios, interaction category; LightGBM + XGBoost rank blend | 0.94225 |
| All features as categorical | CatBoost with ordered target statistics (CTR) on all 13 features | 0.94539 |
| In-fold target encoding | leak-free TE of single features and feature pairs for LightGBM / XGBoost | 0.94532 |
| **Final blend** | equal-weight rank blend of 4 CatBoost + 4 TE-based LightGBM/XGBoost models | **0.94561** |

**Validation.** All models share the same 5-fold `StratifiedKFold` (seed 42), so out-of-fold predictions are comparable and can be blended. Target encoding is computed inside each training fold (with inner folds) to avoid target leakage.

**Blend weights.** Weights optimized with Nelder-Mead on folds 0–2 and evaluated on folds 3–4 (and vice versa) gave no gain over equal weights, so the final blend uses equal weights for robustness on the private leaderboard.

**Speed.** `te_fast.py` reimplements target encoding with `np.bincount` on dense integer codes; it matches `te_feats.py` exactly and runs several times faster.

**What did not help:** extra engineered features in CatBoost, target encoding of triples, all 78 feature pairs at a lower learning rate.

## Repository structure

```
common.py        data loading, shared CV folds, feature engineering, saving OOF/test predictions
baseline_lgb.py  first LightGBM baseline
exp_cb.py        CatBoost, all features as categorical (first version)
exp_cb2.py       configurable CatBoost (--lr --depth --seed --l2 --ctr --fe)
te_feats.py      in-fold target encoding (pandas)
te_fast.py       same encoding, vectorized with NumPy
exp_te.py        LightGBM / XGBoost on target-encoded features
exp_te2.py       same, using te_fast (PAIRS_ALL=1, TRIPLES=1 env flags)
blend.py         rank blend with optimized (Nelder-Mead) or equal weights (EQUAL=1)
blend_check.py   checks whether blend weights are stable across fold halves
experiments.md   full experiment log
```

## Reproduce

1. Download the competition data and put `train.csv`, `test.csv`, `sample_submission.csv` into `data/`.
2. `pip install -r requirements.txt`
3. Run the models **one at a time** (two CatBoost models in parallel do not fit into 8 GB RAM):

```bash
python exp_cb2.py cb_all  --lr 0.08 --depth 7 --seed 42
python exp_cb2.py cb_d8   --lr 0.06 --depth 8 --seed 1
python exp_cb2.py cb_s7   --lr 0.06 --depth 7 --seed 7 --l2 5
python exp_cb2.py cb_lr04 --lr 0.04 --depth 7 --seed 11
python exp_te.py lgb lgb_te2 0.02
python exp_te.py xgb xgb_te2 0.02
PAIRS_ALL=1 TRIPLES=1 python exp_te2.py lgb lgb_te3 0.02
PAIRS_ALL=1 TRIPLES=1 python exp_te2.py xgb xgb_te3 0.02
EQUAL=1 python blend.py s6e9_sub_v5_final cb_all cb_d8 cb_s7 cb_lr04 lgb_te2 xgb_te2 lgb_te3 xgb_te3
```

OOF and test predictions go to `preds/`, the submission file to `submissions/`.

## Ideas not yet tried

- Adding the original ~10k-row dataset to training with an indicator column.
- CatBoost with `max_ctr_complexity=3` and more seeds.
- A neural network with categorical embeddings for blend diversity.
