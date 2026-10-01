# S6E9 experiments (5-fold StratifiedKFold, seed 42)

| ver | model | features | params | OOF AUC | file |
|---|---|---|---|---|---|
| v1 | LightGBM | 13 raw, 6 str cols as pandas category | lr .05, leaves 63, mcs 50, ff .8, bf .8, l2 1, ES 100 (~350-430 it) | 0.94180 | s6e9_sub_v1_lgb.csv |
| - | (original dataset) | not obtainable: Kaggle API blocked by proxy, not in public GitHub repos | - | - | skipped |
| lgb_fe1 | LightGBM | FE level1: +inc/commute floor flags, count enc (inc, commute, age), inc_per_car, stations sum/diff, com_per_st, inc_x_sub, env×sub×home×anxiety combo cat, Age/Stations as cat | lr .05, leaves 63, ff .7 | 0.94208 | - |
| xgb_fe1 | XGBoost hist | FE level1 | lr .05, depth 7, mcw 5, ss .8, cs .6, l2 2, a .5 | 0.94210 | - |
| v2 | rank blend lgb_fe1 .49 + xgb_fe1 .51 | | Nelder-Mead on OOF | 0.94225 | s6e9_sub_v2_lgbxgb.csv |
| cb_all | CatBoost | all 13 feats as cat strings + raw nums | lr .08 depth 7 l2 3 ctr2 | 0.94539 | - |
| v3 | blend cb_all .99 (+xgb .01) | | | 0.94539 | s6e9_sub_v3_catboost.csv |
| lgb_te | LightGBM | FE1 + in-fold TE of 13 raw feats + 55 pairs | lr .05 leaves 63 ff .5 | 0.94517 | - |
| xgb_te | XGBoost | same | lr .05 depth 7 | 0.94524 | - |
| blend | cb_all .56 + lgb_te .14 + xgb_te .31 | | | 0.94556 | - |
| cb_d8 | CatBoost depth 8 | | | OOM-killed (ran in parallel) | - |
| lgb_te_pall | LGB TE all 78 pairs lr .03 | | fold0 0.94439 vs 0.94430, 849s/fold | killed, not worth it | - |
| cb_fe | CatBoost + FE1 | | | fold0 0.94432 < cb_all 0.94449, killed | - |
| cb_d8 | CatBoost depth 8 lr .06 seed 1 | | | 0.94535 | - |
| v4 | blend cb_all .39 cb_d8 .20 lgb_te .12 xgb_te .30 | | | 0.94558 | s6e9_sub_v4_blend4.csv |
| xgb_te2/lgb_te2 | lr .02 | | | 0.94532 / 0.94527 | - |
| blend6 | + te2 models | | | 0.94559 (plateau) | - |
| analysis | income exact-value TE AUC .711 vs raw .670; round-number effect = only 30000 floor | | | | |
| cb_s7 | CatBoost lr .06 l2 5 seed 7 | | | 0.94542 (best single) | - |
| lgb_te3 | LGB TE all pairs + 35 triples lr .02 | | | 0.94523 | - |
| blend7 | 3 cb + 4 gbdt-te | | | 0.94561 | - |
| cb_lr04 | CatBoost lr .04 seed 11 | | | 0.94543 | - |
| xgb_te3 | XGB TE all pairs + triples lr .02 | | | 0.94523 | - |
| blend check | weights fit on folds 0-2 vs eval 3-4: optimized == equal weights (0.94575 vs 0.94575) | | | | |
| **v5 (final)** | equal-weight rank blend: cb_all, cb_d8, cb_s7, cb_lr04, lgb_te2, xgb_te2, lgb_te3, xgb_te3 | | | **0.94561** | s6e9_sub_v5_final.csv |
