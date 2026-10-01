"""LGB or XGB on FE level1 + in-fold target encodings. Usage: python exp_te.py lgb|xgb name lr"""
import sys, time
from te_fast import *
import os

algo, name = sys.argv[1], sys.argv[2]; lr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
tr, te, y = load()
X, Xt = build(tr, te, 1)
codes = te_codes(tr, te, bool(os.environ.get("PAIRS_ALL")), bool(os.environ.get("TRIPLES")))
oof = np.zeros(len(X)); pred = np.zeros(len(Xt)); its = []
for f, (a, b) in enumerate(folds(y)):
    t0 = time.time()
    Xa, Xb, Xc = add_te(X, Xt, y, codes, a, b)
    if algo == "lgb":
        import lightgbm as lgb
        params = dict(objective="binary", metric="auc", learning_rate=lr, num_leaves=int(__import__("os").environ.get("LEAVES",63)),
                      min_child_samples=50, feature_fraction=0.5, bagging_fraction=0.8,
                      bagging_freq=1, lambda_l2=1.0, max_cat_to_onehot=8, cat_smooth=10,
                      min_data_per_group=50, n_jobs=2, verbose=-1, seed=SEED)
        m = lgb.train(params, lgb.Dataset(Xa, y[a]), 10000, valid_sets=[lgb.Dataset(Xb, y[b])],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[b] = m.predict(Xb, num_iteration=m.best_iteration)
        pred += m.predict(Xc, num_iteration=m.best_iteration) / 5
        its.append(m.best_iteration)
    else:
        import xgboost as xgb
        params = dict(objective="binary:logistic", eval_metric="auc", tree_method="hist", learning_rate=lr,
                      max_depth=7, min_child_weight=5, subsample=0.8, colsample_bytree=0.5,
                      reg_lambda=2.0, reg_alpha=0.5, max_cat_to_onehot=8, nthread=2, seed=SEED)
        da = xgb.DMatrix(Xa, y[a], enable_categorical=True); db = xgb.DMatrix(Xb, y[b], enable_categorical=True)
        m = xgb.train(params, da, 10000, evals=[(db, "v")], early_stopping_rounds=150, verbose_eval=False)
        r = (0, m.best_iteration + 1)
        oof[b] = m.predict(db, iteration_range=r)
        pred += m.predict(xgb.DMatrix(Xc, enable_categorical=True), iteration_range=r) / 5
        its.append(m.best_iteration)
    print(f"  {name} fold {f} auc={roc_auc_score(y[b], oof[b]):.5f} it={its[-1]} {time.time()-t0:.0f}s", flush=True)
save(name, oof, pred, y, f"its={its}")
