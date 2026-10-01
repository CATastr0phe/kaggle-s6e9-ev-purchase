import sys, time
from catboost import CatBoostClassifier, Pool
from common import *

tr, te, y = load()
name = sys.argv[1]; lr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.08
nfolds = int(sys.argv[3]) if len(sys.argv) > 3 else 5
# CatBoost: all original features as categorical strings (numeric too) + raw numerics
X, Xt = build(tr, te, 0)
catcols = []
for c in NUMS + CATS:
    X[c + "_s"] = (tr[c] if c in tr else X[c]).astype(str).values
    Xt[c + "_s"] = te[c].astype(str).values
    catcols.append(c + "_s")
for c in CATS:
    X[c] = X[c].astype(str); Xt[c] = Xt[c].astype(str)
X = X.drop(columns=CATS); Xt = Xt.drop(columns=CATS)
oof = np.zeros(len(X)); pred = np.zeros(len(Xt)); its = []
for f, (a, b) in enumerate(folds(y)):
    if f >= nfolds: break
    t0 = time.time()
    m = CatBoostClassifier(iterations=6000, learning_rate=lr, depth=7, l2_leaf_reg=3, eval_metric="AUC",
                           od_type="Iter", od_wait=150, thread_count=2, random_seed=SEED, verbose=0,
                           one_hot_max_size=8, max_ctr_complexity=2, border_count=254)
    m.fit(Pool(X.iloc[a], y[a], cat_features=catcols), eval_set=Pool(X.iloc[b], y[b], cat_features=catcols),
          use_best_model=True)
    oof[b] = m.predict_proba(X.iloc[b])[:, 1]
    pred += m.predict_proba(Pool(Xt, cat_features=catcols))[:, 1] / 5
    its.append(m.get_best_iteration())
    print(f"  {name} fold {f} auc={roc_auc_score(y[b], oof[b]):.5f} it={its[-1]} {time.time()-t0:.0f}s", flush=True)
if nfolds == 5:
    save(name, oof, pred, y, f"its={its}")
