"""Configurable CatBoost: all features as categorical strings (+ raw numerics).
Usage: python exp_cb2.py name --lr 0.08 --depth 7 --seed 42 --ctr 2 --fe 0 --l2 3"""
import argparse, time
from catboost import CatBoostClassifier, Pool
from common import *
ap = argparse.ArgumentParser(); ap.add_argument("name")
ap.add_argument("--lr", type=float, default=0.08); ap.add_argument("--depth", type=int, default=7)
ap.add_argument("--seed", type=int, default=42); ap.add_argument("--ctr", type=int, default=2)
ap.add_argument("--fe", type=int, default=0); ap.add_argument("--l2", type=float, default=3)
ap.add_argument("--rs", type=float, default=1.0)
A = ap.parse_args()
tr, te, y = load()
X, Xt = build(tr, te, A.fe)
catcols = []
for c in NUMS + CATS:
    X[c + "_s"] = tr[c].astype(str).values; Xt[c + "_s"] = te[c].astype(str).values; catcols.append(c + "_s")
X = X.drop(columns=CATS); Xt = Xt.drop(columns=CATS)
for c in list(X.columns):
    if str(X[c].dtype) == "category":
        X[c] = X[c].astype(str); Xt[c] = Xt[c].astype(str); catcols.append(c)
oof = np.zeros(len(X)); pred = np.zeros(len(Xt)); its = []
pt = Pool(Xt, cat_features=catcols)
for f, (a, b) in enumerate(folds(y)):
    t0 = time.time()
    m = CatBoostClassifier(iterations=8000, learning_rate=A.lr, depth=A.depth, l2_leaf_reg=A.l2, eval_metric="AUC",
                           od_type="Iter", od_wait=150, thread_count=2, random_seed=A.seed, verbose=0,
                           one_hot_max_size=8, max_ctr_complexity=A.ctr, border_count=254, random_strength=A.rs)
    m.fit(Pool(X.iloc[a], y[a], cat_features=catcols), eval_set=Pool(X.iloc[b], y[b], cat_features=catcols),
          use_best_model=True)
    oof[b] = m.predict_proba(X.iloc[b])[:, 1]; pred += m.predict_proba(pt)[:, 1] / 5
    its.append(m.get_best_iteration())
    print(f"  {A.name} fold {f} auc={roc_auc_score(y[b], oof[b]):.5f} it={its[-1]} {time.time()-t0:.0f}s", flush=True)
save(A.name, oof, pred, y, f"its={its} args={vars(A)}")
