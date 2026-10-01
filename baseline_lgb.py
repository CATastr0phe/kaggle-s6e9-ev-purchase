"""S6E9 baseline: LightGBM, StratifiedKFold(5), OOF AUC, submission.
Usage: python baseline_lgb.py <data_dir> <out_csv>
"""
import sys, time
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
OUT = sys.argv[2] if len(sys.argv) > 2 else "submissions/sub_v1_lgb.csv"
TARGET = "Will_Buy_EV"
SEED = 42

tr = pd.read_csv(f"{DATA}/train.csv")
te = pd.read_csv(f"{DATA}/test.csv")
ss = pd.read_csv(f"{DATA}/sample_submission.csv")
print(tr.shape, te.shape, ss.shape)
print(tr.dtypes)
print(tr[TARGET].value_counts())

y = tr[TARGET]
if not pd.api.types.is_numeric_dtype(y):
    y = y.map({"Yes": 1, "No": 0, "yes": 1, "no": 0, "True": 1, "False": 0}).astype(int)
y = y.values
feats = [c for c in tr.columns if c not in ("id", TARGET)]
cats = [c for c in feats if not pd.api.types.is_numeric_dtype(tr[c])]
X = tr[feats].copy(); Xt = te[feats].copy()
for c in cats:
    allv = pd.concat([X[c], Xt[c]]).astype("category").cat.categories
    X[c] = pd.Categorical(X[c], categories=allv)
    Xt[c] = pd.Categorical(Xt[c], categories=allv)
print("cats:", cats, {c: X[c].nunique() for c in feats})

params = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=63,
              min_child_samples=50, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, max_cat_to_onehot=8, cat_smooth=10,
              n_jobs=2, verbose=-1, seed=SEED)
oof = np.zeros(len(X)); pred = np.zeros(len(Xt))
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
for f, (a, b) in enumerate(skf.split(X, y)):
    t0 = time.time()
    m = lgb.train(params, lgb.Dataset(X.iloc[a], y[a]), 5000,
                  valid_sets=[lgb.Dataset(X.iloc[b], y[b])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    oof[b] = m.predict(X.iloc[b], num_iteration=m.best_iteration)
    pred += m.predict(Xt, num_iteration=m.best_iteration) / 5
    print(f"fold {f} auc={roc_auc_score(y[b], oof[b]):.5f} it={m.best_iteration} {time.time()-t0:.0f}s", flush=True)
auc = roc_auc_score(y, oof)
print(f"OOF AUC = {auc:.5f}")
np.save(OUT.replace(".csv", "_oof.npy"), oof); np.save(OUT.replace(".csv", "_test.npy"), pred)

sub = ss.copy()
assert (sub["id"].values == te["id"].values).all()
sub[TARGET] = pred
assert len(sub) == len(ss) and sub[TARGET].notna().all() and sub[TARGET].between(0, 1).all()
import os; os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
sub.to_csv(OUT, index=False)
print("saved", OUT, sub.shape)
