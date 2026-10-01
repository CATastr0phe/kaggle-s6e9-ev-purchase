"""Usage: python blend.py <out_name> name1 name2 ... -> OOF-optimized rank blend + validated submission."""
import sys
from scipy.stats import rankdata
from scipy.optimize import minimize
from common import *

out = sys.argv[1]; names = sys.argv[2:]
tr, te, y = load()
ss = pd.read_csv(f"{D}/sample_submission.csv")
O = np.column_stack([rankdata(np.load(f"{W}/{n}_oof.npy")) / len(y) for n in names])
T = np.column_stack([rankdata(np.load(f"{W}/{n}_test.npy")) / len(te) for n in names])
for i, n in enumerate(names):
    print(f"{n}: {roc_auc_score(y, O[:, i]):.5f}")
import os
if len(names) > 1 and not os.environ.get("EQUAL"):
    f = lambda w: -roc_auc_score(y, O @ (np.abs(w) / np.abs(w).sum()))
    res = minimize(f, np.ones(len(names)) / len(names), method="Nelder-Mead",
                   options=dict(maxiter=300, xatol=1e-3, fatol=1e-7))
    w = np.abs(res.x) / np.abs(res.x).sum()
else:
    w = np.ones(len(names)) / len(names)
auc = roc_auc_score(y, O @ w)
print("weights", dict(zip(names, np.round(w, 3))), f"blend OOF AUC = {auc:.5f}")
p = T @ w
p = (p - p.min()) / (p.max() - p.min())
sub = ss.copy(); assert (sub.id.values == te.id.values).all()
sub[TARGET] = p
assert len(sub) == 286571 and sub[TARGET].notna().all() and sub[TARGET].between(0, 1).all()
os.makedirs("submissions", exist_ok=True)
sub.to_csv(f"submissions/{out}.csv", index=False)
print("saved", out, sub.shape)
