"""Robustness check: fit blend weights on half of the folds, evaluate on the other half."""
import sys
from scipy.stats import rankdata
from scipy.optimize import minimize
from common import *
names = sys.argv[1:]
tr, te, y = load(); F = folds(y)
O = np.column_stack([rankdata(np.load(f"{W}/{n}_oof.npy")) / len(y) for n in names])
def fit(idx):
    f = lambda w: -roc_auc_score(y[idx], O[idx] @ (np.abs(w) / np.abs(w).sum()))
    r = minimize(f, np.ones(len(names)) / len(names), method="Nelder-Mead", options=dict(maxiter=400, xatol=1e-3, fatol=1e-7))
    return np.abs(r.x) / np.abs(r.x).sum()
A = np.concatenate([F[i][1] for i in (0, 1, 2)]); B = np.concatenate([F[i][1] for i in (3, 4)])
for fitset, ev, lab in [(A, B, "fit012->eval34"), (B, A, "fit34->eval012")]:
    w = fit(fitset); eq = np.ones(len(names)) / len(names)
    best1 = max(roc_auc_score(y[ev], O[ev, i]) for i in range(len(names)))
    print(lab, "opt", round(roc_auc_score(y[ev], O[ev] @ w), 5), "equal", round(roc_auc_score(y[ev], O[ev] @ eq), 5), "best single", round(best1, 5))
