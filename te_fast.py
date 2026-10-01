"""Fast in-fold target encoding (dense codes + np.bincount). Same semantics as te_feats.py."""
import itertools, os
from common import *
ALL = NUMS + CATS
TOP = ["Environmental_Concern_Level", "Subsidy_Available", "Home_Charging_Possible", "Range_Anxiety_Level",
       "Age", "Charging_Stations_Near_Home", "Charging_Stations_Near_Work", "City_Type",
       "Number_of_Cars_Owned", "Current_Car_Type", "Gender"]
T3 = ["Environmental_Concern_Level", "Subsidy_Available", "Home_Charging_Possible", "Range_Anxiety_Level",
      "City_Type", "Number_of_Cars_Owned", "Current_Car_Type"]


def te_codes(tr, te, pairs_all=False, triples=False):
    full = pd.concat([tr[ALL], te[ALL]], ignore_index=True)
    base = {c: pd.factorize(full[c])[0].astype(np.int64) for c in ALL}
    codes = dict(base)
    for a, b in itertools.combinations(ALL if pairs_all else TOP, 2):
        codes[f"{a}__{b}"] = base[a] * 100000 + base[b]
    if triples:
        for a, b, c in itertools.combinations(T3, 3):
            codes[f"{a}__{b}__{c}"] = (base[a] * 1000 + base[b]) * 1000 + base[c]
    n = len(tr); out = {}
    for k, v in codes.items():
        d = pd.factorize(v)[0]
        out[k] = (d[:n], d[n:], d.max() + 1)
    return out


def _enc(ctr, ytr, cap, prior, K, m=20):
    s = np.bincount(ctr, weights=ytr, minlength=K); c = np.bincount(ctr, minlength=K)
    return ((s + prior * m) / (c + m))[cap]


def add_te(X, Xt, y, codes, tr_idx, va_idx):
    from sklearn.model_selection import StratifiedKFold
    prior = y[tr_idx].mean()
    Xa = X.iloc[tr_idx].copy(); Xb = X.iloc[va_idx].copy(); Xc = Xt.copy()
    inner = list(StratifiedKFold(5, shuffle=True, random_state=7).split(np.zeros(len(tr_idx)), y[tr_idx]))
    new_a, new_b, new_c = {}, {}, {}
    ytr = y[tr_idx].astype(float)
    for k, (ctr_full, cte, K) in codes.items():
        ctr = ctr_full[tr_idx]
        col = np.zeros(len(tr_idx))
        for i, j in inner:
            col[j] = _enc(ctr[i], ytr[i], ctr[j], prior, K)
        new_a["te_" + k] = col
        new_b["te_" + k] = _enc(ctr, ytr, ctr_full[va_idx], prior, K)
        new_c["te_" + k] = _enc(ctr, ytr, cte, prior, K)
    Xa = pd.concat([Xa, pd.DataFrame(new_a, index=Xa.index)], axis=1)
    Xb = pd.concat([Xb, pd.DataFrame(new_b, index=Xb.index)], axis=1)
    Xc = pd.concat([Xc, pd.DataFrame(new_c, index=Xc.index)], axis=1)
    return Xa, Xb, Xc
