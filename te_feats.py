"""In-fold target encoding of all features as categories + pairwise combos (computed inside each outer fold
with inner 5-fold for train rows to avoid leakage)."""
import itertools
from common import *

ALL = NUMS + CATS


def te_codes(tr, te):
    full = pd.concat([tr[ALL], te[ALL]], ignore_index=True)
    codes = {}
    for c in ALL:
        codes[c] = pd.factorize(full[c])[0].astype(np.int64)
    top = ["Environmental_Concern_Level", "Subsidy_Available", "Home_Charging_Possible", "Range_Anxiety_Level",
           "Age", "Charging_Stations_Near_Home", "Charging_Stations_Near_Work", "City_Type",
           "Number_of_Cars_Owned", "Current_Car_Type", "Gender"]
    import os
    if os.environ.get("PAIRS_ALL"):
        top = ALL
    for a, b in itertools.combinations(top, 2):
        codes[f"{a}__{b}"] = codes[a] * 100000 + codes[b]
    import os
    if os.environ.get("TRIPLES"):
        t3 = ["Environmental_Concern_Level", "Subsidy_Available", "Home_Charging_Possible", "Range_Anxiety_Level",
              "City_Type", "Number_of_Cars_Owned", "Current_Car_Type"]
        for a, b, c in itertools.combinations(t3, 3):
            codes[f"{a}__{b}__{c}"] = (codes[a] * 1000 + codes[b]) * 1000 + codes[c]
    n = len(tr)
    return {k: (v[:n], v[n:]) for k, v in codes.items()}


def _enc(ctr, ytr, cap, prior, m=20):
    s = pd.DataFrame({"c": ctr, "y": ytr}).groupby("c").y.agg(["sum", "count"])
    te_map = (s["sum"] + prior * m) / (s["count"] + m)
    return pd.Series(cap).map(te_map).fillna(prior).values


def add_te(X, Xt, y, codes, tr_idx, va_idx):
    """returns (Xtr, Xva, Xte) with TE columns appended for outer fold."""
    from sklearn.model_selection import StratifiedKFold
    prior = y[tr_idx].mean()
    Xa = X.iloc[tr_idx].copy(); Xb = X.iloc[va_idx].copy(); Xc = Xt.copy()
    inner = list(StratifiedKFold(5, shuffle=True, random_state=7).split(np.zeros(len(tr_idx)), y[tr_idx]))
    for k, (ctr_full, cte) in codes.items():
        ctr = ctr_full[tr_idx]; ytr = y[tr_idx]
        col = np.zeros(len(tr_idx))
        for i, j in inner:
            col[j] = _enc(ctr[i], ytr[i], ctr[j], prior)
        Xa["te_" + k] = col
        Xb["te_" + k] = _enc(ctr, ytr, ctr_full[va_idx], prior)
        Xc["te_" + k] = _enc(ctr, ytr, cte, prior)
    return Xa, Xb, Xc
