import time, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

D = "data"
W = "preds"
TARGET = "Will_Buy_EV"
SEED = 42
NUMS = ["Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned",
        "Charging_Stations_Near_Home", "Charging_Stations_Near_Work", "Environmental_Concern_Level"]
CATS = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
        "Subsidy_Available", "Range_Anxiety_Level"]


def load():
    tr = pd.read_csv(f"{D}/train.csv"); te = pd.read_csv(f"{D}/test.csv")
    y = (tr[TARGET] == "Yes").astype(int).values
    return tr, te, y


def folds(y):
    return list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))


def build(tr, te, level=1):
    """level 0: raw (cats as category). level 1: + FE."""
    full = pd.concat([tr.drop(columns=[TARGET]), te], ignore_index=True)
    X = pd.DataFrame(index=full.index)
    for c in NUMS:
        X[c] = full[c]
    for c in CATS:
        X[c] = full[c].astype("category")
    if level >= 1:
        inc, com = full.Annual_Income_USD, full.Daily_Commute_km
        X["inc_floor"] = (inc == 30000).astype(int)
        X["com_floor"] = (com == 5.0).astype(int)
        for c in ["Annual_Income_USD", "Daily_Commute_km", "Age"]:
            X[f"cnt_{c}"] = full[c].map(full[c].value_counts()).astype(float)
        X["inc_per_car"] = inc / full.Number_of_Cars_Owned.clip(lower=1)
        X["st_sum"] = full.Charging_Stations_Near_Home + full.Charging_Stations_Near_Work
        X["st_diff"] = full.Charging_Stations_Near_Home - full.Charging_Stations_Near_Work
        X["com_per_st"] = com / (1 + X["st_sum"])
        X["inc_x_sub"] = inc * (full.Subsidy_Available == "Yes")
        X["env_sub"] = (full.Environmental_Concern_Level.astype(int).astype(str) + "_" +
                        full.Subsidy_Available + "_" + full.Home_Charging_Possible + "_" +
                        full.Range_Anxiety_Level).astype("category")
        for c in ["Age", "Charging_Stations_Near_Home", "Charging_Stations_Near_Work"]:
            X[f"{c}_cat"] = full[c].astype(str).astype("category")
    n = len(tr)
    return X.iloc[:n].reset_index(drop=True), X.iloc[n:].reset_index(drop=True)


def save(name, oof, pred, y, log=""):
    import os
    os.makedirs(W, exist_ok=True)
    np.save(f"{W}/{name}_oof.npy", oof); np.save(f"{W}/{name}_test.npy", pred)
    auc = roc_auc_score(y, oof)
    print(f"[{name}] OOF AUC = {auc:.5f} {log}", flush=True)
    return auc


def run_lgb(name, X, Xt, y, params, rounds=10000, es=150):
    import lightgbm as lgb
    oof = np.zeros(len(X)); pred = np.zeros(len(Xt)); its = []
    for f, (a, b) in enumerate(folds(y)):
        t0 = time.time()
        m = lgb.train(params, lgb.Dataset(X.iloc[a], y[a]), rounds,
                      valid_sets=[lgb.Dataset(X.iloc[b], y[b])],
                      callbacks=[lgb.early_stopping(es, verbose=False)])
        oof[b] = m.predict(X.iloc[b], num_iteration=m.best_iteration)
        pred += m.predict(Xt, num_iteration=m.best_iteration) / 5
        its.append(m.best_iteration)
        print(f"  {name} fold {f} auc={roc_auc_score(y[b], oof[b]):.5f} it={m.best_iteration} {time.time()-t0:.0f}s", flush=True)
    return save(name, oof, pred, y, f"its={its}")
