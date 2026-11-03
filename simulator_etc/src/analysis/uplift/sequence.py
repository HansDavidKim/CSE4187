"""Predict next-day recorded activity; coupon receipt is not randomized treatment."""
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from utils.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
OUT = ROOT / "artifacts/uplift-sequence-pilot"
START = pd.Timestamp("2016-04-16")
END = pd.Timestamp("2016-06-15")
DATES = pd.date_range(START, END)


def features(click, purchase, coupon, target_day):
    """For Y_d, only behaviors through d-2; receipt on d-1 is a separate input."""
    assert target_day >= 16
    last = target_day - 1
    x = [click[:, last-1], purchase[:, last-1], coupon[:, last]]
    for length in (7, 14):
        x.extend([arr[:, last-length:last].sum(axis=1) for arr in (click, purchase, coupon)])
    weekday = DATES[target_day].dayofweek
    x.extend([np.full(len(click), int(weekday == day)) for day in range(1, 7)])
    return np.column_stack(x).astype(float)


def read_panel():
    # Cohort is fixed using activity before every training and evaluation target.
    path = ROOT / "data/raw/tianchi-o2o/ccf_online_stage1_train.csv"
    eligible = set()
    for chunk in pd.read_csv(path, usecols=["User_id", "Action", "Date"], dtype={"Date": "string"}, chunksize=1_000_000):
        day = pd.to_datetime(chunk.Date, format="%Y%m%d", errors="coerce")
        mask = day.between("2016-04-16", "2016-04-30") & chunk.Action.isin([0, 1])
        eligible.update(chunk.loc[mask, "User_id"].tolist())
    rng = np.random.default_rng(2026)
    candidates = np.array(sorted(eligible))
    users = rng.choice(candidates, size=min(15000, len(candidates)), replace=False)
    index = pd.Index(users)
    matrix = np.zeros((3, len(users), len(DATES)), dtype=np.uint8)
    for chunk in pd.read_csv(path, usecols=["User_id", "Action", "Date", "Date_received"],
                             dtype={"Date": "string", "Date_received": "string"}, chunksize=1_000_000):
        chunk = chunk[chunk.User_id.isin(index)].copy()
        date = pd.to_datetime(chunk.Date, format="%Y%m%d", errors="coerce")
        receipt = chunk.Action.eq(2)
        date.loc[receipt] = pd.to_datetime(chunk.loc[receipt, "Date_received"], format="%Y%m%d", errors="coerce")
        mask = date.between(START, END) & chunk.Action.isin([0, 1, 2])
        row = index.get_indexer(chunk.loc[mask, "User_id"])
        col = (date[mask] - START).dt.days.to_numpy()
        action = chunk.loc[mask, "Action"].to_numpy(dtype=int)
        matrix[action, row, col] = 1
    return users, matrix


def indices(first, last):
    return np.flatnonzero((DATES >= first) & (DATES <= last))


def dataset(matrix, users_mask, days):
    click, purchase, coupon = matrix[:, users_mask, :]
    x = np.vstack([features(click, purchase, coupon, day) for day in days])
    y = np.concatenate([click[:, day] + 2 * purchase[:, day] for day in days])
    return x, y


def counts(click, purchase):
    return {"click_day_rate": float(click.mean()), "purchase_day_rate": float(purchase.mean()),
            "any_click_14d": float(click.any(axis=1).mean()), "any_purchase_14d": float(purchase.any(axis=1).mean()),
            "mean_click_days_14d": float(click.sum(axis=1).mean()),
            "mean_purchase_days_14d": float(purchase.sum(axis=1).mean())}


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    users, matrix = read_panel()
    rng = np.random.default_rng(42)
    test_mask = rng.random(len(users)) < 0.2
    train_days = indices("2016-05-02", "2016-05-28")
    test_days = indices("2016-06-01", "2016-06-14")
    x_train, y_train = dataset(matrix, ~test_mask, train_days)
    x_test, y_test = dataset(matrix, test_mask, test_days)
    scaler = StandardScaler().fit(x_train)
    model = LogisticRegression(max_iter=500, C=1.0, random_state=42)
    model.fit(scaler.transform(x_train), y_train)
    assert np.array_equal(model.classes_, np.arange(4))
    probs = model.predict_proba(scaler.transform(x_test))
    keep = [i for i in range(x_train.shape[1]) if i not in (2, 5, 8)]
    no_coupon_model = LogisticRegression(max_iter=500, C=1.0, random_state=42)
    no_coupon_model.fit(scaler.transform(x_train)[:, keep], y_train)
    no_coupon_probs = no_coupon_model.predict_proba(scaler.transform(x_test)[:, keep])
    base = np.bincount(y_train, minlength=4) / len(y_train)
    baseline_loss = log_loss(y_test, np.tile(base, (len(y_test), 1)), labels=np.arange(4))
    # The actual receipt schedule is replayed; not an off-policy causal simulation.
    held = matrix[:, test_mask, :].copy()
    simulated = held.copy()
    simulated[:2, :, test_days] = 0
    for day in test_days:
        p = model.predict_proba(scaler.transform(features(*simulated, day)))
        choices = (rng.random(len(p))[:, None] > np.cumsum(p, axis=1)).sum(axis=1)
        simulated[0, :, day] = (choices % 2).astype(np.uint8)
        simulated[1, :, day] = (choices // 2).astype(np.uint8)
    true = counts(held[0][:, test_days], held[1][:, test_days])
    generated = counts(simulated[0][:, test_days], simulated[1][:, test_days])
    result = {"users_sampled": len(users), "train_users": int((~test_mask).sum()),
              "test_users": int(test_mask.sum()), "train_rows": len(y_train), "test_rows": len(y_test),
              "train_coupon_days": int(x_train[:, 2].sum()), "test_coupon_days": int(x_test[:, 2].sum()),
              "feature_names": ["click_lag2", "purchase_lag2", "coupon_lag1", "click_prev7", "purchase_prev7", "coupon_prev7", "click_prev14", "purchase_prev14", "coupon_prev14"] + [f"weekday_{i}" for i in range(1, 7)],
              "label": "0=no click/purchase record, 1=click only, 2=purchase only, 3=both",
              "test_log_loss": float(log_loss(y_test, probs, labels=np.arange(4))), "baseline_log_loss": float(baseline_loss),
              "no_coupon_history_test_log_loss": float(log_loss(y_test, no_coupon_probs, labels=np.arange(4))),
              "actual_class_rate": np.bincount(y_test, minlength=4).tolist(),
              "one_step_mean_predicted_class_rate": probs.mean(axis=0).tolist(),
              "real_14d": true, "sampled_14d": generated,
              "caveat": "no record is not proven inactivity; receipt is selected; no causal identification; replayed future coupons"}
    result["actual_class_rate"] = [x / len(y_test) for x in result["actual_class_rate"]]
    with (OUT / "model.pkl").open("wb") as stream:
        pickle.dump({"model": model, "scaler": scaler}, stream)
    (OUT / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run()
