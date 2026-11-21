"""시뮬레이터 데이터로 T/X/H의 28일 구매 효과를 학습하고 CSV로 내보낸다."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .common import schema as S
from .common.evaluate import against_truth, ate, purchase_roc_auc, qini_area, qini_curve, qini_decile_curve
from .common.features import build_dataset
from .common.raw import load_raw
from .common.simulator import load_feature_dataset
from .common.synthetic import make_synthetic
from .Hlearner import HLearner
from .Tlearner import TLearner
from .Xlearner import XLearner

LEARNERS = {"tlearner": TLearner, "xlearner": XLearner, "hlearner": HLearner}
MODEL_VERSION = "kit36-purchase28d-v1"


def load_training_input(path, n, seed):
    """SQLite 피처 저장소, 기존 raw 실행 디렉터리 또는 합성 예제를 읽는다."""
    truth = None
    if path is None:
        print("합성 예제의 수치는 파이프라인 점검용입니다.")
        raw, truth = make_synthetic(n, seed)
        truth = truth.set_index(S.CUSTOMER_ID)["uplift"]
    elif (path / "features.sqlite").is_file() or (path.is_file() and path.name != S.RAW_DATABASE):
        return load_feature_dataset(path), truth
    else:
        raw = load_raw(path.parent if path.is_file() else path)
    if not raw.assignments.assignment_probability.eq(0.5).all():
        raise ValueError("고정 1:1 실험의 배정 확률은 0.5여야 한다")
    data = build_dataset(raw)
    data["dataset_id"] = raw.metadata.get("dataset_id")
    return data, truth


def response_segment(pred):
    """프로젝트의 점수·기본 유지확률 기준에 따른 추정 세그먼트."""
    return np.select(
        [pred.uplift_score < 0, pred.p0 >= 0.5, pred.uplift_score > 0.05],
        ["Sleeping Dogs", "Sure Things", "Persuadables"],
        default="Lost Causes",
    )


def train_and_export(data, out, seed, truth=None):
    """같은 분할로 세 모델을 학습하고 전체 고객 점수와 평가 고객 지표를 저장한다."""
    mature = data.dropna(subset=[S.LABEL])
    if mature.empty:
        raise ValueError("28일간 성과가 관측된 적격 고객이 없다")
    train, test = train_test_split(mature, test_size=0.3, stratify=mature[S.TREATMENT],
                                   random_state=seed)
    test_ate = ate(test[S.TREATMENT], test[S.LABEL])
    print(f"train {len(train)}명, test {len(test)}명, test ATE {test_ate:+.4f}")

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    data.to_parquet(out / "dataset.parquet", index=False)
    keys = data[["dataset_id", S.CUSTOMER_ID, S.T0]].copy()
    keys["split"] = "unobserved"
    keys.loc[train.index, "split"] = "train"
    keys.loc[test.index, "split"] = "test"
    rows, curves, decile_curves, predictions = [], [], [], []
    for key, cls in LEARNERS.items():
        model = cls().fit(train[S.FEATURES], train[S.TREATMENT], train[S.LABEL])
        model.save(out / f"{key}.pkl")
        pred = model.predict(data[S.FEATURES])
        predictions.append(keys.join(pred).assign(model=model.name, model_version=MODEL_VERSION,
                                                  segment=response_segment(pred)))
        test_pred = pred.loc[test.index]
        score = test_pred["uplift_score"]
        curve = qini_curve(score, test[S.TREATMENT], test[S.LABEL])
        decile_curve = qini_decile_curve(score, test[S.TREATMENT], test[S.LABEL])
        curves.append(curve.assign(model=model.name))
        decile_curves.append(decile_curve.assign(model=model.name))
        row = {"model": model.name, "n_test": len(test), "ate": test_ate,
               "qini": qini_area(curve), "qini_decile": qini_area(decile_curve),
               "roc_auc": purchase_roc_auc(test_pred, test[S.TREATMENT], test[S.LABEL]),
               "mean_uplift": score.mean()}
        if truth is not None:
            row.update(against_truth(score, truth.loc[test[S.CUSTOMER_ID]]))
        if key == "hlearner":
            print(f"H-learner: lambda = {model.lambda_}, 반복 {model.n_rounds_}회")
        rows.append(row)

    pd.concat(predictions).to_csv(out / "uplift_predictions.csv", index=False)
    pd.concat(curves).to_csv(out / "qini_curve.csv", index=False)
    pd.concat(decile_curves).to_csv(out / "qini_decile_curve.csv", index=False)
    result = pd.DataFrame(rows)
    result.to_csv(out / "evaluation.csv", index=False)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, help="features.sqlite 또는 시뮬레이터의 features/raw 출력 디렉터리")
    ap.add_argument("--n", type=int, default=2000, help="입력 생략 시 합성 예제의 고객 수")
    ap.add_argument("--out", type=Path, default=Path("uplift/artifacts"), help="모델 및 CSV 출력 디렉터리")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    data, truth = load_training_input(args.data, args.n, args.seed)
    result = train_and_export(data, args.out, args.seed, truth)
    print(result.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
