"""T/X/H-learner 학습과 비교 (UC-02, UC-03).

    python -m uplift.train                              # 임시 원시 로그로 파이프라인 점검
    python -m uplift.train --data simulator/artifacts/v5_raw   # 시뮬레이터 실행 디렉터리로 학습

세 모델 모두 같은 입력 단계를 쓴다: common.raw.load_raw -> common.features.build_dataset.
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .common import schema as S
from .common.evaluate import against_truth, ate, qini_coefficient
from .common.features import build_dataset
from .common.raw import load_raw
from .common.synthetic import make_synthetic
from .Hlearner import HLearner
from .Tlearner import TLearner
from .Xlearner import XLearner

LEARNERS = {"tlearner": TLearner, "xlearner": XLearner, "hlearner": HLearner}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", help="시뮬레이터 실행 디렉터리(raw.sqlite가 있는 곳). 없으면 임시 데이터를 만든다")
    ap.add_argument("--n", type=int, default=2000, help="임시 데이터의 고객 수")
    ap.add_argument("--out", default="uplift/artifacts", help="모델과 결과를 저장할 디렉토리")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    truth = None
    if args.data:
        raw = load_raw(args.data)
    else:
        print("[주의] 시뮬레이터 데이터가 아닌 임시 데이터다. 아래 수치는 파이프라인 점검용이다.")
        raw, truth = make_synthetic(args.n, args.seed)
        truth = truth.set_index(S.CUSTOMER_ID)["uplift"]
    propensity = float(raw.assignments["assignment_probability"].iloc[0])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = build_dataset(raw)
    data.to_parquet(out / "dataset.parquet", index=False)
    data = data.dropna(subset=[S.LABEL])  # 결과 28일이 다 관측된 고객만
    train, test = train_test_split(data, test_size=0.3, stratify=data[S.TREATMENT],
                                   random_state=args.seed)
    print(f"train {len(train)}명, test {len(test)}명, test ATE {ate(test[S.TREATMENT], test[S.LABEL]):+.4f}")

    rows = []
    for key, cls in LEARNERS.items():
        model = cls().fit(train[S.FEATURES], train[S.TREATMENT], train[S.LABEL], propensity)
        model.save(out / f"{key}.pkl")
        pred = model.predict(test[S.FEATURES])
        row = {"model": model.name,
               "qini": qini_coefficient(pred["uplift_score"], test[S.TREATMENT], test[S.LABEL]),
               "mean_uplift": pred["uplift_score"].mean()}
        if truth is not None:
            row.update(against_truth(pred["uplift_score"], truth.loc[test[S.CUSTOMER_ID]]))
        if key == "hlearner":
            print(f"H-learner: lambda = {model.lambda_}, 반복 {model.n_rounds_}회")
        rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(out / "evaluation.csv", index=False)
    print(result.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
