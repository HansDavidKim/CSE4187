"""Command line: ``python -m clv <command>`` from the repository root.

Commands
    validate   schema, money, units, coverage and ledger checks
    cutoffs    list cutoffs t0 whose 13 + 52 weeks are observed
    train      fit and evaluate baseline, BG/NBD + Gamma-Gamma and XGBoost
    evaluate   score saved models at a cutoff
    predict    write the prediction CSV (no label needed)
    run        train + evaluate + predict
"""
# WBS 4.7: CLV 명령행 진입점(validate, cutoffs, train, evaluate, predict, run).
# 위 모듈 docstring은 `--help` 설명(description=__doc__)으로 출력되므로 영어 그대로 둔다.
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .io import jsonable


def _inputs(parser):
    """모든 하위 명령에 공통인 입력 인자(주간 패널, 주문 원장, 통화·단위, 체크섬)를 추가한다."""
    parser.add_argument("--weekly", required=True, type=Path,
                        help="weekly_panel.parquet, a simulator features/export directory, or a CSV")
    parser.add_argument("--ledger", type=Path, help="raw.sqlite, a dataset directory, or an order CSV/Parquet")
    parser.add_argument("--currency", help="only for inputs without a currency column")
    parser.add_argument("--amount-unit", help="only for inputs without an amount_unit column")
    parser.add_argument("--skip-checksum", action="store_true", help="do not verify raw.sqlite against its manifest")


def _options(args) -> dict:
    """파싱한 인자에서 ``prepare_inputs``에 넘길 입력 옵션을 만든다."""
    return {"currency": args.currency, "amount_unit": args.amount_unit, "verify_checksum": not args.skip_checksum}


def build_parser() -> argparse.ArgumentParser:
    """``python -m clv``의 하위 명령 파서를 만든다."""
    parser =argparse.ArgumentParser(prog="python -m clv", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "cutoffs"):
        _inputs(commands.add_parser(name))
    train = commands.add_parser("train")
    _inputs(train)
    train.add_argument("--output", type=Path, default=Path("results/clv"))
    train.add_argument("--config", type=Path, help="JSON or YAML settings (a 'clv:' section is allowed)")
    train.add_argument("--cutoff", help="training cutoff t0 (Asia/Seoul Monday), default: most eligible customers")
    evaluate = commands.add_parser("evaluate")
    _inputs(evaluate)
    evaluate.add_argument("--model-dir", type=Path, default=Path("results/clv"))
    evaluate.add_argument("--cutoff", help="default: the training cutoff")
    evaluate.add_argument("--population", choices=("test", "all"), default="test")
    evaluate.add_argument("--output", type=Path, help="write the evaluation JSON here")
    predict = commands.add_parser("predict")
    _inputs(predict)
    predict.add_argument("--model-dir", type=Path, default=Path("results/clv"))
    predict.add_argument("--reference-week", help="prediction cutoff; default: experiment as_of_at or data end")
    predict.add_argument("--model", choices=("baseline", "bgnbd_gg", "xgboost"), help="default: registry final model")
    predict.add_argument("--output", type=Path, default=Path("results/clv_predictions.csv"))
    run = commands.add_parser("run")
    _inputs(run)
    run.add_argument("--output", type=Path, default=Path("results/clv"))
    run.add_argument("--predictions", type=Path, default=Path("results/clv_predictions.csv"))
    run.add_argument("--config", type=Path)
    run.add_argument("--cutoff")
    run.add_argument("--reference-week")
    return parser


def main(argv=None) -> dict:
    """명령을 실행하고 요약을 JSON으로 출력한 뒤 그 요약을 반환한다."""
    args = build_parser().parse_args(argv)
    # pipeline(pandas, xgboost 등)은 인자 파싱이 끝난 뒤에 불러온다.
    from . import pipeline
    from .dataset import find_eligible_cutoffs

    options = _options(args)
    if args.command == "validate":
        # strict=False: 첫 오류에서 멈추지 않고 모든 검증 결과를 모아 보고한다.
        _, _, reports = pipeline.prepare_inputs(args.weekly, args.ledger, strict=False, **options)
        summary = {report.name: report.to_dict() for report in reports}
        summary["ok"] = all(report.ok for report in reports)
    elif args.command == "cutoffs":
        panel, _, _ = pipeline.prepare_inputs(args.weekly, args.ledger, **options)
        table = find_eligible_cutoffs(panel)
        summary = {"cutoffs": table.loc[table["eligible_customers"] > 0].to_dict("records"),
                   "candidate_cutoffs": int(len(table))}
    elif args.command == "train":
        registry = pipeline.train(args.weekly, args.ledger, args.output, load_config(args.config), args.cutoff, **options)
        summary = {key: registry[key] for key in ("reference_week", "dataset", "split_counts", "model_status",
                                                  "final_model", "test_metrics")}
    elif args.command == "evaluate":
        report = pipeline.evaluate(args.weekly, args.ledger, args.model_dir, args.cutoff, args.population,
                                   args.output, **options)
        summary = {"reference_week": report["reference_week"], "customers": report["customers"],
                   "models": {name: {key: value[key] for key in ("mae", "rmse")} | {"top20_overlap_rate": value["top"]["overlap_rate"]}
                              for name, value in report["models"].items()}}
    elif args.command == "predict":
        predictions = pipeline.predict(args.weekly, args.ledger, args.model_dir, args.reference_week, args.model, **options)
        path = pipeline.write_predictions(predictions, args.output)
        summary = {"predictions": str(path), "customers": int(len(predictions)),
                   "reference_week": predictions["reference_week"].iloc[0], "model": predictions["model_name"].iloc[0],
                   "high_value_customers": int(predictions["is_high_value_customer"].sum())}
    else:
        result = pipeline.run(args.weekly, args.ledger, args.output, args.predictions, load_config(args.config),
                              args.cutoff, args.reference_week, **options)
        registry = result.pop("registry")
        summary = {**result, "training_reference_week": registry["reference_week"], "final_model": registry["final_model"],
                   "model_status": registry["model_status"], "test_metrics": registry["test_metrics"]}
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    main()
