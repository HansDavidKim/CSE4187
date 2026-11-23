"""학습 산출물 번들의 모델 카드(``model_card.json``): 버전, 코드·라이브러리, 학습 데이터 출처, 파일 체크섬.

다른 모듈(전략·예산 최적화, 대시보드)에 모델 파일을 넘길 때 함께 전달하는 기록이다.
``train``이 registry를 쓴 뒤 만들고, ``read_model_card``는 불러올 때 파일 체크섬을 다시 대조한다.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess

from . import __version__
from .io import read_json, write_json
from .schema import GROSS_PREDICTION_COLUMN, GROSS_WEEKS, INPUT_WEEKS, LABEL_WEEKS, PREDICTION_COLUMN

CARD_NAME = "model_card.json"
# 2: 학습 모집단 정책(``policy_condition``)과 4주 조건부 총구매액 모델(``gross_4w``)을 기록한다.
BUNDLE_SCHEMA_VERSION = 2
# 읽을 수 있는 번들 형식. 1은 4주 모델이 없는 기존 번들(배정군 혼합 학습)이다.
SUPPORTED_BUNDLE_SCHEMAS = (1, 2)
# 예측 대상 정의 식별자. 정의(구간·금액)가 바뀌면 번호를 올린다.
TARGET_DEFINITION_ID = "clv_paid_amount_52w_v1"
# 모든 모델이 음수 예측을 0으로 자르므로 CLV 값은 0 이상의 금액이다.
VALUE_SEMANTICS = "nonnegative_amount"
TARGET = {
    "definition_id": TARGET_DEFINITION_ID,
    "column": PREDICTION_COLUMN,
    "horizon_value": LABEL_WEEKS,
    "horizon_unit": "weeks",
    "window": f"[t0, t0 + {LABEL_WEEKS} weeks) = 364 days; not 12 calendar months",
    "input_window": f"[t0 - {INPUT_WEEKS} weeks, t0)",
    "definition": "sum of paid amount (gross_revenue - redemption_discount_cost) per customer; "
                  "coupon send cost is not deducted",
    "value_semantics": VALUE_SEMANTICS,
    "column_name_note": f"{PREDICTION_COLUMN} keeps its historical name; the horizon is 52 weeks",
}
GROSS_MODEL = "xgboost_gross_4w"
GROSS_TARGET_DEFINITION_ID = "gross_amount_4w_if_purchase_v1"
GROSS_TARGET = {
    "definition_id": GROSS_TARGET_DEFINITION_ID,
    "column": GROSS_PREDICTION_COLUMN,
    "horizon_value": GROSS_WEEKS,
    "horizon_unit": "weeks",
    "window": f"[t0, t0 + {GROSS_WEEKS} weeks) = 28 days, same customer_id and t0 as {PREDICTION_COLUMN}",
    "input_window": f"[t0 - {INPUT_WEEKS} weeks, t0)",
    "definition": "E[sum of gross amount before discount (gross_revenue, repeat orders included) | at least one "
                  "purchase in the window, X(t0)]; discount and coupon send cost are not deducted",
    "value_semantics": VALUE_SEMANTICS,
    "condition": "conditional on purchase; multiply by a purchase probability or uplift before comparing to amounts",
}
# 4주 모델이 없거나 학습되지 않았을 때 예측 대신 쓰는 상태 값(0이나 52주 값으로 대체하지 않는다).
GROSS_NOT_TRAINED = "not_trained"
# 모델별 저장 파일(번들 디렉터리 기준 상대 경로)
MODEL_FILES = {
    "baseline": ("models/baseline.json",),
    "bgnbd_gg": ("models/bgnbd_gg.json",),
    "xgboost": ("models/xgboost/xgboost_model.json", "models/xgboost/xgboost_meta.json"),
    "xgboost_two_part": ("models/xgboost_two_part/classifier.json", "models/xgboost_two_part/regressor.json",
                         "models/xgboost_two_part/two_part_meta.json"),
    GROSS_MODEL: ("models/xgboost_gross_4w/regressor.json", "models/xgboost_gross_4w/gross_4w_meta.json"),
}
# 모든 모델이 함께 쓰는 파일
SHARED_FILES = ("registry.json", "splits.csv")
# 저장소에 함께 올린 전달용 번들(seed 42 학습 데이터셋 20,000명, api.md 참고)
HANDOFF_BUNDLE = Path(__file__).resolve().parent / "artifacts" / "population-24m-n20000-seed42"


def file_sha256(path: str | Path) -> str:
    """파일의 SHA256(소문자 16진수)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_version() -> dict:
    """``clv`` 버전과, git 저장소 안이면 커밋·``clv/`` 미커밋 변경 여부."""
    folder = Path(__file__).resolve().parent
    result = {"clv_version": __version__, "git_commit": None, "clv_dirty": None}
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=folder, capture_output=True, text=True,
                                timeout=10, check=True).stdout.strip()
        # 전달 번들(artifacts/)은 코드가 아니므로 미커밋 변경 판정에서 뺀다.
        status = subprocess.run(["git", "status", "--porcelain", "--", ".", ":(exclude)artifacts"], cwd=folder,
                                capture_output=True,
                                text=True, timeout=10, check=True).stdout
        result.update(git_commit=commit or None, clv_dirty=bool(status.strip()))
    except (OSError, subprocess.SubprocessError):
        # git이 없거나 저장소 밖(복사한 번들)이면 커밋을 기록하지 않는다.
        pass
    return result


def environment() -> dict:
    """Python과 주요 라이브러리 버전(설치되지 않은 라이브러리는 None)."""
    versions = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "scipy", "xgboost", "pyarrow"):
        try:
            versions[name] = __import__(name).__version__
        except ImportError:
            versions[name] = None
    return versions


def model_version(name: str, checksums: dict) -> str:
    """번들 안 한 모델의 버전 문자열: ``clv-<버전>+<모델>.<registry·모델 파일 체크섬 요약 12자>``."""
    files = [*SHARED_FILES, *MODEL_FILES[name]]
    digest = hashlib.sha256(json.dumps({path: checksums[path] for path in files}, sort_keys=True).encode()).hexdigest()
    return f"clv-{__version__}+{name}.{digest[:12]}"


def gross_status(registry: dict) -> str:
    """registry의 4주 모델 상태. 이 모델이 생기기 전 번들은 ``not_trained``다."""
    return (registry.get("gross_4w") or {}).get("status", GROSS_NOT_TRAINED)


def write_model_card(output_dir: str | Path, registry: dict) -> dict:
    """registry·분할·모델 파일을 모두 쓴 뒤 ``model_card.json``을 만든다."""
    output = Path(output_dir)
    gross = registry.get("gross_4w") or {}
    versioned = [*registry["trained_models"], *([GROSS_MODEL] if gross_status(registry) == "trained" else [])]
    files = [*SHARED_FILES, *(path for name in versioned for path in MODEL_FILES[name])]
    checksums = {path: file_sha256(output / path) for path in files}
    provenance, ledger = registry["provenance"], registry.get("ledger") or {}
    versions = {name: model_version(name, checksums) for name in versioned}
    policy = registry["population"]["policy"]
    card = {
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "final_model": registry["final_model"],
        "model_versions": versions,
        "model_status": registry["model_status"],
        "target": TARGET,
        "predictors": registry["predictors"],
        "code": code_version(),
        "environment": environment(),
        "training_data": {
            "reference_week": registry["reference_week"],
            "dataset_id": provenance.get("dataset_id"),
            "currency": provenance.get("currency"),
            "amount_unit": provenance.get("amount_unit"),
            "raw_database_sha256": ledger.get("raw_database_sha256") or provenance.get("raw_database_sha256"),
            "weekly_panel_sha256": provenance.get("weekly_sha256"),
            "simulator": ledger.get("source_metadata"),
            "training_population": registry["training_population"],
            "split_counts": registry["split_counts"],
            "labelled_customers": registry["dataset"]["labelled_customers"],
            "treatment_in_label_window": registry["dataset"]["treatment_in_label_window"],
            "population": registry["population"],
        },
        # 학습 모집단 정책. control_no_coupon이면 쿠폰을 보내지 않았을 때의 값, observed_experiment_mix면
        # 정답에 학습 실험의 쿠폰 발송(Treatment)이 섞인 값이다.
        "policy_condition": policy,
        "test_metrics": registry["test_metrics"],
        "gross_4w": {"model": GROSS_MODEL, "status": gross_status(registry), "model_version": versions.get(GROSS_MODEL),
                     "target": GROSS_TARGET, "policy_condition": policy, "split_counts": gross.get("split_counts"),
                     "test_metrics": gross.get("test_metrics")},
        "files_sha256": checksums,
    }
    write_json(output / CARD_NAME, card)
    return card


def read_model_card(model_dir: str | Path, verify: bool = True) -> dict:
    """``model_card.json``을 읽고, ``verify``이면 기록된 모든 파일의 체크섬을 다시 대조한다.

    Raises:
        FileNotFoundError: 모델 카드가 없는 번들(이 기능 이전 버전으로 학습). 다시 학습하면 생긴다.
        ValueError: 파일이 없거나 학습 이후 바뀐 경우.
    """
    model_dir = Path(model_dir)
    path = model_dir / CARD_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: pass the shipped bundle (clv.api.HANDOFF_BUNDLE = {HANDOFF_BUNDLE}) "
                                "or a directory written by `python -m clv train`")
    card = read_json(path)
    if card.get("bundle_schema_version") not in SUPPORTED_BUNDLE_SCHEMAS:
        raise ValueError(f"unsupported CLV bundle schema: {card.get('bundle_schema_version')}")
    if verify:
        changed = [name for name, digest in card["files_sha256"].items()
                   if not (model_dir / name).is_file() or file_sha256(model_dir / name) != digest]
        if changed:
            raise ValueError(f"CLV bundle files are missing or changed after training: {changed}")
    return card
