"""시뮬레이터 동봉 population 설정으로 작은 데이터셋을 만든다(시뮬레이터 코드는 읽기만 한다).

``simulator/src/simulate/v4/config/population.yaml``에 쇼핑·반품·쿠폰·리뷰·방문 시점·GMM 집계 모델이
들어 있어 외부 자산이 필요 없다. 수십 명 규모라 raw.sqlite·manifest·weekly_panel 계약과 금액 배선만 본다.
"""
from __future__ import annotations

from pathlib import Path
import sys

SIMULATOR_SRC = Path(__file__).resolve().parents[2] / "simulator" / "src"


def import_simulator():
    """저장소의 ``simulate``를 import할 수 있게 하고 테스트가 쓰는 설정 경로·엔진·feature 생성 함수를 반환한다."""
    if str(SIMULATOR_SRC) not in sys.path:
        sys.path.insert(0, str(SIMULATOR_SRC))
    from simulate.v4.configuration import POPULATION_CONFIG
    from simulate.v4.simulator.engine import EcommerceSimulator
    from simulate.v4.storage import materialize_features
    if not POPULATION_CONFIG.is_file():
        raise FileNotFoundError(POPULATION_CONFIG)
    return POPULATION_CONFIG, EcommerceSimulator, materialize_features


def generate(folder: Path, n_customers: int = 60, months: int = 24, seed: int = 17) -> tuple[Path, Path]:
    """``dataset/``(raw.sqlite와 부속 파일)와 ``features/``(KIT39 + weekly_panel)를 쓰고 두 경로를 반환한다."""
    config, EcommerceSimulator, materialize_features = import_simulator()
    result = EcommerceSimulator(config).generate(n_customers=n_customers, months=months, seed=seed)
    dataset = folder / "dataset"
    result.save(dataset)
    materialize_features(dataset, folder / "features")
    return dataset, folder / "features"
