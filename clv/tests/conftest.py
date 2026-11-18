"""공용 Fixture: 합성 시뮬레이터 산출물, CLV 이름으로 바꾼 주간 패널·주문 원장, 디스크 배치, 행 생성 도우미."""
import pandas as pd
import pytest

from clv.adapter import adapt_orders, adapt_weekly
from clv.dataset import complete_grid

from .fixtures import synthetic_outputs, write_raw_sqlite


@pytest.fixture(scope="session")
def outputs():
    """시뮬레이터 형태의 주간 패널, 주문 원장, 메타데이터."""
    return synthetic_outputs()


@pytest.fixture(scope="session")
def panel(outputs):
    """CLV 이름으로 바꾸고 빠진 고객×주 행을 미관측 주로 채운 주간 패널."""
    weekly, _, _ = outputs
    return complete_grid(adapt_weekly(weekly))


@pytest.fixture(scope="session")
def orders(outputs):
    """CLV 이름으로 바꾼 주문 원장(통화·단위는 메타데이터 값 그대로)."""
    _, ledger, metadata = outputs
    return adapt_orders(ledger, currency=metadata["currency"], amount_unit=metadata["amount_unit"])


@pytest.fixture(scope="session")
def raw_dir(outputs, tmp_path_factory):
    """시뮬레이터 출력 구성(``dataset/raw.sqlite``, ``features/weekly_panel.parquet``)을 흉내 낸 디렉터리."""
    weekly, ledger, metadata = outputs
    directory = tmp_path_factory.mktemp("simulator-like")
    write_raw_sqlite(directory / "dataset", ledger, metadata)
    (directory / "features").mkdir()
    weekly.to_parquet(directory / "features" / "weekly_panel.parquet", index=False)
    return directory


def clv_rows(customer_weeks, start="2024-03-04", currency="SOURCE_UNSPECIFIED", amount_unit="0.01"):
    """합의된 CLV 스키마 행을 만든다.

    인자는 {고객: [주마다 (sessions, purchases, gross, discount) 또는 None]}이고,
    None인 주는 행을 만들지 않는다(미관측 주).
    """
    rows = []
    for customer, weeks in customer_weeks.items():
        for offset, values in enumerate(weeks):
            week = pd.Timestamp(start, tz="Asia/Seoul") + pd.Timedelta(weeks=offset)
            if values is None:
                continue
            sessions, purchases, gross, discount = values
            rows.append({"customer_id": customer, "week": week, "sessions": sessions, "purchases": purchases,
                         "gross_revenue": gross, "redemption_discount_cost": discount, "coupons": 0,
                         "coupon_action": 0, "currency": currency, "amount_unit": amount_unit})
    return pd.DataFrame(rows)
