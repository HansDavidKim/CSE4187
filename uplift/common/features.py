"""원시 로그 -> KIT39 feature와 정답 -> 고객 1명당 1행의 학습 데이터셋 (UC-01).

T/X/H-learner가 공통으로 쓴다. feature 정의는 시뮬레이터 설계서 v5 13·14장(F01-F39)을 따른다.
모든 feature는 event_at < t0 인 기록만 읽고, 정답은 [t0, t0+28d)만 읽는다.

    raw = load_raw(run_dir)
    data = build_dataset(raw)          # customer_id, t0, treatment, KIT39, y
"""

import numpy as np
import pandas as pd

from . import schema as S
from .raw import RawData

C = S.CUSTOMER_ID
LOCAL_TZ = "Asia/Seoul"


def _days(delta) -> pd.Series:
    return delta.dt.total_seconds() / 86400


def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    return (num / den).where(den > 0)


def _replay_cart(events: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """t0 직전의 장바구니 잔존량. 담기 +, 제거 -, 구매는 보유분 한도까지 차감(buy_now는 보존)."""
    moves = events[events["event_type"].isin(["cart", "remove_from_cart"])]
    moves = pd.DataFrame({
        C: moves[C], "product_id": moves["product_id"], "event_at": moves["event_at"],
        "event_id": moves["event_id"], "take": False,
        "qty": moves["quantity"].astype(int) * np.where(moves["event_type"] == "cart", 1, -1),
    })
    bought = items[items["checkout_source"] != "buy_now"]
    bought = (bought.groupby([C, "product_id", "event_at", "purchase_event_id"], as_index=False)["quantity"]
              .sum().rename(columns={"purchase_event_id": "event_id", "quantity": "qty"}))
    bought["take"] = True
    stream = pd.concat([moves, bought]).sort_values([C, "product_id", "event_at", "event_id"])

    customer, product = stream[C].to_numpy(), stream["product_id"].to_numpy()
    qty, take = stream["qty"].to_numpy(int), stream["take"].to_numpy(bool)
    rows, held = [], 0
    for i in range(len(stream)):
        if i and (customer[i] != customer[i - 1] or product[i] != product[i - 1]):
            rows.append((customer[i - 1], product[i - 1], held))
            held = 0
        held = held - min(held, qty[i]) if take[i] else held + qty[i]
        if held < 0:
            raise ValueError(f"장바구니 수량이 음수가 된다: {customer[i]}, {product[i]}")
    if len(stream):
        rows.append((customer[-1], product[-1], held))
    cart = pd.DataFrame(rows, columns=[C, "product_id", "qty"])
    return cart[cart["qty"] > 0]


def _product_reviews(events: pd.DataFrame, orders: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """본인의 실제 구매·상품·주문행에 연결된 상품 별점. 구매 x 상품마다 한 번만 센다."""
    rating = pd.to_numeric(events["rating"], errors="coerce")
    reviews = events[(events["event_type"] == "review") & (events["rating_kind"] == S.PRODUCT_REVIEW_KIND)
                     & events["product_id"].notna() & rating.isin([1, 2, 3, 4, 5])
                     & events["related_purchase_event_id"].notna() & events["order_line_no"].notna()]
    reviews = reviews.assign(
        rating=rating[reviews.index].astype(float),
        purchase_event_id=reviews["related_purchase_event_id"].astype("int64"),
        line_no=reviews["order_line_no"].astype("int64"))
    bought = orders[[C, "purchase_event_id", "order_id", "event_at"]].rename(columns={"event_at": "purchased_at"})
    reviews = reviews.merge(bought, on=[C, "purchase_event_id"])
    reviews = reviews[reviews["purchased_at"] <= reviews["event_at"]]
    reviews = reviews.merge(items[["order_id", "product_id", "line_no"]], on=["order_id", "product_id", "line_no"])
    return reviews.sort_values("event_at", kind="stable").drop_duplicates(["purchase_event_id", "product_id"])


def build_features(tables: dict, snapshots: pd.DataFrame, observation_start=None) -> pd.DataFrame:
    """고객별 기준 시각 t0의 KIT39. snapshots: customer_id, t0 (고객당 1행). index는 customer_id."""
    snap = snapshots[[C, S.T0]]
    if snap[C].duplicated().any():
        raise ValueError("고객당 기준 시각은 하나여야 한다")
    t0 = snap.set_index(C)[S.T0]
    ids = t0.index

    def past(frame, time):  # 고객의 t0를 붙이고 t0 이전 기록만 남긴다
        frame = frame.merge(snap, on=C)
        frame = frame[frame[time] < frame[S.T0]]
        return frame.assign(age=_days(frame[S.T0] - frame[time]))

    def count(frame):
        return frame.groupby(C).size().reindex(ids, fill_value=0).astype(float)

    def agg(frame, column, how, fill=np.nan):
        return frame.groupby(C)[column].agg(how).reindex(ids).astype(float).fillna(fill)

    sessions = past(tables["sessions"], "started_at")
    events = past(tables["events"], "event_at").sort_values(["event_at", "event_id"])
    purchases = events.loc[events["event_type"] == "purchase", ["event_id", C, "session_id", "event_at", "age"]]
    orders = tables["orders"].merge(purchases, left_on="purchase_event_id", right_on="event_id")
    items = tables["order_items"].merge(
        orders[["order_id", C, "purchase_event_id", "checkout_source", "event_at", "age"]], on="order_id")
    view = events["event_type"] == "view"

    recent, prior = (lambda f: f[f["age"] <= S.RECENT_DAYS]), (
        lambda f: f[(f["age"] > S.RECENT_DAYS) & (f["age"] <= 2 * S.RECENT_DAYS)])
    out = pd.DataFrame(index=ids)

    # F01-F08 방문·구매 최근성과 규모
    out["visit_recency_days"] = agg(sessions, "age", "min")
    out["purchase_recency_days"] = agg(orders, "age", "min")
    out["observed_tenure_days"] = agg(sessions, "age", "max")
    out["session_count_30d"] = count(recent(sessions))
    out["view_count_30d"] = count(recent(events[view]))
    out["purchase_count_30d"] = n_orders = count(recent(orders))
    out["net_spend_30d"] = agg(recent(orders), "paid_minor", "sum", 0)
    out["mean_net_order_value_30d"] = agg(recent(orders), "paid_minor", "mean")

    # F09-F11: C30 = 최근 30일에 시작해 t0 전에 끝난 세션
    c30 = recent(sessions)
    c30 = c30[c30["ended_at"] < c30[S.T0]]
    n_c30 = count(c30)
    in_c30 = lambda f: f[f["session_id"].isin(c30["session_id"])]
    out["mean_session_duration_minutes_30d"] = agg(
        c30.assign(minutes=_days(c30["ended_at"] - c30["started_at"]) * 1440), "minutes", "mean")
    out["views_per_session_30d"] = _ratio(count(in_c30(events[view])), n_c30)
    out["orders_per_session_30d"] = _ratio(count(in_c30(orders)), n_c30)

    # F12-F14 구매 간격, 방문 시작 간격
    span = orders.groupby(C)["event_at"].agg(["min", "max", "size"]).reindex(ids)
    out["mean_purchase_gap_days"] = _ratio(_days(span["max"] - span["min"]), span["size"] - 1)
    starts = sessions.sort_values([C, "started_at"])
    starts = starts.assign(gap=_days(starts.groupby(C)["started_at"].diff()))
    gaps = starts[(starts["age"] <= S.GAP_DAYS) & starts["gap"].notna()].groupby(C)["gap"]
    gap_mean = gaps.mean().reindex(ids)
    out["intervisit_start_gap_mean_days_90d"] = gap_mean
    out["intervisit_start_gap_cv_90d"] = (gaps.std().reindex(ids) / gap_mean).where(gap_mean > 0)

    # F15-F17 구매 구성
    out["unique_purchase_products_30d"] = agg(recent(items), "product_id", "nunique", 0)
    out["purchased_qty_30d"] = agg(recent(items), "quantity", "sum", 0)
    out["mean_basket_qty_30d"] = _ratio(out["purchased_qty_30d"], n_orders)

    # F18-F20 t0 시점 장바구니
    cart = _replay_cart(events, items)
    price = tables["products"].set_index("product_id")["unit_price_minor"]
    cart = cart.assign(value=cart["qty"] * cart["product_id"].map(price))
    out["cart_qty_current"] = agg(cart, "qty", "sum", 0)
    out["cart_sku_current"] = count(cart)
    out["cart_gross_value_current"] = agg(cart, "value", "sum", 0)

    # F21-F24 전체 이력에서 t0 전에 끝난 마지막 세션
    done = sessions[sessions["ended_at"] < sessions[S.T0]]
    last = done.sort_values(["ended_at", "started_at", "session_id"]).groupby(C).tail(1)
    last_views = events[view & events["session_id"].isin(last["session_id"])]
    out["last_session_min_viewed_price"] = agg(last_views, "unit_price_minor", "min")
    out["last_session_max_viewed_price"] = agg(last_views, "unit_price_minor", "max")
    local = last.set_index(C)["started_at"].dt.tz_convert(LOCAL_TZ)
    out["last_session_start_hour"] = local.dt.hour.reindex(ids).astype(float)
    out["last_session_start_weekday"] = local.dt.day_of_week.reindex(ids).astype(float)

    # F25-F27 과거 쿠폰 사용
    redeemed = orders[orders["coupon_id"].notna()]
    out["coupon_uses_30d"] = agg(recent(redeemed), "coupon_id", "nunique", 0)
    out["coupon_redemption_recency_days"] = agg(redeemed, "age", "min")
    out["coupon_discount_amount_30d"] = agg(recent(redeemed), "discount_minor", "sum", 0)

    # F28-F32 최근 30일 - 직전 30일
    out["orders_delta_30d"] = n_orders - count(prior(orders))
    out["net_spend_delta_30d"] = out["net_spend_30d"] - agg(prior(orders), "paid_minor", "sum", 0)
    out["mean_net_order_value_delta_30d"] = (
        out["mean_net_order_value_30d"] - agg(prior(orders), "paid_minor", "mean"))
    out["unique_purchase_products_delta_30d"] = (
        out["unique_purchase_products_30d"] - agg(prior(items), "product_id", "nunique", 0))
    out["purchased_qty_delta_30d"] = out["purchased_qty_30d"] - agg(prior(items), "quantity", "sum", 0)

    # F33-F34 도착한 리뷰 전체
    arrived = recent(events[events["event_type"] == "review"])
    out["review_count_30d"] = count(arrived)
    out["mean_review_rating_30d"] = agg(
        arrived.assign(rating=pd.to_numeric(arrived["rating"], errors="coerce")), "rating", "mean")

    # F35-F36 C30 중 주문이 없는 세션의 담기·제거 횟수
    browse = c30[~c30["session_id"].isin(orders["session_id"])]
    in_browse = lambda kind: events[(events["event_type"] == kind) & events["session_id"].isin(browse["session_id"])]
    out["cart_adds_per_nonpurchase_session_30d"] = _ratio(count(in_browse("cart")), count(browse))
    out["cart_removes_per_nonpurchase_session_30d"] = _ratio(count(in_browse("remove_from_cart")), count(browse))

    # F37-F39 유효 상품 별점. 구매 단위로 먼저 평균하고 구매 간에는 같은 비중
    reviews = _product_reviews(events, orders, items)
    reviews = reviews.assign(non5=(reviews["rating"] < 5).astype(float))
    by_order = reviews.groupby([C, "purchase_event_id"]).agg(
        rating=("rating", "mean"), last=("event_at", "max"), age=("age", "min")).reset_index()
    latest = by_order[by_order["last"] == by_order.groupby(C)["last"].transform("max")]
    out["review_last_order_mean_rating"] = agg(latest, "rating", "mean")
    out["review_recency_days"] = agg(latest, "age", "min")
    out["review_non5_share_30d"] = agg(
        recent(reviews).groupby([C, "purchase_event_id"])["non5"].mean().reset_index(), "non5", "mean")

    out = out[S.KIT39].astype(float)
    return _mask_unobserved(out, t0, tables["customers"], observation_start)


def _mask_unobserved(out, t0, customers, observation_start) -> pd.DataFrame:
    """관측되지 않은 구간이 필요한 feature는 NULL (설계서 v5 12.2).

    가입 이전·가입 시각이면 전부 NULL. 30일 창 feature는 max(t0-30d, 가입) 이후가 관측돼야 하고,
    변화량은 60일, 방문 간격은 90일, 현재 장바구니는 가입 시점부터 관측돼야 한다.
    """
    signup = customers.set_index(C)["signed_up_at"].reindex(out.index)
    born = t0 > signup
    start = None if observation_start is None else pd.Timestamp(observation_start)

    def covered(days):
        if start is None:
            return born
        need = (t0 - pd.Timedelta(days=days)).where(lambda v: v > signup, signup)
        return born & (start <= need)

    for name in out.columns:
        if "delta" in name:
            ok = covered(2 * S.RECENT_DAYS)
        elif name.endswith("_90d"):
            ok = covered(S.GAP_DAYS)
        elif name.endswith("_30d"):
            ok = covered(S.RECENT_DAYS)
        elif name.startswith("cart_"):
            ok = born if start is None else born & (start <= signup)
        else:
            ok = born
        out[name] = out[name].where(ok)
    return out


def build_labels(tables: dict, snapshots: pd.DataFrame, observation_end) -> pd.Series:
    """y = [t0, t0+28d)에 주문 1건 이상이면 1. 이 구간이 다 관측되지 않았으면 NaN."""
    snap = snapshots[[C, S.T0]]
    until = snap[S.T0] + pd.Timedelta(days=S.OUTCOME_DAYS)
    events = tables["events"]
    bought = events[events["event_type"] == "purchase"].merge(snap.assign(until=until), on=C)
    bought = bought[(bought["event_at"] >= bought[S.T0]) & (bought["event_at"] < bought["until"])]
    bought = bought[bought["event_id"].isin(tables["orders"]["purchase_event_id"])]
    y = snap[C].isin(bought[C]).astype(float).where((until <= pd.Timestamp(observation_end)).to_numpy())
    return pd.Series(y.to_numpy(), index=pd.Index(snap[C], name=C), name=S.LABEL)


def eligible(tables: dict, snapshots: pd.DataFrame) -> pd.Series:
    """적격 여부: [t0-28d, t0)에 세션 시작이 있는 고객."""
    s = tables["sessions"].merge(snapshots[[C, S.T0]], on=C)
    s = s[(s["started_at"] < s[S.T0]) & (s["started_at"] >= s[S.T0] - pd.Timedelta(days=S.ELIGIBILITY_DAYS))]
    return snapshots[C].isin(s[C])


def build_dataset(raw: RawData) -> pd.DataFrame:
    """적격 고객 1명당 1행: customer_id, t0, treatment, KIT39, y.

    T는 배정군 그대로다(ITT). 결과 28일이 다 관측되지 않은 고객은 y가 NaN이므로 학습 전에 제외한다.
    """
    if raw.metadata.get("time_spec") is not None or raw.metadata.get("feature_version") == "kit39-weekly-4-8-13w-v1":
        raise ValueError("Weekly data requires load_feature_dataset; legacy raw features use 30/60/90-day windows")
    snap = raw.assignments[eligible(raw.tables, raw.assignments).to_numpy()]
    features = build_features(raw.tables, snap, raw.observation_start)
    labels = build_labels(raw.tables, snap, raw.observation_end)
    head = snap.set_index(C)[[S.T0, S.TREATMENT]]
    return pd.concat([head, features, labels], axis=1).reset_index()
