from copy import deepcopy

import pandas as pd
import pytest

from simulate.v4.irl.shopping_history import (
    ShoppingHistory, SHOPPING_HISTORY_FEATURES as F, SHOPPING_HISTORY_DIAGNOSTICS as D,
)


T = pd.Timestamp("2024-01-01T00:00:00Z")
MIN = pd.Timedelta(minutes=1)


def close(history, sid, at, actions, *, completed=True):
    history.begin_session(sid, at)
    for kind, quantity, price in actions:
        history.record_action(sid, at, kind, quantity=quantity, view_price_minor=price)
    history.end_session(sid, at, completed=completed)


def test_completion_is_strictly_after_thirty_minutes_and_current_session_is_excluded():
    h = ShoppingHistory(T)
    h.begin_session("a", T)
    h.record_action("a", T, "cart", quantity=3)
    assert h.snapshot(T)[F[0]] is None
    h.end_session("a", T)
    h.begin_session("b", T + MIN)
    h.record_action("b", T + MIN, "purchase", quantity=5)
    assert h.snapshot(T + 30 * MIN)[F[2]] == 0
    result = h.snapshot(T + 30 * MIN + pd.Timedelta(nanoseconds=1))
    assert result[F[0]] == 1 and result[F[2]] == 1 and result[F[3]] == 0
    assert result[D[0]] == 1  # Neither new-session presence nor its purchase flushes pending history.


def test_no_buy_means_no_checkout_and_action_counts_never_use_quantity():
    h = ShoppingHistory()
    close(h, "a", T, [("cart", 5, None), ("cart", 1, None), ("remove_from_cart", 3, None)])
    close(h, "b", T + MIN, [("view", None, 120), ("cart", 9, None), ("purchase", 7, None)])
    close(h, "c", T + 2 * MIN, [("view", None, 100)])
    result = h.snapshot(T + 33 * MIN)
    assert result[F[0]] == 1  # Two adds across two completed no-buy sessions.
    assert result[F[1]] == .5
    assert result[F[2]] == 3 and result[F[3]] == pytest.approx(1 / 3)
    assert result[D[0]] == 2


def test_purchase_order_count_and_item_rows_do_not_change_session_purchase_share():
    left, right = ShoppingHistory(), ShoppingHistory()
    close(left, 1, T, [("purchase", 12, None)])
    close(right, 1, T, [("purchase", 1, None)] * 12)
    assert left.snapshot(T + 31 * MIN) == right.snapshot(T + 31 * MIN)
    assert left.snapshot(T + 32 * MIN)[F[0]] is None  # No completed nonpurchase denominator.


def test_source_dataframe_integer_session_keys_are_supported():
    h = ShoppingHistory()
    key = pd.Series([1], dtype="int64").iloc[0]
    close(h, key, T, [("view", None, 100)])
    assert h.snapshot(T + 31 * MIN)[F[2]] == 1


def test_window_uses_start_lower_inclusive_and_availability_strict_upper():
    h = ShoppingHistory()
    close(h, "a", T, [("view", 1, 300)])
    assert h.snapshot(T + pd.Timedelta(days=30))[F[2]] == 1
    result = h.snapshot(T + pd.Timedelta(days=30, nanoseconds=1))
    assert result[F[2]] == 0 and result[D[0]] == 0
    assert result[F[3]] is None and result[F[0]] is None
    assert result[F[4]] == 300  # Last completed price is an all-history summary.


@pytest.mark.parametrize("kind", ["max_events", "window_end", "infeasible_initial_action"])
def test_artificial_end_or_censor_is_not_a_no_purchase_completion(kind):
    h = ShoppingHistory()
    close(h, kind, T, [("cart", 5, None)], completed=False)
    result = h.snapshot(T + 31 * MIN)
    assert result == {F[0]: None, F[1]: None, F[2]: 0, F[3]: None, F[4]: None, D[0]: 0}


def test_only_arrived_observed_view_prices_and_no_fallback_to_older_priced_session():
    h = ShoppingHistory()
    close(h, "a", T, [("view", 8, 100), ("view", 1, 300), ("view", None, None)])
    assert h.snapshot(T + 31 * MIN)[F[4]] == 200
    close(h, "b", T + 32 * MIN, [("cart", 5, None)])
    assert h.snapshot(T + 63 * MIN)[F[4]] is None


def test_same_clock_session_order_is_not_treated_as_causal_history():
    histories = []
    for sequence in (("a", "b"), ("b", "a")):
        h = ShoppingHistory()
        for sid in sequence:
            close(h, sid, T, [("view", 1, 100 if sid == "a" else 200)])
        assert h.snapshot(T + 30 * MIN)[F[2]] == 0
        histories.append(h.snapshot(T + 31 * MIN))
    assert histories[0] == histories[1]
    assert histories[0][F[4]] == 200  # Deterministic ID tie convention, not causal order.


def test_future_current_session_outcome_cannot_change_prior_snapshot():
    base = ShoppingHistory()
    close(base, "old", T, [("cart", 2, None)])
    base.begin_session("current", T + 31 * MIN)
    before = base.snapshot(T + 31 * MIN)
    left, right = deepcopy(base), deepcopy(base)
    left.record_action("current", T + 32 * MIN, "purchase")
    right.record_action("current", T + 32 * MIN, "remove")
    assert left.snapshot(T + 32 * MIN) == right.snapshot(T + 32 * MIN) == before


def test_source_and_runtime_online_adapters_get_identical_context_despite_END_clock_knowledge():
    source, runtime = ShoppingHistory(T), ShoppingHistory(T)
    # Source knows completion only after 30m follow-up; runtime has an explicit
    # END at the last action. Both queue the same summary with the shared delay.
    for h in (source, runtime):
        h.begin_session("first", T)
        h.record_action("first", T, "view", view_price_minor=180)
        h.record_action("first", T + MIN, "cart", quantity=6)
        h.end_session("first", T + MIN)
        h.begin_session("next", T + 5 * MIN)
    for at in (T + 5 * MIN, T + 31 * MIN, T + 32 * MIN):
        assert source.snapshot(at) == runtime.snapshot(at)
    assert source.snapshot(T + 33 * MIN)[F[0]] == 1


def test_clock_and_session_contract_rejects_backdating_or_reuse():
    h = ShoppingHistory(T)
    with pytest.raises(ValueError, match="backwards"):
        h.begin_session("before", T - MIN)
    h.begin_session("a", T)
    with pytest.raises(ValueError, match="active"):
        h.begin_session("b", T)
    with pytest.raises(ValueError, match="active"):
        h.record_action("other", T, "view")
    h.record_action("a", T + MIN, "view")
    with pytest.raises(ValueError, match="last observed"):
        h.end_session("a", T + 2 * MIN)
    h.end_session("a", T + MIN)
    with pytest.raises(ValueError, match="reused"):
        h.begin_session("a", T + MIN)
    h.snapshot(T + 32 * MIN)
    with pytest.raises(ValueError, match="backwards"):
        h.snapshot(T + MIN)


@pytest.mark.parametrize("bad", [None, "2024-01-01", pd.NaT])
def test_unknown_or_naive_clocks_are_not_observed(bad):
    with pytest.raises(ValueError):
        ShoppingHistory().snapshot(bad)


@pytest.mark.parametrize("price", [0, -1, float("nan"), float("inf"), True])
def test_unobserved_or_invalid_prices_are_not_fabricated(price):
    h = ShoppingHistory()
    h.begin_session("a", T)
    with pytest.raises(ValueError):
        h.record_action("a", T, "view", view_price_minor=price)


@pytest.mark.parametrize("quantity", [0, -1, 1.5, float("nan"), True])
def test_invalid_quantity_does_not_create_observed_action(quantity):
    h = ShoppingHistory()
    h.begin_session("a", T)
    with pytest.raises(ValueError):
        h.record_action("a", T, "cart", quantity=quantity)
    with pytest.raises(ValueError, match="empty"):
        h.end_session("a", T)


def test_labels_auxiliary_events_and_purchase_prices_are_not_history_inputs():
    h = ShoppingHistory()
    h.begin_session("a", T)
    with pytest.raises(ValueError):
        h.record_action("a", T, "review")
    with pytest.raises(ValueError):
        h.record_action("a", T, "purchase", view_price_minor=500)
    with pytest.raises(TypeError):
        h.snapshot(T, persona="buyer")
