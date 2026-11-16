import pandas as pd
import pytest

from simulate.v4.simulator.lifecycle import build_observed_lifecycle


ORIGIN = pd.Timestamp("2024-01-01T00:00:00Z")


def metadata(days=90):
    return {"observation_start_at": ORIGIN.isoformat(),
            "observation_end_at": (ORIGIN + pd.Timedelta(days=days)).isoformat()}


def visits(days, customer="u"):
    return pd.DataFrame({"customer_id": [customer] * len(days), "session_id": [f"{customer}:{i}" for i in range(len(days))],
                         "started_at": [ORIGIN + pd.Timedelta(days=day) for day in days]})


def test_first_observed_entry_then_first_return_home_without_two_session_rule():
    result = build_observed_lifecycle(visits([1, 2, 3]), metadata(10), {"u": "source_tag"})
    assert result.life_cycle.tolist() == ["ENTRY", "HOME", "HOME"]
    assert result.home_persona.eq("source_tag").all()
    assert result.history_left_censored.all()
    assert "cart_quantity_after" not in result


def test_exit_is_not_emitted_until_twenty_eight_days_have_elapsed():
    assert build_observed_lifecycle(visits([0]), metadata(27.999)).life_cycle.tolist() == ["ENTRY"]
    observed = build_observed_lifecycle(visits([0]), metadata(28.001))
    assert observed.life_cycle.tolist() == ["ENTRY", "EXIT"]
    assert observed.iloc[1].as_of_at == ORIGIN + pd.Timedelta(days=28)
    assert observed.iloc[1].elapsed_since_visit_days == 28
    assert observed.iloc[1].session_id is None


def test_exact_return_deadline_records_exit_then_same_time_home():
    result = build_observed_lifecycle(visits([0, 28]), metadata(30))
    assert result.life_cycle.tolist() == ["ENTRY", "EXIT", "HOME"]
    assert result.as_of_at.iloc[1] == result.as_of_at.iloc[2] == ORIGIN + pd.Timedelta(days=28)
    assert result.transition_order.iloc[1:].tolist() == [0, 1]


def test_return_before_deadline_resets_timer_and_long_return_reactivates_home():
    result = build_observed_lifecycle(visits([0, 27, 70]), metadata(75))
    assert result.life_cycle.tolist() == ["ENTRY", "HOME", "EXIT", "HOME"]
    assert result.loc[result.life_cycle.eq("EXIT"), "as_of_at"].tolist() == [ORIGIN + pd.Timedelta(days=55)]
    assert "CHURNING" not in set(result.life_cycle)


def test_observation_end_is_exclusive_and_final_return_is_right_censored():
    result = build_observed_lifecycle(visits([0]), metadata(28))
    assert result.life_cycle.tolist() == ["ENTRY"]
    assert result.attrs["final_return_gaps_right_censored"] == 1
    assert result.attrs["exit_deadlines_outside_observation"] == 1
    later = build_observed_lifecycle(visits([0]), metadata(50))
    assert later.life_cycle.tolist() == ["ENTRY", "EXIT"]
    assert later.attrs["final_return_gaps_right_censored"] == 1


def test_missing_pre_observation_history_does_not_create_early_exit_or_signup():
    result = build_observed_lifecycle(visits([-50, 40]), metadata(50))
    assert result.life_cycle.tolist() == ["ENTRY"]
    assert result.iloc[0].as_of_at == ORIGIN + pd.Timedelta(days=40)
    assert result.iloc[0].first_observed_at == result.iloc[0].as_of_at
    assert pd.isna(result.iloc[0].home_persona)


def test_future_sessions_do_not_change_observed_history_or_past_timeout():
    past = visits([0])
    original = build_observed_lifecycle(past, metadata(40))
    future = pd.DataFrame({"customer_id": ["u"], "session_id": ["future"],
                           "started_at": [ORIGIN + pd.Timedelta(days=60)]})
    extended_input = pd.concat([past, future], ignore_index=True)
    pd.testing.assert_frame_equal(original, build_observed_lifecycle(extended_input, metadata(40)))
    complete = build_observed_lifecycle(extended_input, metadata(80))
    prefix = complete.loc[complete.as_of_at.lt(ORIGIN + pd.Timedelta(days=40))].reset_index(drop=True)
    pd.testing.assert_frame_equal(original, prefix)


def test_customer_histories_do_not_share_timers_and_input_is_unchanged():
    source = pd.concat([visits([0, 1], "a"), visits([20], "b")], ignore_index=True)
    original = source.copy(deep=True)
    result = build_observed_lifecycle(source.sample(frac=1, random_state=3), metadata(40))
    assert result.loc[result.customer_id.eq("a"), "life_cycle"].tolist() == ["ENTRY", "HOME", "EXIT"]
    assert result.loc[result.customer_id.eq("b"), "life_cycle"].tolist() == ["ENTRY"]
    pd.testing.assert_frame_equal(source, original)


def test_empty_inputs_keep_the_output_contract_and_utc_dtype():
    result = build_observed_lifecycle(visits([]), metadata())
    assert result.empty and {"customer_id", "session_id", "life_cycle", "as_of_at"} <= set(result)
    assert str(result.as_of_at.dtype) == "datetime64[ns, UTC]"


@pytest.mark.parametrize("case", ["naive", "duplicate_id", "missing_time", "invalid_bound"])
def test_invalid_session_boundaries_are_rejected(case):
    source, info = visits([0, 1]), metadata()
    if case == "naive":
        source["started_at"] = source.started_at.dt.tz_localize(None)
    elif case == "duplicate_id":
        source.loc[1, "session_id"] = source.loc[0, "session_id"]
    elif case == "missing_time":
        source.loc[0, "started_at"] = pd.NaT
    else:
        info["observation_end_at"] = info["observation_start_at"]
    with pytest.raises(ValueError):
        build_observed_lifecycle(source, info)
