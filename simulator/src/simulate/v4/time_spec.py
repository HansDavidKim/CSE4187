"""The explicit weekly research clock; legacy calendar datasets keep their contract."""
import pandas as pd


WEEKLY_TIME_SPEC_VERSION = "weekly_104w_v1"
WEEKLY_FEATURE_VERSION = "kit39-weekly-4-8-13w-v1"
WEEKLY_TIME_SPEC = {
    "total_days": 104 * 7, "pre_days": 52 * 7, "post_days": 52 * 7,
    "min_history_days": 13 * 7, "eligibility_days": 4 * 7, "coupon_days": 4 * 7,
    "uplift_days": 4 * 7, "recent_days": 4 * 7, "delta_days": 8 * 7,
    "gap_days": 13 * 7, "clv_input_days": 13 * 7, "clv_label_days": 52 * 7,
}


def is_weekly(metadata):
    spec = metadata.get("time_spec")
    if spec is not None and spec != WEEKLY_TIME_SPEC_VERSION:
        raise ValueError("unknown simulator time_spec")
    return spec == WEEKLY_TIME_SPEC_VERSION or metadata.get("feature_version") == WEEKLY_FEATURE_VERSION


def feature_window_days(metadata):
    if is_weekly(metadata):
        return tuple(WEEKLY_TIME_SPEC[name] for name in ("recent_days", "delta_days", "gap_days"))
    return 30, 60, 90


def weekly_calendar(start_at):
    start = pd.Timestamp(start_at)
    if pd.isna(start) or start.tzinfo is None:
        raise ValueError("weekly start_at must be an explicit timezone-aware Monday midnight in Asia/Seoul")
    local = start.tz_convert("Asia/Seoul")
    if local.weekday() != 0 or local != local.normalize():
        raise ValueError("weekly start_at must be Monday midnight in Asia/Seoul")
    start = start.tz_convert("UTC")
    return (start, start + pd.Timedelta(days=WEEKLY_TIME_SPEC["pre_days"]),
            start + pd.Timedelta(days=WEEKLY_TIME_SPEC["total_days"]))


def weekly_metadata():
    return {"time_spec": WEEKLY_TIME_SPEC_VERSION, "feature_version": WEEKLY_FEATURE_VERSION,
            "time_windows_days": dict(WEEKLY_TIME_SPEC)}
