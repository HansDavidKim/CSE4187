"""Apply the maintained team's human profile gates to validated purchase inputs."""
from dataclasses import asdict
from decimal import Decimal, ROUND_CEILING

import numpy as np
import pandas as pd

from .profile_contract import PolicyCandidate, SegmentationConfig, _duration, _json_value
from .weekly_validation import require


def profile_gate(frame, profile):
    require(isinstance(profile, PolicyCandidate), "profile requires a team PolicyCandidate declaration")
    require(asdict(profile.segmentation) == asdict(SegmentationConfig()), "human profiles require the common development segmentation")
    require(all(_duration(rule.send_delay, "send_delay") == pd.Timedelta(0) for rule in profile.policies),
            "the single-t0 pipeline does not support delayed profile sends")
    known = frame[["churn_prob", "uplift_score", "clv"]].notna().all(axis=1)
    high = pd.Series(False, index=frame.index)
    count = int((Decimal(int(known.sum())) * Decimal(str(profile.segmentation.high_clv_fraction))).to_integral_value(
        rounding=ROUND_CEILING))
    order = frame.loc[known].sort_values(["clv", "customer_id"], ascending=[False, True], kind="stable")
    high.loc[order.head(count).index] = True
    audit = frame[["customer_id"]].copy()
    audit["risk_band"] = np.where(frame.churn_prob.ge(profile.segmentation.churn_threshold), "high", "low")
    audit["value_band"] = np.where(high, "high_value", "standard_value")
    uplift_band = np.select([frame.uplift_score.lt(0), frame.uplift_score.gt(0)],
                            ["negative", "positive"], default="neutral_low")
    identifiers = audit.risk_band + "__" + uplift_band + "__" + audit.value_band
    rules = {rule.segment_id: rule for rule in profile.policies}
    audit["minimum_uplift"] = identifiers.map(lambda key: str(rules[key].min_uplift))
    audit["profile_candidate"] = [bool(valid and rules[key].allow_candidate
        and Decimal(str(effect)) > Decimal(str(rules[key].min_uplift)))
        for valid, key, effect in zip(known, identifiers, frame.uplift_score)]
    audit.loc[~known, ["risk_band", "value_band", "minimum_uplift"]] = pd.NA
    metadata = {"policy_id": profile.policy_id, "version": profile.version, "source": profile.source,
                "description": profile.description, "configuration_fingerprint": profile.configuration_fingerprint(),
                "segmentation": asdict(profile.segmentation), "policies": [asdict(rule) for rule in profile.policies],
                "source_commit": "11a1fcd200af1f779aafd2be689a80ee86cf0671",
                "scope": "additional strict uplift gate; common six-segment eligibility, financial checks and budget stay in force",
                "value_rank_population": "all complete churn/uplift/CLV rows before profile, business-segment or budget selection",
                "policy_effect_verified": False, "actual_dispatch_performed": False}
    return audit.profile_candidate, audit, _json_value(metadata)
