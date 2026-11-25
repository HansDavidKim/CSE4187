"""Customer-level coupon assignment, kept outside behavior RNG and predictors."""
from collections.abc import Iterable, Mapping
from decimal import Decimal
import hashlib
import json
from numbers import Integral, Real

import numpy as np
import pandas as pd


def _customer_ids(values):
    if isinstance(values,(str,bytes,Mapping,pd.DataFrame)) or not isinstance(values,Iterable):
        raise ValueError("coupon customer_ids must be a sequence of unique customer IDs")
    values=list(values)
    if any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(str,Integral)) or not str(v).strip() for v in values):
        raise ValueError("coupon customer_ids must contain nonempty string or integer IDs")
    ids=list(map(str,values))
    if len(set(ids))!=len(ids):raise ValueError("duplicate coupon customer IDs")
    return ids


def resolve_coupon_assignment(value=None):
    if value is None:
        return {"mode": "auto"}
    if not isinstance(value, Mapping) or set(value) - {"mode", "customer_ids", "count", "ratio", "seed"}:
        raise ValueError("coupon_assignment requires mode and optional customer_ids/count/ratio/seed")
    spec = dict(value)
    selectors = set(spec) & {"customer_ids", "count", "ratio"}
    mode = spec.setdefault("mode", "assign" if selectors else "auto")
    if not isinstance(mode,str) or mode not in {"auto", "none", "assign"}:
        raise ValueError("coupon assignment mode must be auto, none or assign")
    if mode != "assign":
        if selectors or "seed" in spec:
            raise ValueError("customer_ids/count/ratio/assignment seed require assign mode")
        return spec
    if len(selectors) != 1:
        raise ValueError("assign requires exactly one of customer_ids, count or ratio")
    if "seed" in spec:
        if isinstance(spec["seed"], (bool, np.bool_)) or not isinstance(spec["seed"], Integral) or spec["seed"] < 0:
            raise ValueError("coupon assignment seed must be a nonnegative integer")
        spec["seed"] = int(spec["seed"])
    if "count" in spec:
        if isinstance(spec["count"], (bool, np.bool_)) or not isinstance(spec["count"], Integral) or spec["count"] < 0:
            raise ValueError("coupon count must be a nonnegative integer")
        spec["count"] = int(spec["count"])
    if "ratio" in spec:
        ratio = spec["ratio"]
        if isinstance(ratio, (bool, np.bool_)) or not isinstance(ratio, Real) or not np.isfinite(ratio) or not 0 <= ratio <= 1:
            raise ValueError("coupon ratio must be finite and between zero and one")
        spec["ratio"] = float(ratio)
    if "customer_ids" in spec:
        spec["customer_ids"] = sorted(_customer_ids(spec["customer_ids"]))
    return spec


def assign_coupons(customer_ids, spec=None, *, seed=42, legacy=None):
    """Return actions, marginal propensities and the replayable assignment design.

    Auto consumes no RNG here: callers supply their unchanged legacy assignment.
    Ratio assignment selects floor(n*ratio) customers without replacement.
    """
    spec = resolve_coupon_assignment(spec)
    ids = _customer_ids(customer_ids)
    n = len(ids)
    if not n or len(set(ids)) != n:
        raise ValueError("coupon assignment requires a nonempty unique customer cohort")
    mode = spec["mode"]
    if mode == "auto":
        if n < 2 or n % 2:
            raise ValueError("auto requires an even customer cohort of at least two")
        actions = np.asarray(legacy)
        if actions.shape != (n,) or not np.isin(actions, [0, 1]).all() or actions.sum() != n // 2:
            raise ValueError("auto requires the original balanced assignment")
        probabilities = np.full(n, .5)
        design, randomized, assignment_seed = "fixed_1to1_randomized", True, None
    elif mode == "none":
        actions = np.zeros(n, dtype=int)
        probabilities = np.zeros(n)
        design, randomized, assignment_seed = "no_new_coupon", False, None
    else:
        assignment_seed = spec.get("seed", int.from_bytes(hashlib.sha256(
            f"coupon-assignment:{seed}".encode()).digest()[:8], "big"))
        spec["seed"] = assignment_seed
        if "customer_ids" in spec:
            unknown = set(spec["customer_ids"]) - set(ids)
            if unknown:
                raise ValueError(f"unknown coupon customer IDs: {sorted(unknown)}")
            selected = set(spec["customer_ids"])
            actions = np.array([int(uid in selected) for uid in ids])
            probabilities = actions.astype(float)
            design, randomized = "specified_customers", False
        else:
            count = spec.get("count", int(n * Decimal(str(spec.get("ratio", 0)))))
            if count > n:
                raise ValueError("coupon count exceeds the customer cohort")
            selected = set(np.random.default_rng(assignment_seed).choice(sorted(ids), count, replace=False))
            actions = np.array([int(uid in selected) for uid in ids])
            probabilities = np.full(n, count / n)
            design, randomized = "fixed_count_randomized", True
    metadata = {"mode": mode, "spec": spec, "design": design, "randomized": randomized,
                "assignment_seed": assignment_seed, "customers": n, "treated_count": int(actions.sum()),
                "ratio_rounding": "floor(n*ratio)", "schedule": "one coupon at t0; no later sends"}
    metadata["assignment_sha256"] = hashlib.sha256(json.dumps(sorted(
        [uid,int(action),float(probability)] for uid,action,probability in zip(ids,actions,probabilities)),
        separators=(",", ":")).encode()).hexdigest()
    return actions.astype(int), probabilities, metadata


def validate_coupon_assignments(frame, metadata):
    """Validate persisted execution-independent assignments against their design."""
    design = metadata.get("coupon_assignment", {"spec": {"mode": "auto"}})
    actions, probabilities, actual = assign_coupons(frame.customer_id, design["spec"],
        seed=metadata.get("seed", 42), legacy=frame.treatment.to_numpy())
    if not np.array_equal(frame.treatment.to_numpy(), actions):
        raise ValueError("coupon treatment differs from its declared assignment")
    observed = pd.to_numeric(frame.assignment_probability, errors="raise").to_numpy(float)
    if not np.array_equal(observed, probabilities):
        raise ValueError("coupon assignment probabilities differ from the declared design")
    if "assignment_sha256" in design and design["assignment_sha256"] != actual["assignment_sha256"]:
        raise ValueError("coupon assignment digest differs from the declared design")


def add_coupon_assignment_arguments(parser):
    parser.add_argument("--coupon-mode", choices=("auto", "none", "assign"))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--coupon-customer-ids", nargs="+", help="exact exported customer IDs to receive a coupon")
    group.add_argument("--coupon-count", type=int, help="randomly select this many customers without replacement")
    group.add_argument("--coupon-ratio", type=float, help="select floor(cohort size*ratio) customers without replacement")
    parser.add_argument("--coupon-assignment-seed", type=int)


def coupon_assignment_from_args(args):
    spec = {key: getattr(args, name) for key, name in (("mode", "coupon_mode"),
        ("customer_ids", "coupon_customer_ids"), ("count", "coupon_count"),
        ("ratio", "coupon_ratio"), ("seed", "coupon_assignment_seed")) if getattr(args, name, None) is not None}
    return resolve_coupon_assignment(spec) if spec else None
