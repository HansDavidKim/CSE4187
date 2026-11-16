"""Observed action-specific item frequencies and within-session repeat mixtures."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import numbers
from pathlib import Path

import pandas as pd


ACTIONS = ("view", "cart", "remove")


def _profile_id(user_id):
    return hashlib.sha256(f"cosmetics:{user_id}".encode()).hexdigest()[:16]


def _pool():
    return {action: Counter() for action in ACTIONS}


def _serial_pool(pool):
    return {action: dict(sorted(pool[action].items())) for action in ACTIONS}


def _serial_pairs(pairs):
    result = {}
    for (previous, action), counts in sorted(pairs.items()):
        result.setdefault(previous, {})[action] = {"same": counts["same"], "total": counts["total"]}
    return result


def fit_item_choice(rows: pd.DataFrame, base_model: dict) -> dict:
    """Aggregate prepared source rows; never changes a source row or base-model field."""
    required = {"user_id", "product_id", "action", "session", "at", "source_event_id", "price_minor"}
    if not required <= set(rows) or rows.empty:
        raise ValueError("item choice requires nonempty prepared source rows")
    if rows[list(required - {"price_minor"})].isna().any().any():
        raise ValueError("item source identity, action, session and clock must be known")
    if not isinstance(rows["at"].dtype, pd.DatetimeTZDtype) or not rows.action.isin((*ACTIONS, "purchase")).all():
        raise ValueError("item source requires UTC-aware time and recognized actions")
    catalog = {}
    for product in base_model["catalog"]:
        pid, price = product["product_id"], product["unit_price_minor"]
        if not isinstance(pid, str) or not pid or pid in catalog or isinstance(price, bool) or not isinstance(price, numbers.Integral) or price < 1:
            raise ValueError("item base catalog requires unique observed IDs with positive integer prices")
        catalog[pid] = product
    positive = rows.loc[rows.price_minor.notna() & rows.price_minor.gt(0), "product_id"]
    if not set(positive) <= catalog.keys():
        raise ValueError("positive-price source product is absent from the base catalog")
    profiles = {profile["profile_id"]: profile for profile in base_model["profiles"]}
    if len(profiles) != len(base_model["profiles"]):
        raise ValueError("duplicate base profile identity")
    source_profiles = {_profile_id(uid) for uid in rows.user_id.unique()}
    if source_profiles != profiles.keys():
        raise ValueError("source users and base hashed profile identities must match exactly")
    profile_pools, group_pools, global_pool = {}, defaultdict(_pool), _pool()
    group_pairs, global_pairs = defaultdict(lambda: defaultdict(Counter)), defaultdict(Counter)
    raw_counts, excluded_counts, audit = Counter(), Counter(), Counter()
    excluded_products = set()
    ordered = rows.sort_values(["user_id", "at", "source_event_id"], kind="stable")
    for uid, history in ordered.groupby("user_id", sort=True):
        profile_id = _profile_id(uid)
        profile = profiles[profile_id]
        purchases = history.loc[history.action.eq("purchase"), "session"].nunique()
        expected_group = ("repeat_buyer" if purchases >= 2 else "observed_buyer" if purchases else
                          "cart_remover_without_purchase" if history.action.eq("remove").any() else
                          "cart_without_purchase" if history.action.eq("cart").any() else "browse_only")
        group = profile["home_group"]
        if group != expected_group or group not in base_model["home_groups"]:
            raise ValueError("source behavior and base profile home_group do not match")
        pool = _pool()
        previous = None
        for event in history.itertuples(index=False):
            supported = event.product_id in catalog
            if not supported:
                excluded_products.add(event.product_id)
                audit["outside_positive_catalog_rows"] += 1
            if event.action in ACTIONS:
                raw_counts[event.action] += 1
                if supported:
                    pool[event.action][event.product_id] += 1
                    group_pools[group][event.action][event.product_id] += 1
                    global_pool[event.action][event.product_id] += 1
                else:
                    excluded_counts[event.action] += 1
                if previous is not None and previous.session == event.session:
                    audit["raw_within_session_pairs"] += 1
                    if supported and previous.product_id in catalog:
                        key = previous.action, event.action
                        same = int(previous.product_id == event.product_id)
                        for target in (group_pairs[group][key], global_pairs[key]):
                            target["same"] += same
                            target["total"] += 1
                    else:
                        audit["outside_positive_catalog_pairs"] += 1
            previous = event
        profile_pools[profile_id] = {"home_group": group, "action_product_counts": _serial_pool(pool)}
        group_pools[group]  # Preserve source groups with no supported action pool.
    return {
        "schema_version": 1, "type": "empirical_action_item_frequency_and_repeat_mixture",
        "actions": list(ACTIONS), "profiles": profile_pools,
        "home_groups": {group: {"action_product_counts": _serial_pool(pool),
                                "repeat_pairs": _serial_pairs(group_pairs[group])}
                        for group, pool in sorted(group_pools.items())},
        "global": {"action_product_counts": _serial_pool(global_pool), "repeat_pairs": _serial_pairs(global_pairs)},
        "source_audit": {"rows": len(rows), "users": len(profile_pools), "positive_catalog_products": len(catalog),
                         "outside_positive_catalog_products": len(excluded_products), **dict(audit),
                         "raw_action_rows": dict(raw_counts), "excluded_action_rows": dict(excluded_counts),
                         "supported_action_rows": {action: sum(global_pool[action].values()) for action in ACTIONS}},
        "interpretation": [
            "All supported observed product counts are retained; no top-N preference truncation or coefficient fitting.",
            "Repeat rates condition on home_group and consecutive actual actions in the same observed session; after checkout the previous product is the final source item.",
            "Full-history donor pools describe retrospective source users, not prospectively inferred personas or causal product effects.",
            "Products outside the positive base catalog are audited and excluded, never assigned fabricated prices or IDs.",
            "Unavailable repeat/other components are conditioned on feasible support. A zero-mass repeat rule falls back to observed action frequencies with a diagnostic.",
            "View/cart without source-supported available products returns no choice. Remove may use positive held-quantity weights as an explicit structural inventory fallback."]}


def item_probabilities(kernel, profileid, homegroup, action, previous_action, previous_product, available_ids, held=None):
    """Return (probabilities, diagnostics); callers may reuse a set for available_ids."""
    if kernel.get("schema_version") != 1 or action not in ACTIONS:
        raise ValueError("incompatible item kernel or unrecognized action")
    profile = kernel["profiles"].get(profileid)
    if profile is None or profile["home_group"] != homegroup or homegroup not in kernel["home_groups"]:
        raise ValueError("item profile and home_group must match the fitted source")
    available = available_ids if isinstance(available_ids, (set, frozenset)) else set(available_ids)
    inventory = {}
    if action == "remove":
        for pid, quantity in (held or {}).items():
            if isinstance(quantity, bool) or not isinstance(quantity, numbers.Integral) or quantity < 0:
                raise ValueError("held quantities must be nonnegative integers")
            if quantity > 0 and pid in available:
                inventory[pid] = int(quantity)
        available = inventory.keys()
    diagnostics = {"preference_scope": None, "repeat_pair_scope": None,
                   "observed_repeat_probability": None, "effective_repeat_probability": None,
                   "support_size": 0, "fallback_reason": None}
    sources = (("profile", profile), ("home_group", kernel["home_groups"][homegroup]), ("source_pool", kernel["global"]))
    def preferences(exclude=None):
        for scope, block in sources:
            weights = {pid: count for pid, count in block["action_product_counts"][action].items()
                       if pid in available and pid != exclude}
            if weights:
                total = sum(weights.values())
                return {pid: count / total for pid, count in weights.items()}, scope
        return {}, None
    def finish(probabilities, scope, reason=None):
        diagnostics.update(preference_scope=scope, support_size=len(probabilities), fallback_reason=reason,
                           effective_repeat_probability=probabilities.get(previous_product, 0.) if previous_product is not None else None)
        return probabilities, diagnostics
    base, base_scope = preferences()
    if not base:
        if inventory:
            total = sum(inventory.values())
            return finish({pid: quantity / total for pid, quantity in inventory.items()}, "held_inventory",
                          "no_observed_remove_support_for_held_products")
        return finish({}, None, "empty_held_inventory" if action == "remove" else "no_source_supported_available_products")
    if previous_action in (None, "start") or previous_product is None:
        return finish(base, base_scope)
    pair, pair_scope = None, None
    for scope, block in (("home_group", kernel["home_groups"][homegroup]), ("source_pool", kernel["global"])):
        candidate = block["repeat_pairs"].get(previous_action, {}).get(action)
        if candidate and candidate["total"]:
            pair, pair_scope = candidate, scope
            break
    if pair is None:
        return finish(base, base_scope, "no_observed_action_pair")
    repeat = pair["same"] / pair["total"]
    diagnostics.update(repeat_pair_scope=pair_scope, observed_repeat_probability=repeat)
    other, other_scope = preferences(exclude=previous_product)
    repeat_supported = previous_product in available and previous_product in kernel["global"]["action_product_counts"][action]
    repeated_mass, other_mass = (repeat if repeat_supported else 0.), (1 - repeat if other else 0.)
    total = repeated_mass + other_mass
    if not total:
        return finish(base, base_scope, "repeat_rule_has_no_feasible_mass")
    result = {pid: probability * other_mass / total for pid, probability in other.items() if other_mass}
    if repeated_mass:
        result[previous_product] = repeated_mass / total
    reason = None if repeat_supported and other else "repeat_mixture_conditioned_on_available_support"
    return finish(result, other_scope or base_scope, reason)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=Path("data/models/shopping_profiles_checkout.json"))
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/models/shopping_profiles_checkout_items.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"item candidate must use a new artifact path: {args.output}")
    base_bytes = args.base_model.read_bytes()
    base = json.loads(base_bytes)
    if base.get("choice_unit") != "source_checkout_episode" or "item_choice" in base:
        raise ValueError("item candidate requires an unmodified checkout base artifact")
    import pyarrow.parquet as pq
    cache = json.loads(pq.read_schema(args.source_cache).metadata[b"quality_source_cache"])
    provenance = cache["source_provenance"]
    if provenance["files"] != base["source"]["files"] or provenance["sample_modulus"] != base["source"]["sample_modulus"]:
        raise ValueError("source cache and checkout base provenance differ")
    rows = pd.read_parquet(args.source_cache)
    kernel = fit_item_choice(rows, base)
    kernel["provenance"] = {"base_artifact_sha256": hashlib.sha256(base_bytes).hexdigest(),
                            "source_cache_fingerprint": cache["fingerprint"],
                            "source_cache_sha256": hashlib.sha256(args.source_cache.read_bytes()).hexdigest(),
                            "fitter_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    candidate = copy.deepcopy(base)
    candidate["item_choice"] = kernel
    assert all(candidate[key] == value for key, value in base.items())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(candidate, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    print(json.dumps({"output": str(args.output), "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      "base_fields_unchanged": True, "source_audit": kernel["source_audit"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
