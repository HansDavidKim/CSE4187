"""Generate products, assortments and customer traits from population counts."""
from collections import Counter, defaultdict
import math

import numpy as np


GAP_SECONDS = (0, 1, 2, 5, 10, 20, 30, 45, 60, 90, 120, 180, 240, 360, 600, 900, 1200, 1800)


def _price_band(price):
    price = int(price)
    step = 10 ** max(1, len(str(price)) - 2)
    return max(1, (price // step) * step), step


def aggregate_population(shopping_model):
    """Calibrate binned summaries; serialized output contains no source IDs or baskets."""
    catalog = shopping_model["catalog"]
    prices = {str(row["product_id"]): int(row["unit_price_minor"]) for row in catalog}
    categories = Counter(row["category_id"] for row in catalog)
    price_counts = defaultdict(Counter)
    for row in catalog:
        category = row["category_id"]
        group = category if categories[category] >= 20 else None
        price_counts[group][_price_band(row["unit_price_minor"])] += 1
    category_bands = sorted(sorted((price, step, count) for (price, step), count in bands.items())
                            for bands in price_counts.values())

    latent = shopping_model["latent_shopping"]
    posteriors = latent["profile_posteriors"]
    members = defaultdict(list)
    for profile in shopping_model["profiles"]:
        members[profile["home_group"]].append(profile)
    groups = {}
    checkout_lines, preferred_lines = Counter(), Counter()
    gap_edges = np.asarray(GAP_SECONDS)
    for name, profiles in sorted(members.items()):
        preferred_prices = Counter(_price_band(profile["preferred_price_minor"])[0] for profile in profiles)
        assortment_sizes = Counter(len(profile["product_weights"]) for profile in profiles)
        leading_weights = [max(weights) / sum(weights) for profile in profiles
                           if len(weights := list(profile["product_weights"].values())) > 1]
        gaps = [gap for profile in profiles for gap in profile.get("choice_gap_seconds", ())]
        gap_indices = np.searchsorted(gap_edges, gaps, side="right") - 1
        gap_counts = Counter(int(gap_edges[i]) for i in gap_indices)
        groups[name] = {
            "customers": len(profiles),
            "component_probabilities": np.mean([posteriors.get(profile["profile_id"], latent["mixture_weights"])
                                                for profile in profiles], axis=0).tolist(),
            "visit_probability_30d": float(np.mean([profile["visit_probability_30d"] for profile in profiles])),
            "preferred_price_counts": sorted(preferred_prices.items()),
            "assortment_size_counts": sorted(assortment_sizes.items()),
            "leading_preference_share": float(np.mean(leading_weights)) if leading_weights else 1.0,
            "choice_gap_counts": sorted(gap_counts.items()),
        }
        for profile in profiles:
            for basket in profile.get("checkout_templates", ()):
                weight = basket["observed_count"]
                checkout_lines[name] += weight * len(basket["items"])
                preferred_lines[name] += weight * sum(
                    str(item["product_id"]) in profile["product_weights"] for item in basket["items"])
    for name, group in groups.items():
        group["checkout_preference_share"] = (preferred_lines[name] / checkout_lines[name]
                                               if checkout_lines[name] else 0.0)

    baskets = Counter()
    for template in shopping_model["distributions"]["checkout_templates"]:
        items = template["items"]
        if any(str(item["product_id"]) not in prices for item in items):
            continue
        units = sum(item["quantity"] for item in items)
        total = sum(prices[str(item["product_id"])] * item["quantity"] for item in items)
        basket_price, _ = _price_band(total / units)
        baskets[(units, len(items), basket_price)] += template["observed_count"]
    if not category_bands or not groups or not baskets:
        raise ValueError("population calibration needs source products, profiles and priced checkouts")
    return {"schema_version": 1, "catalog_bands": category_bands, "home_groups": groups,
            "checkout_counts": [(*shape, count) for shape, count in sorted(baskets.items())]}


def _positive_integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool) and value > 0


def make_catalog(population, rng):
    """Use generated SKU and category identities with prices sampled inside fitted bins."""
    if population.get("schema_version") != 1 or not population.get("catalog_bands"):
        raise ValueError("population catalog needs nonempty calibrated price bins")
    products = []
    for index, bands in enumerate(population["catalog_bands"], 1):
        category_id = f"SC{index:04d}"
        for price, width, count in bands:
            if not all(_positive_integer(value) for value in (price, width, count)):
                raise ValueError("catalog bins require positive integer prices, widths and counts")
            for value in rng.integers(price, price + width, size=count):
                products.append({"product_id": f"SP{len(products) + 1:07d}", "category_id": category_id,
                                 "unit_price_minor": int(value)})
    if not products:
        raise ValueError("population catalog cannot be empty")
    return products


def _histogram(rows):
    if not rows or any(not _positive_integer(count) or not np.isfinite(value) or value < 0 for value, count in rows):
        raise ValueError("population histograms need nonnegative values and positive integer counts")
    values, counts = zip(*rows)
    weights = np.asarray(counts, dtype=float)
    return np.asarray(values), weights / weights.sum()


def compile_population_catalog(products):
    """Reuse the deterministic price index and assortment bounds across resets."""
    if not products:
        raise ValueError("population needs an active catalog")
    by_price = sorted((row["unit_price_minor"], str(row["product_id"])) for row in products)
    prices, ids = map(np.asarray, zip(*by_price))
    prices.setflags(write=False)
    ids.setflags(write=False)
    return prices, ids, {}


def prepare_population(population, products, rng, *, catalog_index=None):
    """Reuse the engine's checkout and item samplers with generated supports."""
    if population.get("schema_version") != 1 or not products or not population.get("home_groups"):
        raise ValueError("population needs an active catalog and calibrated home groups")
    spending = population.get("checkout_spending")
    if spending is not None:
        from ..irl.fit_checkout_spending import validate_spending_model
        validate_spending_model(spending)
    product_prices, ids, bounds = (compile_population_catalog(products)
        if catalog_index is None else catalog_index)

    def assortment(size, price):
        size = min(size, len(ids))
        key = size, price
        if key not in bounds:
            lower, upper = np.searchsorted(product_prices, [price / 1.5, price * 1.5])
            if upper - lower < size:
                center = int(np.searchsorted(product_prices, price))
                lower = max(0, min(center - size // 2, len(ids) - size))
                upper = lower + size
            bounds[key] = lower, upper
        lower, upper = bounds[key]
        return ids[lower + rng.choice(upper - lower, size=size, replace=False)].tolist()

    templates = []
    for units, lines, price, count in population["checkout_counts"]:
        if not all(_positive_integer(value) for value in (units, lines, price, count)) or lines > units:
            raise ValueError("checkout bins require positive quantities, prices and counts")
        # shortcut: generated baskets preserve size and price bands; fit SKU affinities if needed.
        variants = min(count, 4)
        for i in range(variants):
            items = assortment(lines, price)
            quantities = 1 + rng.multinomial(units - len(items), [1 / len(items)] * len(items))
            templates.append({"items": [{"product_id": pid, "quantity": int(quantity)}
                                        for pid, quantity in zip(items, quantities)],
                              "observed_count": count // variants + int(i < count % variants)})
    if not templates:
        raise ValueError("population needs at least one calibrated checkout")

    groups, parameters = {}, {}
    group_names = tuple(population["home_groups"])
    _, group_weights = _histogram([(index, group["customers"])
                                   for index, group in enumerate(population["home_groups"].values())])
    for name, group in population["home_groups"].items():
        gap, gap_weight = _histogram(group["choice_gap_counts"] or [(0, 1)])
        slots = (np.arange(256) + 0.5) / 256
        groups[name] = {"choice_gap_seconds": gap[np.searchsorted(gap_weight.cumsum(), slots)].tolist(),
                        "checkout_templates": templates}
        sizes, size_weights = _histogram(group["assortment_size_counts"])
        prices, price_weights = _histogram(group["preferred_price_counts"])
        if spending is not None and spending.get("assortment_price_copula") is not None:
            order = np.argsort(prices)
            prices, price_weights = prices[order], price_weights[order]
        components = np.asarray(group["component_probabilities"], dtype=float)
        if (components.ndim != 1 or not len(components) or not np.isfinite(components).all()
                or (components < 0).any() or not np.isclose(components.sum(), 1)
                or not all(0 <= group[key] <= 1 for key in (
                    "visit_probability_30d", "leading_preference_share", "checkout_preference_share"))):
            raise ValueError("invalid population profile probabilities")
        parameters[name] = sizes, size_weights, prices, price_weights, components.tolist()

    def draw_profile(candidate_index):
        name = group_names[int(rng.choice(len(group_names), p=group_weights))]
        group = population["home_groups"][name]
        sizes, size_weights, prices, price_weights, components = parameters[name]
        size = int(rng.choice(sizes, p=size_weights))
        preference = None
        copula = spending.get("assortment_price_copula") if spending is not None else None
        if copula is not None:
            preference = rng.multivariate_normal(spending["mean"], spending["between_covariance"], check_valid="raise").tolist()
            standard_price = (preference[1] - spending["mean"][1]) / math.sqrt(spending["between_covariance"][1][1])
            rho = copula["rho"]
            price_rank = .5 * (1 + math.erf((rho * standard_price + math.sqrt(1 - rho ** 2) * rng.normal()) / math.sqrt(2)))
            price = int(prices[min(len(prices) - 1, np.searchsorted(price_weights.cumsum(), price_rank, side="right"))])
        else:
            price = int(rng.choice(prices, p=price_weights))
        favorites = assortment(size, price)
        if len(favorites) > 1:
            head = max(1 / len(favorites), group["leading_preference_share"])
            weights = [head, *[(1 - head) / (len(favorites) - 1)] * (len(favorites) - 1)]
        else:
            weights = [1] * len(favorites)
        profile = {"profile_id": f"SYN{candidate_index:09d}", "home_group": name, "preferred_price_minor": price,
                "visit_probability_30d": group["visit_probability_30d"], "product_weights": dict(zip(favorites, weights)),
                "checkout_preference_share": group["checkout_preference_share"],
                "component_probabilities": components}
        if spending is not None:
            profile["checkout_spending_preference"] = preference if preference is not None else rng.multivariate_normal(
                spending["mean"], spending["between_covariance"], check_valid="raise").tolist()
        return profile

    return groups, draw_profile


def _budgeted_cart_units(reserved, held, direct, quota, products, units, lower, upper):
    """Maximize feasible held units inside the sampled checkout's total price bin."""
    from scipy.optimize import Bounds, LinearConstraint, milp

    held_ids = sorted(pid for pid, qty in held.items() if qty > 0)
    direct_ids = sorted(pid for pid, qty in direct.items() if qty > 0)
    ids = held_ids + direct_ids
    capacities = np.asarray([held[pid] for pid in held_ids] + [direct[pid] for pid in direct_ids])
    prices = np.asarray([products[pid]["unit_price_minor"] for pid in ids])
    held_flags = np.asarray([1] * len(held_ids) + [0] * len(direct_ids))
    reserved_amount = sum(products[pid]["unit_price_minor"] * qty for pid, qty in reserved.items())
    remaining = units - sum(reserved.values())
    solution = milp(-held_flags.astype(float), integrality=np.ones(len(ids)), bounds=Bounds(0, capacities),
        constraints=LinearConstraint(np.vstack([np.ones(len(ids)), prices, held_flags]),
            [remaining, lower - reserved_amount, 0], [remaining, upper - reserved_amount, quota]))
    if not solution.success or solution.x is None or not np.isfinite(solution.x).all():
        raise ValueError("cart allocation could not verify a feasible checkout budget")
    quantities = np.rint(solution.x).astype(np.int64)
    if (np.any(np.abs(solution.x - quantities) > 1e-6) or np.any(quantities < 0)
            or np.any(quantities > capacities) or quantities.sum() != remaining
            or not lower <= reserved_amount + int(prices @ quantities) <= upper
            or int(held_flags @ quantities) > quota):
        raise ValueError("cart allocation violates checkout quantities or budget")
    result = Counter(reserved)
    for pid, quantity in zip(held_ids, quantities[:len(held_ids)]):
        result[pid] += int(quantity)
    rest = Counter({pid: int(quantity) for pid, quantity in zip(direct_ids, quantities[len(held_ids):]) if quantity})
    return result, rest


def align_checkout_basket(basket, profile, products_by_id, rng, *, cart=None, cart_fulfillment=None):
    """Fill checkout quantities from the observed cart-route prior and actual stock."""
    held = {sku: quantity for sku, quantity in (cart or {}).items() if quantity > 0}
    held_by_price = defaultdict(list)
    share = profile.get("checkout_preference_share", 0)
    favorites_by_price = defaultdict(list)
    for sku in reversed(profile.get("product_weights", {})):
        if sku not in basket and (cart_fulfillment is None or not (cart or {}).get(sku, 0)):
            favorites_by_price[products_by_id[sku]["unit_price_minor"]].append(sku)
    result = Counter()
    if cart_fulfillment is not None and held:
        units = sum(basket.values())
        target_price = sum(products_by_id[sku]["unit_price_minor"] * qty for sku, qty in basket.items()) / units
        rows = cart_fulfillment["fraction_counts"]
        weights = np.asarray([count for _, _, count in rows], dtype=float)
        numerator, denominator, _ = rows[int(rng.choice(len(rows), p=weights / weights.sum()))]
        capacity = min(sum(held.values()), units)
        quotient, remainder = divmod(numerator * capacity, denominator)
        quota = quotient + int(bool(remainder) and rng.random() < remainder / denominator)
        direct = Counter(basket)
        for sku, quantity in basket.items():
            matched = min(held.get(sku, 0), quantity)
            result[sku] += matched
            held[sku] = held.get(sku, 0) - matched
            direct[sku] -= matched
        needed = max(0, quota - sum(result.values()))
        reserved, original_direct, extra_quota = result.copy(), direct.copy(), needed
        for sku in sorted(held, key=lambda pid: (abs(products_by_id[pid]["unit_price_minor"] - target_price), pid)):
            quantity = min(held[sku], needed)
            while quantity:
                replaced = min((pid for pid, count in direct.items() if count),
                    key=lambda pid: (abs(products_by_id[pid]["unit_price_minor"] - products_by_id[sku]["unit_price_minor"]), pid))
                consumed = min(quantity, direct[replaced])
                result[sku] += consumed
                direct[replaced] -= consumed
                quantity -= consumed
                needed -= consumed
            if not needed:
                break
        price, width = _price_band(target_price)
        lower, upper = price * units, (price + width) * units - 1
        amount = sum(products_by_id[pid]["unit_price_minor"] * qty for pid, qty in (result + direct).items())
        if not lower <= amount <= upper:
            result, direct = _budgeted_cart_units(reserved, held, original_direct, extra_quota,
                products_by_id, units, lower, upper)
        basket = {sku: quantity for sku, quantity in direct.items() if quantity}
    else:
        for sku in held:
            held_by_price[_price_band(products_by_id[sku]["unit_price_minor"])[0]].append(sku)
    for sku, quantity in basket.items():
        price = products_by_id[sku]["unit_price_minor"]
        candidates = sorted(held_by_price[_price_band(price)[0]],
            key=lambda candidate: (abs(products_by_id[candidate]["unit_price_minor"] - price), candidate != sku, candidate))
        for candidate in candidates:
            consumed = min(held[candidate], quantity)
            result[candidate] += consumed
            held[candidate] -= consumed
            quantity -= consumed
            if not quantity:
                break
        if quantity:
            favorites = favorites_by_price.get(price)
            if favorites and share and rng.random() < share:
                sku = favorites.pop()
            result[sku] += quantity
    return {sku: quantity for sku, quantity in result.items() if quantity > 0}
