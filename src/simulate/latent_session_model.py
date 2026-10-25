"""Small deterministic finite-horizon MaxEnt IRL and discrete return model."""
from itertools import product

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

GAP_STATE_NAMES = (
    'engagement', 'friction', 'satiation', 'inactivity_momentum',
    'gap_acceleration', 'habit_strength'
)


def sample_reward_coefficients(model, rng, persona=None):
    """Sample one user's reward coefficients from a persisted IRL coefficient GMM."""
    weights = np.asarray(model['weights'], dtype=float)
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError('invalid coefficient mixture weights')
    if persona is None:
        component = int(rng.choice(len(weights), p=weights))
    else:
        matches = [row for row in model['component_persona_composition']
                   if int(row['persona']) == int(persona)]
        if not matches:
            raise ValueError('persona has no coefficient-mixture support')
        support = np.zeros(len(weights))
        for row in matches:
            support[int(row['gmm_component'])] = row['size']
        component = int(rng.choice(len(weights), p=support / support.sum()))
    mean = np.asarray(model['standardized_means'][component], dtype=float)
    variance = np.asarray(model['standardized_variances'][component], dtype=float)
    standardized = rng.normal(mean, np.sqrt(variance))
    values = np.asarray(model['scaler_mean']) + standardized * np.asarray(model['scaler_scale'])
    coefficients = {}
    for term, value in zip(model['terms'], values):
        action, feature = term.split(':', 1)
        coefficients.setdefault(action, {})[feature] = float(value)
    dependent = model['dependent_action']
    features = next(iter(coefficients.values()))
    coefficients[dependent] = {
        feature: -sum(action_values[feature] for action_values in coefficients.values())
        for feature in features
    }
    return component, coefficients


def lifecycle_action_probabilities(
    model, persona, action_features, prior_state, new_state, user_coefficients=None
):
    """Evaluate immediate action reward under smoothly changing lifecycle state."""
    action_features = np.asarray(action_features, dtype=float)
    prior_state = np.asarray(prior_state, dtype=float)
    if action_features.shape != (12,) or prior_state.shape != (len(GAP_STATE_NAMES),):
        raise ValueError('invalid action or lifecycle state features')
    raw = np.r_[action_features, prior_state, float(new_state)]
    scaled = raw.copy()
    scaled[1:] = (scaled[1:] - np.asarray(model['mean'])) / np.asarray(model['scale'])
    theta = np.asarray(model['personas'][str(persona)]['coefficient'])
    if user_coefficients is None:
        logits = theta @ scaled
    else:
        actions = model['actions']
        base_names = model['features'][:len(action_features)]
        user_theta = np.asarray([
            [user_coefficients[action][feature] for feature in base_names]
            for action in actions
        ])
        logits = user_theta @ scaled[:len(action_features)]
        logits += theta[:, len(action_features):] @ scaled[len(action_features):]
    return np.exp(logits - logsumexp(logits))


def canonical_type_probabilities(model, persona, normalized_latent=None):
    assignment = model.get('canonical_assignment')
    if assignment is None:
        probabilities = model['personas'][str(persona)]['canonical_type_probabilities']
        return list(probabilities), np.asarray(list(probabilities.values()), dtype=float)
    if normalized_latent is None:
        raise ValueError('normalized latent is required by the canonical assignment matrix')
    latent = np.asarray(normalized_latent, dtype=float)
    matrix = np.asarray(assignment['matrix'], dtype=float)
    if latent.shape != (matrix.shape[0] - 1,):
        raise ValueError('invalid canonical latent dimension')
    logits = np.r_[1., latent] @ matrix
    return assignment['types'], np.exp(logits - logsumexp(logits))


def sample_uplift_profile(model, persona, rng):
    assignment = model.get('canonical_assignment')
    latent = None
    if assignment is not None:
        distribution = assignment['persona_latent_gaussians'][str(persona)]
        latent = (
            np.asarray(distribution['mean'])
            + np.asarray(distribution['cholesky']) @ rng.normal(size=len(distribution['mean']))
        )
    names, values = canonical_type_probabilities(model, persona, latent)
    if np.any(values < 0) or not np.isclose(values.sum(), 1):
        raise ValueError('invalid canonical-type probabilities')
    choice = int(rng.choice(len(names), p=values))
    return {
        'canonical_type': names[choice],
        'canonical_probabilities': values,
        'normalized_latent': latent,
    }


def sample_canonical_type(model, persona, rng):
    return sample_uplift_profile(model, persona, rng)['canonical_type']


def coupon_offsets(model, persona, canonical_type, receipt_ages_days, strength=1.0):
    """Return purchase and return-logit offsets; no active receipt is exactly zero."""
    if strength is None:
        strength = model.get('strength_by_persona', {}).get(
            str(persona), model['default_strength']
        )
    elif isinstance(strength, dict):
        strength = strength[str(persona)]
    if strength < 0 or any(age < 0 for age in receipt_ages_days):
        raise ValueError('coupon strength and ages must be nonnegative')
    multiplier = model['type_multipliers'][canonical_type]
    rho = model['decay']['rho']
    exposure = 0.
    for age in receipt_ages_days:
        week = int(age // 7) + 1
        if week <= 6:
            exposure += max(0., 1 - (week - 1) / 5) ** rho
    exposure = np.sqrt(exposure) * multiplier * strength
    specification = model['personas'][str(persona)]
    return {
        'purchase': exposure * specification['purchase_logit_shift'],
        'return': exposure * specification['return_logit_shift'],
    }


def coupon_action_probabilities(
    base_probabilities, model, persona, canonical_type, receipt_ages_days, strength=1.0
):
    probabilities = np.asarray(base_probabilities, dtype=float)
    if probabilities.shape != (5,) or np.any(probabilities <= 0):
        raise ValueError('five positive baseline action probabilities are required')
    offset = coupon_offsets(
        model, persona, canonical_type, receipt_ages_days, strength
    )['purchase']
    logits = np.log(probabilities)
    logits[3] += offset
    return np.exp(logits - logsumexp(logits))


def candidate_paths(n_actions, max_events=3):
    if n_actions < 2 or max_events < 1:
        raise ValueError('invalid action space or horizon')
    return [p for length in range(1, max_events + 1) for p in product(range(n_actions), repeat=length)]


def path_features(path, latent, n_actions, max_events=3):
    """Sum reward features over actions and inferred END; view is zero reference."""
    latent = np.asarray(latent, dtype=float)
    if not 1 <= len(path) <= max_events or any(a < 0 or a >= n_actions for a in path):
        raise ValueError('invalid trajectory')
    width = 3 + len(latent)
    features = np.zeros((n_actions, width))  # one row per non-view action incl END
    previous = -1
    for step, action in enumerate([*path, n_actions]):
        if action > 0:
            features[action - 1] += np.r_[1., step / max_events,
                                           float(previous == 2), latent]
        previous = action
    return features.ravel()


def fit_reward(latents, observed, n_actions, l1=.1):
    paths = candidate_paths(n_actions)
    index = {p: i for i, p in enumerate(paths)}
    latents = np.asarray(latents)
    if len(latents) != len(observed) or not len(observed) or l1 < 0:
        raise ValueError('invalid training data')
    # Features are affine in latent probabilities: construct once per candidate.
    base = np.stack([path_features(p, np.zeros(latents.shape[1]), n_actions) for p in paths])
    basis = [np.stack([path_features(p, np.eye(latents.shape[1])[j], n_actions) for p in paths]) - base
             for j in range(latents.shape[1])]
    empirical_index = np.array([index[tuple(p)] for p in observed])
    d = base.shape[1]

    def objective(v):
        w = v[:d] - v[d:]
        scores = np.broadcast_to(base @ w, (len(latents), len(paths))).copy()
        if basis:
            scores += latents @ np.stack([b @ w for b in basis])
        logz = logsumexp(scores, axis=1)
        probs = np.exp(scores - logz[:, None])
        resid = probs
        resid[np.arange(len(resid)), empirical_index] -= 1
        grad = resid.sum(axis=0) @ base
        for j in range(latents.shape[1]):
            grad += (resid.T @ latents[:, j]) @ basis[j]
        loss = float(logz.sum() - scores[np.arange(len(scores)), empirical_index].sum())
        return loss + l1*v.sum(), np.r_[grad+l1, -grad+l1]

    fit = minimize(objective, np.zeros(2*d), method='L-BFGS-B', jac=True,
                   bounds=[(0, None)]*(2*d), options={'maxiter': 1000, 'ftol': 1e-9})
    if not fit.success:
        raise RuntimeError(fit.message)
    return fit.x[:d] - fit.x[d:], {'iterations': int(fit.nit), 'objective': float(fit.fun)}


def path_probabilities(latents, reward, n_actions):
    paths = candidate_paths(n_actions)
    probs = []
    for z in np.asarray(latents):
        f = np.stack([path_features(p, z, n_actions) for p in paths])
        scores = f @ reward
        probs.append(np.exp(scores-logsumexp(scores)))
    return paths, np.asarray(probs)


def sequential_choice(paths, probs, prefix):
    """Conditional choices from exact trajectory weights (soft Q, not raw reward)."""
    if not paths or len(paths) != len(probs) or not np.isclose(np.sum(probs), 1):
        raise ValueError('invalid trajectory distribution')
    prefix = tuple(prefix)
    mass = {}
    for p, value in zip(paths, probs):
        if p[:len(prefix)] != prefix or len(p) < len(prefix):
            continue
        action = p[len(prefix)] if len(p) > len(prefix) else 'END'
        mass[action] = mass.get(action, 0.) + value
    total = sum(mass.values())
    if total <= 0:
        raise ValueError('prefix has no support')
    return {key: value/total for key, value in mass.items()}


def coupon_outcomes(p0, p1, shared_uniform):
    """Explicit monotone coupling scenario, not identified canonical labels."""
    if not (0 <= p0 <= 1 and 0 <= p1 <= 1 and 0 <= shared_uniform <= 1):
        raise ValueError('probability outside [0,1]')
    return shared_uniform < p0, shared_uniform < p1


def persona_gap_features(summary):
    """Past-only session-end state used by the persona return model."""
    previous = summary.get('previous_gap_hours')
    has_previous = previous is not None and np.isfinite(previous) and previous >= 0
    if not has_previous:
        previous = summary.get('initial_gap_hours', 0)
    user_mean = summary.get('user_mean_gap_hours', summary.get('initial_gap_hours', 0))
    values = [summary.get(action, 0) for action in ('view', 'cart', 'remove', 'purchase')]
    values += [summary.get('duration_hours', 0), summary.get('events', 0),
               previous, user_mean]
    if any(not np.isfinite(value) or value < 0 for value in values):
        raise ValueError('gap state values must be finite and nonnegative')
    return np.r_[np.log1p(values), float(has_previous)]


def sample_persona_gap(model, persona, summary, rng):
    """Sample a return interval; None denotes no return within the fitted horizon."""
    specification = model['personas'][str(persona)]
    summary = dict(summary)
    summary.setdefault('initial_gap_hours', model.get('initial_gap_hours', 0))
    summary.setdefault(
        'user_mean_gap_hours',
        model.get('default_user_mean_gap_hours', model.get('initial_gap_hours', 0)),
    )
    state = persona_gap_features(summary)
    mean, scale = np.asarray(specification['mean']), np.asarray(specification['scale'])
    state = (state - mean) / scale
    coefficient = np.asarray(specification['coefficient'])
    edges = np.asarray(model['edges_hours'])
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        bins = np.zeros(len(edges) - 1); bins[i] = 1
        logit = float(np.r_[bins, state] @ coefficient)
        probability = 1 / (1 + np.exp(-np.clip(logit, -30, 30)))
        if rng.random() < probability:
            return float(rng.uniform(lo, hi))
    return None


def update_gap_internal_state(state, summary, memory, initial_gap_hours=0.0):
    """Update interpretable session-level latent state using information known at END."""
    state = np.asarray(state, dtype=float)
    if state.shape != (len(GAP_STATE_NAMES),) or not 0 <= memory < 1:
        raise ValueError('invalid internal state or memory')
    events = max(float(summary.get('events', 0)), 1.0)
    view = float(summary.get('view', 0))
    cart = float(summary.get('cart', 0))
    remove = float(summary.get('remove', 0))
    purchase = float(summary.get('purchase', 0))
    previous = summary.get('previous_gap_hours')
    if previous is None or not np.isfinite(previous) or previous < 0:
        previous = initial_gap_hours
    if any(not np.isfinite(x) or x < 0 for x in (view, cart, remove, purchase, events, previous)):
        raise ValueError('session signals must be finite and nonnegative')
    inactivity = min(np.log1p(previous) / np.log1p(720), 1.0)
    signal = np.array([
        min(np.log1p(events) / np.log(65), 1.0),
        min((remove + max(cart - purchase, 0.0)) / events, 1.0),
        min(purchase / events, 1.0),
        inactivity,
        max(inactivity - state[3], 0.0),
        1.0,
    ])
    return signal, memory * state + (1 - memory) * signal


def churn_sequence_features(signal, history=()):
    """Current and two previous core signals; history is ordered oldest to newest."""
    signal = np.asarray(signal, dtype=float)[:4]
    history = [np.asarray(value, dtype=float)[:4] for value in history]
    lag1 = history[-1] if history else np.zeros(4)
    lag2 = history[-2] if len(history) > 1 else np.zeros(4)
    return np.r_[signal, lag1, lag2, float(bool(history)), float(len(history) > 1)]


def sample_home_persona(model, rng):
    """Draw the mature persona fixed at birth from the calibrated population mix."""
    probabilities = model['lifecycle']['home_persona_prior']
    personas = list(probabilities)
    values = np.asarray(list(probabilities.values()), dtype=float)
    if any(int(persona) not in model['lifecycle']['mature_personas'] for persona in personas):
        raise ValueError('home persona prior contains a non-mature persona')
    if np.any(values < 0) or not np.isclose(values.sum(), 1):
        raise ValueError('invalid home persona prior')
    return int(personas[int(rng.choice(len(personas), p=values))])


def sample_new_user(model, rng):
    """Sample a home persona while lifecycle starts from NEW."""
    return {
        'home_persona': sample_home_persona(model, rng),
        'lifecycle_state': 'NEW',
    }


def sample_internal_gap(model, persona, state, summary, rng, return_logit_offset=0.0):
    """Choose a MaxEnt gap token and update latent state; None means no return in 30d."""
    specification = model['personas'][str(persona)]
    signal, updated = update_gap_internal_state(
        state, summary, specification['memory'], model['initial_gap_hours']
    )
    if specification['features'] == 'state_interaction':
        features = np.r_[signal, updated, signal * updated]
    else:
        features = updated if specification['features'] == 'internal_state' else signal
    mean = np.asarray(specification['mean'])
    scale = np.asarray(specification['scale'])
    features = (features - mean) / scale
    churn_inputs = {
        'persona_only': np.zeros(1),
        'core_session_signal': signal[:4],
        'session_signal': signal,
        'internal_state': updated,
        'state_interaction': np.r_[signal, updated, signal * updated],
        'churn_sequence': churn_sequence_features(
            signal, summary.get('churn_signal_history', ())
        ),
    }[specification['churn_features']]
    churn_inputs = (
        churn_inputs - np.asarray(specification['churn_mean'])
    ) / np.asarray(specification['churn_scale'])
    churn_logit = float(np.asarray(specification['churn_coefficient']) @ churn_inputs)
    churn_logit += specification['churn_intercept'] - return_logit_offset
    if rng.random() < 1 / (1 + np.exp(-np.clip(churn_logit, -30, 30))):
        return None, updated, model['gap_tokens'][-1]
    logits = np.asarray(specification['coefficient']) @ features
    logits += np.asarray(specification['intercept'])
    probability = np.exp(logits - logsumexp(logits))
    choice = int(rng.choice(len(probability), p=probability))
    lo, hi = model['edges_hours'][choice:choice + 2]
    return float(rng.uniform(lo, hi)), updated, model['gap_tokens'][choice]


def sample_lifecycle_step(
    model, home_persona, lifecycle_state, state, summary, rng,
    completed_sessions=1, return_logit_offset=0.0,
):
    """Advance the no-coupon lifecycle while keeping the birth persona fixed."""
    mature = set(model['lifecycle']['mature_personas'])
    if home_persona not in mature:
        raise ValueError('invalid fixed home persona')
    if lifecycle_state in ('EXIT', 'MISSION_EXIT'):
        return None, np.asarray(state, dtype=float), 'NO_RETURN_30D', lifecycle_state
    allowed = {'NEW', 'CHURNING', *(f'P{x}' for x in mature)}
    if lifecycle_state not in allowed:
        raise ValueError('invalid lifecycle state')
    gap, updated, token = sample_internal_gap(
        model, home_persona, state, summary, rng, return_logit_offset
    )
    if gap is None:
        if float(summary.get('purchase', 0)) > 0:
            destination = 'MISSION_EXIT'
        else:
            destination = 'EXIT' if lifecycle_state == 'CHURNING' else 'CHURNING'
    elif lifecycle_state == 'NEW':
        onboarding_sessions = int(model['lifecycle'].get('onboarding_sessions', 2))
        destination = f'P{home_persona}' if completed_sessions >= onboarding_sessions else 'NEW'
    else:
        destination = f'P{home_persona}'
    return gap, updated, token, destination


def _simulate_users(
    coefficient_model, action_model, gap_model, users=1000, seed=42,
    max_sessions=50, max_events=100, uplift_model=None,
    coupon_receipt_days=(), coupon_strength=1.0,
):
    """Roll out the fitted model without replaying observed contexts."""
    if users < 1 or max_sessions < 1 or max_events < 1:
        raise ValueError('rollout sizes must be positive')
    typical_event_gap = max(float(np.expm1(action_model['mean'][2])), 1e-3)
    sessions, transitions, profiles = [], [], []
    forced_ends = 0
    truncated_users = 0
    for user_id in range(users):
        # Per-user streams keep calibration comparisons stable when another
        # user's treatment changes its trajectory length.
        rng = np.random.default_rng(np.random.SeedSequence([seed, user_id]))
        canonical_rng = np.random.default_rng(
            np.random.SeedSequence([seed + 1_000_003, user_id])
        )
        user = sample_new_user(gap_model, rng)
        persona = user['home_persona']
        lifecycle = user['lifecycle_state']
        uplift_profile = (
            sample_uplift_profile(uplift_model, persona, canonical_rng)
            if uplift_model is not None else None
        )
        canonical_type = None if uplift_profile is None else uplift_profile['canonical_type']
        if uplift_profile is not None:
            profiles.append({
                'user_id': user_id, 'home_persona': persona,
                'canonical_type': canonical_type,
                'canonical_probabilities': uplift_profile['canonical_probabilities'].tolist(),
                'normalized_latent': uplift_profile['normalized_latent'].tolist(),
            })
        component, coefficients = sample_reward_coefficients(
            coefficient_model, rng, persona=persona
        )
        initial_inactivity = np.log1p(gap_model['initial_gap_hours']) / np.log1p(
            gap_model['edges_hours'][-1]
        )
        state = np.array([0., 0., 0., initial_inactivity, 0., 0.])
        previous_gap = None
        elapsed_days = 0.
        signal_history = []
        for session_index in range(max_sessions):
            receipt_ages = [
                elapsed_days - day for day in coupon_receipt_days
                if day <= elapsed_days
            ]
            offsets = (
                coupon_offsets(
                    uplift_model, persona, canonical_type, receipt_ages, coupon_strength
                ) if uplift_model is not None else {'purchase': 0., 'return': 0.}
            )
            counts = np.zeros(4)
            previous_action = -1
            elapsed = 0.
            ended = False
            for position in range(max_events):
                # END was fitted with a zero inter-event gap.  First decide whether
                # the session ends; only continuing events receive a sampled delay.
                end_features = _rollout_action_features(
                    position, elapsed, 0., counts, previous_action
                )
                probability = lifecycle_action_probabilities(
                    action_model, persona, end_features, state,
                    lifecycle == 'NEW', coefficients,
                )
                if uplift_model is not None:
                    probability = coupon_action_probabilities(
                        probability, uplift_model, persona, canonical_type,
                        receipt_ages, coupon_strength,
                    )
                if position and rng.random() < probability[-1]:
                    ended = True
                    break
                event_gap = 0. if position == 0 else min(
                    float(rng.exponential(typical_event_gap)), 30.
                )
                elapsed += event_gap
                features = _rollout_action_features(
                    position, elapsed, event_gap, counts, previous_action
                )
                event_probability = lifecycle_action_probabilities(
                    action_model, persona, features, state,
                    lifecycle == 'NEW', coefficients,
                )
                if uplift_model is not None:
                    event_probability = coupon_action_probabilities(
                        event_probability, uplift_model, persona, canonical_type,
                        receipt_ages, coupon_strength,
                    )
                event_probability = event_probability[:-1]
                event_probability /= event_probability.sum()
                action = int(rng.choice(4, p=event_probability))
                counts[action] += 1
                previous_action = action
            if not ended:
                forced_ends += 1
            summary = {
                'view': counts[0], 'cart': counts[1], 'remove': counts[2],
                'purchase': counts[3], 'events': counts.sum(),
                'duration_hours': elapsed / 60,
                'previous_gap_hours': previous_gap,
                'churn_signal_history': signal_history[-2:],
            }
            origin = lifecycle
            gap, updated, token, lifecycle = sample_lifecycle_step(
                gap_model, persona, lifecycle, state, summary, rng,
                completed_sessions=session_index + 1,
                return_logit_offset=offsets['return'],
            )
            signal, _ = update_gap_internal_state(
                state, summary, gap_model['personas'][str(persona)]['memory'],
                gap_model['initial_gap_hours'],
            )
            signal_history.append(signal)
            sessions.append({
                'user_id': user_id, 'home_persona': persona,
                'coefficient_component': component, 'session_index': session_index,
                'calendar_day': elapsed_days,
                'canonical_type': canonical_type,
                'coupon_purchase_offset': offsets['purchase'],
                'coupon_return_offset': offsets['return'],
                'lifecycle_state': origin, **summary, 'gap_hours': gap,
                'gap_token': token, 'next_lifecycle_state': lifecycle,
            })
            transitions.append((origin, lifecycle))
            state, previous_gap = updated, gap
            if gap is None:
                break
            elapsed_days += gap / 24
        else:
            truncated_users += 1
    return {
        'sessions': sessions,
        'transitions': transitions,
        'profiles': profiles,
        'users': users,
        'forced_end_rate': forced_ends / len(sessions),
        'truncated_user_rate': truncated_users / users,
        'typical_event_gap_minutes': typical_event_gap,
    }


def simulate_no_uplift_users(
    coefficient_model, action_model, gap_model, users=1000, seed=42,
    max_sessions=50, max_events=100,
):
    return _simulate_users(
        coefficient_model, action_model, gap_model, users, seed,
        max_sessions, max_events,
    )


def simulate_uplift_users(
    coefficient_model, action_model, gap_model, uplift_model,
    coupon_receipt_days=(0.,), coupon_strength=None, users=1000, seed=42,
    max_sessions=50, max_events=100,
):
    return _simulate_users(
        coefficient_model, action_model, gap_model, users, seed,
        max_sessions, max_events, uplift_model, coupon_receipt_days,
        coupon_strength,
    )


def _rollout_action_features(position, elapsed, gap, counts, previous):
    one_hot = np.zeros(4)
    if 0 <= previous < 4:
        one_hot[previous] = 1.
    return np.r_[
        1., np.log1p(position), np.log1p(elapsed), np.log1p(gap),
        np.log1p(counts), one_hot,
    ]


class LatentSessionUser:
    """One simulated user; latent responsibilities held fixed after two-session prefix."""

    def __init__(self, model, prefix, rng):
        import pandas as pd
        self.model, self.rng = model, rng
        row = pd.DataFrame([prefix], columns=model['prefix_cols'])
        if row.isna().any().any():
            raise ValueError('complete two-session prefix required')
        self.latent = model['gmm'].predict_proba(model['scaler'].transform(row))[0, :2]
        self.previous_gap_hours = float(prefix['previous_gap_hours']) * 720
        self.completed_sessions = 2

    def sample_session(self):
        paths, p = path_probabilities([self.latent], self.model['reward'], len(self.model['actions']))
        chosen = paths[self.rng.choice(len(paths), p=p[0])]
        events = [self.model['actions'][a] for a in chosen]
        return events

    def finish_session(self, events):
        """Update end-of-session state and sample delay; None means >30d, not permanent exit."""
        allowed = self.model['actions']
        if not events or len(events) > self.model.get('max_session_events', 3) or any(a not in allowed for a in events):
            raise ValueError('session has unsupported length or events')
        counts = [min(events.count(a), 20)/20 for a in ('view', 'cart', 'purchase', 'remove')]
        x = np.array([*counts, min(len(events), 20)/20,
                      min(max(self.previous_gap_hours, 0), 720)/720, *self.latent])
        edges = self.model['hazard_edges_hours']
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            bins = np.zeros(len(edges)-1); bins[i] = 1
            p = self.model['hazard'].predict_proba(np.r_[bins, x][None, :])[0, 1]
            if self.rng.random() < p:
                delay = float(self.rng.uniform(lo, hi))  # assumed within-bin uniformity
                self.previous_gap_hours = delay
                self.completed_sessions += 1
                return delay
        self.completed_sessions += 1
        return None
