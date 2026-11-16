"""Observable, online shopping history shared by source replay and runtime.

This is an applied MDP context, not a redefinition of KIT45. A session is
available only strictly after its last observed action plus 30 minutes, in both
domains. Its current/future outcome is never included in a snapshot. Artificial
caps, observation-window truncation and other censored sessions must close with
``completed=False``. Completion is an observation rule, not a latent persona.

The 30-day denominators contain available sessions whose starts are in
``[at-30d, at)``. Counts describe observed exposure, including a short history;
they do not assert 30 days of complete customer observation. Empty denominators
produce None. The last completed session's price summary uses the whole past.

Cart/remove counts are action counts (one call), never purchased/item units.
A source checkout's contiguous purchase rows should be collapsed before one
purchase call; only its observed purchase-presence flag is used here. Prices are
positive prices from observed view actions, averaged per view, not catalog or
purchase prices filled in for unobserved views. No learned effect is imposed.

The completed no-buy interaction means adapt Kuric et al. (2024), Table 3;
historical session/purchase/price summaries use the existing KIT variable family.
See docs/feature_registry_events_v1.json for the source papers and D/A boundary.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real

import pandas as pd


SHOPPING_HISTORY_VERSION = "shopping_observed_history_v1"
SHOPPING_HISTORY_FEATURES = (
    "mdp_history_cart_adds_per_nonpurchase_session_30d",
    "mdp_history_cart_removes_per_nonpurchase_session_30d",
    "mdp_history_completed_session_count_30d",
    "mdp_history_purchase_session_share_30d",
    "mdp_history_last_completed_session_mean_view_price_minor",
)
SHOPPING_HISTORY_DIAGNOSTICS = ("mdp_history_nonpurchase_session_count_30d",)
_AVAILABILITY_DELAY = pd.Timedelta(minutes=30)
_WINDOW = pd.Timedelta(days=30)


def _utc(value) -> pd.Timestamp:
    clock = pd.Timestamp(value)
    if pd.isna(clock) or clock.tzinfo is None:
        raise ValueError("shopping history requires known timezone-aware clocks")
    return clock.tz_convert("UTC")


def _session_key(value) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, Integral)) or not str(value).strip():
        raise ValueError("session_id must be a nonempty string or integer")
    return str(value)


@dataclass
class _OpenSession:
    session_id: str
    started_at: pd.Timestamp
    last_at: pd.Timestamp
    action_count: int = 0
    cart_adds: int = 0
    cart_removes: int = 0
    has_purchase: bool = False
    view_price_sum: float = 0.0
    view_price_count: int = 0


@dataclass(frozen=True)
class _CompletedSession:
    session_id: str
    started_at: pd.Timestamp
    ended_at: pd.Timestamp
    available_at: pd.Timestamp
    cart_adds: int
    cart_removes: int
    has_purchase: bool
    mean_view_price_minor: float | None

    @property
    def last_key(self):
        # ID is a deterministic tie convention, not within-clock causality.
        return self.available_at, self.ended_at, self.started_at, self.session_id


class ShoppingHistory:
    """One customer's observed prefix; call methods in nondecreasing clock order.

    ``snapshot(at)`` precedes the target action. After observing that action,
    call ``record_action`` at its actual clock. Tied clocks are permitted but a
    completed session at the exact availability boundary is still excluded.
    A newly begun session does not make any earlier pending session available.
    The helper accepts no customer label, propensity, donor or hidden-state input.
    """

    def __init__(self, observation_start_at=None):
        self.observation_start_at = _utc(observation_start_at) if observation_start_at is not None else None
        self._watermark = self.observation_start_at
        self._open: _OpenSession | None = None
        self._seen: set[str] = set()
        self._pending: deque[_CompletedSession] = deque()
        self._completed: deque[_CompletedSession] = deque()
        self._last_completed: _CompletedSession | None = None

    def _clock(self, at):
        clock = _utc(at)
        if self._watermark is not None and clock < self._watermark:
            raise ValueError("shopping history clocks cannot move backwards")
        return clock

    def _session(self, session_id):
        if self._open is None or self._open.session_id != _session_key(session_id):
            raise ValueError("action/end must refer to the active session")
        return self._open

    def begin_session(self, session_id, at) -> None:
        clock, key = self._clock(at), _session_key(session_id)
        if self._open is not None:
            raise ValueError("close the active session before beginning another")
        if key in self._seen:
            raise ValueError("session_id cannot be reused")
        self._open = _OpenSession(key, clock, clock)
        self._seen.add(key)
        self._watermark = clock

    def record_action(self, session_id, at, action, *, quantity=1, view_price_minor=None) -> None:
        """Record one observed shopping action; quantity never weights a count.

        ``remove`` and raw ``remove_from_cart`` are equivalent. Quantity may be
        unspecified (None); when supplied it must be a positive integer. Only a
        view may provide ``view_price_minor``, which must be finite and positive.
        """
        clock, session = self._clock(at), self._session(session_id)
        action = "remove" if action == "remove_from_cart" else action
        if action not in {"view", "cart", "remove", "purchase"}:
            raise ValueError("history records only view/cart/remove/purchase actions")
        if quantity is not None and (isinstance(quantity, bool) or not isinstance(quantity, Real)
                                     or not isfinite(quantity) or quantity <= 0 or int(quantity) != quantity):
            raise ValueError("quantity must be a positive integer when supplied")
        price = None
        if view_price_minor is not None:
            if action != "view" or isinstance(view_price_minor, bool) or not isinstance(view_price_minor, Real):
                raise ValueError("only a view may provide a positive observed view price")
            price = float(view_price_minor)
            if not isfinite(price) or price <= 0 or not isfinite(session.view_price_sum + price):
                raise ValueError("observed view price must be finite and positive")
        session.action_count += 1
        session.cart_adds += action == "cart"
        session.cart_removes += action == "remove"
        session.has_purchase |= action == "purchase"
        if price is not None:
            session.view_price_sum += price
            session.view_price_count += 1
        session.last_at = clock
        self._watermark = clock

    def end_session(self, session_id, at, *, completed=True) -> None:
        """Close at the last action clock; caps/windows/censoring use False.

        A real/source-inferred completion enters the pending bank at ``at+30m``.
        It becomes visible only when a later snapshot is strictly beyond that
        boundary. A censored session contributes to neither denominator nor the
        last-completed price, even if its observed prefix contains no purchase.
        """
        clock, session = self._clock(at), self._session(session_id)
        if not isinstance(completed, bool):
            raise ValueError("completed must explicitly be True or False")
        if clock != session.last_at:
            raise ValueError("end clock must equal the last observed action clock")
        if completed and not session.action_count:
            raise ValueError("an empty session cannot be an observed completion")
        if completed:
            mean = session.view_price_sum / session.view_price_count if session.view_price_count else None
            self._pending.append(_CompletedSession(session.session_id, session.started_at, clock,
                clock + _AVAILABILITY_DELAY, session.cart_adds, session.cart_removes, session.has_purchase, mean))
        self._open = None
        self._watermark = clock

    def snapshot(self, at) -> dict[str, float | int | None]:
        """Return five observed features plus one nonpurchase-exposure count."""
        clock = self._clock(at)
        while self._pending and self._pending[0].available_at < clock:
            session = self._pending.popleft()
            self._completed.append(session)
            if self._last_completed is None or session.last_key > self._last_completed.last_key:
                self._last_completed = session
        low = clock - _WINDOW
        while self._completed and self._completed[0].started_at < low:
            self._completed.popleft()
        completed = list(self._completed)
        nonpurchase = [session for session in completed if not session.has_purchase]
        total, no_buy = len(completed), len(nonpurchase)
        self._watermark = clock
        return {
            SHOPPING_HISTORY_FEATURES[0]: sum(s.cart_adds for s in nonpurchase) / no_buy if no_buy else None,
            SHOPPING_HISTORY_FEATURES[1]: sum(s.cart_removes for s in nonpurchase) / no_buy if no_buy else None,
            SHOPPING_HISTORY_FEATURES[2]: total,
            SHOPPING_HISTORY_FEATURES[3]: sum(s.has_purchase for s in completed) / total if total else None,
            SHOPPING_HISTORY_FEATURES[4]: self._last_completed.mean_view_price_minor if self._last_completed else None,
            SHOPPING_HISTORY_DIAGNOSTICS[0]: no_buy,
        }
