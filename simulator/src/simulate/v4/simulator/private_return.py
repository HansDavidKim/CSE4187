"""Persistent retail return state whose session updates wait for observed END."""
from copy import deepcopy
import heapq

import pandas as pd

from simulate.v4.irl import return_state
from simulate.v4.irl.source_contract import utc_cutoff


class PrivateReturnState:
    """Keep private eta across visits; inferred END is known strictly after30min."""

    def __init__(self, model, rng):
        return_state.validate_model(model)
        self.model, self.rng = deepcopy(model), rng
        self.eta = None
        self.updates = 0
        self.pending = []
        self.clock = None

    def value_at(self, at):
        at = utc_cutoff(at)
        if self.clock is not None and at < self.clock:
            raise ValueError('private return state cannot move its observation clock backwards')
        self.clock = at
        while self.pending and self.pending[0] < at:
            heapq.heappop(self.pending)
            self.eta = return_state.advance_state(self.model, self.eta, self.rng)
            self.updates += 1
        return 0. if self.eta is None else self.eta

    def begin_session(self, at):
        value = self.value_at(at)
        if self.eta is None:
            self.eta = return_state.sample_initial_state(self.model, self.rng)
            value = self.eta
        return value

    def complete_session(self, ended_at, reason, truncated):
        ended_at = utc_cutoff(ended_at)
        if type(truncated) is not bool or self.eta is None or (self.clock is not None and ended_at < self.clock):
            raise ValueError('session completion requires initialized private state and an ordered clock')
        if not truncated and reason in {'causal_policy_end', 'donor_length'}:
            heapq.heappush(self.pending, ended_at + pd.Timedelta(minutes=30))
