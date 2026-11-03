"""Session-based, non-overlapping behavior windows for retrospective analysis."""
from __future__ import annotations


def session_tokens(sessions, max_tokens=64):
    if not sessions or max_tokens < 1:
        raise ValueError('nonempty sessions and positive token limit required')
    tokens = []
    for actions in sessions:
        if not actions or any(action not in (0, 1, 2) for action in actions):
            raise ValueError('sessions must contain view/cart/purchase actions')
        tokens.extend(actions)
        tokens.append(3)
    return tokens[:max_tokens]


def summarize_tokens(tokens):
    events = sum(tokens.count(i) for i in range(3))
    if not events:
        raise ValueError('at least one event required')
    return {'views': tokens.count(0), 'carts': tokens.count(1), 'purchases': tokens.count(2),
            'events': events, 'tokens': len(tokens)}
