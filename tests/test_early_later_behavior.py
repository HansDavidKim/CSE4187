import pandas as pd
from analysis.persona.behavior import summarize_windows


def test_window_separation():
    events = pd.DataFrame({'user': [1]*4 + [2]*4, 'session': [1, 2, 3, 4]*2,
                           'action': [0, 0, 1, 2, 1, 2, 0, 0]})
    result = summarize_windows(events, ['view', 'cart', 'purchase'])
    assert result['users'] == 2
    assert result['counts']['early_view_only_later_cart_or_purchase'] == 1
    assert result['counts']['early_cart_or_purchase_later_view_only'] == 1
