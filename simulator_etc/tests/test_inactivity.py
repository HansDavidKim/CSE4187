import pandas as pd
import pytest

from analysis.eda import inactivity_label


def test_inactivity_censoring_and_boundary():
    landmarks = pd.Series(pd.to_datetime(['2020-01-01', '2020-01-01', '2020-01-25'], utc=True))
    nexts = pd.Series(pd.to_datetime(['2020-01-31', '2020-02-01', None], utc=True))
    labels = inactivity_label(landmarks, nexts, pd.Timestamp('2020-02-15', tz='UTC'), 30)
    assert labels.iloc[0] == False
    assert labels.iloc[1] == True
    assert pd.isna(labels.iloc[2])


def test_rejects_past_next_session():
    with pytest.raises(ValueError):
        inactivity_label(pd.Series(['2020-01-03']), pd.Series(['2020-01-01']),
                         pd.Timestamp('2020-03-01', tz='UTC'), 30)
