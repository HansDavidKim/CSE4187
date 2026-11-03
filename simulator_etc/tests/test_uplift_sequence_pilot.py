import numpy as np

from analysis.uplift import sequence as module


def test_features_exclude_day_before_and_future_behavior():
    arr = np.zeros((1, 61), dtype=np.uint8)
    click, purchase, coupon = arr.copy(), arr.copy(), arr.copy()
    click[0, 17] = 1
    purchase[0, 18] = 1
    coupon[0, 18] = 1
    x = module.features(click, purchase, coupon, 19)[0]
    assert x[0] == 1 and x[1] == 0 and x[2] == 1
    purchase[0, 19:] = 1
    assert np.array_equal(x, module.features(click, purchase, coupon, 19)[0])
