

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tools.eval_measures import aic  # could also use bic

# generate synthetic data
np.random.seed(42)
n = 100
series = np.random.randn(n)

for t in range(1, n):
    series[t] = 0.7 * series[t-1] + series[t]


def fit_setar_and_calculate_aic(series, threshold_value, lag=1):
    series_below_threshold = series[:-lag][series[lag:] < threshold_value]
    series_above_threshold = series[:-lag][series[lag:] >= threshold_value]

    model_below_threshold = AutoReg(series_below_threshold, lags=lag).fit()
    model_above_threshold = AutoReg(series_above_threshold, lags=lag).fit()

    nobs = len(series)
    rss_below = np.sum(model_below_threshold.resid ** 2)
    rss_above = np.sum(model_above_threshold.resid ** 2)
    rss_total = rss_below + rss_above

    k = 2 * (lag + 1)  # total number of parameters in both models
    setar_aic = aic(rss_total, nobs, k)
    return setar_aic


threshold_values = np.linspace(np.percentile(
    series, 10), np.percentile(series, 90), 100)

aic_values = [fit_setar_and_calculate_aic(
    series, threshold_value) for threshold_value in threshold_values]
optimal_threshold_value = threshold_values[np.argmin(aic_values)]

lag = 1  # choose an appropriate lag value based on your data and domain knowledge
series_below_threshold = series[:-lag][series[lag:] < optimal_threshold_value]
series_above_threshold = series[:-lag][series[lag:] >= optimal_threshold_value]

model_below_threshold = AutoReg(series_below_threshold, lags=lag).fit()
model_above_threshold = AutoReg(series_above_threshold, lags=lag).fit()

last_lagged_value = series[-lag]

if last_lagged_value < optimal_threshold_value:
    prediction = model_below_threshold.predict(start=len(series_below_threshold), end=len(series_below_threshold))
else:
    prediction = model_above_threshold.predict(start=len(series_above_threshold), end=len(series_above_threshold))

print("Next prediction:", prediction[0])
