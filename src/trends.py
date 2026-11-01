"""
trends.py — Price trend + forecast visualization for the user Trends
dashboard.

HONESTY NOTE: there is no real historical price record for this catalog
(prices are a single snapshot from the dataset). The "history" and
"forecast" lines here are a deterministic random-walk SIMULATION
anchored to today's real average price — useful for demonstrating a
trend/forecast dashboard UI, but should be labeled as illustrative
in your report, not presented as real market data.
"""

import hashlib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def simulate_price_trend(category: str, current_avg_price: float,
                          weeks_history: int = 12, weeks_forecast: int = 4) -> pd.DataFrame:
    """
    Deterministic (seeded by category name) random-walk series ending
    at today's real average price, plus a short forward projection.
    Same category always produces the same simulated curve, so it
    doesn't flicker between reruns.
    """
    seed = int(hashlib.md5(category.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)

    # Walk backward from today's real price to build a plausible history
    steps = rng.normal(loc=0, scale=current_avg_price * 0.025, size=weeks_history)
    history = [current_avg_price]
    for s in steps:
        history.append(max(history[-1] - s, current_avg_price * 0.5))
    history = list(reversed(history))  # oldest -> newest, ending at today's real price

    # Forecast forward from today with a slight seasonal/random drift
    forecast_steps = rng.normal(loc=current_avg_price * 0.002, scale=current_avg_price * 0.02,
                                 size=weeks_forecast)
    forecast = [history[-1]]
    for s in forecast_steps:
        forecast.append(max(forecast[-1] + s, current_avg_price * 0.5))
    forecast = forecast[1:]

    today = datetime.utcnow()
    hist_dates = [today - timedelta(weeks=(weeks_history - i)) for i in range(weeks_history + 1)]
    fcst_dates = [today + timedelta(weeks=i + 1) for i in range(weeks_forecast)]

    df = pd.DataFrame({
        "date": hist_dates + fcst_dates,
        "price": history + forecast,
        "type": ["Historical (simulated)"] * len(hist_dates) + ["Forecast (simulated)"] * len(fcst_dates),
    })
    return df
