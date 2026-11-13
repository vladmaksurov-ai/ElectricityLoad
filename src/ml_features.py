"""Признаки общего суточного прогноза, известные к 00:00 прогнозируемого дня."""

import pandas as pd

from .data_validation import validate_power_series


HISTORY_DAYS = 7
LAG_DAYS = (1, 2, 7)


# Строит все суточные точки старта векторно; первые семь суток оставляет для истории.
def build_day_ahead_features(
    power: pd.Series,
    interval_minutes: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_power_series(power, interval_minutes, complete_days=True)
    per_day = 1440 // interval_minutes
    origins = power.index.normalize()
    step = pd.Timedelta(minutes=interval_minutes)
    features = pd.DataFrame({
        "time_slot": (power.index.hour * 60 + power.index.minute) // interval_minutes,
        "weekday": power.index.dayofweek,
        "is_weekend": (power.index.dayofweek >= 5).astype(int),
        "month": power.index.month,
    }, index=power.index)
    audit = pd.DataFrame({
        "forecast_origin": origins,
        "history_start": origins - pd.Timedelta(days=HISTORY_DAYS),
        "history_end": origins - step,
    }, index=power.index)
    for days in LAG_DAYS:
        features[f"power_lag_{days}d"] = power.shift(days * per_day)
        audit[f"lag_{days}d_source"] = power.index - pd.Timedelta(days=days)
    features["profile_change_1d_7d"] = features["power_lag_1d"] - features["power_lag_7d"]
    features["last_known_power"] = power.reindex(origins - step).to_numpy()

    # Сначала сводки полных дней, затем сдвиг на день: факты текущих суток не попадут в признаки.
    grouped = power.groupby(origins, sort=True)
    daily = pd.DataFrame({
        "mean": grouped.mean(), "median": grouped.median(), "min": grouped.min(),
        "max": grouped.max(), "std": grouped.std(ddof=0),
    })
    previous = daily.shift(1)
    for statistic in daily.columns:
        features[f"previous_day_{statistic}"] = previous[statistic].reindex(origins).to_numpy()
    weekly = previous["mean"].rolling(HISTORY_DAYS, min_periods=HISTORY_DAYS)
    features["previous_week_mean"] = weekly.mean().reindex(origins).to_numpy()
    features["previous_week_daily_std"] = weekly.std(ddof=0).reindex(origins).to_numpy()
    features["daily_mean_change"] = (daily["mean"].shift(1) - daily["mean"].shift(2)).reindex(origins).to_numpy()

    if not audit.drop(columns="forecast_origin").lt(audit["forecast_origin"], axis=0).all().all():
        raise AssertionError("Источники признаков должны предшествовать началу прогнозируемых суток.")
    return features, audit
