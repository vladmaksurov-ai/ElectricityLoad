"""Суточные базовые прогнозы с явной границей доступной истории."""

import numpy as np
import pandas as pd

from .data_validation import validate_power_series


METHOD_LABELS = {
    "last_value": "Последнее значение",
    "previous_day": "Как вчера",
    "previous_week": "Как неделю назад",
}
METRIC_LABELS = {"mae_kw": "MAE, кВт", "rmse_kw": "RMSE, кВт", "wape_pct": "WAPE, %"}
SOURCE_COLUMNS = [f"{method}_source" for method in METHOD_LABELS]


# Строит сутки по проверенной истории, сохраняя контроль доступности каждого источника.
def _forecast_from_checked_history(
    history: pd.Series,
    forecast_day: str | pd.Timestamp,
    interval_minutes: int = 15,
) -> pd.DataFrame:
    origin = pd.Timestamp(forecast_day)
    if pd.isna(origin) or origin.tzinfo is not None or origin != origin.normalize():
        raise ValueError("Прогноз должен начинаться в 00:00 локальных суток.")
    step = pd.Timedelta(minutes=interval_minutes)
    if history.index[-1] >= origin:
        raise ValueError("История содержит значения из прогнозируемых суток или будущего.")
    if history.index[-1] != origin - step:
        raise ValueError("История должна заканчиваться непосредственно перед прогнозом.")

    target_index = pd.date_range(origin, periods=1440 // interval_minutes,
                                 freq=f"{interval_minutes}min", name="timestamp")
    result = pd.DataFrame(index=target_index)
    result["forecast_origin"] = origin
    result["last_value"] = float(history.iloc[-1])
    result["last_value_source"] = history.index[-1]
    for method, days in (("previous_day", 1), ("previous_week", 7)):
        sources = target_index - pd.Timedelta(days=days)
        values = history.reindex(sources)
        if values.isna().any():
            raise ValueError(f"Недостаточно истории для метода {METHOD_LABELS[method]}.")
        result[method] = values.to_numpy(dtype=float)
        result[f"{method}_source"] = sources
    if not result[SOURCE_COLUMNS].lt(result["forecast_origin"], axis=0).all().all():
        raise AssertionError("Источник прогноза должен быть известен до старта суток.")
    return result


# Проверяет внешнюю историю целиком перед самостоятельным прогнозом следующих суток.
def forecast_next_day(
    history: pd.Series,
    forecast_day: str | pd.Timestamp,
    interval_minutes: int = 15,
) -> pd.DataFrame:
    validate_power_series(history, interval_minutes)
    return _forecast_from_checked_history(history, forecast_day, interval_minutes)


# Ежедневно прогнозирует валидацию, добавляя фактический день в историю после прогноза.
def backtest_baselines(
    history: pd.Series,
    validation: pd.Series,
    interval_minutes: int = 15,
) -> pd.DataFrame:
    validate_power_series(history, interval_minutes)
    validate_power_series(validation, interval_minutes, complete_days=True)
    if history.index[-1] != validation.index[0] - pd.Timedelta(minutes=interval_minutes):
        raise ValueError("История и валидация должны граничить без разрывов и пересечений.")
    # Для этих методов достаточно последних семи суток; весь вход уже проверен выше.
    history_intervals = 7 * (1440 // interval_minutes)
    observed = history.iloc[-history_intervals:].copy()
    forecasts = []
    for day, actual_day in validation.groupby(validation.index.normalize(), sort=True):
        forecast = _forecast_from_checked_history(observed, day, interval_minutes)
        if not forecast.index.equals(actual_day.index):
            raise AssertionError("Для всех методов нужны одинаковые интервалы оценки.")
        forecast["actual_kw"] = actual_day
        forecasts.append(forecast)
        observed = pd.concat([observed, actual_day]).iloc[-history_intervals:]
    result = pd.concat(forecasts)
    if not result.index.equals(validation.index):
        raise AssertionError("Прогнозы должны покрывать всю валидацию без потерь.")
    return result


# Считает метрики на строго совпадающих интервалах; WAPE при нулевой сумме факта не определён.
def calculate_forecast_metrics(actual: pd.Series, prediction: pd.Series) -> pd.Series:
    if not isinstance(actual, pd.Series) or not isinstance(prediction, pd.Series):
        raise ValueError("Факт и прогноз должны быть рядами с одинаковым индексом.")
    if actual.empty or not actual.index.is_unique or not actual.index.equals(prediction.index):
        raise ValueError("Нужны непустые факт и прогноз на одинаковых уникальных интервалах.")
    actual_values = actual.to_numpy(dtype=float)
    prediction_values = prediction.to_numpy(dtype=float)
    if not np.isfinite(actual_values).all() or not np.isfinite(prediction_values).all():
        raise ValueError("Метрики требуют конечных значений без пропусков.")
    errors = prediction_values - actual_values
    absolute_errors = np.abs(errors)
    denominator = np.abs(actual_values).sum()
    return pd.Series({
        "mae_kw": absolute_errors.mean(),
        "rmse_kw": np.sqrt(np.mean(errors ** 2)),
        "wape_pct": 100 * absolute_errors.sum() / denominator if denominator else np.nan,
    })


# Сравнивает все методы на всех интервалах одним расчётом, без усреднения суточных WAPE.
def compare_baselines(forecasts: pd.DataFrame) -> pd.DataFrame:
    metrics = pd.DataFrame({
        method: calculate_forecast_metrics(forecasts["actual_kw"], forecasts[method])
        for method in METHOD_LABELS
    }).T
    metrics.index.name = "method"
    return metrics.sort_values("mae_kw", kind="stable")
