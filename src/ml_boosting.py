"""Один воспроизводимый эксперимент HistGradientBoosting на временной валидации."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from .data_validation import validate_power_series
from .ml_baselines import METHOD_LABELS, backtest_baselines, calculate_forecast_metrics, compare_baselines
from .ml_features import HISTORY_DAYS, build_day_ahead_features


BOOSTING_METHOD = "hist_gradient_boosting"
BOOSTING_LABEL = "HistGradientBoosting"


@dataclass
class BoostingExperiment:
    model: HistGradientBoostingRegressor
    comparison: pd.DataFrame
    predictions: pd.DataFrame
    daily_mae: pd.DataFrame
    training_summary: pd.Series
    feature_names: tuple[str, ...]


# Обучает только на train и оценивает все дни validation без обновления весов модели.
def run_boosting_experiment(
    train: pd.Series,
    validation: pd.Series,
    model_parameters: dict,
    baseline_forecasts: pd.DataFrame | None = None,
    interval_minutes: int = 15,
) -> BoostingExperiment:
    validate_power_series(train, interval_minutes, complete_days=True)
    validate_power_series(validation, interval_minutes, complete_days=True)
    step = pd.Timedelta(minutes=interval_minutes)
    if train.index[-1] + step != validation.index[0]:
        raise ValueError("Train и validation должны граничить без пересечений или разрывов.")
    if model_parameters.get("early_stopping", "auto") is not False:
        raise ValueError("Отключите early_stopping: случайную внутреннюю валидацию не используем.")

    features, audit = build_day_ahead_features(pd.concat([train, validation]), interval_minutes)
    warmup_intervals = HISTORY_DAYS * (1440 // interval_minutes)
    training_index = train.index[warmup_intervals:]
    if training_index.empty:
        raise ValueError("Для обучения нужны полные сутки после семи дней исходной истории.")
    x_train = features.loc[training_index]
    y_train = train.loc[training_index]
    x_validation = features.loc[validation.index]
    if not np.isfinite(x_train.to_numpy()).all() or not np.isfinite(x_validation.to_numpy()).all():
        raise ValueError("Недостаточно доступной истории для всех признаков; строки не исключаем автоматически.")
    if not (training_index < validation.index[0]).all():
        raise AssertionError("Обучающие цели должны заканчиваться до начала валидации.")

    if baseline_forecasts is None:
        baseline_forecasts = backtest_baselines(train, validation, interval_minutes)
    if not baseline_forecasts.index.equals(validation.index):
        raise ValueError("Все методы должны оцениваться на всей одинаковой валидации.")
    if not np.array_equal(baseline_forecasts["actual_kw"].to_numpy(), validation.to_numpy()):
        raise ValueError("Фактические значения базовых методов не совпадают с валидацией.")

    model = HistGradientBoostingRegressor(**model_parameters)
    with threadpool_limits(limits=1):
        model.fit(x_train, y_train)
        predicted = model.predict(x_validation)
    predictions = baseline_forecasts.copy()
    predictions[BOOSTING_METHOD] = predicted
    for column in audit.columns.difference(["forecast_origin"]):
        predictions[f"model_{column}"] = audit.loc[validation.index, column]
    model_metrics = calculate_forecast_metrics(validation, predictions[BOOSTING_METHOD])
    comparison = pd.concat([
        compare_baselines(baseline_forecasts), model_metrics.to_frame(BOOSTING_METHOD).T,
    ]).sort_values("mae_kw", kind="stable")
    comparison.index.name = "method"
    daily_mae = predictions[[*METHOD_LABELS, BOOSTING_METHOD]].sub(
        predictions["actual_kw"], axis=0,
    ).abs().groupby(predictions["forecast_origin"]).mean()
    summary = pd.Series({
        "Начало обучающих целей": training_index[0],
        "Конец обучающих целей": training_index[-1],
        "Обучающих интервалов": len(training_index),
        "Обучающих суток": training_index.normalize().nunique(),
        "Признаков": len(features.columns),
        "Начальных суток только для истории": HISTORY_DAYS,
        "Интервалов валидации": len(validation),
        "Обучений модели": 1,
        "Ранняя остановка": model.early_stopping,
    }, name="Значение")
    return BoostingExperiment(model, comparison, predictions, daily_mae, summary, tuple(features.columns))


# Сохраняет проверяемые результаты в отдельные производные файлы, не меняя исходный датасет.
def save_boosting_results(experiment: BoostingExperiment, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment.predictions.to_csv(output_dir / "hgb_validation_predictions.csv")
    experiment.comparison.to_csv(output_dir / "hgb_validation_metrics.csv")
    experiment.daily_mae.to_csv(output_dir / "hgb_daily_mae.csv")
