"""Проверки суточных прогнозов, общей оценки и границы доступной информации."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ml_baselines import (
    METHOD_LABELS, SOURCE_COLUMNS, backtest_baselines,
    calculate_forecast_metrics, compare_baselines, forecast_next_day,
)


class BaselineForecastTests(unittest.TestCase):
    # Небольшой искусственный ряд служит только проверке алгоритмов, не экспериментам.
    def setUp(self):
        index = pd.date_range("2013-12-24", periods=17 * 96, freq="15min", name="timestamp")
        self.power = pd.Series(np.arange(len(index), dtype=float), index=index, name="power")
        self.history = self.power.iloc[:8 * 96]
        self.validation = self.power.iloc[8 * 96:]
        self.origin = self.validation.index[0]

    # Проверяет весь горизонт и точные профили, включая последний интервал суток.
    def test_next_day_values_and_source_timestamps(self):
        forecast = forecast_next_day(self.history, self.origin)
        self.assertEqual(len(forecast), 96)
        self.assertEqual(forecast.index[-1], self.origin + pd.Timedelta(hours=23, minutes=45))
        np.testing.assert_array_equal(forecast.last_value, np.repeat(self.history.iloc[-1], 96))
        np.testing.assert_array_equal(forecast.previous_day, self.history.iloc[-96:])
        np.testing.assert_array_equal(forecast.previous_week, self.history.iloc[96:192])
        self.assertTrue(forecast[SOURCE_COLUMNS].lt(forecast.forecast_origin, axis=0).all().all())
        self.assertTrue(forecast.last_value_source.eq(self.origin - pd.Timedelta(minutes=15)).all())
        self.assertTrue((forecast.previous_day_source == forecast.index - pd.Timedelta(days=1)).all())
        self.assertTrue((forecast.previous_week_source == forecast.index - pd.Timedelta(days=7)).all())

    # Не позволяет скрыто подставить прогнозируемые или тестовые сутки в историю.
    def test_history_containing_current_or_future_data_is_rejected(self):
        for history in (self.power.iloc[:8 * 96 + 1], self.power):
            with self.subTest(length=len(history)), self.assertRaises(ValueError):
                forecast_next_day(history, self.origin)

    # Отклоняет другой момент старта, неполную историю, разрыв и недостающую неделю.
    def test_invalid_origin_or_history_is_rejected(self):
        with self.assertRaises(ValueError):
            forecast_next_day(self.history, self.origin + pd.Timedelta(minutes=15))
        with self.assertRaises(ValueError):
            forecast_next_day(self.history, pd.NaT)
        for history in (self.history.iloc[:-1], self.history.drop(self.history.index[8]),
                        self.history.iloc[-6 * 96:]):
            with self.subTest(length=len(history)), self.assertRaises(ValueError):
                forecast_next_day(history, self.origin)

    # Проверяет полное покрытие валидации и совпадение первого опыта с общей оценкой.
    def test_backtest_uses_same_intervals_and_preserves_inputs(self):
        history_before, validation_before = self.history.copy(), self.validation.copy()
        forecasts = backtest_baselines(self.history, self.validation)
        self.assertTrue(forecasts.index.equals(self.validation.index))
        self.assertFalse(forecasts.isna().any().any())
        first = forecast_next_day(self.history, self.origin)
        pd.testing.assert_frame_equal(forecasts.iloc[:96].drop(columns="actual_kw"), first)
        pd.testing.assert_series_equal(forecasts.actual_kw, self.validation, check_names=False)
        pd.testing.assert_series_equal(self.history, history_before)
        pd.testing.assert_series_equal(self.validation, validation_before)

    # Изменение текущих и будущих фактов не должно менять уже выпущенные прогнозы.
    def test_current_and_future_actuals_cannot_change_issued_forecasts(self):
        original = backtest_baselines(self.history, self.validation)
        for offset in (0, 3, 8):
            day = self.origin + pd.Timedelta(days=offset)
            poisoned = self.validation.copy()
            poisoned.loc[poisoned.index >= day] += 1_000_000
            changed = backtest_baselines(self.history, poisoned)
            issued = original.forecast_origin <= day
            pd.testing.assert_frame_equal(
                original.loc[issued].drop(columns="actual_kw"),
                changed.loc[issued].drop(columns="actual_kw"),
            )

    # Завершившаяся валидация становится историей завтра, а через неделю — недельным лагом.
    def test_completed_validation_days_become_available_history(self):
        original = backtest_baselines(self.history, self.validation)
        changed_actual = self.validation.copy()
        changed_actual.iloc[:96] += 1000
        changed = backtest_baselines(self.history, changed_actual)
        np.testing.assert_array_equal(changed.previous_day.iloc[96:192],
                                      original.previous_day.iloc[96:192] + 1000)
        np.testing.assert_array_equal(changed.last_value.iloc[96:192],
                                      original.last_value.iloc[96:192] + 1000)
        np.testing.assert_array_equal(changed.previous_week.iloc[7 * 96:8 * 96],
                                      original.previous_week.iloc[7 * 96:8 * 96] + 1000)

    # Не допускает тихого исключения неудобных интервалов из общей оценки методов.
    def test_incomplete_invalid_or_overlapping_validation_is_rejected(self):
        invalid_series = [self.validation.iloc[1:], self.validation.drop(self.validation.index[100]),
                          self.power.iloc[7 * 96:]]
        for value in (np.nan, np.inf, -1):
            invalid = self.validation.copy()
            invalid.iloc[5] = value
            invalid_series.append(invalid)
        for validation in invalid_series:
            with self.subTest(length=len(validation)), self.assertRaises(ValueError):
                backtest_baselines(self.history, validation)

    # Сверяет общую таблицу с независимыми формулами на одной и той же выборке.
    def test_comparison_uses_all_validation_intervals(self):
        forecasts = backtest_baselines(self.history, self.validation)
        comparison = compare_baselines(forecasts)
        self.assertEqual(set(comparison.index), set(METHOD_LABELS))
        self.assertTrue(comparison.mae_kw.is_monotonic_increasing)
        for method in METHOD_LABELS:
            errors = forecasts[method].to_numpy() - self.validation.to_numpy()
            self.assertAlmostEqual(comparison.loc[method, "mae_kw"], np.abs(errors).mean())
            self.assertAlmostEqual(comparison.loc[method, "rmse_kw"], np.sqrt(np.mean(errors ** 2)))
            self.assertAlmostEqual(comparison.loc[method, "wape_pct"],
                                   np.abs(errors).sum() / self.validation.abs().sum() * 100)


class ForecastMetricTests(unittest.TestCase):
    # Контрольный пример отдельно проверяет квадрат ошибки и абсолютный знаменатель WAPE.
    def test_known_metrics_with_signed_actuals(self):
        actual = pd.Series([-2.0, 0.0, 2.0])
        prediction = pd.Series([0.0, 1.0, 1.0])
        metrics = calculate_forecast_metrics(actual, prediction)
        self.assertAlmostEqual(metrics.mae_kw, 4 / 3)
        self.assertAlmostEqual(metrics.rmse_kw, np.sqrt(2))
        self.assertAlmostEqual(metrics.wape_pct, 100)

    # Нулевой знаменатель означает неопределённый WAPE даже у идеального прогноза.
    def test_zero_denominator_returns_nan_without_infinity(self):
        for prediction in (pd.Series([0.0, 0.0]), pd.Series([1.0, 2.0])):
            metrics = calculate_forecast_metrics(pd.Series([0.0, 0.0]), prediction)
            self.assertTrue(np.isnan(metrics.wape_pct))
            self.assertTrue(np.isfinite(metrics[["mae_kw", "rmse_kw"]]).all())

    # При разном объёме нагрузки общий WAPE отличается от среднего суточных процентов.
    def test_wape_is_ratio_of_sums_not_average_daily_percentage(self):
        actual = pd.Series(np.repeat([1.0, 100.0], 96))
        prediction = pd.Series(np.repeat([0.0, 100.0], 96))
        overall = calculate_forecast_metrics(actual, prediction).wape_pct
        daily_average = np.mean([
            calculate_forecast_metrics(actual.iloc[:96], prediction.iloc[:96]).wape_pct,
            calculate_forecast_metrics(actual.iloc[96:], prediction.iloc[96:]).wape_pct,
        ])
        self.assertAlmostEqual(overall, 100 / 101)
        self.assertAlmostEqual(daily_average, 50)

    # Метрики не должны автоматически выравнивать, удалять или переставлять интервалы.
    def test_misaligned_empty_duplicate_and_nonfinite_inputs_are_rejected(self):
        pairs = [(pd.Series([1, 2]), pd.Series([1, 2], index=[1, 0])),
                 (pd.Series(dtype=float), pd.Series(dtype=float)),
                 (pd.Series([1, 2], index=[0, 0]), pd.Series([1, 2], index=[0, 0])),
                 (pd.Series([1, 2]), pd.Series([1])),
                 (pd.Series([1, np.nan]), pd.Series([1, 2])),
                 (pd.Series([1, 2]), pd.Series([1, np.inf]))]
        for actual, prediction in pairs:
            with self.subTest(actual=actual.tolist()), self.assertRaises(ValueError):
                calculate_forecast_metrics(actual, prediction)


if __name__ == "__main__":
    unittest.main()
