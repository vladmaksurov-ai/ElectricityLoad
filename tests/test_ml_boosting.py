"""Проверки доступности признаков, временного обучения и сравнения первой ML-модели."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ml_baselines import METHOD_LABELS, backtest_baselines
from src.ml_boosting import BOOSTING_METHOD, run_boosting_experiment
from src.ml_features import build_day_ahead_features


class DayAheadFeatureTests(unittest.TestCase):
    # Искусственный ряд используется только в тестах алгоритмов и защиты от утечки.
    def setUp(self):
        index = pd.date_range("2013-12-20", periods=16 * 96, freq="15min", name="timestamp")
        self.power = pd.Series(10 + np.arange(len(index), dtype=float) ** 1.2, index=index)

    # Сверяет лаги и сводки с прямыми срезами истории до начала прогнозируемого дня.
    def test_features_match_only_completed_history(self):
        before = self.power.copy()
        features, audit = build_day_ahead_features(self.power)
        origin = self.power.index[10 * 96]
        target = origin + pd.Timedelta(hours=12)
        previous_day = self.power.loc[origin - pd.Timedelta(days=1):origin - pd.Timedelta(minutes=15)]
        previous_week = self.power.loc[origin - pd.Timedelta(days=7):origin - pd.Timedelta(minutes=15)]
        row = features.loc[target]
        for days in (1, 2, 7):
            self.assertEqual(row[f"power_lag_{days}d"], self.power.loc[target - pd.Timedelta(days=days)])
        self.assertEqual(row.last_known_power, self.power.loc[origin - pd.Timedelta(minutes=15)])
        self.assertAlmostEqual(row.previous_day_mean, previous_day.mean())
        self.assertAlmostEqual(row.previous_day_median, previous_day.median())
        self.assertAlmostEqual(row.previous_day_std, previous_day.std(ddof=0))
        self.assertAlmostEqual(row.previous_week_mean, previous_week.mean())
        self.assertEqual(row.time_slot, 48)
        self.assertEqual(row.weekday, target.dayofweek)
        self.assertTrue(audit.drop(columns="forecast_origin").lt(audit.forecast_origin, axis=0).all().all())
        pd.testing.assert_series_equal(self.power, before)

    # Любое изменение текущего дня и будущего не меняет признаки уже выпущенного прогноза.
    def test_current_and_future_values_cannot_change_features_for_issued_days(self):
        features, audit = build_day_ahead_features(self.power)
        for offset in (7, 10, 15):
            origin = self.power.index[offset * 96]
            poisoned = self.power.copy()
            poisoned.loc[poisoned.index >= origin] += 1_000_000
            changed, changed_audit = build_day_ahead_features(poisoned)
            issued = audit.forecast_origin <= origin
            pd.testing.assert_frame_equal(features.loc[issued], changed.loc[issued], check_exact=True)
            pd.testing.assert_frame_equal(audit, changed_audit)

    # Факт завершившегося дня становится лагом следующего дня, не текущего.
    def test_completed_day_updates_following_days_features(self):
        features, _ = build_day_ahead_features(self.power)
        origin = self.power.index[8 * 96]
        changed_power = self.power.copy()
        changed_power.loc[origin:origin + pd.Timedelta(hours=23, minutes=45)] += 1000
        changed, _ = build_day_ahead_features(changed_power)
        next_day = slice(origin + pd.Timedelta(days=1), origin + pd.Timedelta(days=2, minutes=-15))
        np.testing.assert_allclose(changed.loc[next_day, "power_lag_1d"],
                                   features.loc[next_day, "power_lag_1d"] + 1000)
        np.testing.assert_allclose(changed.loc[next_day, "previous_day_mean"],
                                   features.loc[next_day, "previous_day_mean"] + 1000)

    # Слоты и лаги определяются шагом; первые семь дней остаются историей без подстановок.
    def test_warmup_and_nondefault_interval(self):
        index = pd.date_range("2014-01-01", periods=10 * 48, freq="30min")
        power = pd.Series(np.arange(len(index), dtype=float), index=index)
        features, _ = build_day_ahead_features(power, interval_minutes=30)
        self.assertTrue(features.power_lag_7d.iloc[:7 * 48].isna().all())
        self.assertTrue(np.isfinite(features.iloc[7 * 48:].to_numpy()).all())
        self.assertEqual(features.time_slot.iloc[-1], 47)
        self.assertEqual(features.power_lag_7d.iloc[-1], power.iloc[-1 - 7 * 48])

    # Не позволяет скрыть разрывы, неполный день или недопустимые исходные значения.
    def test_invalid_source_is_rejected(self):
        invalid = [self.power.iloc[1:], self.power.drop(self.power.index[100])]
        for value in (np.nan, np.inf, -1):
            power = self.power.copy()
            power.iloc[10] = value
            invalid.append(power)
        for power in invalid:
            with self.subTest(length=len(power)), self.assertRaises(ValueError):
                build_day_ahead_features(power)


class BoostingExperimentTests(unittest.TestCase):
    # Небольшой нелинейный ряд позволяет проверить настоящее обучение без дорогого расчёта.
    def setUp(self):
        index = pd.date_range("2013-12-20", periods=14 * 96, freq="15min", name="timestamp")
        power = pd.Series(100 + np.arange(len(index)) / 10 + 20 * np.sin(np.arange(len(index)) / 15),
                          index=index, name="total_power_kw")
        self.train, self.validation = power.iloc[:10 * 96], power.iloc[10 * 96:]
        self.parameters = {"max_iter": 3, "max_leaf_nodes": 7, "min_samples_leaf": 5,
                           "early_stopping": False, "random_state": 42, "loss": "absolute_error"}

    # Перехват фактического fit подтверждает: модель видит только train после прогрева.
    def test_fit_uses_only_training_targets_and_no_random_early_stopping(self):
        seen = []
        original_fit = HistGradientBoostingRegressor.fit

        def record_fit(model, features, target, *args, **kwargs):
            seen.append((features.copy(), target.copy()))
            return original_fit(model, features, target, *args, **kwargs)

        before_train, before_validation = self.train.copy(), self.validation.copy()
        with patch.object(HistGradientBoostingRegressor, "fit", record_fit):
            experiment = run_boosting_experiment(self.train, self.validation, self.parameters)
        self.assertEqual(len(seen), 1)
        x_train, y_train = seen[0]
        self.assertTrue((x_train.index < self.validation.index[0]).all())
        pd.testing.assert_series_equal(y_train, self.train.iloc[7 * 96:])
        self.assertFalse(experiment.model.early_stopping)
        self.assertEqual(experiment.model.n_iter_, 3)
        self.assertTrue(experiment.predictions.index.equals(self.validation.index))
        sources = experiment.predictions.filter(regex="^model_")
        self.assertTrue(sources.lt(experiment.predictions.forecast_origin, axis=0).all().all())
        pd.testing.assert_series_equal(self.train, before_train)
        pd.testing.assert_series_equal(self.validation, before_validation)

    # Независимые формулы проверяют метрики всех четырёх методов на всей одинаковой выборке.
    def test_comparison_and_daily_errors_use_the_same_validation(self):
        baselines = backtest_baselines(self.train, self.validation)
        before = baselines.copy(deep=True)
        experiment = run_boosting_experiment(self.train, self.validation, self.parameters, baselines)
        self.assertEqual(set(experiment.comparison.index), {*METHOD_LABELS, BOOSTING_METHOD})
        self.assertTrue(experiment.comparison.mae_kw.is_monotonic_increasing)
        self.assertEqual(len(experiment.daily_mae), 4)
        for method in experiment.comparison.index:
            error = experiment.predictions[method].to_numpy() - self.validation.to_numpy()
            metrics = experiment.comparison.loc[method]
            self.assertAlmostEqual(metrics.mae_kw, np.abs(error).mean())
            self.assertAlmostEqual(metrics.rmse_kw, np.sqrt(np.mean(error ** 2)))
            self.assertAlmostEqual(metrics.wape_pct, np.abs(error).sum() / self.validation.abs().sum() * 100)
        pd.testing.assert_frame_equal(baselines, before)

    # Случайная внутренняя валидация запрещена даже при изменении параметров извне.
    def test_random_early_stopping_is_rejected(self):
        for value in (True, "auto"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                run_boosting_experiment(self.train, self.validation,
                                        {**self.parameters, "early_stopping": value})

    # Исключение части валидации или подмена факта в базовых результатах останавливает опыт.
    def test_misaligned_or_changed_baseline_actuals_are_rejected(self):
        baselines = backtest_baselines(self.train, self.validation)
        changed = baselines.copy()
        changed.iloc[0, changed.columns.get_loc("actual_kw")] += 1
        for invalid in (baselines.iloc[1:], changed):
            with self.subTest(length=len(invalid)), self.assertRaises(ValueError):
                run_boosting_experiment(self.train, self.validation, self.parameters, invalid)

    # Семь дней достаточны для истории, но не заменяют отдельные обучающие цели.
    def test_training_without_post_warmup_targets_is_rejected(self):
        train = self.train.iloc[:7 * 96]
        validation = self.train.iloc[7 * 96:]
        with self.assertRaises(ValueError):
            run_boosting_experiment(train, validation, self.parameters)


if __name__ == "__main__":
    unittest.main()
