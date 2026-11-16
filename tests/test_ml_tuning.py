"""Проверки временного CV, воспроизводимости поиска и независимой внешней оценки."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ml_boosting import BOOSTING_METHOD, run_boosting_experiment
from src.ml_tuning import (
    TUNED_METHOD, aggregate_cv_scores, evaluate_tuned_boosting, make_boosting_candidates,
    make_temporal_folds, search_boosting_parameters,
)


class TemporalTuningTests(unittest.TestCase):
    # Малый ряд используется только для проверки алгоритма, не для результатов эксперимента.
    def setUp(self):
        index = pd.date_range("2013-01-01", periods=36 * 96, freq="15min", name="timestamp")
        values = 200 + np.arange(len(index)) / 20 + 15 * np.sin(np.arange(len(index)) / 13)
        power = pd.Series(values, index=index, name="total_power_kw")
        self.train, self.validation = power.iloc[:32 * 96], power.iloc[32 * 96:]
        self.windows = (("2013-01-12", "2013-01-20"), ("2013-01-20", "2013-02-02"))
        self.parameters = {"loss": "absolute_error", "max_iter": 3, "max_leaf_nodes": 7,
                           "min_samples_leaf": 5, "early_stopping": False, "random_state": 42}
        self.space = {"max_leaf_nodes": [3, 7]}

    # Исходная модель включается один раз; выбор параметров повторяем и не меняет настройки.
    def test_candidates_are_unique_reproducible_and_include_original(self):
        before = dict(self.parameters)
        space = {"max_leaf_nodes": [3, 7, 15], "min_samples_leaf": [5, 10]}
        first = make_boosting_candidates(self.parameters, space, 5, 42)
        second = make_boosting_candidates(self.parameters, space, 5, 42)
        self.assertEqual(first, second)
        self.assertEqual(first["original"], self.parameters)
        self.assertEqual(len(first), 5)
        self.assertEqual(len({tuple(sorted(value.items())) for value in first.values()}), 5)
        self.assertEqual(self.parameters, before)
        with self.assertRaises(ValueError):
            make_boosting_candidates(self.parameters, {"early_stopping": [True]}, 2)
        with self.assertRaises(ValueError):
            make_boosting_candidates({**self.parameters, "early_stopping": True}, self.space, 2)

    # Блоки должны состоять из полных дней и заканчиваться внутри предоставленного train.
    def test_fold_boundaries_reject_future_overlap_and_partial_days(self):
        folds = make_temporal_folds(self.train, self.windows)
        self.assertTrue((folds.train_end < folds.validation_start).all())
        self.assertEqual(folds.intervals.tolist(), [8 * 96, 13 * 96])
        invalid = [(("2013-01-12", "2013-02-03"),),
                   (("2013-01-12 00:15", "2013-01-20"),),
                   (("2013-01-08", "2013-01-10"),),
                   (("2013-01-12", "2013-01-21"), ("2013-01-20", "2013-01-25")), ()]
        for windows in invalid:
            with self.subTest(windows=windows), self.assertRaises(ValueError):
                make_temporal_folds(self.train, windows)

    # Фактические вызовы fit не получают цели текущего или следующих проверяемых блоков.
    def test_every_fit_ends_before_its_validation_block(self):
        seen = []
        original_fit = HistGradientBoostingRegressor.fit

        def record_fit(model, features, target, *args, **kwargs):
            seen.append((features.index.copy(), target.copy(), model.early_stopping))
            return original_fit(model, features, target, *args, **kwargs)

        before = self.train.copy()
        with patch.object(HistGradientBoostingRegressor, "fit", record_fit):
            search = search_boosting_parameters(self.train, self.parameters, self.space, self.windows, 2)
        self.assertEqual(len(seen), 4)
        for position, (index, targets, stopping) in enumerate(seen):
            boundary = search.folds.iloc[position % 2].validation_start
            self.assertTrue((index < boundary).all())
            self.assertFalse(stopping)
            pd.testing.assert_series_equal(targets, self.train.loc[index])
        pd.testing.assert_series_equal(self.train, before)

    # Проверяет все общие метрики непосредственно по сохранённым внеобучающим прогнозам.
    def test_search_scores_match_independent_oof_metrics_and_best_choice(self):
        search = search_boosting_parameters(self.train, self.parameters, self.space, self.windows, 2)
        actual = search.oof_predictions.actual_kw.to_numpy()
        self.assertEqual(len(actual), 21 * 96)
        for candidate in search.summary.index:
            errors = search.oof_predictions[candidate].to_numpy() - actual
            score = search.summary.loc[candidate]
            self.assertAlmostEqual(score.cv_mae_kw, np.abs(errors).mean())
            self.assertAlmostEqual(score.cv_rmse_kw, np.sqrt(np.mean(errors ** 2)))
            self.assertAlmostEqual(score.cv_wape_pct, np.abs(errors).sum() / np.abs(actual).sum() * 100)
        self.assertEqual(search.best_candidate_id, search.summary.cv_mae_kw.idxmin())

    # Неравные блоки объединяются по интервалам; процент и корень не усредняются.
    def test_pooled_metrics_are_not_unweighted_fold_averages(self):
        scores = pd.DataFrame([
            {"candidate": "x", "intervals": 2, "mae_kw": 1.0, "rmse_kw": 1.0, "absolute_actual_sum": 2},
            {"candidate": "x", "intervals": 3, "mae_kw": 0.0, "rmse_kw": 0.0, "absolute_actual_sum": 300},
        ])
        row = aggregate_cv_scores(scores).loc["x"]
        self.assertAlmostEqual(row.cv_mae_kw, 2 / 5)
        self.assertAlmostEqual(row.cv_rmse_kw, np.sqrt(2 / 5))
        self.assertAlmostEqual(row.cv_wape_pct, 2 / 302 * 100)
        scores.absolute_actual_sum = 0
        self.assertTrue(np.isnan(aggregate_cv_scores(scores).loc["x", "cv_wape_pct"]))

    # Внешняя оценка обучает выбранные по CV параметры и сохраняет исходные прогнозы.
    def test_external_evaluation_keeps_selection_and_reference_unchanged(self):
        search = search_boosting_parameters(self.train, self.parameters, self.space, self.windows, 2)
        reference = run_boosting_experiment(self.train, self.validation, self.parameters)
        predictions_before = reference.predictions.copy(deep=True)
        parameters_before = dict(search.best_parameters)
        result = evaluate_tuned_boosting(self.train, self.validation, search, reference)
        self.assertTrue(result.predictions.index.equals(self.validation.index))
        for name, value in parameters_before.items():
            self.assertEqual(result.selected.model.get_params()[name], value)
        np.testing.assert_array_equal(result.predictions[BOOSTING_METHOD], reference.predictions[BOOSTING_METHOD])
        pd.testing.assert_frame_equal(reference.predictions, predictions_before)
        self.assertEqual(search.best_parameters, parameters_before)
        self.assertIn(TUNED_METHOD, result.comparison.index)

    # Результат поиска нельзя незаметно применить к изменённому обучающему ряду.
    def test_search_cannot_be_reused_with_changed_training_values(self):
        search = search_boosting_parameters(self.train, self.parameters, self.space, self.windows, 2)
        reference = run_boosting_experiment(self.train, self.validation, self.parameters)
        changed = self.train.copy()
        changed.iloc[0] += 1
        with self.assertRaises(ValueError):
            evaluate_tuned_boosting(changed, self.validation, search, reference)

    # Ход поиска и выбранные параметры сохраняются как читаемые проверяемые артефакты.
    def test_search_progress_and_artifacts_are_complete(self):
        with TemporaryDirectory() as directory:
            search = search_boosting_parameters(self.train, self.parameters, self.space,
                                                 self.windows, 2, output_dir=directory)
            progress = json.loads((Path(directory) / "progress.json").read_text())
            self.assertEqual(progress["phase"], "complete")
            self.assertEqual(progress["completed_fits"], 4)
            metadata = json.loads((Path(directory) / "selected_parameters.json").read_text())
            self.assertEqual(metadata["parameters"], search.best_parameters)
            self.assertEqual(len(pd.read_csv(Path(directory) / "cv_fold_scores.csv")), 4)


if __name__ == "__main__":
    unittest.main()
