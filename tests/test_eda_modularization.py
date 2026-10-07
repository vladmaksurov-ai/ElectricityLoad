"""Регрессии функций, вынесенных из notebook, на небольших исходных рядах."""

import contextlib
import io
from pathlib import Path
import sys
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import cluster_visualizations as cluster_viz
from src import eda_reporting as reporting
from src import profile_clustering as clustering


class EdaModularizationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(10)
        slots = np.arange(96)
        values = np.vstack([
            1 + 0.6 * np.cos((slots - peak) * 2 * np.pi / 96)
            + rng.normal(0, 0.01, 96)
            for peak in [20, 20, 20, 60, 60, 60]
        ])
        index = pd.date_range("2024-01-01 00:15", periods=3 * 96, freq="15min")
        self.active = pd.DataFrame(
            np.tile(values.T, (3, 1)), index=index,
            columns=[f"client_{i}" for i in range(6)],
        )
        self.active["excluded"] = 0.0
        self.profiles = clustering.get_client_typical_profiles(self.active)
        self.scale, self.normalized = clustering.normalize_client_profiles(self.profiles)
        self.quality, self.excluded, self.features = clustering.prepare_clustering_features(
            self.active, self.profiles, self.normalized, min_complete_days=2,
        )

    def tearDown(self):
        plt.close("all")

    def cluster_result(self):
        _, labels = clustering.fit_profile_clusters(self.features, 2)
        energy = pd.DataFrame({
            "client_total_energy_kwh": [10.0] * 6 + [40.0],
            "mean_power_kw": [1.0] * 7,
        }, index=self.profiles.index)
        summary, members, profiles, q25, q75 = clustering.summarize_profile_clusters(
            self.features, labels, energy,
        )
        coverage = clustering.get_clustering_energy_coverage(
            energy, members, self.excluded, self.features,
        )
        return labels, summary, members, profiles, q25, q75, coverage

    def test_profile_alignment_and_normalization_preserve_input(self):
        before = self.active.copy(deep=True)
        aligned = self.active.copy()
        aligned.index -= pd.Timedelta(minutes=15)
        expected = aligned.groupby(aligned.index.hour * 4 + aligned.index.minute // 15).median().T
        np.testing.assert_allclose(self.profiles.to_numpy(), expected.to_numpy())
        self.assertEqual(self.profiles.columns[0], "00:00")
        self.assertEqual(self.profiles.columns[-1], "23:45")
        self.assertTrue(pd.isna(self.scale.loc["excluded"]))
        self.assertNotIn("excluded", self.normalized.index)
        np.testing.assert_allclose(self.normalized.mean(axis=1), 1)
        clustering.get_client_typical_profiles(self.active)
        pd.testing.assert_frame_equal(self.active, before)

    def test_feature_preparation_and_thresholds(self):
        self.assertEqual(self.features.index.tolist(), self.active.columns[:-1].tolist())
        self.assertEqual(self.excluded.index.tolist(), ["excluded"])
        self.assertTrue(self.quality["complete_days"].eq(3).all())
        table = clustering.get_history_threshold_sensitivity(self.quality, (0, 2, 4))
        self.assertEqual(table["included_clients"].tolist(), [6, 6, 0])

    def test_energy_coverage_and_result_table(self):
        labels, summary, members, _, _, _, coverage = self.cluster_result()
        all_energy, included, excluded, table = coverage
        self.assertEqual((all_energy, included, excluded), (100, 60, 40))
        self.assertEqual(table["energy_share_all_pct"].tolist(), [60, 40])
        result = clustering.validate_clustering_result(
            self.quality, members, self.profiles, self.features,
            labels.value_counts().sort_index(), summary, all_energy, included, excluded,
        )
        self.assertEqual(len(result), 7)
        self.assertTrue(pd.isna(result.loc["excluded", "cluster"]))
        self.assertEqual(str(result["cluster"].dtype), "Int64")

    def test_inconsistent_energy_still_fails_original_check(self):
        labels, summary, members, _, _, _, coverage = self.cluster_result()
        all_energy, included, excluded, _ = coverage
        with self.assertRaises(AssertionError):
            clustering.validate_clustering_result(
                self.quality, members, self.profiles, self.features,
                labels.value_counts().sort_index(), summary, all_energy + 1, included, excluded,
            )

    def test_all_extracted_plots_keep_features_unchanged(self):
        labels, summary, members, profiles, q25, q75, _ = self.cluster_result()
        before = self.features.copy(deep=True)
        representatives = clustering.cluster_representatives(members)
        comparison = pd.DataFrame({
            "silhouette_mean": [0.9], "silhouette_std": [0.01],
            "inertia": [10.0], "ari_min": [1.0],
        }, index=[2])
        calls = [
            (cluster_viz.plot_cluster_count_comparison, (comparison,)),
            (cluster_viz.plot_candidate_cluster_profiles, (self.features, {2: labels}, [2])),
            (cluster_viz.plot_cluster_profiles, (profiles, q25, q75, labels.value_counts())),
            (cluster_viz.plot_cluster_examples, (self.features, profiles, members, representatives)),
            (cluster_viz.plot_cluster_silhouettes, (members,)),
            (cluster_viz.plot_cluster_energy_summary, (summary,)),
        ]
        for plot, args in calls:
            fig, _ = plot(*args)
            self.assertGreater(len(fig.axes), 0)
        pd.testing.assert_frame_equal(self.features, before)

    def test_reporting_preserves_exclusion_and_window_text(self):
        text = reporting.format_profile_exclusions(self.excluded)[0]
        self.assertTrue(text.startswith("**excluded:** нулевой медианный профиль. "))
        self.assertIn("нулевых слотов типичного профиля: 96/96", text)
        self.assertEqual(reporting.format_profile_window_note(),
            "Ночь: 00:00–06:00, утро: 06:00–12:00, день: 12:00–18:00, "
            "вечер: 18:00–24:00; правая граница не включена. "
            "При нескольких равных максимумах/минимумах таблица показывает первый слот, "
            "а не уникальный момент пика. Средняя ширина IQR описывает разброс "
            "типичных профилей внутри группы, не нестабильность клиента по дням.")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            reporting.print_activity_overview({
                "clients_with_non_zero": 6, "total_clients": 7,
                "starts_after_data_start": 2, "ends_before_data_end": 1,
                "median_observed_duration": pd.Timedelta(days=3),
            })
        self.assertIn("Клиентов с ненулевыми показаниями: 6 из 7\n", output.getvalue())


if __name__ == "__main__":
    unittest.main()
