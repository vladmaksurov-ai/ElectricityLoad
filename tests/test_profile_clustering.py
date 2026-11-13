"""Проверки подготовки профилей и кластеризации на небольших данных."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.profile_clustering import (
    audit_profile_quality,
    cluster_representatives,
    compare_cluster_counts,
    fit_profile_clusters,
    get_client_typical_profiles,
    get_history_threshold_sensitivity,
    summarize_profile_clusters,
    validate_features,
)


class ProfileClusteringTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(10)
        slots = np.arange(96)
        values = np.vstack([
            1 + 0.6 * np.cos((slots - peak) * 2 * np.pi / 96)
            + rng.normal(0, 0.01, 96)
            for peak in [20, 20, 20, 60, 60, 60]
        ])
        self.features = pd.DataFrame(
            values / values.mean(axis=1, keepdims=True),
            index=[f"client_{i}" for i in range(6)],
            columns=[f"{i // 4:02d}:{i % 4 * 15:02d}" for i in range(96)],
        )

    def test_exclusion_reasons_and_flat_profile(self):
        index = pd.date_range("2024-01-01 00:15", periods=32 * 96, freq="15min")
        active = pd.DataFrame(1.0, index=index, columns=["flat", "rare", "short", "missing"])
        active.loc[index[10 * 96:], "rare"] = 0
        active.loc[index[5 * 96:], "short"] = np.nan
        active.loc[index[::96], "missing"] = np.nan
        aligned = active.copy()
        aligned.index -= pd.Timedelta(minutes=15)
        profiles = aligned.groupby(aligned.index.hour * 4 + aligned.index.minute // 15).median().T
        audit = audit_profile_quality(active, profiles, min_complete_days=30)
        self.assertTrue(audit.loc["flat", "included"])
        self.assertIn("нулевой", audit.loc["rare", "exclusion_reason"])
        self.assertGreater(audit.loc["rare", "nonzero_intervals"], 0)
        self.assertIn("дней", audit.loc["short", "exclusion_reason"])
        self.assertIn("слоты", audit.loc["missing", "exclusion_reason"])
        self.assertEqual(audit.loc["flat", "complete_days"], 32)

    def test_feature_validation(self):
        validate_features(self.features)
        invalid = self.features.copy()
        invalid.iloc[0, 0] = np.inf
        with self.assertRaises(ValueError):
            validate_features(invalid)
        with self.assertRaises(ValueError):
            validate_features(self.features * 2)

    # Аудит и чувствительность должны одинаково учитывать каждую причину исключения.
    def test_threshold_sensitivity_matches_audit_for_all_exclusion_reasons(self):
        index = pd.date_range("2024-01-01 00:15", periods=3 * 96, freq="15min")
        active = pd.DataFrame(1.0, index=index, columns=[
            "good", "short", "missing", "infinite", "negative", "zero", "empty",
        ])
        active.loc[index[96:], "short"] = np.nan
        active.loc[index[::96], "missing"] = np.nan
        active.loc[index[1::96], "infinite"] = np.inf
        active.loc[index[2::96], "negative"] = -1.0
        active["zero"] = 0.0
        active["empty"] = np.nan
        profiles = get_client_typical_profiles(active)
        quality = audit_profile_quality(active, profiles, min_complete_days=2)
        before = quality.copy(deep=True)
        thresholds = (0, 2, 4)
        sensitivity = get_history_threshold_sensitivity(quality, thresholds)
        self.assertEqual(sensitivity.included_clients.tolist(), [2, 1, 0])
        for threshold, count in zip(thresholds, sensitivity.included_clients):
            with self.subTest(threshold=threshold):
                audit = audit_profile_quality(active, profiles, min_complete_days=threshold)
                self.assertEqual(count, int(audit.included.sum()))
                pd.testing.assert_series_equal(audit.included, audit.exclusion_reason.eq(""),
                                               check_names=False)
        self.assertIn("бесконечные значения", quality.loc["infinite", "exclusion_reason"])
        self.assertNotIn("бесконечные значения", quality.loc["missing", "exclusion_reason"])
        self.assertIn("отрицательный профиль", quality.loc["negative", "exclusion_reason"])
        self.assertIn("нулевой медианный профиль", quality.loc["zero", "exclusion_reason"])
        self.assertIn("нет наблюдений", quality.loc["empty", "exclusion_reason"])
        pd.testing.assert_frame_equal(quality, before)

    # Нельзя включать клиента без исходных наблюдений только из-за заполненного профиля.
    def test_sensitivity_keeps_clients_without_observations_excluded(self):
        index = pd.date_range("2024-01-01 00:15", periods=96, freq="15min")
        active = pd.DataFrame(np.nan, index=index, columns=["empty"])
        profiles = pd.DataFrame(1.0, index=active.columns, columns=range(96))
        quality = audit_profile_quality(active, profiles, min_complete_days=0)
        self.assertFalse(quality.loc["empty", "included"])
        self.assertEqual(get_history_threshold_sensitivity(quality, (0,)).included_clients.iloc[0], 0)

    # Оба пути отвергают отрицательный порог вместо разных правил включения.
    def test_audit_and_sensitivity_reject_negative_history_threshold(self):
        index = pd.date_range("2024-01-01 00:15", periods=96, freq="15min")
        active = pd.DataFrame(1.0, index=index, columns=["good"])
        profiles = get_client_typical_profiles(active)
        quality = audit_profile_quality(active, profiles, min_complete_days=0)
        with self.assertRaises(ValueError):
            audit_profile_quality(active, profiles, min_complete_days=-1)
        with self.assertRaises(ValueError):
            get_history_threshold_sensitivity(quality, (-1,))

    def test_constant_profiles_have_no_valid_k(self):
        constant = pd.DataFrame(1.0, index=range(5), columns=range(96))
        with self.assertRaises(ValueError):
            compare_cluster_counts(constant)

    def test_comparison_and_reproducibility(self):
        table, _, assignments = compare_cluster_counts(self.features, [1, 2, 3, 6])
        self.assertEqual(table.index.tolist(), [2, 3])
        self.assertGreater(table.loc[2, "silhouette"], 0.9)
        self.assertAlmostEqual(table.loc[2, "ari_min"], 1)
        _, labels = fit_profile_clusters(self.features, 2)
        pd.testing.assert_series_equal(assignments[2], labels)
        self.assertTrue(labels.index.equals(self.features.index))

    def test_energy_denominators_and_representatives(self):
        _, labels = fit_profile_clusters(self.features, 2)
        energy = pd.DataFrame({
            "client_total_energy_kwh": [10.0] * 6 + [40.0],
            "mean_power_kw": [1.0] * 7,
        }, index=[*self.features.index, "excluded"])
        summary, members, medians, q25, q75 = summarize_profile_clusters(
            self.features, labels, energy,
        )
        self.assertEqual(summary.clients.sum(), 6)
        self.assertAlmostEqual(summary.energy_share_all_pct.sum(), 60)
        self.assertAlmostEqual(summary.energy_share_included_pct.sum(), 100)
        self.assertTrue((q25 <= medians).all().all())
        self.assertTrue((medians <= q75).all().all())
        representatives = cluster_representatives(members, per_cluster=2)
        self.assertEqual(representatives.cluster.nunique(), 2)
        self.assertTrue(representatives.index.is_unique)
        self.assertTrue(representatives.index.isin(self.features.index).all())

    def test_zero_energy_is_unknown_share(self):
        _, labels = fit_profile_clusters(self.features, 2)
        energy = pd.DataFrame({
            "client_total_energy_kwh": 0.0, "mean_power_kw": 0.0,
        }, index=self.features.index)
        summary, *_ = summarize_profile_clusters(self.features, labels, energy)
        self.assertTrue(summary.energy_share_all_pct.isna().all())

    def test_mismatched_label_order_is_rejected(self):
        _, labels = fit_profile_clusters(self.features, 2)
        with self.assertRaises(ValueError):
            summarize_profile_clusters(self.features, labels.iloc[::-1], pd.DataFrame())


if __name__ == "__main__":
    unittest.main()
