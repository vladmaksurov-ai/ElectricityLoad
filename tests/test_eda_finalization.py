"""Проверки новых диагностик и представления; исходные значения не исправляются."""

import sys
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_validation import get_value_quality
from src.eda_utils import get_clock_change_audit
from src.visualizations import (
    configure_eda_plot_style,
    plot_clock_change_examples,
    plot_longest_zero_periods,
    plot_spike_case_contexts,
    plot_yearly_normalized_daily_profiles,
    plot_zero_period_context,
)


class EdaFinalizationTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_value_quality_keeps_input_unchanged(self):
        data = pd.DataFrame({"zero": [0.0, 0.0, 0.0], "other": [-1.0, np.inf, np.nan]})
        before = data.copy(deep=True)
        quality = get_value_quality(data)
        self.assertEqual(quality["negative_values"], 1)
        self.assertEqual(quality["nonfinite_values"], 2)
        self.assertEqual(quality["all_zero_clients"], 1)
        pd.testing.assert_frame_equal(data, before)

    def test_clock_audit_uses_only_neighbouring_dates(self):
        index = pd.date_range("2014-03-29", periods=3 * 96, freq="15min")
        data = pd.DataFrame(10.0, index=index, columns=["client"])
        selected = pd.date_range("2014-03-30 01:00", periods=4, freq="15min")
        data.loc[selected] = 0
        before = data.copy(deep=True)
        row = get_clock_change_audit(data).iloc[0]
        self.assertEqual(row["reference_mean_kw"], 10)
        self.assertEqual(row["mean_power_kw"], 0)
        self.assertEqual(row["window_to_reference"], 0)
        self.assertEqual(row["all_zero_intervals"], 4)
        pd.testing.assert_frame_equal(data, before)

    def test_empty_clock_windows_have_defined_schema(self):
        data = pd.DataFrame({"client": [1.0]}, index=pd.to_datetime(["2014-01-01"]))
        audit = get_clock_change_audit(data)
        self.assertTrue(audit.empty)
        self.assertIn("window_to_reference", audit)

    def test_style_and_yearly_normalization_label(self):
        configure_eda_plot_style()
        self.assertEqual(plt.rcParams["axes.facecolor"], "white")
        profiles = pd.DataFrame({2011: np.ones(96), 2012: np.ones(96)}, index=range(96))
        _, ax = plot_yearly_normalized_daily_profiles(profiles)
        self.assertIn("медианного профиля", ax.get_ylabel())
        self.assertEqual([line.get_linestyle() for line in ax.lines[:2]], ["-", "--"])

    def test_zero_ranking_uses_positions_not_log_bar_lengths(self):
        summary = pd.DataFrame({"longest_zero_hours": [24.0, 2640.0]}, index=["a", "b"])
        _, ax = plot_longest_zero_periods(summary)
        self.assertEqual(ax.get_xscale(), "log")
        self.assertEqual(len(ax.patches), 0)
        np.testing.assert_allclose(ax.collections[0].get_offsets()[:, 0], [24, 2640])

    def test_local_plots_keep_input_and_expected_scales(self):
        configure_eda_plot_style()
        index = pd.date_range("2014-03-29", periods=3 * 96, freq="15min")
        data = pd.DataFrame(10.0, index=index, columns=["client"])
        timestamp = pd.Timestamp("2014-03-30 12:00")
        events = pd.DataFrame({"current_kw": [10.0]}, index=pd.MultiIndex.from_tuples(
            [("client", timestamp)], names=["client", "timestamp"]))
        before = data.copy(deep=True)
        _, axes = plot_spike_case_contexts(data, events)
        self.assertEqual(axes[0].get_ylabel(), "Мощность, кВт")
        plot_clock_change_examples(data, dates=("2014-03-30",))
        period = pd.Series({"client": "client", "start": timestamp,
                            "last_zero": timestamp + pd.Timedelta(minutes=45)})
        _, ax = plot_zero_period_context(data, period)
        self.assertEqual(ax.get_xlabel(), "Дата по началу интервала")
        pd.testing.assert_frame_equal(data, before)


if __name__ == "__main__":
    unittest.main()
