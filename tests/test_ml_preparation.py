"""Проверки качества, сохранения нагрузки и временного разделения для ML."""

from contextlib import redirect_stdout
import hashlib
import io
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_load import get_data, get_file_sha256
from src.data_validation import validate_ml_source_data, validate_regular_time_index
from src.ml_preparation import (
    get_split_summary,
    prepare_total_load,
    save_prepared_load,
    split_load_by_time,
)


class MlPreparationTests(unittest.TestCase):
    # Создаёт три полных дня с двумя клиентами и нулевым показанием внутри истории.
    def setUp(self):
        self.raw_data = pd.DataFrame(
            {"a": np.arange(3 * 96, dtype=float), "b": np.full(3 * 96, 2.0)},
            index=pd.date_range("2014-01-01 00:15", periods=3 * 96, freq="15min"),
        )

    # Проверяет суммы всех клиентов, единицы энергии, сдвиг времени и неизменность входа.
    def test_preparation_preserves_source_and_energy(self):
        before = self.raw_data.copy(deep=True)
        load = prepare_total_load(self.raw_data)
        self.assertEqual(load.index[0], pd.Timestamp("2014-01-01"))
        self.assertEqual(load.index[-1], pd.Timestamp("2014-01-03 23:45"))
        np.testing.assert_allclose(load.total_power_kw, self.raw_data.a + self.raw_data.b)
        np.testing.assert_allclose(load.total_energy_kwh, load.total_power_kw / 4)
        pd.testing.assert_frame_equal(self.raw_data, before)

    # Не допускает, чтобы сумма нагрузки скрывала NaN, бесконечности или отрицательные значения.
    def test_invalid_values_are_rejected(self):
        for value in (np.nan, np.inf, -1.0):
            with self.subTest(value=value):
                invalid = self.raw_data.copy()
                invalid.iloc[10, 0] = value
                with redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                    validate_ml_source_data(invalid)
                with self.assertRaises(ValueError):
                    prepare_total_load(invalid)

    # Проверяет сохранение нулевых значений и повторное использование диагностик EDA.
    def test_quality_accepts_zero_values(self):
        before = self.raw_data.copy(deep=True)
        with redirect_stdout(io.StringIO()):
            quality = validate_ml_source_data(self.raw_data)
        self.assertEqual(quality["zero_values"], 1)
        self.assertEqual(quality["nonfinite_values"], 0)
        pd.testing.assert_frame_equal(self.raw_data, before)

    # Отклоняет разрывы, дубликаты и неполные сутки перед прогнозированием.
    def test_broken_time_grid_is_rejected(self):
        load = prepare_total_load(self.raw_data)
        for invalid in (load.drop(load.index[5]), load.iloc[1:],
                        pd.concat([load, load.iloc[-1:]])):
            with self.subTest(rows=len(invalid)):
                with self.assertRaises(ValueError):
                    validate_regular_time_index(invalid, require_complete_days=True)

    # Проверяет отсутствие пересечений и потерь на границах train, validation и test.
    def test_temporal_splits_cover_input_without_overlap(self):
        load = prepare_total_load(self.raw_data)
        before = load.copy(deep=True)
        splits = split_load_by_time(load, "2014-01-02", "2014-01-03")
        self.assertEqual([len(frame) for frame in splits.values()], [96, 96, 96])
        self.assertEqual(splits["validation"].index[0], pd.Timestamp("2014-01-02"))
        self.assertEqual(splits["test"].index[0], pd.Timestamp("2014-01-03"))
        pd.testing.assert_frame_equal(pd.concat(splits.values()), load)
        self.assertEqual(get_split_summary(splits).days.tolist(), [1, 1, 1])
        splits["train"].iloc[0, 0] = -99
        pd.testing.assert_frame_equal(load, before)

    # Не разрешает обратные границы, пустые выборки и разделение внутри суток.
    def test_invalid_boundaries_are_rejected(self):
        load = prepare_total_load(self.raw_data)
        for validation, test in (("2014-01-03", "2014-01-02"),
                                 ("2014-01-02 12:00", "2014-01-03"),
                                 ("2014-01-01", "2014-01-03"),
                                 ("2014-01-02", "2014-01-04"),
                                 (pd.NaT, "2014-01-03")):
            with self.subTest(validation=validation, test=test):
                with self.assertRaises(ValueError):
                    split_load_by_time(load, validation, test)

    # Проверяет повторяемость экспорта и сохранение меток выборок при чтении CSV.
    def test_csv_round_trip_and_repeatability(self):
        splits = split_load_by_time(prepare_total_load(self.raw_data), "2014-01-02", "2014-01-03")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "ml" / "load.csv"
            prepared = save_prepared_load(splits, path)
            original_bytes = path.read_bytes()
            pd.testing.assert_frame_equal(save_prepared_load(splits, path), prepared)
            self.assertEqual(path.read_bytes(), original_bytes)
            self.assertEqual(prepared.groupby("split").size().to_dict(),
                             {"train": 96, "validation": 96, "test": 96})

    # Проверяет загрузку по явному пути с десятичной запятой и контрольную сумму файла.
    def test_loader_and_checksum(self):
        content = ';"client"\n"2014-01-01 00:15:00";1,5\n'
        with TemporaryDirectory() as directory:
            path = Path(directory) / "source.txt"
            path.write_text(content, encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                data = get_data(path)
            self.assertEqual(data.iloc[0, 0], 1.5)
            self.assertEqual(data.index.name, "timestamp")
            self.assertEqual(get_file_sha256(path), hashlib.sha256(content.encode()).hexdigest())


if __name__ == "__main__":
    unittest.main()
