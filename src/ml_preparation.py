"""Подготовка совокупной нагрузки и временных выборок для прогноза на сутки."""

from pathlib import Path

import numpy as np
import pandas as pd

from .data_validation import validate_regular_time_index
from .eda_utils import get_total_load, shift_to_interval_start


# Суммирует всех клиентов функциями EDA и проверяет мощность, энергию и полноту суток.
def prepare_total_load(
    raw_data: pd.DataFrame,
    interval_minutes: int = 15,
) -> pd.DataFrame:
    validate_regular_time_index(raw_data, interval_minutes)
    if raw_data.shape[1] == 0 or not raw_data.columns.is_unique:
        raise ValueError("Нужны клиентские столбцы с уникальными именами.")
    values = raw_data.to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Исходная мощность должна быть конечной и неотрицательной.")
    interval_hours = interval_minutes / 60
    total_load = shift_to_interval_start(
        get_total_load(raw_data, interval_hours), interval_minutes,
    )
    total_load.index.name = "timestamp"
    validate_regular_time_index(total_load, interval_minutes, require_complete_days=True)
    np.testing.assert_allclose(total_load["total_power_kw"], values.sum(axis=1))
    np.testing.assert_allclose(
        total_load["total_energy_kwh"], total_load["total_power_kw"] * interval_hours,
    )
    return total_load


# Делит полные сутки на train, validation и test без перемешивания и потери интервалов.
def split_load_by_time(
    total_load: pd.DataFrame,
    validation_start: str | pd.Timestamp,
    test_start: str | pd.Timestamp,
    interval_minutes: int = 15,
) -> dict[str, pd.DataFrame]:
    validate_regular_time_index(total_load, interval_minutes, require_complete_days=True)
    validation_start, test_start = pd.Timestamp(validation_start), pd.Timestamp(test_start)
    if pd.isna(validation_start) or pd.isna(test_start):
        raise ValueError("Границы выборок не должны быть пустыми.")
    if validation_start >= test_start:
        raise ValueError("Начало валидации должно предшествовать началу теста.")
    if (validation_start != validation_start.normalize()
            or test_start != test_start.normalize()):
        raise ValueError("Границы выборок должны проходить по началу суток.")
    splits = {
        "train": total_load.loc[total_load.index < validation_start].copy(),
        "validation": total_load.loc[
            (total_load.index >= validation_start) & (total_load.index < test_start)
        ].copy(),
        "test": total_load.loc[total_load.index >= test_start].copy(),
    }
    for name, frame in splits.items():
        if frame.empty:
            raise ValueError(f"Пустая выборка: {name}")
        validate_regular_time_index(frame, interval_minutes, require_complete_days=True)
    pd.testing.assert_frame_equal(pd.concat(splits.values()), total_load)
    return splits


# Показывает границы, количество интервалов и полных суток каждой выборки.
def get_split_summary(
    splits: dict[str, pd.DataFrame],
    interval_minutes: int = 15,
) -> pd.DataFrame:
    rows = []
    for name, frame in splits.items():
        validate_regular_time_index(frame, interval_minutes, require_complete_days=True)
        rows.append({
            "split": name, "start": frame.index.min(), "end": frame.index.max(),
            "intervals": len(frame), "days": len(frame) // (1440 // interval_minutes),
        })
    return pd.DataFrame(rows).set_index("split")


# Сохраняет производный CSV с метками выборок и проверяет его повторным чтением.
def save_prepared_load(
    splits: dict[str, pd.DataFrame],
    output_path: str | Path,
) -> pd.DataFrame:
    prepared_data = pd.concat([frame.assign(split=name) for name, frame in splits.items()])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared_data.to_csv(output_path, index=True)
    restored = pd.read_csv(output_path, parse_dates=["timestamp"], index_col="timestamp")
    pd.testing.assert_frame_equal(restored, prepared_data, check_freq=False)
    return prepared_data
