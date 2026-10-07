"""Проверки структуры и качества исходного датасета энергопотребления."""

from typing import Any

import numpy as np
import pandas as pd


# Собирает основные характеристики таблицы без вывода всех клиентских столбцов.
def get_dataset_overview(
    data: pd.DataFrame,
    preview_clients: int = 10,
) -> dict[str, Any]:
    """Печатает и возвращает структуру исходного датасета."""
    rows, columns = data.shape

    overview = {
        "rows": rows,
        "columns": columns,
        "start_date": data.index.min(),
        "end_date": data.index.max(),
        "first_clients": data.columns[:preview_clients].tolist(),
        "type_counts": data.dtypes.value_counts(),
    }

    print("ОБЩАЯ ИНФОРМАЦИЯ О ДАТАСЕТЕ")
    print("-" * 45)
    print(f"Строки (временные отметки): {overview['rows']:,}")
    print(f"Столбцы (клиенты / точки учёта): {overview['columns']}")
    print(f"Начало наблюдений: {overview['start_date']}")
    print(f"Конец наблюдений:  {overview['end_date']}")
    print(f"Первые {preview_clients} клиентов: {overview['first_clients']}")
    print()
    print("Количество столбцов по типам данных:")
    print(overview["type_counts"])

    return overview


# Проверяет временную сетку, дубликаты и явные пропуски во всём датасете.
def get_time_index_quality(
    data: pd.DataFrame,
    common_intervals_count: int = 5,
) -> dict[str, Any]:
    """Печатает и возвращает показатели целостности временного индекса."""
    # Разности соседних меток позволяют одновременно увидеть основной шаг и разрывы.
    common_intervals = (
        data.index.to_series().diff().value_counts().head(common_intervals_count)
    )

    quality = {
        "is_sorted": data.index.is_monotonic_increasing,
        "duplicate_dates": int(data.index.duplicated().sum()),
        "missing_values": int(data.isna().sum().sum()),
        "common_intervals": common_intervals,
    }

    print("ПРОВЕРКА ВРЕМЕННОГО ИНДЕКСА И ДАННЫХ")
    print("-" * 50)
    print(f"Индекс расположен по возрастанию: {quality['is_sorted']}")
    print(f"Повторяющиеся временные отметки: {quality['duplicate_dates']:,}")
    print(f"Явные пропуски NaN: {quality['missing_values']:,}")
    print()
    print("Наиболее частые интервалы между наблюдениями:")

    for interval, count in quality["common_intervals"].items():
        minutes = interval.total_seconds() / 60
        print(f"{minutes:g} минут — {count:,} раз")

    return quality


def get_value_quality(data: pd.DataFrame) -> pd.Series:
    """Проверяет числовые значения всего файла, а не только медианные профили."""
    values = data.to_numpy()
    return pd.Series(
        {
            "negative_values": int((values < 0).sum()),
            "nonfinite_values": int((~np.isfinite(values)).sum()),
            "duplicate_client_names": int(data.columns.duplicated().sum()),
            "all_zero_clients": int(data.eq(0).all(axis=0).sum()),
            "all_zero_timestamps": int(data.eq(0).all(axis=1).sum()),
        },
        name="count",
    )


# Рассчитывает долю нулевых значений отдельно для каждого клиента.
def calculate_zero_share(
    data: pd.DataFrame,
    preview_clients: int = 10,
) -> pd.Series:
    """Печатает первые доли нулей и возвращает результат для всех клиентов."""
    zero_share = data.eq(0).mean().rename("zero_share")

    print("ДОЛЯ НУЛЕВЫХ ЗНАЧЕНИЙ В ИСХОДНОМ ФАЙЛЕ")
    print("-" * 47)

    # Процентный формат применяется только к компактному выводу, не меняя значения Series.
    print(
        zero_share.head(preview_clients).to_string(
            float_format=lambda value: f"{value:.2%}",
        )
    )

    return zero_share
