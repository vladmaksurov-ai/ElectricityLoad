from pathlib import Path
import pandas as pd


def get_data():
    candidate_paths = [
        Path("../data/LD2011_2014.txt"),
        Path("data/LD2011_2014.txt"),
        Path("startup-saas/ElectricityLoadDiagrams/data/LD2011_2014.txt"),
    ]

    data_path = next(
        (path.resolve() for path in candidate_paths if path.exists()),
        None,
    )

    if data_path is None:
        raise FileNotFoundError("Файл LD2011_2014.txt не найден.")

    data = pd.read_csv(
        data_path,
        sep=";",
        decimal=",",
        parse_dates=[0],
        index_col=0,
        encoding="utf-8",
        low_memory=False,
    )

    data.index.name = "timestamp"

    print(f"Датасет загружен из: {data_path}")
    return data