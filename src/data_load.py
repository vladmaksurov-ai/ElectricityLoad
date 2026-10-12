from pathlib import Path
import hashlib
import pandas as pd


# Загружает исходный датасет из указанного файла или стандартных папок проекта.
def get_data(data_path: str | Path | None = None):
    candidate_paths = [
        Path("../data/LD2011_2014.txt"),
        Path("data/LD2011_2014.txt"),
        Path("startup-saas/ElectricityLoadDiagrams/data/LD2011_2014.txt"),
    ]

    if data_path is None:
        data_path = next(
            (path.resolve() for path in candidate_paths if path.is_file()),
            None,
        )
    else:
        data_path = Path(data_path).resolve()

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


# Фиксирует контрольную сумму использованного файла без загрузки его целиком в память.
def get_file_sha256(data_path: str | Path) -> str:
    with Path(data_path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()
