"""Download the original UCI dataset without overwriting an existing data file."""

import shutil
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = (
    "https://archive.ics.uci.edu/static/public/321/"
    "electricityloaddiagrams20112014.zip"
)


def main():
    destination = PROJECT_ROOT / "data" / "LD2011_2014.txt"
    if destination.is_file() and destination.stat().st_size > 0:
        print(f"Data file already exists: {destination}")
        return

    cache = PROJECT_ROOT / ".cache"
    cache.mkdir(parents=True, exist_ok=True)
    archive_path = cache / "electricityloaddiagrams20112014.zip"
    if not archive_path.exists():
        partial_archive = archive_path.with_suffix(".zip.part")
        print(f"Downloading {SOURCE_URL}", flush=True)
        with urlopen(SOURCE_URL, timeout=60) as response:
            with partial_archive.open("wb") as output:
                shutil.copyfileobj(response, output)
        # Validate the ZIP directory before caching the completed download.
        with ZipFile(partial_archive) as archive:
            archive.getinfo(destination.name)
        partial_archive.replace(archive_path)

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_data = destination.with_suffix(".txt.part")
    try:
        with ZipFile(archive_path) as archive:
            info = archive.getinfo(destination.name)
            with archive.open(info) as source, partial_data.open("wb") as output:
                # Reading to EOF also verifies the ZIP member's CRC.
                shutil.copyfileobj(source, output)
            if info.file_size == 0 or partial_data.stat().st_size != info.file_size:
                raise ValueError("The extracted dataset is empty or incomplete")
        partial_data.replace(destination)
    finally:
        partial_data.unlink(missing_ok=True)
    print(f"Data ready: {destination} ({destination.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
