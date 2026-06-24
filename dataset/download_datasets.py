"""
Dataset download helpers for the public experiments.

The module stores raw data files in a local data directory and keeps the
original public filenames when possible.
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable


CONCRETE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
SUPERCONDUCTIVITY_URLS = (
    "https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00464/superconductivty+data.zip",
)


class DatasetDownloadError(RuntimeError):
    """
    Error raised when a public dataset cannot be downloaded.
    """


def _download_file(url: str, output_path: Path, timeout: int = 120) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response, tmp_path.open("wb") as out_file:
            shutil.copyfileobj(response, out_file)
        tmp_path.replace(output_path)
    except Exception as exc:  # pragma: no cover - network behavior depends on the runtime
        if tmp_path.exists():
            tmp_path.unlink()
        raise DatasetDownloadError(f"Download failed for {url}") from exc

    return output_path


def _first_success(urls: Iterable[str], output_path: Path) -> Path:
    last_error: Exception | None = None
    for url in urls:
        try:
            return _download_file(url, output_path)
        except Exception as exc:  # pragma: no cover - network behavior depends on the runtime
            last_error = exc
    raise DatasetDownloadError(f"All download sources failed for {output_path.name}") from last_error


def download_concrete(data_dir: str | Path = "data/raw", overwrite: bool = False) -> Path:
    """
    Download the concrete compressive strength dataset.
    """
    raw_dir = Path(data_dir)
    output_path = raw_dir / "Concrete_Data.xls"
    if output_path.exists() and not overwrite:
        return output_path
    return _download_file(CONCRETE_URL, output_path)


def download_superconductivity(data_dir: str | Path = "data/raw", overwrite: bool = False) -> Path:
    """
    Download and extract the superconductivity dataset archive.
    """
    raw_dir = Path(data_dir)
    archive_path = raw_dir / "superconductivity_data.zip"
    extracted_dir = raw_dir / "superconductivity"
    train_path = extracted_dir / "train.csv"

    if train_path.exists() and not overwrite:
        return train_path

    raw_dir.mkdir(parents=True, exist_ok=True)
    _first_success(SUPERCONDUCTIVITY_URLS, archive_path)

    if extracted_dir.exists() and overwrite:
        shutil.rmtree(extracted_dir)
    extracted_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        zip_ref.extractall(extracted_dir)

    candidates = sorted(extracted_dir.rglob("train.csv"))
    if not candidates:
        raise DatasetDownloadError("The superconductivity archive did not contain train.csv.")

    if candidates[0] != train_path:
        shutil.copy2(candidates[0], train_path)

    return train_path


def download_all(data_dir: str | Path = "data/raw", overwrite: bool = False) -> dict[str, Path]:
    """
    Download all tabular datasets used by the experiments.
    """
    return {
        "concrete": download_concrete(data_dir=data_dir, overwrite=overwrite),
        "superconductivity": download_superconductivity(data_dir=data_dir, overwrite=overwrite),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download public tabular datasets for the experiments.")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    paths = download_all(data_dir=args.data_dir, overwrite=args.overwrite)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
