"""Download the Disc Golf dataset from Kaggle with validation and logging."""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Iterable, List

if __package__ in (None, ""):
    # Allow running as `python src/data_loading/download_kaggle_data.py`
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

from common.logging_utils import get_logger
from common.path_utils import ensure_directories

DATA_DIRECTORIES: Iterable[Path] = (
    Path("data/raw"),
    Path("data/processed"),
    Path("data/features"),
)
DATASET_NAME = "jakestrasler/disc-golf-disc-flight-numbers-and-dimensions"
RAW_DIR = Path("data/raw")

logger = get_logger(__name__)


def setup_directories() -> None:
    """Ensure all required data directories exist."""
    ensure_directories(DATA_DIRECTORIES)
    for directory in DATA_DIRECTORIES:
        logger.info("Ensured directory %s", directory)


def check_kaggle_credentials() -> Path:
    """Return the Kaggle credentials path if available, otherwise raise."""
    kaggle_json_path = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json_path.exists():
        raise FileNotFoundError(
            "Kaggle API credentials were not found. Please place kaggle.json in ~/.kaggle/."
        )

    logger.info("Kaggle credentials located at %s", kaggle_json_path)
    return kaggle_json_path


def download_dataset(dataset_name: str = DATASET_NAME, output_dir: Path = RAW_DIR) -> Path:
    """Download the dataset via Kaggle CLI and return the downloaded archive path."""

    logger.info("Downloading dataset %s into %s", dataset_name, output_dir)
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_name, "-p", str(output_dir)],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        logger.error("Kaggle download failed: %s", exc.stderr.strip())
        raise RuntimeError("Dataset download failed. See logs for details.") from exc
    except FileNotFoundError as exc:
        raise RuntimeError("Kaggle CLI is not installed. Run `pip install kaggle`." ) from exc

    archives = sorted(output_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not archives:
        raise FileNotFoundError("Download reported success but no ZIP archive was found.")

    logger.info("Download complete: %s", archives[0].name)
    return archives[0]


def extract_dataset(archive_path: Path) -> List[Path]:
    """Extract the provided archive and return the list of extracted CSV files."""

    logger.info("Extracting archive %s", archive_path.name)
    extracted_files: List[Path] = []
    try:
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(RAW_DIR)
            extracted_files = [RAW_DIR / member for member in zip_ref.namelist()]
    except zipfile.BadZipFile as exc:
        raise RuntimeError("Failed to extract dataset: corrupted archive.") from exc

    archive_path.unlink(missing_ok=True)
    logger.info("Extraction complete; removed archive %s", archive_path.name)
    return extracted_files


def validate_data(data_dir: Path = RAW_DIR) -> List[Path]:
    """Ensure at least one non-empty CSV exists and return their paths."""

    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir} after extraction.")

    valid_csvs: List[Path] = []
    for csv_path in csv_files:
        size_bytes = csv_path.stat().st_size
        if size_bytes == 0:
            raise ValueError(f"File {csv_path} is empty.")
        valid_csvs.append(csv_path)
        logger.info("Validated %s (%.2f MB)", csv_path.name, size_bytes / (1024 * 1024))

    return valid_csvs


def main() -> None:
    """Run the download workflow with structured logging and error handling."""

    logger.info("Starting Disc Golf dataset download workflow")
    try:
        setup_directories()
        check_kaggle_credentials()
        archive_path = download_dataset()
        extract_dataset(archive_path)
        csv_files = validate_data()
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Download workflow failed: %s", exc)
        raise SystemExit(1) from exc

    logger.info("Dataset ready for analysis: %d CSV file(s) available", len(csv_files))
    logger.info("Next step: run notebooks/01_exploratory_analysis.ipynb")


if __name__ == "__main__":
    main()
