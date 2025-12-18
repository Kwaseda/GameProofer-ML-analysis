"""Disc golf data cleaning utilities with logging and validation."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd

from common.logging_utils import get_logger
from common.path_utils import ensure_directories, validate_file

logger = get_logger(__name__)


def clean_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace and remove newline characters from string columns."""

    df_clean = df.copy()
    string_columns = df_clean.select_dtypes(include=["object"]).columns

    for column in string_columns:
        df_clean[column] = df_clean[column].str.strip()
        df_clean[column] = df_clean[column].str.replace("\n", "", regex=False)
        df_clean[column] = df_clean[column].str.replace("\r", "", regex=False)

    logger.info("Cleaned %d string columns", len(string_columns))
    return df_clean


def validate_numeric_ranges(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Validate numeric columns against known disc golf ranges."""

    df_clean = df.copy()
    issues: List[str] = []

    expected_ranges = {
        "SPEED": (1, 15),
        "GLIDE": (1, 7),
        "TURN": (-5, 2),
        "FADE": (0, 6),
        "DIAMETER (cm)": (20, 25),
    }

    for column, (minimum, maximum) in expected_ranges.items():
        if column not in df_clean.columns:
            continue
        invalid_rows = df_clean[(df_clean[column] < minimum) | (df_clean[column] > maximum)]
        if not invalid_rows.empty:
            issues.append(f"{column} values outside [{minimum}, {maximum}] range: {len(invalid_rows)} records")

    if issues:
        for issue in issues:
            logger.warning(issue)
    else:
        logger.info("All numeric values within expected ranges")

    return df_clean, issues


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to snake_case for easier downstream processing."""

    column_mapping = {
        "MOLD": "mold",
        "DISC TYPE": "disc_type",
        "SPEED": "speed",
        "GLIDE": "glide",
        "TURN": "turn",
        "FADE": "fade",
        "STABILITY": "stability",
        "DIAMETER (cm)": "diameter_cm",
        "HEIGHT (cm)": "height_cm",
        "RIM DEPTH (cm)": "rim_depth_cm",
        "RIM WIDTH (cm)": "rim_width_cm",
        "INSIDE RIM DIAMETER (cm)": "inside_rim_diameter_cm",
        "RIM DEPTH / DIAMETER RATION (%)": "rim_ratio_pct",
        "RIM CONFIGURATION": "rim_config",
        "BEAD": "bead",
    }

    renamed = df.rename(columns=column_mapping)
    logger.info("Standardized %d column names", len(column_mapping))
    return renamed


def create_feature_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Add categorical features derived from numeric values."""

    df_clean = df.copy()

    if "speed" in df_clean.columns:
        df_clean["speed_category"] = pd.cut(
            df_clean["speed"],
            bins=[0, 3, 5, 9, 15],
            labels=["Putter", "Midrange", "Fairway", "Distance"],
            include_lowest=True,
        )

    if "stability" in df_clean.columns:
        df_clean["stability_category"] = pd.cut(
            df_clean["stability"],
            bins=[-10, -1, 0, 1, 10],
            labels=["Understable", "Neutral", "Stable", "Overstable"],
            include_lowest=True,
        )

    logger.info("Feature categories created")
    return df_clean


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load the raw dataset after validating the file path."""

    validated_path = validate_file(csv_path, allowed_suffixes=[".csv"])
    logger.info("Loading dataset from %s", validated_path)
    return pd.read_csv(validated_path)


def clean_disc_golf_data(input_path: Path, output_path: Path | None = None) -> pd.DataFrame:
    """Run the complete cleaning workflow and optionally persist results."""

    data = load_dataset(input_path)
    logger.info("Dataset shape prior to cleaning: %s", data.shape)

    cleaned = clean_string_columns(data)
    cleaned = standardize_column_names(cleaned)
    cleaned, issues = validate_numeric_ranges(cleaned)
    cleaned = create_feature_categories(cleaned)

    if output_path:
        ensure_directories([output_path.parent])
        cleaned.to_csv(output_path, index=False)
        logger.info("Cleaned dataset saved to %s", output_path)

    logger.info(
        "Cleaning complete. Rows: %d, Columns: %d (validation issues: %d)",
        len(cleaned),
        len(cleaned.columns),
        len(issues),
    )
    return cleaned


def main() -> None:
    """CLI entry point for cleaning the default dataset."""

    raw_data_path = Path("data/raw/disc-data.csv")
    cleaned_output = Path("data/processed/disc_golf_cleaned.csv")

    try:
        clean_disc_golf_data(raw_data_path, cleaned_output)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Data cleaning failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
