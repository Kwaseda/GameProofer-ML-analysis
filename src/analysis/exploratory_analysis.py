"""Exploratory data analysis utilities for the disc golf dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    import sys

    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

from common.logging_utils import get_logger

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
pd.set_option("display.max_columns", None)
pd.set_option("display.precision", 3)

logger = get_logger(__name__)


def load_and_clean_data(file_path: Path) -> pd.DataFrame:
    """Load CSV data and clean string columns."""

    logger.info("Loading dataset from %s", file_path)
    df = pd.read_csv(file_path)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip().str.replace("\n", "", regex=False).str.replace("\r", "", regex=False)

    logger.info("Dataset loaded: %d rows, %d columns", len(df), len(df.columns))
    return df


def initial_inspection(df: pd.DataFrame) -> None:
    """
    Perform initial data inspection.
    
    Checks:
    - Dataset shape
    - Column names and types
    - First few rows
    - Basic statistics
    """
    print("\n" + "=" * 70)
    print("[Step 2/8] Initial Data Inspection")
    print("=" * 70)
    
    # Dataset shape
    print(f"\nDataset Shape: {df.shape}")
    print(f"Total Records: {len(df):,}")
    print(f"Total Features: {len(df.columns)}")
    
    # Column names
    print("\nColumn Names:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    # Data types
    print("\nData Types Summary:")
    print(f"  Numeric columns: {df.select_dtypes(include=[np.number]).shape[1]}")
    print(f"  String columns: {df.select_dtypes(include=['object']).shape[1]}")
    
    # First rows
    print("\nFirst 5 Rows:")
    print(df.head())
    
    # Basic statistics
    print("\nNumeric Feature Statistics:")
    print(df.describe())


def assess_data_quality(df: pd.DataFrame) -> None:
    """
    Comprehensive data quality assessment.
    
    Checks:
    - Missing values
    - Duplicate rows
    - Data type consistency
    - Value range validation
    """
    print("\n" + "=" * 70)
    print("[Step 3/8] Data Quality Assessment")
    print("=" * 70)
    
    # Missing values
    print("\nMissing Values Analysis:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    missing_report = pd.DataFrame({
        'Missing_Count': missing,
        'Percentage': missing_pct
    }).sort_values('Missing_Count', ascending=False)
    
    missing_cols = missing_report[missing_report['Missing_Count'] > 0]
    
    if len(missing_cols) > 0:
        print("WARNING: Columns with missing values:")
        print(missing_cols)
    else:
        print("PASS: No missing values found")
    
    # Duplicates
    print("\nDuplicate Detection:")
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows: {duplicates} ({duplicates/len(df)*100:.2f}%)")
    
    if duplicates > 0:
        print("WARNING: Duplicates found - may need deduplication")
    else:
        print("PASS: No duplicate rows")
    
    # Domain validation
    print("\nDomain Validation (Disc Golf Standards):")
    validation_issues = []
    
    if 'SPEED' in df.columns:
        invalid_speed = df[(df['SPEED'] < 1) | (df['SPEED'] > 15)]
        if len(invalid_speed) > 0:
            validation_issues.append(f"Invalid SPEED values: {len(invalid_speed)}")
        else:
            print("  PASS: SPEED values in valid range [1-15]")
    
    if 'GLIDE' in df.columns:
        invalid_glide = df[(df['GLIDE'] < 1) | (df['GLIDE'] > 7)]
        if len(invalid_glide) > 0:
            validation_issues.append(f"Invalid GLIDE values: {len(invalid_glide)}")
        else:
            print("  PASS: GLIDE values in valid range [1-7]")
    
    if 'DIAMETER (cm)' in df.columns:
        invalid_diameter = df[(df['DIAMETER (cm)'] < 20) | (df['DIAMETER (cm)'] > 25)]
        if len(invalid_diameter) > 0:
            validation_issues.append(f"Invalid DIAMETER values: {len(invalid_diameter)}")
        else:
            print("  PASS: DIAMETER values in valid range [20-25 cm]")
    
    if validation_issues:
        print("\nWARNING: Validation issues found:")
        for issue in validation_issues:
            print(f"  - {issue}")
    else:
        print("\nPASS: All validation checks passed")


def _identify_skewed_columns(df: pd.DataFrame, threshold: float = 0.75) -> Iterable[str]:
    """Return numeric columns whose absolute skew exceeds the threshold."""

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for column in numeric_cols:
        skew_val = df[column].skew()
        if abs(skew_val) >= threshold:
            yield column


def analyze_distributions(df: pd.DataFrame) -> None:
    """
    Analyze feature distributions.
    
    Focuses on:
    - Flight numbers (SPEED, GLIDE, TURN, FADE)
    - Physical dimensions
    - Distribution statistics
    """
    print("\n" + "=" * 70)
    print("[Step 4/8] Feature Distribution Analysis")
    print("=" * 70)
    
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"\nAnalyzing {len(numeric_cols)} numeric features")
    
    # Distribution statistics
    dist_stats = pd.DataFrame({
        'mean': df[numeric_cols].mean(),
        'median': df[numeric_cols].median(),
        'std': df[numeric_cols].std(),
        'min': df[numeric_cols].min(),
        'max': df[numeric_cols].max(),
        'skewness': df[numeric_cols].skew(),
        'kurtosis': df[numeric_cols].kurtosis()
    })
    
    print("\nDistribution Statistics:")
    print(dist_stats)
    
    # Identify skewed distributions
    print("\nDistribution Characteristics:")
    for col in numeric_cols:
        skew_val = df[col].skew()
        if abs(skew_val) > 1:
            direction = "right" if skew_val > 0 else "left"
            print(f"  {col}: Highly {direction}-skewed (skewness = {skew_val:.2f})")
        elif abs(skew_val) > 0.5:
            direction = "right" if skew_val > 0 else "left"
            print(f"  {col}: Moderately {direction}-skewed (skewness = {skew_val:.2f})")
        else:
            print(f"  {col}: Approximately symmetric (skewness = {skew_val:.2f})")
    
    # Violin plot overview for all numeric features
    melted = df[numeric_cols].melt(var_name="feature", value_name="value")
    plt.figure(figsize=(12, 0.6 * len(numeric_cols)))
    sns.violinplot(data=melted, x="value", y="feature", inner="quartile", cut=0, scale="width")
    plt.title("Feature Distributions (Violin View)", fontsize=14, fontweight="bold")
    plt.xlabel("Value")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig("data/processed/feature_distributions_violin.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Violin plots for skewed features
    skewed_columns = list(_identify_skewed_columns(df))
    if skewed_columns:
        plt.figure(figsize=(12, 0.5 + 0.6 * len(skewed_columns)))
        sns.violinplot(data=df[skewed_columns], orient="h", inner="quartile", cut=0)
        plt.title("Violin Plots for Skewed Features", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig("data/processed/skewed_feature_violins.png", dpi=150, bbox_inches="tight")
        plt.close()


def analyze_categorical_features(df):
    """
    Analyze categorical features.
    
    Examines:
    - Disc types
    - Stability categories
    - Bead presence
    """
    print("\n" + "=" * 70)
    print("[Step 5/8] Categorical Feature Analysis")
    print("=" * 70)
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_cols:
        print(f"\nFound {len(categorical_cols)} categorical features\n")
        
        for col in categorical_cols:
            print(f"\n{col}:")
            print(f"  Unique values: {df[col].nunique()}")
            
            value_counts = df[col].value_counts()
            print(f"\n  Distribution:")
            for val, count in value_counts.head(10).items():
                pct = (count / len(df)) * 100
                print(f"    {val}: {count} ({pct:.1f}%)")
            
            if df[col].nunique() > 10:
                print(f"    ... and {df[col].nunique() - 10} more categories")
    else:
        print("No categorical features found")


def analyze_correlations(df):
    """
    Correlation analysis between numeric features.
    
    Identifies:
    - Strong correlations (multicollinearity concerns)
    - Weak correlations
    - Feature relationships
    """
    print("\n" + "=" * 70)
    print("[Step 6/8] Correlation Analysis")
    print("=" * 70)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    print("\nCorrelation Matrix:")
    print(corr_matrix)
    
    # Identify highly correlated pairs
    high_corr_threshold = 0.8
    high_corr_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > high_corr_threshold:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_val
                ))
    
    if high_corr_pairs:
        print(f"\nWARNING: Highly Correlated Features (|r| > {high_corr_threshold}):")
        print("These may cause multicollinearity in regression models:")
        for feat1, feat2, corr_val in high_corr_pairs:
            print(f"  {feat1} <-> {feat2}: r = {corr_val:.3f}")
        print("\nRecommendation: Consider removing one feature from each pair")
    else:
        print(f"\nPASS: No highly correlated pairs found (threshold: {high_corr_threshold})")
    
    # Create correlation heatmap
    print("\nGenerating correlation heatmap...")
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        center=0,
        vmin=-1,
        vmax=1,
    )
    plt.title("Feature Correlation Matrix", fontsize=16, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("data/processed/correlation_heatmap.png", dpi=150, bbox_inches="tight")
    print("Saved: data/processed/correlation_heatmap.png")
    plt.close()


def analyze_disc_types(df):
    """
    Specific analysis for disc types and flight characteristics.
    
    This addresses the actual data content (disc catalog, not throws).
    """
    print("\n" + "=" * 70)
    print("[Step 7/8] Disc Type Analysis")
    print("=" * 70)
    
    if 'DISC TYPE' not in df.columns:
        print("DISC TYPE column not found - skipping this analysis")
        return
    
    print("\nDisc Type Distribution:")
    disc_type_dist = df['DISC TYPE'].value_counts()
    print(disc_type_dist)
    
    # Analyze flight characteristics by disc type
    if all(col in df.columns for col in ['SPEED', 'GLIDE', 'TURN', 'FADE']):
        print("\nFlight Characteristics by Disc Type:")
        
        flight_stats = df.groupby('DISC TYPE')[['SPEED', 'GLIDE', 'TURN', 'FADE']].agg([
            'mean', 'std', 'min', 'max'
        ])
        
        print(flight_stats)
        
        # Create visualization
        print("\nGenerating disc type comparison...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        for idx, metric in enumerate(['SPEED', 'GLIDE', 'TURN', 'FADE']):
            ax = axes[idx // 2, idx % 2]
            df.boxplot(column=metric, by='DISC TYPE', ax=ax)
            ax.set_title(f'{metric} by Disc Type', fontsize=12, fontweight='bold')
            ax.set_xlabel('Disc Type')
            ax.set_ylabel(metric)
            plt.sca(ax)
            plt.xticks(rotation=45, ha='right')
        
        plt.suptitle('Flight Characteristics by Disc Type', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('data/processed/disc_type_analysis.png', dpi=150, bbox_inches='tight')
        print("Saved: data/processed/disc_type_analysis.png")
        plt.close()


def generate_key_findings(df):
    """
    Generate summary of key findings from EDA.
    
    This is the most important output for the professor meeting.
    """
    print("\n" + "=" * 70)
    print("[Step 8/8] KEY FINDINGS SUMMARY")
    print("=" * 70)
    
    print("\nDATASET OVERVIEW:")
    print(f"  - Total records: {len(df):,}")
    print(f"  - Total features: {len(df.columns)}")
    print(f"  - Missing values: {df.isnull().sum().sum()}")
    print(f"  - Duplicate rows: {df.duplicated().sum()}")
    
    # Categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        print("\nCATEGORICAL FEATURES:")
        for col in categorical_cols:
            print(f"  - {col}: {df[col].nunique()} unique values")
    
    # Numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\nNUMERIC FEATURES: {len(numeric_cols)} total")
    
    # Correlations
    corr_matrix = df[numeric_cols].corr()
    high_corr_count = 0
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                high_corr_count += 1
    
    if high_corr_count > 0:
        print(f"\nCORRELATION WARNINGS:")
        print(f"  - Found {high_corr_count} highly correlated feature pairs")
        print(f"  - Recommendation: Feature selection needed before modeling")
    
    # Data quality summary
    print("\nDATA QUALITY SUMMARY:")
    if df.isnull().sum().sum() == 0:
        print("  - PASS: No missing values")
    else:
        print(f"  - WARNING: {df.isnull().sum().sum()} missing values require handling")
    
    if df.duplicated().sum() == 0:
        print("  - PASS: No duplicate rows")
    else:
        print(f"  - WARNING: {df.duplicated().sum()} duplicate rows")
    
    # Important note about data type
    print("\nCRITICAL FINDING:")
    print("  This dataset contains DISC SPECIFICATIONS, not throw performance data.")
    print("  Impact on ML solutions:")
    print("    - Cannot test throw quality prediction (no throw data)")
    print("    - Cannot cluster players by skill (no player data)")
    print("    - CAN implement content-based disc recommendation")
    print("    - CAN demonstrate clustering methodology on disc specs")
    print("    - CAN show regression approach predicting speed from dimensions")
    
    print("\nRECOMMENDATIONS FOR NEXT PHASE:")
    print("  1. Focus on content-based recommendation system (fully feasible)")
    print("  2. Use disc clustering as methodology demonstration")
    print("  3. Create regression model: predict speed from physical dimensions")
    print("  4. Document data preparation pipeline for future GameProofer data")
    print("  5. Discuss with professor: access to real throw performance data")


def main():
    """
    Main execution function for EDA.
    """
    # Define paths
    data_path = Path("data/raw/disc-data.csv")
    
    # Create output directory
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    # Check if data exists
    if not data_path.exists():
        print(f"ERROR: Data file not found at {data_path}")
        print("Please run: python src/data_loading/download_kaggle_data.py")
        return
    
    # Run EDA pipeline
    df = load_and_clean_data(data_path)
    initial_inspection(df)
    assess_data_quality(df)
    analyze_distributions(df)
    analyze_categorical_features(df)
    analyze_correlations(df)
    analyze_disc_types(df)
    generate_key_findings(df)
    
    # Save cleaned dataset
    output_path = Path("data/processed/disc_golf_cleaned.csv")
    df.to_csv(output_path, index=False)
    print(f"\nCleaned dataset saved to: {output_path}")
    
    print("\n" + "=" * 70)
    print("EXPLORATORY DATA ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nNext Steps:")
    print("  1. Review generated visualizations in data/processed/")
    print("  2. Read docs/03_EDA_FINDINGS_AND_ML_FEASIBILITY.md")
    print("  3. Prepare for professor meeting with key findings")
    print("  4. Focus on feasible ML solutions (content-based recommendation)")


if __name__ == "__main__":
    main()
