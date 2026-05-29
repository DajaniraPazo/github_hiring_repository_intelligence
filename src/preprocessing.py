"""
Feature engineering and preprocessing for raw GitHub repository data.

Signals extracted:
  - stars_count, forks_count, contributors_count, weekly_commit_freq
  - open_issues_count, closed_prs_count, releases_count, readme_length
  - repo_age_days, last_activity_days, language_count, topics_count
  - has_ci_cd, has_tests, has_license (boolean -> int)
  - is_fork, is_template

Engineered composite features:
  - stars_per_fork_ratio, commit_density, pr_velocity
  - activity_score, complexity_score, maturity_score
  - log-transformed versions of skewed signals
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.utils import setup_logging, RAW_DIR, PROCESSED_DIR, LABEL2ID

logger = setup_logging("preprocessing")

NUMERIC_FEATURES = [
    "stars_count", "forks_count", "contributors_count", "weekly_commit_freq",
    "open_issues_count", "closed_prs_count", "releases_count", "readme_length",
    "repo_age_days", "last_activity_days", "language_count", "topics_count",
    "watchers_count", "network_count",
]

BOOL_FEATURES = ["has_ci_cd", "has_tests", "has_license", "is_fork", "is_template"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)

    for col in BOOL_FEATURES:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: int(str(x).lower() in ("true", "1", "1.0", "yes"))
            )

    eps = 1e-6

    df["stars_per_fork_ratio"] = df["stars_count"] / (df["forks_count"] + 1)
    df["commit_density"] = df["weekly_commit_freq"] / (df["repo_age_days"] / 7 + eps)
    df["pr_velocity"] = df["closed_prs_count"] / (df["repo_age_days"] / 30 + eps)
    df["issue_pr_ratio"] = df["open_issues_count"] / (df["closed_prs_count"] + 1)
    df["readme_density"] = df["readme_length"] / (df["repo_age_days"] + eps)

    max_last = df["last_activity_days"].max() + 1
    df["activity_score"] = (
        np.log1p(df["weekly_commit_freq"]) * 0.30
        + np.log1p(df["closed_prs_count"]) * 0.30
        + (1 - df["last_activity_days"] / max_last) * 0.40
    )

    df["complexity_score"] = (
        np.log1p(df["contributors_count"]) * 0.25
        + np.log1p(df["stars_count"]) * 0.20
        + df["has_ci_cd"] * 0.20
        + df["has_tests"] * 0.20
        + np.log1p(df["releases_count"]) * 0.15
    )

    df["maturity_score"] = (
        np.log1p(df["repo_age_days"]) * 0.20
        + np.log1p(df["contributors_count"]) * 0.25
        + df["has_license"] * 0.15
        + np.log1p(df["closed_prs_count"]) * 0.20
        + np.log1p(df["releases_count"]) * 0.20
    )

    for col in ["stars_count", "forks_count", "contributors_count", "closed_prs_count",
                "readme_length", "releases_count", "watchers_count", "weekly_commit_freq"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col])

    return df


def create_label_encoding(df: pd.DataFrame) -> pd.DataFrame:
    if "category" in df.columns:
        df["label"] = df["category"].map(LABEL2ID).fillna(-1).astype(int)
    return df


def preprocess(input_path=None, output_path=None) -> pd.DataFrame:
    input_path = Path(input_path) if input_path else RAW_DIR / "repos_raw.csv"
    output_path = Path(output_path) if output_path else PROCESSED_DIR / "repos_processed.csv"

    logger.info(f"Loading {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Raw shape: {df.shape}")

    if "topics" in df.columns:
        df["topics"] = df["topics"].apply(
            lambda x: x if isinstance(x, list)
            else (eval(str(x)) if pd.notna(x) and str(x).startswith("[") else [])
        )

    df = engineer_features(df)
    df = create_label_encoding(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved processed data → {output_path}  shape={df.shape}")
    return df


if __name__ == "__main__":
    df = preprocess()
    print(df[["complexity_score", "maturity_score", "activity_score"]].describe())
