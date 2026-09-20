"""
train_model.py
==============
Trains baseline Random Forest classification models for IPL match prediction:
- Model A: Pre-Toss Model (features available before the toss)
- Model B: Post-Toss Model (pre-toss features + toss winner & decision)

Outputs:
- ml/models/pre_toss_model.joblib
- ml/models/post_toss_model.joblib
- data/processed/feature_importance_pre_toss.csv
- data/processed/feature_importance_post_toss.csv
- data/processed/feature_importance_pre_toss.png
- data/processed/feature_importance_post_toss.png
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# File paths
FEATURES_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "ipl_matches_features.csv"
MODELS_DIR = PROJECT_ROOT / "ml" / "models"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

PRE_TOSS_MODEL_PATH = MODELS_DIR / "pre_toss_model.joblib"
POST_TOSS_MODEL_PATH = MODELS_DIR / "post_toss_model.joblib"

PRE_TOSS_IMPORTANCE_CSV = PROCESSED_DATA_DIR / "feature_importance_pre_toss.csv"
POST_TOSS_IMPORTANCE_CSV = PROCESSED_DATA_DIR / "feature_importance_post_toss.csv"

PRE_TOSS_IMPORTANCE_PNG = PROCESSED_DATA_DIR / "feature_importance_pre_toss.png"
POST_TOSS_IMPORTANCE_PNG = PROCESSED_DATA_DIR / "feature_importance_post_toss.png"

# Feature definitions
CATEGORICAL_FEATURES: List[str] = [
    "team1",
    "team2",
    "venue",
    "city",
    "season",
]

PRE_TOSS_NUMERICAL_FEATURES: List[str] = [
    "team1_prior_win_rate",
    "team2_prior_win_rate",
    "team1_last_5_win_rate",
    "team2_last_5_win_rate",
    "team1_last_10_win_rate",
    "team2_last_10_win_rate",
    "h2h_team1_win_rate",
    "h2h_matches_before",
    "team1_venue_win_rate",
    "team2_venue_win_rate",
    "venue_matches_before",
    "venue_chase_win_rate",
    "team1_elo",
    "team2_elo",
    "elo_diff",
    "prior_win_rate_diff",
    "last_5_win_rate_diff",
    "last_10_win_rate_diff",
    "venue_win_rate_diff",
    "team1_home_advantage",
]

PRE_TOSS_FEATURES: List[str] = CATEGORICAL_FEATURES + PRE_TOSS_NUMERICAL_FEATURES

POST_TOSS_NUMERICAL_FEATURES: List[str] = PRE_TOSS_NUMERICAL_FEATURES + [
    "team1_toss_winner",
    "toss_decision_field",
]

POST_TOSS_FEATURES: List[str] = CATEGORICAL_FEATURES + POST_TOSS_NUMERICAL_FEATURES

TARGET_COLUMN: str = "team1_win"

# Random Forest Baseline Hyperparameters
RF_PARAMS: Dict[str, Any] = {
    "n_estimators": 500,
    "random_state": 42,
    "class_weight": "balanced",
    "min_samples_leaf": 2,
    "n_jobs": -1,
}


def load_and_split_data(features_csv: Path = FEATURES_CSV_PATH) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load features dataset and split into chronological Train, Validation, and Test sets:
    - Train: Seasons 2008 - 2023 (year <= 2023)
    - Validation: Season 2024 (year == 2024)
    - Test: Seasons 2025 - 2026 (year >= 2025)
    """
    df = pd.read_csv(features_csv)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["date", "match_id"]).reset_index(drop=True)
    df["year"] = df["date"].dt.year

    train_df = df[df["year"] <= 2023].copy()
    val_df = df[df["year"] == 2024].copy()
    test_df = df[df["year"] >= 2025].copy()

    # Drop temporary year column
    train_df = train_df.drop(columns=["year"])
    val_df = val_df.drop(columns=["year"])
    test_df = test_df.drop(columns=["year"])

    return train_df, val_df, test_df


def build_pipeline(feature_cols: List[str], cat_cols: List[str], rf_params: Dict[str, Any]) -> Pipeline:
    """Build an sklearn Pipeline with ColumnTransformer preprocessing and RandomForestClassifier."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(**rf_params)),
        ]
    )
    return pipeline


def extract_feature_importances(
    pipeline: Pipeline,
    feature_cols: List[str],
    cat_cols: List[str],
) -> pd.DataFrame:
    """
    Extract and format feature importances from a trained pipeline.
    Maps encoded feature names back to readable names.
    """
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    classifier: RandomForestClassifier = pipeline.named_steps["classifier"]

    # Get feature names from ColumnTransformer
    raw_feature_names = preprocessor.get_feature_names_out()

    # Clean prefix names (cat__, remainder__)
    cleaned_feature_names: List[str] = []
    for name in raw_feature_names:
        if name.startswith("cat__"):
            cleaned_feature_names.append(name.replace("cat__", ""))
        elif name.startswith("remainder__"):
            cleaned_feature_names.append(name.replace("remainder__", ""))
        else:
            cleaned_feature_names.append(name)

    importances = classifier.feature_importances_
    importance_df = pd.DataFrame(
        {
            "feature": cleaned_feature_names,
            "importance": importances,
        }
    ).sort_values(by="importance", ascending=False).reset_index(drop=True)

    return importance_df


def plot_feature_importance(importance_df: pd.DataFrame, title: str, output_png: Path, top_n: int = 15) -> None:
    """Create and save horizontal bar chart of top N feature importances."""
    top_df = importance_df.head(top_n).iloc[::-1]  # Reverse for ascending order on plot

    plt.figure(figsize=(10, 7))
    plt.barh(top_df["feature"], top_df["importance"], color="#1f77b4", edgecolor="#0d47a1")
    plt.xlabel("Importance (Gini Impurity Reduction)", fontsize=11, fontweight="bold")
    plt.ylabel("Feature", fontsize=11, fontweight="bold")
    plt.title(f"{title} - Top {top_n} Features", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()
    print(f"Saved feature importance chart to: {output_png}")


def train_models() -> Tuple[Pipeline, Pipeline, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train Pre-Toss and Post-Toss Random Forest models and save artifacts."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data and creating chronological splits...")
    train_df, val_df, test_df = load_and_split_data()

    print(f"  Train set samples (2008-2023):      {len(train_df)}")
    print(f"  Validation set samples (2024):       {len(val_df)}")
    print(f"  Test set samples (2025-2026):        {len(test_df)}")

    # -------------------------------------------------------------
    # 1. MODEL A: PRE-TOSS
    # -------------------------------------------------------------
    print("\nTraining Model A (Pre-Toss Random Forest)...")
    pipe_pre_toss = build_pipeline(
        feature_cols=PRE_TOSS_FEATURES,
        cat_cols=CATEGORICAL_FEATURES,
        rf_params=RF_PARAMS,
    )
    pipe_pre_toss.fit(train_df[PRE_TOSS_FEATURES], train_df[TARGET_COLUMN])

    joblib.dump(pipe_pre_toss, PRE_TOSS_MODEL_PATH)
    print(f"Saved pre-toss model to: {PRE_TOSS_MODEL_PATH}")

    # Feature Importance Model A
    imp_pre_toss = extract_feature_importances(
        pipe_pre_toss,
        PRE_TOSS_FEATURES,
        CATEGORICAL_FEATURES,
    )
    imp_pre_toss.to_csv(PRE_TOSS_IMPORTANCE_CSV, index=False)
    print(f"Saved pre-toss feature importance CSV to: {PRE_TOSS_IMPORTANCE_CSV}")

    plot_feature_importance(
        imp_pre_toss,
        title="Pre-Toss Random Forest",
        output_png=PRE_TOSS_IMPORTANCE_PNG,
    )

    # -------------------------------------------------------------
    # 2. MODEL B: POST-TOSS
    # -------------------------------------------------------------
    print("\nTraining Model B (Post-Toss Random Forest)...")
    pipe_post_toss = build_pipeline(
        feature_cols=POST_TOSS_FEATURES,
        cat_cols=CATEGORICAL_FEATURES,
        rf_params=RF_PARAMS,
    )
    pipe_post_toss.fit(train_df[POST_TOSS_FEATURES], train_df[TARGET_COLUMN])

    joblib.dump(pipe_post_toss, POST_TOSS_MODEL_PATH)
    print(f"Saved post-toss model to: {POST_TOSS_MODEL_PATH}")

    # Feature Importance Model B
    imp_post_toss = extract_feature_importances(
        pipe_post_toss,
        POST_TOSS_FEATURES,
        CATEGORICAL_FEATURES,
    )
    imp_post_toss.to_csv(POST_TOSS_IMPORTANCE_CSV, index=False)
    print(f"Saved post-toss feature importance CSV to: {POST_TOSS_IMPORTANCE_CSV}")

    plot_feature_importance(
        imp_post_toss,
        title="Post-Toss Random Forest",
        output_png=POST_TOSS_IMPORTANCE_PNG,
    )

    return pipe_pre_toss, pipe_post_toss, train_df, val_df, test_df


if __name__ == "__main__":
    train_models()
