"""Simple Parquet-based feature storage with versioning.

Replaces Feast (which is overkill for <1000 cases). Provides:
- Versioned feature tables saved as Parquet files
- Feature metadata (names, dtypes, basic stats)
- Train/val/test split management based on temporal splits

Usage:
    store = FeatureStore()
    store.save_features(df, version="v1")
    df = store.load_features(version="v1")
    train, val, test = store.temporal_split(df)
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from configs.settings import settings

logger = logging.getLogger(__name__)


class FeatureStore:
    """File-based feature storage with versioning and temporal splits."""

    def __init__(self, base_path: Path | None = None):
        self.base_path = base_path or settings.data.processed_dir
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save_features(
        self,
        df: pd.DataFrame,
        version: str,
        description: str = "",
    ) -> Path:
        """Save a feature DataFrame as versioned Parquet.

        Also saves metadata (feature names, dtypes, stats) as JSON.
        """
        parquet_path = self.base_path / f"features_{version}.parquet"
        meta_path = self.base_path / f"features_{version}_meta.json"

        df.to_parquet(parquet_path, index=True)

        metadata = {
            "version": version,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "null_counts": df.isnull().sum().to_dict(),
            "numeric_stats": {},
        }

        for col in df.select_dtypes(include="number").columns:
            metadata["numeric_stats"][col] = {
                "mean": float(df[col].mean()) if not df[col].isna().all() else None,
                "std": float(df[col].std()) if not df[col].isna().all() else None,
                "min": float(df[col].min()) if not df[col].isna().all() else None,
                "max": float(df[col].max()) if not df[col].isna().all() else None,
            }

        meta_path.write_text(json.dumps(metadata, indent=2, default=str))
        logger.info("Saved features %s: %d rows x %d cols", version, len(df), len(df.columns))
        return parquet_path

    def load_features(self, version: str) -> pd.DataFrame:
        """Load a versioned feature DataFrame."""
        parquet_path = self.base_path / f"features_{version}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Feature version not found: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        logger.info("Loaded features %s: %d rows x %d cols", version, len(df), len(df.columns))
        return df

    def list_versions(self) -> list[str]:
        """List all available feature versions."""
        versions = []
        for path in sorted(self.base_path.glob("features_*.parquet")):
            version = path.stem.replace("features_", "")
            versions.append(version)
        return versions

    def get_metadata(self, version: str) -> dict:
        """Load metadata for a feature version."""
        meta_path = self.base_path / f"features_{version}_meta.json"
        if not meta_path.exists():
            return {}
        return json.loads(meta_path.read_text())

    def temporal_split(
        self,
        df: pd.DataFrame,
        year_column: str = "filing_year",
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split features into train/val/test by year (temporal split).

        Uses year ranges from settings.model:
        - Train: 2015-2020
        - Validation: 2021
        - Test: 2022-2023

        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        cfg = settings.model

        train = df[df[year_column].isin(cfg.train_years)]
        val = df[df[year_column].isin(cfg.val_years)]
        test = df[df[year_column].isin(cfg.test_years)]

        logger.info(
            "Temporal split: train=%d (%s), val=%d (%s), test=%d (%s)",
            len(train),
            cfg.train_years,
            len(val),
            cfg.val_years,
            len(test),
            cfg.test_years,
        )
        return train, val, test
