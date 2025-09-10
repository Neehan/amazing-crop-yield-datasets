"""ML imputation for missing crop calendar data using land surface features"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

logger = logging.getLogger(__name__)


class CropCalendarMLImputation:
    """ML-based imputation for missing crop calendar data using environmental features"""

    def __init__(self, country: str, admin_level: int, data_dir: Path):
        """Initialize ML imputation

        Args:
            country: Country name
            admin_level: Administrative level
            data_dir: Data directory path
        """
        self.country = country
        self.admin_level = admin_level
        self.data_dir = data_dir
        self.scaler = StandardScaler()
        self.planted_model = None
        self.harvested_model = None
        self.feature_columns = None

    def _load_land_surface_data(self, year: int) -> pd.DataFrame:
        """Load land surface data for the specified year"""
        features_dir = (
            self.data_dir / self.country.lower() / "intermediate" / "aggregated"
        )

        if not features_dir.exists():
            raise FileNotFoundError(
                f"Intermediate aggregated directory not found: {features_dir}"
            )

        logger.info(
            f"Loading features from intermediate aggregated data for year {year}"
        )

        land_surface_files = [
            f"land_surface_{year}-{year}_ndvi_weekly_weighted_admin{self.admin_level}.csv",
            f"land_surface_{year}-{year}_lai_low_weekly_weighted_admin{self.admin_level}.csv",
            f"land_surface_{year}-{year}_lai_high_weekly_weighted_admin{self.admin_level}.csv",
        ]

        dataframes = []
        for file_pattern in land_surface_files:
            file_path = features_dir / file_pattern
            if file_path.exists():
                df = pd.read_csv(file_path)
                df["year"] = year
                dataframes.append(df)
                logger.info(f"Loaded {file_path.name}: {len(df)} records")
            else:
                logger.warning(f"File not found: {file_path}")

        if not dataframes:
            raise ValueError(f"No feature files found for year {year}")

        # Merge all dataframes
        features_df = dataframes[0]
        merge_cols = [
            "country",
            "admin_level_1",
            "admin_level_2",
            "year",
            "latitude",
            "longitude",
        ]
        common_cols = [col for col in merge_cols if col in features_df.columns]

        for df in dataframes[1:]:
            common_cols_df = [col for col in common_cols if col in df.columns]
            features_df = features_df.merge(df, on=common_cols_df, how="outer")

        logger.info(f"Loaded {len(features_df)} administrative units for year {year}")
        return features_df

    def _scale_ndvi(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Scale NDVI values by dividing by 10000"""
        ndvi_cols = [col for col in features_df.columns if col.startswith("ndvi_week_")]
        for col in ndvi_cols:
            if col in features_df.columns:
                features_df[col] = features_df[col] / 10000.0
                logger.info(f"Scaled NDVI column {col} by dividing by 10000")
        return features_df

    def _get_feature_columns(self, features_df: pd.DataFrame) -> List[str]:
        """Get feature column names for land surface data"""
        feature_columns = []
        land_surface_features = ["ndvi", "lai_low", "lai_high"]

        for feature in land_surface_features:
            for week in range(1, 53):
                col_name = f"{feature}_week_{week}"
                if col_name in features_df.columns:
                    feature_columns.append(col_name)

        logger.info(
            f"Using only NDVI, LAI_LOW, LAI_HIGH features ({len(feature_columns)} total)"
        )
        return feature_columns

    def load_features_data(self, year: int = 2000) -> pd.DataFrame:
        """Load aggregated features data for the specified year from intermediate directory"""
        features_df = self._load_land_surface_data(year)
        features_df = self._scale_ndvi(features_df)
        return features_df

    def prepare_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature columns for ML training with proper scaling"""
        features_df = features_df.copy()
        feature_columns = self._get_feature_columns(features_df)

        # Create features DataFrame
        features_subset = features_df[
            ["country", "admin_level_1", "admin_level_2"] + feature_columns
        ].copy()

        # Remove rows with any missing values
        features_subset = features_subset.dropna()
        self.feature_columns = feature_columns

        logger.info(
            f"Prepared {len(feature_columns)} features for {len(features_subset)} administrative units"
        )
        return features_subset

    def prepare_targets(self, crop_calendar_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare target variables (planting/harvesting distributions) for ML training"""
        target_columns = ["country_name", "admin_level_1_name", "admin_level_2_name"]

        # Add monthly planted and harvested columns
        for month in range(1, 13):
            target_columns.extend(
                [f"planted_month_{month}", f"harvested_month_{month}"]
            )

        targets_df = crop_calendar_df[target_columns].copy()

        # Remove rows where all monthly values are 0 (no data)
        monthly_cols = [col for col in targets_df.columns if "month_" in col]
        targets_df = targets_df[targets_df[monthly_cols].sum(axis=1) > 0]

        logger.info(
            f"Prepared targets for {len(targets_df)} administrative units with crop calendar data"
        )
        return targets_df

    def _normalize_probability_distribution(self, y: np.ndarray) -> np.ndarray:
        """Normalize array to sum to 1 (probability distribution)"""
        y = y / np.sum(y, axis=1, keepdims=True)
        return y

    def _apply_threshold_and_normalize(
        self, y: np.ndarray, threshold: float = 0.1
    ) -> np.ndarray:
        """Zero out small values and renormalize to 1"""
        y[y < threshold] = 0.0
        return self._normalize_probability_distribution(y)

    def _calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Tuple[float, float, List[float]]:
        """Calculate MSE and R² metrics"""
        mse_scores = []
        for month in range(12):
            mse = np.mean((y_true[:, month] - y_pred[:, month]) ** 2)
            mse_scores.append(mse)

        avg_mse = float(np.mean(mse_scores))
        avg_r2 = 1.0 - avg_mse
        return avg_mse, avg_r2, mse_scores

    def _train_single_model(
        self, X_scaled: np.ndarray, y: np.ndarray, target_type: str
    ) -> MultiOutputRegressor:
        """Train a single Random Forest model"""
        # Normalize targets to ensure they sum to 1
        y = self._normalize_probability_distribution(y)

        # Split data for training and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # Train Random Forest with MultiOutputRegressor
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )

        model = MultiOutputRegressor(rf)
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_val)
        y_pred = self._apply_threshold_and_normalize(y_pred)

        # Calculate metrics
        avg_mse, avg_r2, mse_scores = self._calculate_metrics(y_val, y_pred)

        logger.info(
            f"Model {target_type}: Average MSE={avg_mse:.6f} (monthly MSE: {[f'{r:.6f}' for r in mse_scores]})"
        )
        logger.info(
            f"Model {target_type}: Average R²={avg_r2:.4f} (1-MSE, monthly R²: {[f'{1-r:.3f}' for r in mse_scores]})"
        )
        logger.info(
            f"Model {target_type}: MSE shows prediction error, R²=1-MSE shows variance explained"
        )

        return model

    def train_models(
        self, features_df: pd.DataFrame, targets_df: pd.DataFrame
    ) -> Dict[str, MultiOutputRegressor]:
        """Train Random Forest models for planted and harvested distributions"""
        # Merge features and targets
        features_df_renamed = features_df.rename(
            columns={
                "country": "country_name",
                "admin_level_1": "admin_level_1_name",
                "admin_level_2": "admin_level_2_name",
            }
        )

        merged_df = features_df_renamed.merge(
            targets_df,
            on=["country_name", "admin_level_1_name", "admin_level_2_name"],
            how="inner",
        )

        if merged_df.empty:
            raise ValueError("No matching data between features and targets")

        logger.info(f"Training models on {len(merged_df)} samples")

        # Prepare feature matrix
        feature_cols = [
            col
            for col in merged_df.columns
            if col not in ["country_name", "admin_level_1_name", "admin_level_2_name"]
            and "month_" not in col
        ]

        X = merged_df[feature_cols].values
        X_scaled = self.scaler.fit_transform(X)

        # Train models
        models = {}
        for target_type in ["planted", "harvested"]:
            target_cols = [f"{target_type}_month_{month}" for month in range(1, 13)]
            y = merged_df[target_cols].values.astype(np.float64)
            models[target_type] = self._train_single_model(X_scaled, y, target_type)

        self.planted_model = models["planted"]
        self.harvested_model = models["harvested"]
        return models

    def _get_matching_units(
        self, no_crop_units: pd.DataFrame, features_df_renamed: pd.DataFrame
    ) -> pd.Series:
        """Get units that have both no crop data AND matching features"""
        no_crop_identifiers = no_crop_units[
            ["admin_level_1_name", "admin_level_2_name"]
        ].apply(
            lambda x: f"{x['admin_level_1_name']}_{x['admin_level_2_name']}", axis=1
        )

        features_identifiers = features_df_renamed[
            ["admin_level_1_name", "admin_level_2_name"]
        ].apply(
            lambda x: f"{x['admin_level_1_name']}_{x['admin_level_2_name']}", axis=1
        )

        return no_crop_identifiers[no_crop_identifiers.isin(features_identifiers)]

    def predict_missing(
        self, features_df: pd.DataFrame, crop_calendar_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Predict crop calendar data for units with total_area=0"""
        if not self.planted_model or not self.harvested_model:
            raise ValueError("Models not trained. Call train_models() first.")

        # Find units with no crop data
        no_crop_units = crop_calendar_df[crop_calendar_df["total_area"] == 0.0].copy()
        logger.info(
            f"Found {len(no_crop_units)} administrative units with no crop data (total_area=0)"
        )

        if no_crop_units.empty:
            logger.info("No units with zero area to impute")
            return crop_calendar_df

        # Rename features columns to match crop calendar format
        features_df_renamed = features_df.rename(
            columns={
                "country": "country_name",
                "admin_level_1": "admin_level_1_name",
                "admin_level_2": "admin_level_2_name",
            }
        )

        # Get matching units
        matching_units = self._get_matching_units(no_crop_units, features_df_renamed)
        logger.info(
            f"Found {len(matching_units)} units with no crop data that have matching features"
        )

        if matching_units.empty:
            logger.info("No matching features found for units with zero area")
            return crop_calendar_df

        # Get features for matching units
        features_identifiers = features_df_renamed[
            ["admin_level_1_name", "admin_level_2_name"]
        ].apply(
            lambda x: f"{x['admin_level_1_name']}_{x['admin_level_2_name']}", axis=1
        )

        missing_features = features_df_renamed[
            features_identifiers.isin(matching_units)
        ].copy()

        # Prepare feature matrix
        feature_cols = [
            col
            for col in missing_features.columns
            if col not in ["country_name", "admin_level_1_name", "admin_level_2_name"]
            and "month_" not in col
        ]

        X_missing = missing_features[feature_cols].values
        X_missing_scaled = self.scaler.transform(X_missing)

        # Create predictions
        no_crop_identifiers = no_crop_units[
            ["admin_level_1_name", "admin_level_2_name"]
        ].apply(
            lambda x: f"{x['admin_level_1_name']}_{x['admin_level_2_name']}", axis=1
        )
        predicted_df = no_crop_units[no_crop_identifiers.isin(matching_units)].copy()

        for target_type, model in [
            ("planted", self.planted_model),
            ("harvested", self.harvested_model),
        ]:
            pred = model.predict(X_missing_scaled)
            pred = self._apply_threshold_and_normalize(pred)

            # Add monthly columns
            for month in range(1, 13):
                predicted_df[f"{target_type}_month_{month}"] = pred[:, month - 1]

        # Update the original dataframe
        matching_mask = (crop_calendar_df["total_area"] == 0.0) & (
            crop_calendar_df["admin_level_1_name"]
            + "_"
            + crop_calendar_df["admin_level_2_name"]
        ).isin(matching_units)

        crop_calendar_df.loc[matching_mask, predicted_df.columns] = predicted_df

        logger.info(
            f"Imputed crop calendar data for {len(predicted_df)} administrative units with zero area"
        )
        return crop_calendar_df

    def impute_crop_calendar(
        self, crop_calendar_path: Path, year: int = 2000
    ) -> pd.DataFrame:
        """Complete pipeline for crop calendar imputation"""
        logger.info(f"Starting ML imputation for crop calendar: {crop_calendar_path}")

        # Load existing crop calendar data
        crop_calendar_df = pd.read_csv(crop_calendar_path)
        logger.info(f"Loaded {len(crop_calendar_df)} existing crop calendar records")

        # Load and prepare features
        features_df = self.load_features_data(year)
        features_df = self.prepare_features(features_df)

        # Prepare targets
        targets_df = self.prepare_targets(crop_calendar_df)

        # Train models
        self.train_models(features_df, targets_df)

        # Predict missing data
        imputed_df = self.predict_missing(features_df, crop_calendar_df)

        logger.info(f"ML imputation complete. Total records: {len(imputed_df)}")
        return imputed_df
