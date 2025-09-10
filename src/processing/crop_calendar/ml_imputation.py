"""ML imputation for missing crop calendar data using land surface features"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

from src.processing.crop_calendar.config import ML_IMPUTATION_CONFIG

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
        self.pca_models = {}  # Store PCA models for each group

        # Load ML imputation configuration
        self.pca_variance_retention = ML_IMPUTATION_CONFIG["pca_variance_retention"]
        self.n_neighbors = ML_IMPUTATION_CONFIG["n_neighbors"]
        self.weights = ML_IMPUTATION_CONFIG["weights"]
        self.p = ML_IMPUTATION_CONFIG["p"]

    def _load_features_data(self, year: int) -> pd.DataFrame:
        """Load land surface and weather data for the specified year"""
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

        # Define all feature files to load
        feature_files = [
            # Land surface data
            f"land_surface_{year}-{year}_ndvi_weekly_weighted_admin{self.admin_level}.csv",
            f"land_surface_{year}-{year}_lai_low_weekly_weighted_admin{self.admin_level}.csv",
            f"land_surface_{year}-{year}_lai_high_weekly_weighted_admin{self.admin_level}.csv",
            # Weather data
            f"weather_{year}-{year}_t2m_min_weekly_weighted_admin{self.admin_level}.csv",
            f"weather_{year}-{year}_t2m_max_weekly_weighted_admin{self.admin_level}.csv",
            f"weather_{year}-{year}_precipitation_weekly_weighted_admin{self.admin_level}.csv",
            # Soil data (no year column)
            "soil_clay_weighted_admin2.csv",
            "soil_sand_weighted_admin2.csv",
            "soil_silt_weighted_admin2.csv",
            "soil_ph_h2o_weighted_admin2.csv",
            "soil_organic_carbon_weighted_admin2.csv",
            "soil_bulk_density_weighted_admin2.csv",
        ]

        dataframes = []
        for file_pattern in feature_files:
            file_path = features_dir / file_pattern
            if file_path.exists():
                df = pd.read_csv(file_path)
                # Only add year column for time-series data (not soil)
                if not file_pattern.startswith("soil_"):
                    df["year"] = year
                dataframes.append(df)
                logger.info(f"Loaded {file_path.name}: {len(df)} records")
            else:
                logger.warning(f"File not found: {file_path}")

        if not dataframes:
            raise ValueError(f"No feature files found for year {year}")

        # Merge all dataframes
        features_df = dataframes[0]

        # Define merge columns - soil data doesn't have year
        merge_cols = [
            "country",
            "admin_level_1",
            "admin_level_2",
            "latitude",
            "longitude",
        ]
        # Add year to merge cols only if it exists in the first dataframe
        if "year" in features_df.columns:
            merge_cols.append("year")

        common_cols = [col for col in merge_cols if col in features_df.columns]

        for df in dataframes[1:]:
            common_cols_df = [col for col in common_cols if col in df.columns]
            features_df = features_df.merge(df, on=common_cols_df, how="inner")

        logger.info(f"Loaded {len(features_df)} administrative units for year {year}")
        return features_df

    def _scale_ndvi(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Scale NDVI values by dividing by 10000"""
        ndvi_cols = [col for col in features_df.columns if col.startswith("ndvi_week_")]
        for col in ndvi_cols:
            if col in features_df.columns:
                features_df[col] = features_df[col] / 10000.0
        return features_df

    def _get_feature_columns(self, features_df: pd.DataFrame) -> List[str]:
        """Get feature column names for land surface, weather, and soil data"""
        feature_columns = []

        # Time-series features (weekly data)
        time_series_features = [
            "ndvi",
            "lai_low",
            "lai_high",
            "t2m_min",
            "t2m_max",
            "precipitation",
        ]

        for feature in time_series_features:
            for week in range(1, 53):
                col_name = f"{feature}_week_{week}"
                if col_name in features_df.columns:
                    feature_columns.append(col_name)

        # Soil features (static data)
        soil_features = [
            "clay_0_5cm",
            "clay_5_15cm",
            "clay_15_30cm",
            "clay_30_60cm",
            "clay_60_100cm",
            "clay_100_200cm",
            "sand_0_5cm",
            "sand_5_15cm",
            "sand_15_30cm",
            "sand_30_60cm",
            "sand_60_100cm",
            "sand_100_200cm",
            "silt_0_5cm",
            "silt_5_15cm",
            "silt_15_30cm",
            "silt_30_60cm",
            "silt_60_100cm",
            "silt_100_200cm",
            "ph_h2o_0_5cm",
            "ph_h2o_5_15cm",
            "ph_h2o_15_30cm",
            "ph_h2o_30_60cm",
            "ph_h2o_60_100cm",
            "ph_h2o_100_200cm",
            "organic_carbon_0_5cm",
            "organic_carbon_5_15cm",
            "organic_carbon_15_30cm",
            "organic_carbon_30_60cm",
            "organic_carbon_60_100cm",
            "organic_carbon_100_200cm",
            "bulk_density_0_5cm",
            "bulk_density_5_15cm",
            "bulk_density_15_30cm",
            "bulk_density_30_60cm",
            "bulk_density_60_100cm",
            "bulk_density_100_200cm",
        ]

        for feature in soil_features:
            if feature in features_df.columns:
                feature_columns.append(feature)

        logger.info(
            f"Using NDVI, LAI_LOW, LAI_HIGH, T2M_MIN, T2M_MAX, PRECIPITATION, SOIL features ({len(feature_columns)} total)"
        )
        return feature_columns

    def _apply_group_pca(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Apply group-wise PCA to reduce noise while keeping coordinates raw"""
        features_df = features_df.copy()

        # Define feature groups
        feature_groups = {
            "weather": ["t2m_min", "t2m_max", "precipitation"],
            "land_surface": ["ndvi", "lai_low", "lai_high"],
            "soil": [
                "clay",
                "sand",
                "silt",
                "ph_h2o",
                "organic_carbon",
                "bulk_density",
            ],
        }

        # Keep coordinates as raw features
        coord_cols = ["latitude", "longitude"]
        pca_features = []

        for group_name, group_prefixes in feature_groups.items():
            # Find all columns for this group
            group_cols = []
            for prefix in group_prefixes:
                group_cols.extend(
                    [col for col in features_df.columns if col.startswith(f"{prefix}_")]
                )

            if not group_cols:
                logger.warning(f"No columns found for group {group_name}")
                continue

            # Extract group data
            group_data = features_df[group_cols].values

            # Check if we have enough data
            if group_data.shape[0] < 2 or group_data.shape[1] < 2:
                logger.warning(
                    f"Not enough data for PCA on group {group_name}, keeping original features"
                )
                pca_features.extend(group_cols)
                continue

            # Apply PCA to the group
            pca_data, pca = self._apply_pca_to_group(group_data, group_cols, group_name)

            # Store PCA model for later use
            self.pca_models[group_name] = pca

            # Create new column names and add to dataframe
            n_components = pca_data.shape[1]
            new_cols = [f"{group_name}_pca_{i+1}" for i in range(n_components)]

            # Create DataFrame with PCA data and concatenate to avoid fragmentation
            pca_df = pd.DataFrame(pca_data, columns=new_cols, index=features_df.index)
            features_df = pd.concat([features_df, pca_df], axis=1)

            pca_features.extend(new_cols)

        logger.info(
            f"Group {group_name}: {len(group_cols)} features -> {n_components} PCA components ({self.pca_variance_retention*100:.1f}% variance)"
        )

        # Add coordinates as raw features
        pca_features.extend(coord_cols)

        # Select only PCA features and coordinates
        final_features = [col for col in pca_features if col in features_df.columns]

        logger.info(
            f"Group-wise PCA complete: {len(final_features)} final features (including {len(coord_cols)} raw coordinates)"
        )
        return features_df[
            final_features + ["country", "admin_level_1", "admin_level_2"]
        ]

    def _fill_nan_values(
        self, group_data: np.ndarray, group_cols: List[str], group_name: str
    ) -> np.ndarray:
        """Fill NaN values using forward fill and backward fill for temporal data"""
        if not np.isnan(group_data).any():
            return group_data

        logger.info(
            f"Filling {np.isnan(group_data).sum()} NaN values in {group_name} group with forward fill"
        )

        # Convert to DataFrame for easier ffill operation
        group_df = pd.DataFrame(group_data, columns=group_cols)
        # Forward fill first, then backward fill for any remaining NaN
        group_df = group_df.ffill(axis=1).bfill(axis=1)

        return group_df.values

    def _apply_pca_to_group(
        self,
        group_data: np.ndarray,
        group_cols: List[str],
        group_name: str,
    ) -> Tuple[np.ndarray, PCA]:
        """Apply PCA to a feature group with configured variance retention"""
        # Fill NaN values
        group_data_clean = self._fill_nan_values(group_data, group_cols, group_name)

        # Apply PCA with configured variance retention
        pca = PCA(n_components=self.pca_variance_retention, random_state=42)
        pca_data = pca.fit_transform(group_data_clean)

        return pca_data, pca

    def load_features_data(self, year: int = 2000) -> pd.DataFrame:
        """Load aggregated features data for the specified year from intermediate directory"""
        features_df = self._load_features_data(year)
        features_df = self._scale_ndvi(features_df)
        return features_df

    def prepare_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature columns for ML training with group-wise PCA and scaling"""
        features_df = features_df.copy()

        # Apply group-wise PCA
        features_df = self._apply_group_pca(features_df)

        # Get feature columns (excluding metadata columns)
        feature_columns = [
            col
            for col in features_df.columns
            if col not in ["country", "admin_level_1", "admin_level_2"]
        ]

        # Remove rows with any missing values
        features_subset = features_df.dropna()
        self.feature_columns = feature_columns

        logger.info(
            f"Prepared {len(feature_columns)} features (after group-wise PCA) for {len(features_subset)} administrative units"
        )
        return pd.DataFrame(features_subset)

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
        return pd.DataFrame(targets_df)

    def _normalize_probability_distribution(self, y: np.ndarray) -> np.ndarray:
        """Normalize array to sum to 1 (probability distribution)"""
        y = y / np.sum(y, axis=1, keepdims=True)
        return y

    def _apply_threshold_and_normalize(
        self, y: np.ndarray, threshold: float = 0.01
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
        """Train a single KNN model"""
        # Normalize targets to ensure they sum to 1
        y = self._normalize_probability_distribution(y)

        # Split data for training and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # Train KNN with MultiOutputRegressor
        knn = KNeighborsRegressor(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            algorithm="auto",
            p=self.p,
            n_jobs=-1,
        )

        model = MultiOutputRegressor(knn)
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
        """Train KNN models for planted and harvested distributions"""
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

        # The features are already processed with PCA, so we can use them directly

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
