"""ML imputation for missing crop calendar data using land surface features"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
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
        self.country = country
        self.admin_level = admin_level
        self.data_dir = data_dir
        self.scaler = StandardScaler()
        self.models = {}
        self.pca_models = {}

        # Load config
        config = ML_IMPUTATION_CONFIG
        self.pca_variance_retention = config["pca_variance_retention"]
        self.n_neighbors = config["n_neighbors"]
        self.weights = config["weights"]
        self.p = config["p"]
        self.test_size = config["test_size"]
        self.random_state = config["random_state"]

    def _load_and_merge_features(self, year: int) -> pd.DataFrame:
        """Load and merge all feature files for the specified year"""
        features_dir = (
            self.data_dir / self.country.lower() / "intermediate" / "aggregated"
        )

        if not features_dir.exists():
            raise FileNotFoundError(
                f"Intermediate aggregated directory not found: {features_dir}"
            )

        # Define feature files to load
        feature_files = {
            # Land surface data (with year)
            f"land_surface_ndvi_weekly_weighted_admin{self.admin_level}.csv": True,
            f"land_surface_lai_low_weekly_weighted_admin{self.admin_level}.csv": True,
            f"land_surface_lai_high_weekly_weighted_admin{self.admin_level}.csv": True,
            # Weather data (with year)
            f"weather_t2m_min_weekly_weighted_admin{self.admin_level}.csv": True,
            f"weather_t2m_max_weekly_weighted_admin{self.admin_level}.csv": True,
            f"weather_precipitation_weekly_weighted_admin{self.admin_level}.csv": True,
            f"weather_vapour_pressure_weekly_weighted_admin{self.admin_level}.csv": True,
            f"weather_snow_lwe_weekly_weighted_admin{self.admin_level}.csv": True,
            f"weather_reference_et_weekly_weighted_admin{self.admin_level}.csv": True,
            f"weather_wind_speed_weekly_weighted_admin{self.admin_level}.csv": True,
            f"weather_solar_radiation_weekly_weighted_admin{self.admin_level}.csv": True,
            # Soil data (no year)
            "soil_bulk_density_weighted_admin2.csv": False,
            "soil_cec_weighted_admin2.csv": False,
            "soil_clay_weighted_admin2.csv": False,
            "soil_coarse_fragments_weighted_admin2.csv": False,
            "soil_nitrogen_weighted_admin2.csv": False,
            "soil_organic_carbon_weighted_admin2.csv": False,
            "soil_organic_carbon_density_weighted_admin2.csv": False,
            "soil_ph_h2o_weighted_admin2.csv": False,
            "soil_sand_weighted_admin2.csv": False,
            "soil_silt_weighted_admin2.csv": False,
        }

        dataframes = []
        for file_pattern, has_year in feature_files.items():
            file_path = features_dir / file_pattern
            if file_path.exists():
                df = pd.read_csv(file_path)
                if has_year:
                    df = df.loc[df["year"] == year].copy()
                    if df.empty:
                        raise ValueError(
                            f"No data found for year {year} in {file_path}"
                        )
                dataframes.append(df)
                logger.info(f"Loaded {file_path.name}: {len(df)} records")

        if not dataframes:
            raise ValueError(f"No feature files found for year {year}")

        # Merge all dataframes
        merge_cols = [
            "country",
            "admin_level_1",
            "admin_level_2",
            "latitude",
            "longitude",
        ]
        features_df = dataframes[0]

        for df in dataframes[1:]:
            # Use common columns present in both dataframes
            common_cols = [
                col
                for col in merge_cols
                if col in features_df.columns and col in df.columns
            ]
            if "year" in features_df.columns and "year" in df.columns:
                common_cols.append("year")
            features_df = features_df.merge(df, on=common_cols, how="inner")

        # Scale NDVI values
        ndvi_cols = [col for col in features_df.columns if col.startswith("ndvi_week_")]
        features_df[ndvi_cols] = features_df[ndvi_cols] / 10000.0

        logger.info(f"Loaded {len(features_df)} administrative units for year {year}")
        return features_df

    def _apply_group_pca(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Apply group-wise PCA to reduce dimensionality"""
        features_df = features_df.copy()

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

        pca_features = ["latitude", "longitude"]  # Keep coordinates as raw features

        for group_name, prefixes in feature_groups.items():
            # Find all columns for this group
            group_cols = []
            for prefix in prefixes:
                group_cols.extend(
                    [col for col in features_df.columns if col.startswith(f"{prefix}_")]
                )

            if not group_cols:
                continue

            # Apply PCA to the group
            group_data = features_df[group_cols].values

            # Fill NaN values with forward/backward fill
            if np.isnan(group_data).any():
                group_df = pd.DataFrame(group_data, columns=group_cols)
                group_data = group_df.ffill(axis=1).bfill(axis=1).values

            # Apply PCA
            pca = PCA(
                n_components=self.pca_variance_retention, random_state=self.random_state
            )
            pca_data = pca.fit_transform(group_data)
            self.pca_models[group_name] = pca

            # Add PCA components to dataframe
            n_components = pca_data.shape[1]
            new_cols = [f"{group_name}_pca_{i+1}" for i in range(n_components)]
            pca_df = pd.DataFrame(pca_data, columns=new_cols, index=features_df.index)
            features_df = pd.concat([features_df, pca_df], axis=1)
            pca_features.extend(new_cols)

            logger.info(
                f"Group {group_name}: {len(group_cols)} -> {n_components} PCA components"
            )

        # Return only PCA features and admin identifiers
        keep_cols = pca_features + ["country", "admin_level_1", "admin_level_2"]
        return features_df[[col for col in keep_cols if col in features_df.columns]]

    def _prepare_data(
        self, features_df: pd.DataFrame, targets_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare and split data for training"""
        # Rename features columns to match targets
        features_renamed = features_df.rename(
            columns={
                "country": "country_name",
                "admin_level_1": "admin_level_1_name",
                "admin_level_2": "admin_level_2_name",
            }
        )

        # Merge features and targets
        merged_df = features_renamed.merge(
            targets_df,
            on=["country_name", "admin_level_1_name", "admin_level_2_name"],
            how="inner",
        )

        if merged_df.empty:
            raise ValueError("No matching data between features and targets")

        # Prepare feature matrix
        feature_cols = [
            col
            for col in merged_df.columns
            if col not in ["country_name", "admin_level_1_name", "admin_level_2_name"]
            and "month_" not in col
        ]

        X = merged_df[feature_cols].values

        # Prepare target matrices
        planted_cols = [f"planted_month_{month}" for month in range(1, 13)]
        harvested_cols = [f"harvested_month_{month}" for month in range(1, 13)]

        y_planted = merged_df[planted_cols].values.astype(np.float64)
        y_harvested = merged_df[harvested_cols].values.astype(np.float64)

        logger.info(
            f"Prepared data: {len(merged_df)} samples, {len(feature_cols)} features"
        )
        return X, y_planted, y_harvested, merged_df

    def _normalize_predictions(self, y: np.ndarray) -> np.ndarray:
        """Normalize predictions to sum to 1 and apply threshold"""
        y = np.maximum(y, 0)  # Ensure non-negative
        y[y < 0.01] = 0.0  # Apply threshold
        row_sums = np.sum(y, axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid division by zero
        return y / row_sums

    def _train_model(
        self, X: np.ndarray, y: np.ndarray, target_type: str
    ) -> MultiOutputRegressor:
        """Train a single KNN model with train-test split"""
        # Normalize targets
        y = self._normalize_predictions(y)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        knn = KNeighborsRegressor(
            n_neighbors=self.n_neighbors, weights=self.weights, p=self.p, n_jobs=-1
        )
        model = MultiOutputRegressor(knn)
        model.fit(X_train_scaled, y_train)

        # Evaluate on test set
        y_pred = model.predict(X_test_scaled)
        y_pred = self._normalize_predictions(y_pred)

        # Calculate R² metrics
        mse_scores = [np.mean((y_test[:, i] - y_pred[:, i]) ** 2) for i in range(12)]
        r2_scores = [1 - mse for mse in mse_scores]
        avg_r2 = np.mean(r2_scores)

        logger.info(
            f"Model {target_type}: Average R²={avg_r2:.4f} (monthly R²: {[f'{r2:.3f}' for r2 in r2_scores]})"
        )
        return model

    def train_models(
        self, features_df: pd.DataFrame, targets_df: pd.DataFrame
    ) -> Dict[str, MultiOutputRegressor]:
        """Train models for planted and harvested distributions"""
        X, y_planted, y_harvested, _ = self._prepare_data(features_df, targets_df)

        models = {}
        for target_type, y in [("planted", y_planted), ("harvested", y_harvested)]:
            models[target_type] = self._train_model(X, y, target_type)

        self.models = models
        return models

    def predict_missing(
        self, features_df: pd.DataFrame, crop_calendar_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Predict crop calendar data for units with total_area=0"""
        if not self.models:
            raise ValueError("Models not trained. Call train_models() first.")

        # Find units with no crop data
        no_crop_units = crop_calendar_df[crop_calendar_df["total_area"] == 0.0].copy()
        logger.info(f"Found {len(no_crop_units)} units with no crop data")

        if no_crop_units.empty:
            return crop_calendar_df

        # Rename features columns
        features_renamed = features_df.rename(
            columns={
                "country": "country_name",
                "admin_level_1": "admin_level_1_name",
                "admin_level_2": "admin_level_2_name",
            }
        )

        # Create identifier for matching
        def create_id(df):
            return df["admin_level_1_name"] + "_" + df["admin_level_2_name"]

        no_crop_ids = create_id(no_crop_units)
        features_ids = create_id(features_renamed)
        matching_ids = no_crop_ids[no_crop_ids.isin(features_ids)]

        if matching_ids.empty:
            logger.info("No matching features found for units with zero area")
            return crop_calendar_df

        # Get features for prediction
        matching_features = features_renamed[features_ids.isin(matching_ids)]
        feature_cols = [
            col
            for col in matching_features.columns
            if col not in ["country_name", "admin_level_1_name", "admin_level_2_name"]
        ]

        X_missing = matching_features[feature_cols].values
        X_missing_scaled = self.scaler.transform(X_missing)

        # Make predictions
        predicted_df = no_crop_units[no_crop_ids.isin(matching_ids)].copy()

        for target_type, model in self.models.items():
            pred = model.predict(X_missing_scaled)
            pred = self._normalize_predictions(pred)

            for month in range(1, 13):
                predicted_df[f"{target_type}_month_{month}"] = pred[:, month - 1]

        # Update original dataframe
        matching_mask = (crop_calendar_df["total_area"] == 0.0) & create_id(
            crop_calendar_df
        ).isin(matching_ids)

        crop_calendar_df.loc[matching_mask, predicted_df.columns] = predicted_df

        logger.info(f"Imputed crop calendar data for {len(predicted_df)} units")
        return crop_calendar_df

    def impute_crop_calendar(
        self, crop_calendar_path: Path, year: int = 2000
    ) -> pd.DataFrame:
        """Complete pipeline for crop calendar imputation"""
        logger.info(f"Starting ML imputation for: {crop_calendar_path}")

        # Load data
        crop_calendar_df = pd.read_csv(crop_calendar_path)
        features_df = self._load_and_merge_features(year)
        features_df = self._apply_group_pca(features_df)

        # Prepare targets
        target_columns = ["country_name", "admin_level_1_name", "admin_level_2_name"]
        target_columns.extend(
            [
                f"{target_type}_month_{month}"
                for target_type in ["planted", "harvested"]
                for month in range(1, 13)
            ]
        )

        targets_df = crop_calendar_df[target_columns].copy()
        monthly_cols = [col for col in targets_df.columns if "month_" in col]
        targets_df = targets_df[targets_df[monthly_cols].sum(axis=1) > 0]

        # Train and predict
        self.train_models(features_df, targets_df)
        imputed_df = self.predict_missing(features_df, crop_calendar_df)

        logger.info(f"ML imputation complete. Total records: {len(imputed_df)}")
        return imputed_df
