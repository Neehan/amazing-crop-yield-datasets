"""Simple ML imputation for missing crop calendar data"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.decomposition import PCA

from src.processing.crop_calendar.config import ML_IMPUTATION_CONFIG, TEST_SIZE

logger = logging.getLogger(__name__)


class CropCalendarMLImputation:
    """Simple ML-based imputation for missing crop calendar data"""

    def __init__(self, country: str, admin_level: int, data_dir: Path):
        self.country = country
        self.admin_level = admin_level
        self.data_dir = data_dir
        self.scaler = StandardScaler()
        self.planted_model = None
        self.harvested_model = None
        self.pca_models = {}
        self.feature_columns = None

        # Load config
        self.pca_variance = ML_IMPUTATION_CONFIG["pca_variance_retention"]
        self.n_neighbors = ML_IMPUTATION_CONFIG["n_neighbors"]
        self.weights = ML_IMPUTATION_CONFIG["weights"]

    def load_features(self, year: int = 2000) -> pd.DataFrame:
        """Load all feature data from intermediate directories"""
        weather_df = self._load_weather_data(year)
        land_surface_df = self._load_land_surface_data(year)
        soil_df = self._load_soil_data()

        # Merge: weather has lat/lon, others don't need them
        features_df = weather_df.merge(
            land_surface_df.drop(columns=["latitude", "longitude"]),
            on=["country_name", "admin_level_1_name", "admin_level_2_name"],
            how="inner",
        ).merge(
            soil_df.drop(columns=["latitude", "longitude"]),
            on=["country_name", "admin_level_1_name", "admin_level_2_name"],
            how="inner",
        )

        # Apply PCA
        features_df = self._apply_pca(features_df)

        logger.info(f"Loaded features for {len(features_df)} admin units")
        return features_df

    def _load_weather_data(self, year: int) -> pd.DataFrame:
        """Load weather data from intermediate/weather"""
        weather_dir = self.data_dir / self.country.lower() / "intermediate" / "weather"

        weather_vars = ["t2m_min", "t2m_max", "precipitation"]
        weather_dfs = []

        for var in weather_vars:
            file_path = (
                weather_dir
                / f"{year}_{var}_weekly_weighted_admin{self.admin_level}.csv"
            )
            df = pd.read_csv(file_path)
            df["week"] = pd.to_datetime(df["time"]).dt.isocalendar().week

            # Pivot to wide format with lat/lon in index
            pivot_df = df.pivot_table(
                index=[
                    "country_name",
                    "admin_level_1_name",
                    "admin_level_2_name",
                    "latitude",
                    "longitude",
                ],
                columns="week",
                values="value",
                aggfunc="mean",
            )
            pivot_df.columns = [f"{var}_week_{col}" for col in pivot_df.columns]
            pivot_df = pivot_df.reset_index()
            weather_dfs.append(pivot_df)

        # Merge weather variables, first has lat/lon, rest don't need them
        weather_df = weather_dfs[0]
        for df in weather_dfs[1:]:
            weather_df = weather_df.merge(
                df.drop(columns=["latitude", "longitude"]),
                on=["country_name", "admin_level_1_name", "admin_level_2_name"],
                how="inner",
            )

        return weather_df

    def _load_land_surface_data(self, year: int) -> pd.DataFrame:
        """Load land surface data from intermediate/weather"""
        weather_dir = self.data_dir / self.country.lower() / "intermediate" / "weather"

        land_vars = ["ndvi", "lai_low", "lai_high"]
        land_dfs = []

        for var in land_vars:
            file_path = (
                weather_dir
                / f"{year}_{var}_weekly_weighted_admin{self.admin_level}.csv"
            )
            df = pd.read_csv(file_path)
            df["week"] = pd.to_datetime(df["time"]).dt.isocalendar().week

            # Scale NDVI
            if var == "ndvi":
                df["value"] = df["value"] / 10000.0

            # Pivot to wide format with lat/lon in index
            pivot_df = df.pivot_table(
                index=[
                    "country_name",
                    "admin_level_1_name",
                    "admin_level_2_name",
                    "latitude",
                    "longitude",
                ],
                columns="week",
                values="value",
                aggfunc="mean",
            )
            pivot_df.columns = [f"{var}_week_{col}" for col in pivot_df.columns]
            pivot_df = pivot_df.reset_index()
            land_dfs.append(pivot_df)

        # Merge land surface variables, first has lat/lon, rest don't need them
        land_df = land_dfs[0]
        for df in land_dfs[1:]:
            land_df = land_df.merge(
                df.drop(columns=["latitude", "longitude"]),
                on=["country_name", "admin_level_1_name", "admin_level_2_name"],
                how="inner",
            )

        return land_df

    def _load_soil_data(self) -> pd.DataFrame:
        """Load soil data from intermediate/aggregated"""
        aggregated_dir = (
            self.data_dir / self.country.lower() / "intermediate" / "aggregated"
        )

        # All 10 soil variables
        soil_vars = [
            "bulk_density",
            "cec",
            "clay",
            "coarse_fragments",
            "nitrogen",
            "organic_carbon",
            "organic_carbon_density",
            "ph_h2o",
            "sand",
            "silt",
        ]

        soil_dfs = []
        for var in soil_vars:
            file_path = (
                aggregated_dir / f"soil_{var}_weighted_admin{self.admin_level}.csv"
            )
            if not file_path.exists():
                continue

            df = pd.read_csv(file_path)
            # Rename admin columns
            df = df.rename(
                columns={
                    "country": "country_name",
                    "admin_level_1": "admin_level_1_name",
                    "admin_level_2": "admin_level_2_name",
                }
            )

            # Keep admin + depth columns, rename depth columns
            depth_cols = [col for col in df.columns if col.endswith("cm")]
            keep_cols = [
                "country_name",
                "admin_level_1_name",
                "admin_level_2_name",
                "latitude",
                "longitude",
            ] + depth_cols
            df = df[keep_cols]

            for col in depth_cols:
                df = df.rename(columns={col: f"{var}_{col}"})

            soil_dfs.append(df)

        # Merge all soil variables, first has lat/lon, rest don't need them
        soil_df = soil_dfs[0]
        for df in soil_dfs[1:]:
            soil_df = soil_df.merge(
                df.drop(columns=["latitude", "longitude"]),
                on=["country_name", "admin_level_1_name", "admin_level_2_name"],
                how="inner",
            )

        return soil_df

    def _apply_pca(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Apply PCA to feature groups"""
        feature_groups = {
            "weather": ["t2m_min", "t2m_max", "precipitation"],
            "land_surface": ["ndvi", "lai_low", "lai_high"],
            "soil": [
                "bulk_density",
                "cec",
                "clay",
                "coarse_fragments",
                "nitrogen",
                "organic_carbon",
                "organic_carbon_density",
                "ph_h2o",
                "sand",
                "silt",
            ],
        }

        pca_features = []
        logger.info(f"Applying PCA to retain {self.pca_variance * 100:.2f}% variance")
        for group_name, prefixes in feature_groups.items():
            # Get all columns for this group
            group_cols = []
            for prefix in prefixes:
                group_cols.extend(
                    [col for col in features_df.columns if col.startswith(f"{prefix}_")]
                )

            if len(group_cols) < 2:
                pca_features.extend(group_cols)
                continue

            # Apply PCA
            group_data = features_df[group_cols].fillna(0).values
            pca = PCA(n_components=self.pca_variance, random_state=42)
            pca_data = pca.fit_transform(group_data)

            # Store PCA model
            self.pca_models[group_name] = (pca, group_cols)

            # Add PCA features
            pca_cols = [f"{group_name}_pca_{i+1}" for i in range(pca_data.shape[1])]
            pca_df = pd.DataFrame(pca_data, columns=pca_cols, index=features_df.index)
            features_df = pd.concat([features_df, pca_df], axis=1)
            pca_features.extend(pca_cols)

            logger.info(
                f"{group_name}: {len(group_cols)} -> {pca_data.shape[1]} components"
            )

        # Keep admin info, coordinates, and PCA features
        keep_cols = [
            "country_name",
            "admin_level_1_name",
            "admin_level_2_name",
            "latitude",
            "longitude",
        ] + pca_features
        return features_df[keep_cols]

    def prepare_data(
        self, features_df: pd.DataFrame, crop_calendar_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare feature matrix and target arrays"""
        crop_with_data = crop_calendar_df[crop_calendar_df["total_area"] > 0].copy()
        merged_df = features_df.merge(
            crop_with_data.drop(columns=["latitude", "longitude"]),
            on=["country_name", "admin_level_1_name", "admin_level_2_name"],
            how="inner",
        )

        # Get feature columns (exclude admin, target, and other metadata columns)
        exclude_cols = {
            "country_name",
            "admin_level_1_name",
            "admin_level_2_name",
            "crop_name",
            "total_area",
        }

        # Add target columns to exclusion
        for prefix in ["planted", "harvested"]:
            for month in range(1, 13):
                exclude_cols.add(f"{prefix}_month_{month}")

        feature_cols = [col for col in merged_df.columns if col not in exclude_cols]
        self.feature_columns = feature_cols
        X = merged_df[feature_cols].fillna(0).values

        # Get target columns
        target_cols = []
        for prefix in ["planted", "harvested"]:
            target_cols.extend([f"{prefix}_month_{month}" for month in range(1, 13)])

        y = merged_df[target_cols].fillna(0).values

        logger.info(
            f"Prepared {X.shape[0]} samples with {X.shape[1]} features and {y.shape[1]} targets"
        )
        return X, y

    def normalize_predictions(
        self, y: np.ndarray, threshold: float = 0.01
    ) -> np.ndarray:
        """Normalize predictions to sum to 1 and remove small values"""
        n_months = 12
        result = np.zeros_like(y)

        for i in range(0, y.shape[1], n_months):
            y_type = y[:, i : i + n_months]
            y_type[y_type < threshold] = 0.0
            row_sums = y_type.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            result[:, i : i + n_months] = y_type / row_sums

        return result

    def train_and_evaluate(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train separate models for planted and harvested with train/test split and show metrics"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=42
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Split targets into planted (0:12) and harvested (12:24)
        y_train_planted = y_train[:, :12]
        y_train_harvested = y_train[:, 12:]
        y_test_planted = y_test[:, :12]
        y_test_harvested = y_test[:, 12:]

        # Train separate KNN models
        knn_planted = KNeighborsRegressor(
            n_neighbors=self.n_neighbors, weights=self.weights, n_jobs=-1
        )
        knn_harvested = KNeighborsRegressor(
            n_neighbors=self.n_neighbors, weights=self.weights, n_jobs=-1
        )
        
        planted_model = MultiOutputRegressor(knn_planted)
        harvested_model = MultiOutputRegressor(knn_harvested)
        
        planted_model.fit(X_train_scaled, y_train_planted)
        harvested_model.fit(X_train_scaled, y_train_harvested)

        # Predict separately
        y_pred_planted = planted_model.predict(X_test_scaled)
        y_pred_harvested = harvested_model.predict(X_test_scaled)
        
        # Normalize predictions
        y_pred_planted = self.normalize_predictions(y_pred_planted)
        y_pred_harvested = self.normalize_predictions(y_pred_harvested)

        # Overall R² scores
        r2_planted = r2_score(y_test_planted, y_pred_planted, multioutput="uniform_average")
        r2_harvested = r2_score(y_test_harvested, y_pred_harvested, multioutput="uniform_average")

        logger.info(
            f"Test metrics - Planted R²: {r2_planted:.3f}, Harvested R²: {r2_harvested:.3f}"
        )

        # Month-over-month breakdown for planted
        planted_monthly = r2_score(y_test_planted, y_pred_planted, multioutput="raw_values")
        logger.info(
            "Planted month-over-month R²:\n *"
            + "\n *".join(
                f"  Month {month+1}: {planted_monthly[month]:.3f}"
                for month in range(12)
            )
        )

        # Month-over-month breakdown for harvested
        harvested_monthly = r2_score(y_test_harvested, y_pred_harvested, multioutput="raw_values")
        logger.info(
            "Harvested month-over-month R²:\n *"
            + "\n *".join(
                f"  Month {month+1}: {harvested_monthly[month]:.3f}"
                for month in range(12)
            )
        )

        # Train final models on full data
        X_full_scaled = self.scaler.fit_transform(X)
        y_planted_full = y[:, :12]
        y_harvested_full = y[:, 12:]
        
        self.planted_model = MultiOutputRegressor(knn_planted)
        self.harvested_model = MultiOutputRegressor(knn_harvested)
        
        self.planted_model.fit(X_full_scaled, y_planted_full)
        self.harvested_model.fit(X_full_scaled, y_harvested_full)

        logger.info("Trained separate models on full dataset")

    def predict_missing(
        self, features_df: pd.DataFrame, crop_calendar_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Predict crop calendar for units with total_area = 0"""
        no_crop_units = crop_calendar_df[crop_calendar_df["total_area"] == 0.0].copy()

        if no_crop_units.empty:
            logger.info("No units with zero area to impute")
            return crop_calendar_df

        missing_features = features_df.merge(
            no_crop_units[["country_name", "admin_level_1_name", "admin_level_2_name"]],
            on=["country_name", "admin_level_1_name", "admin_level_2_name"],
            how="inner",
        )

        if missing_features.empty:
            logger.info("No matching features for units with zero area")
            return crop_calendar_df

        X_missing = missing_features[self.feature_columns].fillna(0).values
        X_missing_scaled = self.scaler.transform(X_missing)

        # Predict with separate models
        y_pred_planted = self.planted_model.predict(X_missing_scaled)
        y_pred_harvested = self.harvested_model.predict(X_missing_scaled)
        
        # Normalize predictions separately
        y_pred_planted = self.normalize_predictions(y_pred_planted)
        y_pred_harvested = self.normalize_predictions(y_pred_harvested)

        # Update crop calendar dataframe
        result_df = crop_calendar_df.copy()

        for i, (_, row) in enumerate(missing_features.iterrows()):
            mask = (
                (result_df["country_name"] == row["country_name"])
                & (result_df["admin_level_1_name"] == row["admin_level_1_name"])
                & (result_df["admin_level_2_name"] == row["admin_level_2_name"])
            )

            # Update planted months
            for month in range(1, 13):
                result_df.loc[mask, f"planted_month_{month}"] = y_pred_planted[i, month - 1]
            
            # Update harvested months
            for month in range(1, 13):
                result_df.loc[mask, f"harvested_month_{month}"] = y_pred_harvested[i, month - 1]

        logger.info(f"Imputed crop calendar for {len(missing_features)} units")
        return result_df

    def impute_crop_calendar(
        self, crop_calendar_path: Path, year: int = 2000
    ) -> pd.DataFrame:
        """Complete imputation pipeline"""
        logger.info(f"Starting ML imputation for {crop_calendar_path}")

        crop_calendar_df = pd.read_csv(crop_calendar_path)
        features_df = self.load_features(year)

        X, y = self.prepare_data(features_df, crop_calendar_df)
        self.train_and_evaluate(X, y)
        result_df = self.predict_missing(features_df, crop_calendar_df)

        logger.info("ML imputation complete")
        return result_df
