"""Coupled spatiotemporal matrix factorization for CCI cropland mask imputation using MatCoupLy"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import xarray as xr
from matcouply.decomposition import cmf_aoadmm
from sklearn.metrics import mean_squared_error, r2_score

logger = logging.getLogger(__name__)


class MatrixFactorizationConstants:
    """Constants for matrix factorization imputation"""

    # Low-rank dimensions
    SPATIAL_RANK = 50
    TEMPORAL_RANK = 20

    # MatCoupLy parameters
    MAX_ITER = 500
    TOL = 1e-6
    TEMPORAL_REG_WEIGHT = 5.0
    SPATIAL_REG_WEIGHT = 1.0
    COUPLING_WEIGHT = 10.0

    # HYDE-CCI overlap period (1992-2022)
    OVERLAP_YEARS = 5


class CoupledMatrixFactorization:
    """Coupled spatiotemporal matrix factorization using MatCoupLy for extending CCI backwards with HYDE"""

    def __init__(
        self,
        spatial_rank: int = MatrixFactorizationConstants.SPATIAL_RANK,
        temporal_rank: int = MatrixFactorizationConstants.TEMPORAL_RANK,
    ):
        """Initialize coupled matrix factorization model"""
        self.spatial_rank = spatial_rank
        self.temporal_rank = temporal_rank

        # MatCoupLy model
        self.model = None
        self.is_trained = False
        self.train_metrics: Dict[str, float] = {}

        # Data info
        self.n_pixels = None
        self.hyde_times = None
        self.cci_times = None
        self.overlap_start_idx = None

        logger.info(
            f"Initialized MatCoupLy coupled matrix factorization: spatial_rank={spatial_rank}, temporal_rank={temporal_rank}"
        )

    def fit(self, hyde_data: xr.DataArray, cci_data: xr.DataArray) -> Dict[str, float]:
        """
        Train coupled matrix factorization model

        Args:
            hyde_data: HYDE cropland data (lat, lon, time) - full 1979-2024 history
            cci_data: CCI cropland data (lat, lon, time) - 1992-2022 period
        """
        logger.info(
            f"Training on HYDE shape: {hyde_data.shape}, CCI shape: {cci_data.shape}"
        )

        # Reshape to matrix form (pixels, time)
        hyde_matrix = self._reshape_to_matrix(hyde_data)
        cci_matrix = self._reshape_to_matrix(cci_data)

        self.n_pixels = hyde_matrix.shape[0]
        self.hyde_times = hyde_data.time.values
        self.cci_times = cci_data.time.values

        # Find overlap period indices
        self.overlap_start_idx = self._find_overlap_start(
            self.hyde_times, self.cci_times
        )
        hyde_overlap = hyde_matrix[
            :, self.overlap_start_idx : self.overlap_start_idx + len(self.cci_times)
        ]

        logger.info(
            f"Overlap period: HYDE[{self.overlap_start_idx}:{self.overlap_start_idx + len(self.cci_times)}], CCI full"
        )

        # Use MatCoupLy CMF decomposition
        coupled_matrices = [hyde_overlap, cci_matrix]

        # Run coupled matrix factorization
        self.model = cmf_aoadmm(
            coupled_matrices, rank=self.spatial_rank, verbose=-1, random_state=42
        )

        # Extract results
        weights, (A, B_matrices, C) = self.model
        cci_reconstructed = weights[1] * (A @ B_matrices[1] @ C.T)
        self.train_metrics = {
            "mse": mean_squared_error(
                cci_matrix.flatten(), cci_reconstructed.flatten()
            ),
            "r2": r2_score(cci_matrix.flatten(), cci_reconstructed.flatten()),
            "n_pixels": self.n_pixels,
            "overlap_years": len(self.cci_times),
        }

        self.is_trained = True
        logger.info(
            f"Training complete - MSE: {self.train_metrics['mse']:.6f}, R²: {self.train_metrics['r2']:.6f}"
        )

        return self.train_metrics

    def predict_extended(self, hyde_data: xr.DataArray) -> xr.DataArray:
        """
        Predict fine-resolution CCI for full HYDE historical period

        Args:
            hyde_data: Full HYDE data (lat, lon, time) - 1979-2024

        Returns:
            Extended CCI data with same spatial/temporal dimensions as HYDE
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        # Reshape HYDE to matrix form
        hyde_matrix = self._reshape_to_matrix(hyde_data)

        # Use trained model to extend CCI backwards
        # Extract factorization components
        weights, (A, B_matrices, C) = self.model

        # Transform full HYDE matrix using learned components
        hyde_transformed = A @ hyde_matrix @ C.T

        # Apply fine matrix transformation (use B_matrices[1] for fine resolution)
        extended_cci_matrix = weights[1] * (hyde_transformed @ B_matrices[1].T)

        # Reshape back to spatial dimensions
        extended_cci = self._reshape_to_spatial(extended_cci_matrix, hyde_data)

        logger.info(f"Generated extended CCI time series: {extended_cci.shape}")
        return extended_cci

    def _reshape_to_matrix(self, data: xr.DataArray) -> np.ndarray:
        """Reshape (lat, lon, time) to (pixels, time) matrix"""
        return data.stack(pixel=("lat", "lon")).transpose("pixel", "time").values

    def _reshape_to_spatial(
        self, matrix: np.ndarray, reference: xr.DataArray
    ) -> xr.DataArray:
        """Reshape (pixels, time) matrix back to (lat, lon, time) spatial array"""
        n_pixels, n_times = matrix.shape
        n_lat, n_lon = len(reference.lat), len(reference.lon)

        # Reshape matrix to (lat, lon, time)
        spatial_array = matrix.reshape(n_lat, n_lon, n_times)

        # Create new DataArray with same coordinates as reference
        return xr.DataArray(
            spatial_array,
            coords={"lat": reference.lat, "lon": reference.lon, "time": reference.time},
            dims=["lat", "lon", "time"],
        )

    def _find_overlap_start(self, hyde_times: np.ndarray, cci_times: np.ndarray) -> int:
        """Find starting index of overlap period in HYDE time series"""
        cci_start_year = int(cci_times[0])

        for i, hyde_time in enumerate(hyde_times):
            hyde_year = hyde_time.year
            if hyde_year == cci_start_year:
                return i

        raise ValueError(
            f"CCI start year {cci_start_year} not found in HYDE time series"
        )


class MatrixFactorizationProcessor:
    """Memory-efficient processor for matrix factorization imputation"""

    def __init__(self, config):
        """Initialize processor with configuration"""
        self.config = config
        self.model = CoupledMatrixFactorization()

    def train_and_predict(
        self, hyde_data: xr.DataArray, cci_data: xr.DataArray, missing_years: List[int]
    ) -> Dict[int, xr.DataArray]:
        """
        Train model on overlap period and predict for missing years

        Args:
            hyde_data: Full HYDE time series
            cci_data: Available CCI data
            missing_years: Years to predict

        Returns:
            Dictionary mapping year -> predicted CCI data
        """
        # Train on available data
        self.model.fit(hyde_data, cci_data)

        # Generate extended time series
        extended_cci = self.model.predict_extended(hyde_data)

        # Extract predictions for missing years
        predictions = {}
        hyde_years = [t.year for t in hyde_data.time.values]

        for year in missing_years:
            try:
                year_idx = hyde_years.index(year)
                predictions[year] = extended_cci.isel(time=year_idx)
                logger.debug(f"Extracted prediction for year {year}")
            except ValueError:
                logger.warning(f"Year {year} not found in HYDE time series")

        return predictions
