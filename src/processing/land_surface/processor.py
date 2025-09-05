"""Land surface data processor"""

import logging
from pathlib import Path
from typing import List

from src.processing.base.processor import BaseProcessor
from src.processing.base.spatial_aggregator import SpatialAggregator
from src.processing.base.temporal_aggregator import TemporalAggregator
from src.processing.land_surface.config import LandSurfaceConfig
from src.processing.land_surface.formatter import LandSurfaceFormatter
from src.processing.base.tiff_converter import TiffConverter

logger = logging.getLogger(__name__)


class LandSurfaceProcessor(BaseProcessor):
    """Land surface processor - follows same pattern as weather but with TIF files"""

    def __init__(self, config: LandSurfaceConfig):
        super().__init__(config.country, config.admin_level, config.data_dir)
        self.config = config
        self.spatial_aggregator = SpatialAggregator(
            self, config.country, cropland_filter=True
        )
        self.temporal_aggregator = TemporalAggregator()
        self.formatter = LandSurfaceFormatter()

        # Set up logging
        log_level = logging.DEBUG if config.debug else logging.INFO
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def process(self) -> List[Path]:
        """Process land surface data - same pattern as weather"""
        logger.info(
            f"Processing land surface data for {self.country_full_name} ({self.config.start_year}-{self.config.end_year})"
        )

        self.config.validate()

        # Get directories
        land_surface_dir = self.config.get_land_surface_directory()
        processed_dir = self.data_dir / self.country_full_name.lower() / "processed"
        ls_processed_dir = processed_dir / "land_surface"
        ls_processed_dir.mkdir(parents=True, exist_ok=True)

        # Initialize TIF converter
        tiff_converter = TiffConverter(ls_processed_dir)

        # Get variables to process
        variables_to_process = self.config.variables
        if not variables_to_process:
            # Auto-detect from TIF files
            tiff_files = list(land_surface_dir.glob("*.tif"))
            variables_to_process = []
            for tiff_file in tiff_files:
                parts = tiff_file.stem.split("_")
                year_idx = next(
                    i for i, p in enumerate(parts) if p.isdigit() and len(p) == 4
                )
                weekly_idx = parts.index("weekly") if "weekly" in parts else len(parts)
                variable = "_".join(parts[year_idx + 1 : weekly_idx])
                if variable not in variables_to_process:
                    variables_to_process.append(variable)

            logger.info(f"Auto-detected variables: {variables_to_process}")

        output_files = []

        # Process each variable
        for variable in variables_to_process:
            logger.info(f"Processing variable: {variable}")

            # Step 1: Convert TIF files to weekly NetCDF
            weekly_nc_files = tiff_converter.process_all_tiffs(
                land_surface_dir=land_surface_dir,
                year_range=(self.config.start_year, self.config.end_year),
                variables=[variable],
            )

            # Step 2: Combine annual files (already weekly, no temporal aggregation needed)
            combined_ds = self.combine_annual_files_in_memory(weekly_nc_files)

            # Step 3: Spatial aggregation to admin boundaries
            aggregated_df = self.spatial_aggregator.aggregate_dataset(combined_ds)

            # Step 4: Format and save output
            aggregated_df["variable"] = variable

            pivoted_df = self.formatter.pivot_to_final_format(
                aggregated_df, self.config.admin_level
            )

            # Save output file
            filename = f"land_surface_{self.config.start_year}-{self.config.end_year-1}_{variable}_weekly_weighted_admin{self.config.admin_level}.{self.config.output_format}"
            output_file = self.save_output(
                pivoted_df, filename, self.config.output_format, processed_dir
            )

            output_files.append(output_file)
            logger.debug(f"Completed processing for {variable}: {output_file}")

            # Log coverage statistics
            self._log_coverage_statistics(pivoted_df, variable)

        logger.debug(f"Land surface processing complete. Output files: {output_files}")
        return output_files

    def _log_coverage_statistics(self, df, variable: str):
        """Log admin unit coverage statistics"""

        # Calculate expected years
        expected_years = self.config.end_year - self.config.start_year

        # Count records per admin pair
        # Note: After formatting, columns are renamed from admin_level_X_name to admin_level_X
        admin_cols = [f"admin_level_{i}" for i in range(1, self.config.admin_level + 1)]
        records_per_admin = df.groupby(admin_cols).size()

        # Count complete vs incomplete coverage
        complete_coverage = (records_per_admin == expected_years).sum()
        total_admin_pairs = len(records_per_admin)
        coverage_pct = complete_coverage / total_admin_pairs * 100

        # Log concise coverage summary
        incomplete_pairs = records_per_admin[records_per_admin < expected_years]
        if len(incomplete_pairs) > 0:
            logger.info(
                f"Coverage for {variable}: {complete_coverage}/{total_admin_pairs} admin pairs ({coverage_pct:.1f}%) have complete {expected_years}-year coverage"
            )
        else:
            logger.info(
                f"Coverage for {variable}: All {total_admin_pairs} admin pairs have complete {expected_years}-year coverage"
            )
