"""Base downloader class with flexible functionality for all data types"""

import logging
from pathlib import Path
from typing import List, Optional, Any, Dict, Union
from datetime import datetime
from tqdm import tqdm
import asyncio
from abc import ABC, abstractmethod

from src.utils.geography import Geography, GeoBounds

# Set up logger
logger = logging.getLogger(__name__)


class BaseDownloader(ABC):
    """Flexible base class for all downloaders with different API patterns"""

    def __init__(self, data_dir: str, country: str, subdir: str, max_concurrent: int):
        """Initialize downloader with data directory and concurrency settings

        Args:
            data_dir: Base directory to save downloaded files
            country: Country name for subdirectory (optional)
            subdir: Additional subdirectory (e.g., 'weather', 'soil')
            max_concurrent: Maximum concurrent downloads
        """
        self.data_dir = Path(data_dir)
        self.country = country
        self.subdir = Path(data_dir) / country / subdir
        self.geography = Geography()
        self.max_concurrent = max_concurrent

        self.subdir.mkdir(parents=True, exist_ok=True)

    # --- Abstract Methods ---
    @abstractmethod
    async def _handle_api_response(self, result: Any, **kwargs):
        """Handle API response - implemented by subclasses

        Args:
            result: API response object
            **kwargs: Original request parameters
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _build_request(self, **kwargs) -> Dict[str, Any]:
        """Build API request parameters - implemented by subclasses

        Args:
            **kwargs: Parameters specific to the request

        Returns:
            Dict with parameters for _make_api_request
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _make_api_request(self, **kwargs) -> Any:
        """Make the actual API request - implemented by subclasses

        Args:
            **kwargs: Request parameters from _build_request

        Returns:
            API response object
        """
        raise NotImplementedError("Subclasses must implement this method")

    # --- Utility Methods ---
    def get_country_bounds(self) -> Optional[GeoBounds]:
        """Get country bounds (utility method for subclasses)"""
        return self.geography.get_country_bounds(self.country)

    async def download(self, download_tasks: List[Dict[str, Any]], description: str):
        """Execute multiple download tasks concurrently with progress tracking

        Args:
            download_tasks: List of task dictionaries with download parameters
            description: Description for progress bar
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)
        progress_bar = tqdm(total=len(download_tasks), desc=description)

        tasks = []
        for task_params in download_tasks:
            task = self._download_with_progress(semaphore, progress_bar, task_params)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        progress_bar.close()
        return results

    async def _download_with_progress(self, semaphore, progress_bar, task_params):
        """Download single item with progress tracking"""
        try:
            success = await self._download_single_item(semaphore, **task_params)
            progress_bar.update(1)
            return success
        except Exception as e:
            logger.error(f"Download task failed: {e}")
            progress_bar.update(1)
            return False

    def check_file_exists(self, file_path: Union[str, Path]) -> bool:
        """Check if file exists (utility for subclasses)"""
        path = Path(file_path)
        if path.exists():
            logger.debug(f"File exists: {path}")
            return True
        return False

    async def _download_single_item(
        self, semaphore: asyncio.Semaphore, **kwargs
    ) -> bool:
        """Download a single item with semaphore control

        Args:
            semaphore: Async semaphore for concurrency control
            **kwargs: Parameters specific to the download task

        Returns:
            bool: Success status
        """
        async with semaphore:
            try:
                # Check if file already exists
                file_path = kwargs.get("file_path")
                if file_path and self.check_file_exists(file_path):
                    logger.debug(f"Skipping existing file: {file_path}")
                    return True

                # Build request parameters
                request_params = self._build_request(**kwargs)
                logger.debug(f"API Request: {request_params}")

                # Make API request in executor to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, lambda: self._make_api_request(**request_params)
                )

                # Handle the result (download, save, etc.)
                await self._handle_api_response(result, **kwargs)

                logger.debug(f"Successfully downloaded: {file_path}")
                return True

            except Exception as e:
                item_id = kwargs.get("item_id", "unknown")
                logger.error(f"Failed to download {item_id}: {e}")
                return False
