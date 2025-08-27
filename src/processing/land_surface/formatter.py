"""Land surface data formatter"""

from src.processing.weather.formatter import WeatherFormatter


class LandSurfaceFormatter(WeatherFormatter):
    """Formatter for land surface data - reuses weather formatter logic"""

    def __init__(self):
        super().__init__()
        # Land surface data has same structure as weather data
        # (time series with admin boundaries), so we can reuse the formatter
