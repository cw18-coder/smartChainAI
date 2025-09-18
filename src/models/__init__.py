"""
Models package for time series forecasting
"""

# Import key classes to make them available at package level
try:
    from .univariate import MSTLUnivariateForecaster
    from .tsutils import mstl_season_selector
    
    __all__ = [
        'MSTLUnivariateForecaster',
        'mstl_season_selector'
    ]
except ImportError as e:
    # Handle import errors gracefully during development
    print(f"Warning: Could not import all models components: {e}")
    __all__ = []