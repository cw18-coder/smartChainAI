#!/usr/bin/env python3
"""
Unit tests for MSTLUnivariateForecaster to debug initialization and prediction issues.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import pytest
import traceback

# Add the project root to Python path for direct script execution
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import from src package
from src.models import MSTLUnivariateForecaster


class TestMSTLUnivariateForecaster:
    """Test class for MSTLUnivariateForecaster"""
    
    @pytest.fixture
    def test_data(self):
        """Create test time series data similar to train.csv"""
        data = []
        
        # Create 3 time series with 14 daily observations each
        for series_id in [1, 2, 3]:
            base_value = series_id * 100
            for day in range(1, 15):  # 14 days
                date_str = f"2025-09-{day:02d}"
                value = base_value + day
                data.append({
                    'unique_id': series_id,
                    'ds': date_str,
                    'y': value
                })
        
        df = pd.DataFrame(data)
        df['ds'] = pd.to_datetime(df['ds'])
        return df
    
    @pytest.fixture
    def forecaster_params(self, test_data):
        """Common forecaster parameters"""
        min_ts_len = test_data.groupby('unique_id').size().min()
        return {
            'min_ts_len': min_ts_len,
            'freq': "D",
            'h': 7,
            'level': [80, 95]
        }
    
    def test_data_creation(self, test_data):
        """Test that test data is created correctly"""
        assert test_data.shape[0] == 42  # 3 series * 14 observations
        assert len(test_data['unique_id'].unique()) == 3
        assert test_data['ds'].dtype == 'datetime64[ns]'
        assert all(col in test_data.columns for col in ['unique_id', 'ds', 'y'])
        
        # Check minimum time series length
        min_ts_len = test_data.groupby('unique_id').size().min()
        assert min_ts_len == 14
        
        print(f"✅ Test data created successfully:")
        print(f"   Shape: {test_data.shape}")
        print(f"   Unique series: {test_data['unique_id'].unique()}")
        print(f"   Date range: {test_data['ds'].min()} to {test_data['ds'].max()}")
        print(f"   Min time series length: {min_ts_len}")
    
    def test_forecaster_initialization(self, forecaster_params):
        """Test MSTLUnivariateForecaster initialization"""
        print(f"✅ Testing forecaster initialization with params:")
        print(f"   {forecaster_params}")
        
        forecaster = MSTLUnivariateForecaster(**forecaster_params)
        
        # Check basic attributes
        assert forecaster.freq == forecaster_params['freq']
        assert forecaster.h == forecaster_params['h']
        assert forecaster.level == forecaster_params['level']
        assert hasattr(forecaster, 'season_lengths')
        assert hasattr(forecaster, 'models')
        
        print(f"✅ Forecaster initialized successfully!")
        print(f"   - freq: {forecaster.freq}")
        print(f"   - h: {forecaster.h}")
        print(f"   - level: {forecaster.level}")
        print(f"   - season_lengths: {forecaster.season_lengths}")
        print(f"   - models count: {len(forecaster.models)}")
    
    def test_forecaster_models(self, forecaster_params):
        """Test that MSTL models are properly initialized"""
        forecaster = MSTLUnivariateForecaster(**forecaster_params)
        
        # Check that models list is not empty
        assert len(forecaster.models) > 0
        
        print(f"✅ Checking {len(forecaster.models)} MSTL models:")
        
        # Check each model
        for i, model in enumerate(forecaster.models):
            print(f"   Model {i+1}: {model}")
            assert model is not None, f"Model {i+1} is None!"
            
            if hasattr(model, 'alias'):
                print(f"     - alias: {model.alias}")
            if hasattr(model, 'season_length'):
                print(f"     - season_length: {model.season_length}")
            
            # This is likely where the error occurs - check prediction_intervals
            if hasattr(model, 'prediction_intervals'):
                print(f"     - has prediction_intervals: {model.prediction_intervals}")
            else:
                print(f"     - no prediction_intervals attribute")
    
    def test_forecaster_prediction(self, test_data, forecaster_params):
        """Test MSTLUnivariateForecaster prediction"""
        forecaster = MSTLUnivariateForecaster(**forecaster_params)
        
        print(f"✅ Testing prediction...")
        
        forecasts = forecaster.predict(
            df=test_data,
            id_col='unique_id',
            time_col='ds',
            target_col='y'
        )
        
        # Check results
        assert isinstance(forecasts, list)
        assert len(forecasts) > 0
        
        print(f"✅ Prediction successful!")
        print(f"   Forecast results: {len(forecasts)} records")
        
        if forecasts:
            print(f"   Sample forecast record keys: {list(forecasts[0].keys())}")
            print(f"   First forecast: {forecasts[0]}")
    
    def test_debug_step_by_step(self, test_data, forecaster_params):
        """Debug step by step to find the exact issue"""
        print("\n" + "="*60)
        print("🔍 DEBUG: Step-by-step analysis")
        print("="*60)
        
        try:
            # Step 1: Test season length calculation
            from src.models import mstl_season_selector
            
            print("Step 1: Testing season length calculation...")
            season_lengths = mstl_season_selector(forecaster_params['min_ts_len'], forecaster_params['freq'])
            print(f"   Season lengths: {season_lengths}")
            assert season_lengths is not None
            
            # Step 2: Test individual model components
            print("\nStep 2: Testing individual model imports...")
            from statsforecast.models import AutoARIMA, AutoCES, AutoETS, AutoTheta, MSTL
            
            print("   AutoARIMA:", AutoARIMA())
            print("   AutoCES:", AutoCES())
            print("   AutoETS:", AutoETS())
            print("   AutoTheta:", AutoTheta())
            
            # Step 3: Test MSTL wrapper creation
            print("\nStep 3: Testing MSTL wrapper creation...")
            test_model = MSTL(
                season_length=season_lengths,
                trend_forecaster=AutoARIMA(),
                alias="TEST_MSTL_ARIMA"
            )
            print(f"   Test MSTL model: {test_model}")
            print(f"   Has prediction_intervals: {hasattr(test_model, 'prediction_intervals')}")
            
            if hasattr(test_model, 'prediction_intervals'):
                print(f"   prediction_intervals value: {test_model.prediction_intervals}")
            
            print("\n✅ All debug steps passed!")
            
        except Exception as e:
            print(f"\n❌ Debug step failed:")
            print(f"   Error: {e}")
            traceback.print_exc()
            raise


def test_manual_run():
    """Manual test function that can be run directly"""
    test_class = TestMSTLUnivariateForecaster()
    
    # Create test data
    test_data = []
    for series_id in [1, 2, 3]:
        base_value = series_id * 100
        for day in range(1, 15):
            date_str = f"2025-09-{day:02d}"
            value = base_value + day
            test_data.append({
                'unique_id': series_id,
                'ds': date_str,
                'y': value
            })
    
    df = pd.DataFrame(test_data)
    df['ds'] = pd.to_datetime(df['ds'])
    
    min_ts_len = df.groupby('unique_id').size().min()
    params = {
        'min_ts_len': min_ts_len,
        'freq': "D",
        'h': 7,
        'level': [80, 95]
    }
    
    # Run tests
    test_class.test_data_creation(df)
    test_class.test_forecaster_initialization(params)
    test_class.test_forecaster_models(params)
    test_class.test_debug_step_by_step(df, params)
    test_class.test_forecaster_prediction(df, params)


if __name__ == "__main__":
    test_manual_run()