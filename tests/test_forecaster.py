import pytest
import pandas as pd
import numpy as np
import torch
from unittest.mock import patch, MagicMock
from src.forecaster import TSFMForecaster

@patch('src.forecaster.ChronosPipeline.from_pretrained')
def test_forecaster_predict_shape(mock_from_pretrained):
    # 1. Setup: Mock the Chronos pipeline so it doesn't download the real model
    mock_pipeline_instance = MagicMock()
    
    # Create a dummy tensor of shape (batch_size=1, num_samples=20, prediction_length=48)
    dummy_forecast_tensor = torch.tensor(np.random.rand(1, 20, 48))
    mock_pipeline_instance.predict.return_value = dummy_forecast_tensor
    
    mock_from_pretrained.return_value = mock_pipeline_instance

    # 2. Execution
    forecaster = TSFMForecaster(model_id="dummy-model")
    
    # Create a fake 100-hour historical context
    dates = pd.date_range(start="2024-01-01", periods=100, freq='h')
    context_series = pd.Series(np.random.rand(100), index=dates)
    
    # Ask for a 48-hour prediction
    forecast_df = forecaster.predict(context_series, prediction_length=48)

    # 3. Assertions
    assert isinstance(forecast_df, pd.DataFrame), "Output must be a DataFrame"
    assert len(forecast_df) == 48, "Forecast should be exactly 48 steps long"
    assert all(col in forecast_df.columns for col in ['forecast_low_p10', 'forecast_median', 'forecast_high_p90'])
    
    # Verify the first predicted timestamp is exactly 1 hour after our context ended
    expected_start = pd.to_datetime("2024-01-05 04:00:00")
    assert forecast_df.index[0] == expected_start