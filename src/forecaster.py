import torch
import pandas as pd
import numpy as np
from chronos import ChronosPipeline

class TSFMForecaster:
    """
    A wrapper for the Amazon Chronos Foundation Model to generate 
    zero-shot probabilistic forecasts.
    """
    def __init__(self, model_id: str = "amazon/chronos-t5-small"):
        """
        Initializes the Chronos pipeline. 
        Using 'small' for local development speed, but can easily be swapped 
        to 'base' or 'large' in production.
        """
        print(f"Loading Time Series Foundation Model: {model_id}...")
        self.pipeline = ChronosPipeline.from_pretrained(
            model_id,
            device_map="auto", # Automatically uses GPU if available, else CPU
            dtype=torch.bfloat16,
        )

    def predict(self, context_series: pd.Series, prediction_length: int = 48) -> pd.DataFrame:
        """
        Takes historical context and predicts the specified number of future steps.
        Returns the 10th, 50th (median), and 90th percentiles.
        """
        print(f"Generating {prediction_length}-hour forecast...")
        
        # Chronos expects a PyTorch tensor
        context_tensor = torch.tensor(context_series.values)
        
        # Generate 20 sample paths to create a probabilistic distribution
        forecast_samples = self.pipeline.predict(
            context_tensor,
            prediction_length=prediction_length,
            num_samples=20,
        )
        
        # Chronos returns shape (batch_size, num_samples, prediction_length)
        # We extract the 1st batch and calculate quantiles for our confidence intervals
        low, median, high = np.quantile(forecast_samples[0].numpy(), [0.1, 0.5, 0.9], axis=0)
        
        # Generate the future timestamps
        last_date = context_series.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(hours=1),
            periods=prediction_length,
            freq='h'
        )
        
        # Package into a clean DataFrame
        forecast_df = pd.DataFrame({
            'datetime': future_dates,
            'forecast_low_p10': low,
            'forecast_median': median,
            'forecast_high_p90': high
        }).set_index('datetime')
        
        return forecast_df