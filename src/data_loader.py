import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

class EnergyDataLoader:
    """
    A utility class to load and preprocess commercial building energy data.
    """
    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        self.df: Optional[pd.DataFrame] = None

    def load_and_clean(self, target_building_id: str) -> pd.DataFrame:
        """
        Loads the raw energy dataset, filters for a specific building, 
        and cleans the time series index.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found at {self.file_path}")

        # Load the dataset (Assuming the standard BDG2 format or similar)
        print(f"Loading data from {self.file_path}...")
        raw_df = pd.read_csv(self.file_path)

        # Basic standardizations
        raw_df.rename(columns={
            'timestamp': 'datetime', 
            target_building_id: 'energy_kwh'
        }, inplace=True, errors='ignore')

        # Filter down to just our datetime and target building
        if 'energy_kwh' not in raw_df.columns:
            raise ValueError(f"Building ID '{target_building_id}' not found in dataset.")
            
        df = raw_df[['datetime', 'energy_kwh']].copy()

        # Convert to proper datetime index
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)

        # Handle missing data (Forward fill short gaps, drop massive ones)
        df['energy_kwh'] = df['energy_kwh'].ffill(limit=3)
        df.dropna(inplace=True)

        self.df = df
        print(f"Loaded {len(self.df)} hourly records for building: {target_building_id}")
        return self.df

    def get_train_test_split(self, test_days: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Splits the data sequentially for time series validation."""
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load_and_clean() first.")
            
        split_point = self.df.index[-1] - pd.Timedelta(days=test_days)
        train = self.df[self.df.index <= split_point]
        test = self.df[self.df.index > split_point]
        return train, test