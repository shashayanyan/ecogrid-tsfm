import pytest
import pandas as pd
from pathlib import Path
from src.data_loader import EnergyDataLoader

def test_load_and_clean(tmp_path: Path):
    # 1. Setup: Create a temporary mock CSV file
    mock_csv = tmp_path / "mock_data.csv"
    mock_data = """timestamp,building_1,building_2
2016-01-01 00:00:00,100.5,200.1
2016-01-01 01:00:00,105.2,198.5
2016-01-01 02:00:00,,199.0
2016-01-01 03:00:00,110.0,205.0"""
    mock_csv.write_text(mock_data)

    # 2. Execution: Run our loader
    loader = EnergyDataLoader(mock_csv)
    df = loader.load_and_clean(target_building_id='building_1')

    # 3. Assertions: Verify the logic worked
    assert not df.empty, "DataFrame should not be empty"
    assert len(df) == 4, "Should have loaded all 4 rows"
    assert 'energy_kwh' in df.columns, "Target column should be renamed"
    
    # Check that our forward-fill logic handled the missing value at 02:00:00
    assert df.loc['2016-01-01 02:00:00', 'energy_kwh'] == 105.2