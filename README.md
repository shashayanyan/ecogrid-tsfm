# EcoGrid-TSFM: Smart Building Energy Forecasting & Fault Detection

[![CI/CD Pipeline](https://github.com/shashayanyan/ecogrid-tsfm/actions/workflows/test.yml/badge.svg)](https://github.com/shashayanyan/ecogrid-tsfm/actions)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Model: Amazon Chronos](https://img.shields.io/badge/Foundation_Model-Chronos_T5-orange)](https://github.com/amazon-science/chronos-forecasting)

## Executive Summary
EcoGrid-TSFM is an end-to-end, industrial-grade AI pipeline designed for commercial microgrids. It leverages **Time Series Foundation Models (TSFMs)** to deliver zero-shot probabilistic energy forecasts and automated anomaly detection, drastically reducing onboarding time for new building deployments.

Traditional deep learning models (LSTMs, TFTs) require weeks of historical data to fine-tune for *each specific building*. By treating time series data like language tokens, this project uses **Amazon Chronos** to accurately forecast energy loads and detect HVAC/VFD equipment failures without requiring any building-specific training data.

## Business Value
* **Peak Demand Shaving:** Forecasts the next 48 hours of energy usage with 90% confidence intervals, allowing microgrid battery systems to discharge during expensive peak-tariff hours.
* **Predictive Maintenance:** Automatically flags equipment faults (e.g., a Variable Frequency Drive failing to spin down at night) when real-time telemetry breaks the Foundation Model's zero-shot prediction bounds.
* **Rapid Industrialization:** Zero-shot deployment means zero cold-start problems for new commercial customers.

## Technical Architecture
* **Core AI:** PyTorch, Hugging Face `chronos-forecasting`
* **Data Engineering:** Object-Oriented Pandas (`EnergyDataLoader`)
* **Visual Analytics & Storytelling:** Plotly, Streamlit
* **Software Engineering:** Pytest (with Mocking), GitHub Actions CI/CD
* **Environment:** Fully container-ready and tested in isolated Python environments.

## Quick Start

### 1. Installation
Clone the repository and spin up the environment:
```bash
git clone [https://github.com/shashayanyan/ecogrid-tsfm.git](https://github.com/shashayanyan/ecogrid-tsfm.git)
cd ecogrid-tsfm
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Dashboard
```bash
streamlit run app.py
```

### 3. Run tests
```bash
pytest tests/ -v
```

## Repo Structure
```txt
├── .github/workflows/   # CI/CD pipelines (test.yml)
├── data/                # Raw commercial building datasets (BDG2)
├── notebooks/           # Jupyter notebooks for EDA and Data Storytelling
├── src/
│   ├── data_loader.py   # OOP data ingestion and cleaning
│   └── forecaster.py    # Hugging Face TSFM pipeline wrapper
├── tests/               # Pytest suite with mock tensors
├── app.py               # Streamlit interactive dashboard
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```