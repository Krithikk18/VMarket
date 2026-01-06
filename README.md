# 📈 VMarket - Volatility Prediction Platform

AI-powered stock volatility prediction using real-time news sentiment analysis.

## 🎯 Project Overview

This system predicts stock market volatility by analyzing:
- Real-time news sentiment (FinBERT)
- Historical volatility patterns (GARCH)
- Temporal patterns (LSTM)

## 🏢 Companies Tracked

- **AAPL** - Apple
- **NVDA** - NVIDIA  
- **MSFT** - Microsoft
- **GOOGL** - Google
- **META** - Meta
- **AMZN** - Amazon
- **NFLX** - Netflix

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## 📊 Tech Stack

- **Data Sources:** NewsAPI, Yahoo Finance
- **Models:** GARCH, FinBERT, LSTM
- **Framework:** PyTorch, Streamlit
- **ML Libraries:** scikit-learn, transformers, arch

## 📁 Project Structure

```
VMarket/
├── data/              # Stock data CSVs
├── models/            # Trained models
├── src/               # Source code
│   ├── data_pipeline.py
│   ├── sentiment_analyzer.py
│   ├── volatility_predictor.py
│   └── model_trainer.py
├── app.py             # Streamlit dashboard
└── README.md
```
