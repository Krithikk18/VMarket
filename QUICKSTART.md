# VMarket - Quick Start Guide

## 🚀 How to Run

```bash
streamlit run app.py
```

## 📊 What You'll See

### Main Dashboard Features:
1. **TradingView-Style Candlestick Charts** - Professional OHLC visualization
2. **Real-Time Volatility Metrics** - Current vs historical comparisons
3. **AI Predictions** - GARCH model forecasts with confidence intervals
4. **Multi-Stock Heat Map** - Compare volatility across all 7 stocks
5. **Interactive Controls** - Time period selection, display options

### Available Stocks:
- AAPL (Apple)
- NVDA (NVIDIA)
- MSFT (Microsoft)
- GOOGL (Google)
- META (Meta)
- AMZN (Amazon)
- NFLX (Netflix)

## 📈 Features Implemented

### ✅ Phase 1: Core Data Pipeline
- Volatility calculator (Garman-Klass, Parkinson, Close-to-Close)
- News fetcher with NewsAPI integration
- FinBERT sentiment analyzer
- Data preprocessing pipeline

### ✅ Phase 2: Predictive Models  
- GARCH(1,1) volatility forecasting
- LSTM neural network (ready for training)
- Model ensemble system

### ✅ Phase 3: TradingView-Style Frontend
- Dark theme with gradient effects
- Interactive candlestick charts with volume
- Volatility analysis dashboard
- Real-time GARCH predictions
- Multi-stock volatility heat map
- Responsive metrics cards

## 🔧 Next Steps (To Complete the System)

### 1. Fetch News Data (Required for Sentiment)
```bash
# You'll need a NewsAPI key - Get free at: https://newsapi.org/
# Add to .env file: NEWS_API_KEY=your_key_here

python -c "from src.news_fetcher import save_news_data; import datetime; end = datetime.datetime.now().strftime('%Y-%m-%d'); start = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime('%Y-%m-%d'); save_news_data('AAPL', start, end)"
```

### 2. Run Sentiment Analysis
```bash
python -c "from src.sentiment_analyzer import process_news_sentiment; process_news_sentiment('AAPL_news.csv', 'AAPL_sentiment.csv')"
```

### 3. Train LSTM Model (Optional)
```bash
# Create training script and run for better predictions
python train_lstm.py
```

## 🎯 APIs Needed

### Essential:
1. **NewsAPI** (Free tier: 100 requests/day)
   - Sign up: https://newsapi.org/
   - Get API key
   - Add to `.env` as `NEWS_API_KEY=your_key`

### Current Status (Works Without API):
- ✅ Stock price data (already have 5 years of data)
- ✅ Volatility calculations (works on existing data)
- ✅ GARCH predictions (works on existing data)
- ⏳ News sentiment (needs API key for new data)

### Optional (For Production):
- Finnhub API (alternative news source)
- Alpha Vantage (real-time stock data)
- Twitter/Reddit API (social sentiment)

## 💡 Demo Strategy

###For Presentation (No API Needed):
1. Run `streamlit run app.py`
2. Show the beautiful interface
3. Demonstrate GARCH predictions (works immediately!)
4. Show historical volatility analysis
5. Explain the architecture

### With News API (Better):
1. Fetch last 30 days of news for all stocks
2. Run sentiment analysis
3. Train LSTM model
4. Show full integrated predictions

## ⚠️ Important Notes

- The dashboard works IMMEDIATELY with existing stock data
- GARCH predictions work without any APIs
- News sentiment requires NewsAPI key (free)
- LSTM needs training data (news + sentiment)

## 🎨 Visual Features

- Dark TradingView-inspired theme
- Gradient metric cards with hover effects
- Color-coded volatility (green = low, red = high)
- Interactive Plotly charts with zoom/pan
- Responsive layout for all screen sizes
- Professional typography and spacing

Enjoy your volatility prediction platform! 📈
