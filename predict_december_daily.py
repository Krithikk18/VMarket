"""
Daily December 2025 Predictions
Generate volatility predictions for each trading day in December
Using data through Nov 28 + daily sentiment updates
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

sys.path.append('src')
from garch_model import VolatilityForecaster
from lstm_model import LSTMVolatilityPredictor
from volatility_calculator import prepare_stock_data

TICKERS = ['AAPL', 'NVDA', 'MSFT', 'GOOGL', 'META', 'AMZN', 'NFLX']

print("="*80)
print("DAILY DECEMBER 2025 PREDICTIONS")
print("="*80)
print("Generating predictions for each trading day in December")
print("Using data through Nov 28, 2025 + daily sentiment updates")
print("="*80)

# Get December trading days
december_dates = pd.date_range(start='2025-12-01', end='2025-12-31', freq='B')  # Business days

all_daily_predictions = []

for ticker in TICKERS:
    print(f"\n{'='*80}")
    print(f"DAILY PREDICTIONS FOR {ticker}")
    print(f"{'='*80}")
    
    # Load data
    stock_file = f"{ticker}_stock_data.csv"
    training_file = f"{ticker}_training_data.csv"
    sentiment_file = f"{ticker}_daily_sentiment.csv"
    model_file = f"models/{ticker}_lstm.pth"
    
    if not os.path.exists(stock_file):
        print(f"⚠️  Stock data not found")
        continue
    
    # Load stock data through Nov 28
    _, stock_df = prepare_stock_data(stock_file)
    
    # Train GARCH once
    forecaster = VolatilityForecaster()
    forecaster.train_model(ticker, stock_df)
    
    # Load LSTM model
    lstm_available = False
    if os.path.exists(training_file) and os.path.exists(model_file):
        try:
            predictor = LSTMVolatilityPredictor(sequence_length=60, hidden_dim=128, num_layers=2)
            predictor.load_model(model_file)
            training_df = pd.read_csv(training_file, index_col=0, parse_dates=True)
            lstm_available = True
        except Exception as e:
            print(f"⚠️  LSTM not available: {str(e)}")
    
    # Load sentiment data
    sentiment_df = None
    if os.path.exists(sentiment_file):
        sentiment_df = pd.read_csv(sentiment_file)
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        sentiment_df.set_index('date', inplace=True)
    
    print(f"\nGenerating daily predictions...")
    print(f"{'Date':<12} {'GARCH':<10} {'LSTM':<10} {'Ensemble':<10} {'Sentiment':<12}")
    print("-" * 80)
    
    for pred_date in december_dates:
        # GARCH prediction (same for all days since we're not updating with Dec data)
        garch_pred = forecaster.predict_volatility(ticker, horizon=1)
        garch_vol = garch_pred['predicted_volatility']
        
        # LSTM prediction with daily sentiment
        lstm_vol = None
        sentiment_score = 0.0
        
        if lstm_available and sentiment_df is not None:
            # Get sentiment for this date if available
            if pred_date in sentiment_df.index:
                day_sentiment = sentiment_df.loc[pred_date]
                sentiment_score = day_sentiment.get('sentiment_mean', 0.0)
                
                try:
                    # Create synthetic row with Nov 28 prices + this day's sentiment
                    last_row = training_df.iloc[-1:].copy()
                    last_row.index = [pred_date]
                    
                    # Update sentiment columns
                    for col in ['sentiment_mean', 'sentiment_std', 'news_count', 
                               'sentiment_intensity', 'sentiment_ma_3']:
                        if col in day_sentiment.index:
                            last_row[col] = day_sentiment[col]
                    
                    # Prepare sequence
                    extended_df = pd.concat([training_df, last_row])
                    X, _ = predictor.prepare_data(extended_df.iloc[-100:], target_col='Vol_GK')
                    
                    if len(X) > 0:
                        lstm_pred = predictor.predict(X[-1:])
                        lstm_vol = lstm_pred[0]
                except:
                    pass
        
        # Ensemble
        if garch_vol is not None and lstm_vol is not None:
            ensemble_vol = 0.4 * garch_vol + 0.6 * lstm_vol
        else:
            ensemble_vol = garch_vol if garch_vol is not None else lstm_vol
        
        # Display
        date_str = pred_date.strftime('%b %d')
        garch_str = f"{garch_vol:.2%}" if garch_vol else "N/A"
        lstm_str = f"{lstm_vol:.2%}" if lstm_vol else "N/A"
        ensemble_str = f"{ensemble_vol:.2%}" if ensemble_vol else "N/A"
        sentiment_str = f"{sentiment_score:+.3f}"
        
        print(f"{date_str:<12} {garch_str:<10} {lstm_str:<10} {ensemble_str:<10} {sentiment_str:<12}")
        
        # Store
        all_daily_predictions.append({
            'Ticker': ticker,
            'Date': pred_date.strftime('%Y-%m-%d'),
            'DayOfWeek': pred_date.strftime('%A'),
            'GARCH_Prediction': garch_vol,
            'LSTM_Prediction': lstm_vol,
            'Ensemble_Prediction': ensemble_vol,
            'Sentiment': sentiment_score
        })

# Save all predictions
print(f"\n{'='*80}")
print("SAVING PREDICTIONS")
print(f"{'='*80}")

predictions_df = pd.DataFrame(all_daily_predictions)
predictions_df.to_csv('december_daily_predictions.csv', index=False)

print(f"\n✓ Daily predictions saved to: december_daily_predictions.csv")
print(f"\nTotal predictions: {len(predictions_df)} (across {len(TICKERS)} stocks)")
print(f"Trading days in December: {len(december_dates)}")

print(f"\n{'='*80}")
print("VALIDATION INSTRUCTIONS")
print(f"{'='*80}")
print(f"\n1. Fetch actual December 2025 stock data for each stock")
print(f"2. Calculate actual daily volatility for each date")
print(f"3. Compare predicted vs actual for each day")
print(f"4. Calculate daily accuracy metrics")
print(f"\nThese predictions use:")
print(f"  ✓ Stock prices: Through Nov 28, 2025 ONLY")
print(f"  ✓ Sentiment: Daily updates for December")
print(f"  ✓ Models: Trained on data through Nov 28")
print(f"\n🎯 True daily forward predictions!")
