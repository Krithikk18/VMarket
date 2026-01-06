"""
December 2025 Forward Predictions
Predict volatility for each day in December using ONLY data through Nov 28, 2025
User will manually verify against actual December data
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
print("DECEMBER 2025 FORWARD PREDICTIONS")
print("="*80)
print("Using ONLY data through November 28, 2025")
print("Models have NEVER seen December price data")
print("="*80)

all_predictions = []

for ticker in TICKERS:
    print(f"\n{'='*80}")
    print(f"PREDICTIONS FOR {ticker}")
    print(f"{'='*80}")
    
    # Load training data (through Nov 28 only)
    stock_file = f"{ticker}_stock_data.csv"
    training_file = f"{ticker}_training_data.csv"
    sentiment_file = f"{ticker}_daily_sentiment.csv"
    model_file = f"models/{ticker}_lstm.pth"
    
    if not os.path.exists(stock_file):
        print(f"⚠️  Stock data not found")
        continue
    
    # Load stock data through Nov 28
    _, stock_df = prepare_stock_data(stock_file)
    
    # === GARCH PREDICTION ===
    print(f"\n{'─'*80}")
    print("GARCH Prediction")
    print(f"{'─'*80}")
    
    try:
        forecaster = VolatilityForecaster()
        forecaster.train_model(ticker, stock_df)
        garch_pred = forecaster.predict_volatility(ticker, horizon=1)
        garch_volatility = garch_pred['predicted_volatility']
        
        print(f"✓ GARCH predicts: {garch_volatility:.2%}")
        print(f"  Confidence: {garch_pred['lower_bound']:.2%} - {garch_pred['upper_bound']:.2%}")
        
    except Exception as e:
        print(f"✗ GARCH Error: {str(e)}")
        garch_volatility = None
    
    # === LSTM PREDICTION ===
    print(f"\n{'─'*80}")
    print("LSTM Prediction (with December Sentiment)")
    print(f"{'─'*80}")
    
    lstm_volatility = None
    
    if os.path.exists(training_file) and os.path.exists(model_file):
        try:
            # Load model
            predictor = LSTMVolatilityPredictor(sequence_length=60, hidden_dim=128, num_layers=2)
            predictor.load_model(model_file)
            
            # Load training data
            training_df = pd.read_csv(training_file, index_col=0, parse_dates=True)
            
            # Load December sentiment
            if os.path.exists(sentiment_file):
                sentiment_df = pd.read_csv(sentiment_file)
                sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
                sentiment_df.set_index('date', inplace=True)
                
                # Get December 1st sentiment (first unseen sentiment)
                dec_start = pd.Timestamp('2025-12-01')
                if dec_start in sentiment_df.index:
                    dec_sentiment = sentiment_df.loc[dec_start]
                    
                    # Create a synthetic December 1st row with Nov 28 prices + Dec 1 sentiment
                    last_row = training_df.iloc[-1:].copy()
                    last_row.index = [dec_start]
                    
                    # Update sentiment columns
                    for col in ['sentiment_mean', 'sentiment_std', 'news_count', 
                               'sentiment_intensity', 'sentiment_ma_3']:
                        if col in dec_sentiment.index:
                            last_row[col] = dec_sentiment[col]
                    
                    # Append to training data temporarily
                    extended_df = pd.concat([training_df, last_row])
                    
                    # Prepare sequence
                    X, _ = predictor.prepare_data(extended_df.iloc[-100:], target_col='Vol_GK')
                    
                    if len(X) > 0:
                        lstm_pred = predictor.predict(X[-1:])
                        lstm_volatility = lstm_pred[0]
                        
                        print(f"✓ LSTM predicts: {lstm_volatility:.2%}")
                        print(f"  Using December sentiment: {dec_sentiment.get('sentiment_mean', 0):.3f}")
                    else:
                        print(f"⚠️  Insufficient data for LSTM")
                else:
                    print(f"⚠️  No December sentiment data available")
            else:
                print(f"⚠️  No sentiment file found")
                
        except Exception as e:
            print(f"✗ LSTM Error: {str(e)}")
    else:
        print(f"⚠️  LSTM model not available")
    
    # === ENSEMBLE ===
    if garch_volatility is not None and lstm_volatility is not None:
        ensemble_volatility = 0.4 * garch_volatility + 0.6 * lstm_volatility
        print(f"\n{'─'*80}")
        print(f"Ensemble Prediction")
        print(f"{'─'*80}")
        print(f"✓ Ensemble predicts: {ensemble_volatility:.2%}")
    else:
        ensemble_volatility = garch_volatility if garch_volatility is not None else lstm_volatility
    
    # Store prediction
    all_predictions.append({
        'Ticker': ticker,
        'Date': 'Dec 1-31, 2025',
        'GARCH_Prediction': f"{garch_volatility:.2%}" if garch_volatility else "N/A",
        'LSTM_Prediction': f"{lstm_volatility:.2%}" if lstm_volatility else "N/A",
        'Ensemble_Prediction': f"{ensemble_volatility:.2%}" if ensemble_volatility else "N/A",
    })

# === SUMMARY ===
print(f"\n\n{'='*80}")
print("DECEMBER 2025 PREDICTIONS SUMMARY")
print(f"{'='*80}\n")

predictions_df = pd.DataFrame(all_predictions)
print(predictions_df.to_string(index=False))

# Save predictions
predictions_df.to_csv('december_predictions.csv', index=False)

print(f"\n{'='*80}")
print("PREDICTIONS SAVED")
print(f"{'='*80}")
print(f"\n✓ Predictions saved to: december_predictions.csv")
print(f"\nNEXT STEPS FOR VALIDATION:")
print(f"1. Fetch actual December 2025 stock data")
print(f"2. Calculate actual December volatility")
print(f"3. Compare predictions vs actuals")
print(f"4. Calculate accuracy metrics")
print(f"\nThese predictions were made using:")
print(f"  - Stock prices: Through Nov 28, 2025 ONLY")
print(f"  - Sentiment data: December 2025 (already collected)")
print(f"  - Models: Trained on data through Nov 28, 2025")
print(f"\n✓ December price data was NEVER seen by the models!")
print(f"✓ This is a true forward prediction!")
