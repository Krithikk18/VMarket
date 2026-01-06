"""
Complete Backtesting Suite
Validates GARCH and LSTM model accuracy on recent data
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import sys

sys.path.append('src')
from garch_model import VolatilityForecaster
from lstm_model import LSTMVolatilityPredictor
from volatility_calculator import prepare_stock_data

print("="*80)
print("VMARKET - COMPREHENSIVE BACKTESTING SUITE")
print("="*80)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

TICKERS = ['AAPL', 'NVDA', 'MSFT', 'GOOGL', 'META', 'AMZN', 'NFLX']

# Store results
all_results = []

for ticker in TICKERS:
    print(f"\n{'='*80}")
    print(f"BACKTESTING: {ticker}")
    print(f"{'='*80}")
    
    # Load data
    stock_file = f"{ticker}_stock_data.csv"
    training_file = f"{ticker}_training_data.csv"
    model_file = f"models/{ticker}_lstm.pth"
    
    if not os.path.exists(stock_file):
        print(f"⚠️  Stock data not found, skipping...")
        continue
    
    # Load stock data
    _, stock_df = prepare_stock_data(stock_file)
    
    # Use last 30 days as test set (Nov data)
    test_df = stock_df.iloc[-30:]
    train_df = stock_df.iloc[:-30]
    
    print(f"Training data: {len(train_df)} days")
    print(f"Test data: {len(test_df)} days")
    
    # === GARCH BACKTESTING ===
    print(f"\n{'─'*80}")
    print("GARCH Model Backtesting")
    print(f"{'─'*80}")
    
    try:
        garch_forecaster = VolatilityForecaster()
        garch_forecaster.train_model(ticker, train_df)
        
        # Get prediction for last day
        garch_pred = garch_forecaster.predict_volatility(ticker, horizon=1)
        garch_predicted = garch_pred['predicted_volatility']
        
        # Actual volatility (last day in test)
        actual_vol = test_df['Vol_GK'].iloc[-1]
        
        # Calculate metrics
        garch_error = abs(garch_predicted - actual_vol)
        garch_error_pct = (garch_error / actual_vol) * 100
        garch_direction = 1 if garch_predicted > train_df['Vol_GK'].iloc[-1] else -1
        actual_direction = 1 if actual_vol > train_df['Vol_GK'].iloc[-1] else -1
        garch_dir_correct = (garch_direction == actual_direction)
        
        print(f"✓ GARCH Results:")
        print(f"  Predicted: {garch_predicted:.2%}")
        print(f"  Actual:    {actual_vol:.2%}")
        print(f"  Error:     {garch_error:.2%} ({garch_error_pct:.1f}%)")
        print(f"  Direction: {'✓ Correct' if garch_dir_correct else '✗ Wrong'}")
        
    except Exception as e:
        print(f"✗ GARCH Error: {str(e)}")
        garch_predicted = None
        garch_error = None
        garch_dir_correct = None
    
    # === LSTM BACKTESTING ===
    print(f"\n{'─'*80}")
    print("LSTM Model Backtesting")
    print(f"{'─'*80}")
    
    if os.path.exists(training_file) and os.path.exists(model_file):
        try:
            # Load training data and model
            training_df = pd.read_csv(training_file, index_col=0, parse_dates=True)
            
            lstm_predictor = LSTMVolatilityPredictor(sequence_length=60, hidden_dim=128, num_layers=2)
            lstm_predictor.load_model(model_file)
            
            # Prepare test sequences
            X_test, y_test = lstm_predictor.prepare_data(training_df.iloc[-100:], target_col='Vol_GK')
            
            if len(X_test) > 0:
                # Predict
                lstm_predictions = lstm_predictor.predict(X_test)
                lstm_predicted = lstm_predictions[-1]  # Last prediction
                
                # Calculate metrics
                lstm_error = abs(lstm_predicted - actual_vol)
                lstm_error_pct = (lstm_error / actual_vol) * 100
                lstm_direction = 1 if lstm_predicted > training_df['Vol_GK'].iloc[-61] else -1
                lstm_dir_correct = (lstm_direction == actual_direction)
                
                # Overall accuracy on test set
                mae = np.mean(np.abs(lstm_predictions - y_test.flatten()))
                
                print(f"✓ LSTM Results:")
                print(f"  Predicted: {lstm_predicted:.2%}")
                print(f"  Actual:    {actual_vol:.2%}")
                print(f"  Error:     {lstm_error:.2%} ({lstm_error_pct:.1f}%)")
                print(f"  Direction: {'✓ Correct' if lstm_dir_correct else '✗ Wrong'}")
                print(f"  Test MAE:  {mae:.4f}")
            else:
                print(f"⚠️  Insufficient data for LSTM testing")
                lstm_predicted = None
                lstm_error = None
                lstm_dir_correct = None
                
        except Exception as e:
            print(f"✗ LSTM Error: {str(e)}")
            lstm_predicted = None
            lstm_error = None
            lstm_dir_correct = None
    else:
        print(f"⚠️  LSTM model or training data not found")
        lstm_predicted = None
        lstm_error = None
        lstm_dir_correct = None
    
    # === ENSEMBLE ===
    ensemble_pred = None
    ensemble_error = None
    ensemble_error_pct = None
    ensemble_dir_correct = None
    
    if garch_predicted is not None and lstm_predicted is not None:
        ensemble_pred = 0.4 * garch_predicted + 0.6 * lstm_predicted
        ensemble_error = abs(ensemble_pred - actual_vol)
        ensemble_error_pct = (ensemble_error / actual_vol) * 100
        ensemble_direction = 1 if ensemble_pred > train_df['Vol_GK'].iloc[-1] else -1
        ensemble_dir_correct = (ensemble_direction == actual_direction)
        
        print(f"\n{'─'*80}")
        print("Ensemble (40% GARCH + 60% LSTM)")
        print(f"{'─'*80}")
        print(f"✓ Ensemble Results:")
        print(f"  Predicted: {ensemble_pred:.2%}")
        print(f"  Actual:    {actual_vol:.2%}")
        print(f"  Error:     {ensemble_error:.2%} ({ensemble_error_pct:.1f}%)")
        print(f"  Direction: {'✓ Correct' if ensemble_dir_correct else '✗ Wrong'}")
    else:
        ensemble_pred = garch_predicted if garch_predicted is not None else lstm_predicted
        ensemble_error = garch_error if garch_error is not None else lstm_error
        ensemble_dir_correct = garch_dir_correct if garch_dir_correct is not None else lstm_dir_correct
    
    # Store results
    all_results.append({
        'Ticker': ticker,
        'Actual_Vol': actual_vol if 'actual_vol' in locals() else None,
        'GARCH_Pred': garch_predicted,
        'GARCH_Error_%': garch_error_pct if garch_error is not None else None,
        'GARCH_Dir': '✓' if garch_dir_correct else '✗' if garch_dir_correct is not None else '-',
        'LSTM_Pred': lstm_predicted,
        'LSTM_Error_%': lstm_error_pct if lstm_error is not None else None,
        'LSTM_Dir': '✓' if lstm_dir_correct else '✗' if lstm_dir_correct is not None else '-',
        'Ensemble_Pred': ensemble_pred,
        'Ensemble_Error_%': ensemble_error_pct if ensemble_error is not None else None,
        'Ensemble_Dir': '✓' if ensemble_dir_correct else '✗' if ensemble_dir_correct is not None else '-'
    })

# === SUMMARY REPORT ===
print(f"\n\n{'='*80}")
print("BACKTESTING SUMMARY - ALL STOCKS")
print(f"{'='*80}\n")

results_df = pd.DataFrame(all_results)
results_df.to_csv('backtest_results.csv', index=False)

print(results_df.to_string(index=False))

# Calculate overall metrics
print(f"\n{'='*80}")
print("OVERALL PERFORMANCE METRICS")
print(f"{'='*80}")

garch_dir_acc = results_df['GARCH_Dir'].value_counts().get('✓', 0) / len(results_df) * 100
lstm_dir_acc = results_df['LSTM_Dir'].value_counts().get('✓', 0) / len(results_df) * 100
ensemble_dir_acc = results_df['Ensemble_Dir'].value_counts().get('✓', 0) / len(results_df) * 100

print(f"\nDirection Accuracy (Predicting Up/Down):")
print(f"  GARCH:    {garch_dir_acc:.1f}%")
print(f"  LSTM:     {lstm_dir_acc:.1f}%")
print(f"  Ensemble: {ensemble_dir_acc:.1f}%")

avg_garch_error = results_df['GARCH_Error_%'].mean()
avg_lstm_error = results_df['LSTM_Error_%'].mean()
avg_ensemble_error = results_df['Ensemble_Error_%'].mean()

print(f"\nAverage Prediction Error:")
print(f"  GARCH:    {avg_garch_error:.1f}%")
print(f"  LSTM:     {avg_lstm_error:.1f}%")
print(f"  Ensemble: {avg_ensemble_error:.1f}%")

print(f"\n{'='*80}")
print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*80}")
print(f"\n✓ Results saved to: backtest_results.csv")
print(f"✓ Backtesting complete!")
