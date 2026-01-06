"""
Fetch December 2025 Actual Stock Data
Independent verification - fetches fresh data and calculates daily volatility
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

TICKERS = ['AAPL', 'NVDA', 'MSFT', 'GOOGL', 'META', 'AMZN', 'NFLX']

print("="*80)
print("FETCHING DECEMBER 2025 ACTUAL DATA")
print("="*80)
print("Downloading real stock data from Yahoo Finance...")
print("="*80)

all_daily_volatilities = []

for ticker in TICKERS:
    print(f"\n{'='*80}")
    print(f"FETCHING {ticker}")
    print(f"{'='*80}")
    
    try:
        # Download December 2025 data
        stock = yf.Ticker(ticker)
        df = stock.history(start="2025-12-01", end="2025-12-31")
        
        if len(df) == 0:
            print(f"⚠️  No data available for {ticker}")
            continue
        
        # Save raw data
        df.to_csv(f"verification/{ticker}_december_raw.csv")
        print(f"✓ Downloaded {len(df)} days of data")
        
        # Calculate daily volatility using Garman-Klass method
        print(f"\nCalculating daily volatility...")
        
        # Garman-Klass volatility estimator
        df['GK_Volatility'] = np.sqrt(
            0.5 * (np.log(df['High'] / df['Low']))**2 - 
            (2 * np.log(2) - 1) * (np.log(df['Close'] / df['Open']))**2
        )
        
        # Annualize (252 trading days)
        df['GK_Volatility_Annualized'] = df['GK_Volatility'] * np.sqrt(252)
        
        # Also calculate simple close-to-close volatility
        df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['CC_Volatility'] = df['Returns'].rolling(window=5).std() * np.sqrt(252)
        
        # Create summary for each day
        for idx, row in df.iterrows():
            all_daily_volatilities.append({
                'Ticker': ticker,
                'Date': idx.strftime('%Y-%m-%d'),
                'DayOfWeek': idx.strftime('%A'),
                'Open': row['Open'],
                'High': row['High'],
                'Low': row['Low'],
                'Close': row['Close'],
                'Volume': row['Volume'],
                'GK_Volatility': row['GK_Volatility_Annualized'],
                'CC_Volatility': row['CC_Volatility'] if not pd.isna(row['CC_Volatility']) else None
            })
        
        # Display summary
        print(f"\n{'Date':<12} {'Open':<10} {'High':<10} {'Low':<10} {'Close':<10} {'Volatility':<12}")
        print("-" * 80)
        for idx, row in df.iterrows():
            date_str = idx.strftime('%b %d')
            vol_str = f"{row['GK_Volatility_Annualized']:.2%}" if not pd.isna(row['GK_Volatility_Annualized']) else "N/A"
            print(f"{date_str:<12} ${row['Open']:<9.2f} ${row['High']:<9.2f} ${row['Low']:<9.2f} ${row['Close']:<9.2f} {vol_str:<12}")
        
        avg_vol = df['GK_Volatility_Annualized'].mean()
        print(f"\n✓ Average December Volatility: {avg_vol:.2%}")
        
    except Exception as e:
        print(f"✗ Error fetching {ticker}: {str(e)}")

# Save all daily volatilities
print(f"\n{'='*80}")
print("SAVING RESULTS")
print(f"{'='*80}")

results_df = pd.DataFrame(all_daily_volatilities)
results_df.to_csv('verification/december_actual_volatilities.csv', index=False)

print(f"\n✓ All data saved to verification/december_actual_volatilities.csv")
print(f"  Total records: {len(results_df)}")
print(f"  Stocks: {results_df['Ticker'].nunique()}")
print(f"  Date range: {results_df['Date'].min()} to {results_df['Date'].max()}")

# Create summary by stock
print(f"\n{'='*80}")
print("DECEMBER 2025 VOLATILITY SUMMARY")
print(f"{'='*80}\n")

for ticker in TICKERS:
    ticker_data = results_df[results_df['Ticker'] == ticker]
    if len(ticker_data) > 0:
        avg_vol = ticker_data['GK_Volatility'].mean()
        min_vol = ticker_data['GK_Volatility'].min()
        max_vol = ticker_data['GK_Volatility'].max()
        days = len(ticker_data)
        print(f"{ticker:6s}: Avg={avg_vol:.2%}, Range={min_vol:.2%} to {max_vol:.2%}, Days={days}")

print(f"\n{'='*80}")
print("VERIFICATION DATA COMPLETE")
print(f"{'='*80}")
print(f"\nAll files saved in 'verification/' folder:")
print(f"  - {len(TICKERS)} stock raw data files (*_december_raw.csv)")
print(f"  - 1 combined daily volatility file (december_actual_volatilities.csv)")
print(f"\nYou can now manually compare these actual values")
print(f"with your predictions in december_daily_predictions.csv")
