"""
Clean stock data - Remove December 2025 data for backtesting
This keeps Dec 2025 as "unseen future data" for testing model accuracy
"""

import pandas as pd
import os

# All stock files
TICKERS = ['AAPL', 'NVDA', 'MSFT', 'GOOGL', 'META', 'AMZN', 'NFLX']

# Cutoff date (keep everything before Dec 1, 2025)
CUTOFF_DATE = '2025-12-01'

print("=" * 60)
print("CLEANING STOCK DATA - REMOVING DEC 2025 ONWARDS")
print("=" * 60)
print(f"Cutoff date: {CUTOFF_DATE}")
print(f"This will keep data up to November 30, 2025\n")

for ticker in TICKERS:
    filename = f"{ticker}_stock_data.csv"
    
    if not os.path.exists(filename):
        print(f"⚠️  {filename} not found, skipping...")
        continue
    
    # Load data
    df = pd.read_csv(filename)
    df['Date'] = pd.to_datetime(df['Date'])
    
    original_rows = len(df)
    
    # Filter out December 2025 onwards
    df_cleaned = df[df['Date'] < CUTOFF_DATE]
    
    new_rows = len(df_cleaned)
    removed_rows = original_rows - new_rows
    
    # Save cleaned data
    df_cleaned.to_csv(filename, index=False)
    
    print(f"✓ {ticker:6s} - Removed {removed_rows:3d} rows | Kept {new_rows:4d} rows | Last date: {df_cleaned['Date'].max().strftime('%Y-%m-%d')}")

print("\n" + "=" * 60)
print("✅ DATA CLEANING COMPLETE")
print("=" * 60)
print("All stock data now ends on November 30, 2025")
print("December 2025 is reserved as your test period!")
