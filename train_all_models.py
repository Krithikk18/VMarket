"""
Complete Training Pipeline
1. Sentiment analysis with FinBERT (GPU)
2. Data merging
3. LSTM training
"""

import os
import sys
import pandas as pd
import torch
from datetime import datetime

# Force GPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print("="*70)
print("VMARKET - COMPLETE TRAINING PIPELINE")
print("="*70)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print("="*70)

# Configuration
TICKERS = ['AAPL', 'NVDA', 'MSFT', 'GOOGL', 'META', 'AMZN', 'NFLX']

# Import modules
sys.path.append('src')
from sentiment_analyzer import SentimentAnalyzer, process_news_sentiment
from volatility_calculator import prepare_stock_data
from lstm_model import LSTMVolatilityPredictor

print("\n" + "="*70)
print("PHASE 1: SENTIMENT ANALYSIS (FinBERT on GPU)")
print("="*70)

analyzer = SentimentAnalyzer()  # Will use GPU if available

for ticker in TICKERS:
    news_file = f"{ticker}_news.csv"
    sentiment_file = f"{ticker}_daily_sentiment.csv"
    
    if not os.path.exists(news_file):
        print(f"⚠️  {ticker}: News file not found, skipping...")
        continue
    
    if os.path.exists(sentiment_file):
        print(f"✓ {ticker}: Sentiment already processed, skipping...")
        continue
    
    print(f"\n{'─'*70}")
    print(f"Processing {ticker}...")
    print(f"{'─'*70}")
    
    # Process sentiment
    daily_sentiment = process_news_sentiment(news_file, sentiment_file)
    print(f"✓ {ticker}: Saved {len(daily_sentiment)} days of sentiment data")

print("\n" + "="*70)
print("PHASE 2: DATA MERGING (Stock + Sentiment)")
print("="*70)

for ticker in TICKERS:
    stock_file = f"{ticker}_stock_data.csv"
    sentiment_file = f"{ticker}_daily_sentiment.csv"
    training_file = f"{ticker}_training_data.csv"
    
    if not os.path.exists(stock_file):
        print(f"⚠️  {ticker}: Stock data not found, skipping...")
        continue
    
    if os.path.exists(training_file):
        print(f"✓ {ticker}: Training data already exists, skipping...")
        continue
    
    print(f"\nMerging data for {ticker}...")
    
    # Load stock data with volatility
    _, stock_df = prepare_stock_data(stock_file)
    
    # Load sentiment data if available
    if os.path.exists(sentiment_file):
        sentiment_df = pd.read_csv(sentiment_file)
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        sentiment_df.set_index('date', inplace=True)
        
        # Merge
        merged = stock_df.join(sentiment_df, how='left')
        
        # Forward fill sentiment (persist last known sentiment)
        sentiment_cols = ['sentiment_mean', 'sentiment_std', 'news_count', 
                         'sentiment_intensity', 'sentiment_ma_3']
        for col in sentiment_cols:
            if col in merged.columns:
                merged[col] = merged[col].fillna(method='ffill').fillna(0)
    else:
        print(f"  ⚠️  No sentiment data, using zeros...")
        merged = stock_df.copy()
        merged['sentiment_mean'] = 0
        merged['sentiment_std'] = 0
        merged['news_count'] = 0
        merged['sentiment_intensity'] = 0
        merged['sentiment_ma_3'] = 0
    
    # Save
    merged.to_csv(training_file)
    print(f"✓ {ticker}: Created training data with {len(merged)} rows")

print("\n" + "="*70)
print("PHASE 3: LSTM TRAINING (GPU Accelerated)")
print("="*70)

# Create models directory
os.makedirs('models', exist_ok=True)

training_results = {}

for ticker in TICKERS:
    training_file = f"{ticker}_training_data.csv"
    model_file = f"models/{ticker}_lstm.pth"
    
    if not os.path.exists(training_file):
        print(f"⚠️  {ticker}: No training data, skipping...")
        continue
    
    print(f"\n{'═'*70}")
    print(f"TRAINING LSTM FOR {ticker}")
    print(f"{'═'*70}")
    
    # Load training data
    df = pd.read_csv(training_file, index_col=0, parse_dates=True)
    
    # Initialize predictor
    predictor = LSTMVolatilityPredictor(
        sequence_length=60,
        hidden_dim=128,
        num_layers=2,
        learning_rate=0.001
    )
    
    # Prepare sequences
    print(f"Preparing sequences...")
    X, y = predictor.prepare_data(df, target_col='Vol_GK')
    print(f"✓ Created {len(X)} sequences with {X.shape[2]} features")
    
    # Train
    print(f"\nTraining (this will take a while on GPU)...")
    losses = predictor.train(X, y, epochs=100, batch_size=32, validation_split=0.2)
    
    # Save model
    predictor.save_model(model_file)
    
    training_results[ticker] = {
        'final_loss': losses[-1],
        'sequences': len(X),
        'features': X.shape[2]
    }
    
    print(f"\n✓ {ticker}: Training complete!")
    print(f"  Final Loss: {losses[-1]:.6f}")
    print(f"  Model saved to: {model_file}")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print("\nResults Summary:")
for ticker, results in training_results.items():
    print(f"{ticker:6s} - Loss: {results['final_loss']:.6f} | Sequences: {results['sequences']:4d} | Features: {results['features']}")

print(f"\n{'='*70}")
print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*70}")
print("\n✓ All models trained and saved to models/ directory")
print("✓ Ready to integrate with dashboard!")
