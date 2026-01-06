import pandas as pd

df = pd.read_csv('december_daily_predictions.csv')

print("="*70)
print("ENSEMBLE PREDICTIONS - DECEMBER 2025")
print("="*70)

for ticker in ['AAPL', 'NVDA', 'MSFT', 'GOOGL', 'META', 'AMZN', 'NFLX']:
    stock_data = df[df['Ticker'] == ticker]
    ensemble_vals = stock_data['Ensemble_Prediction']
    
    print(f"\n{ticker}:")
    print(f"  Average: {ensemble_vals.mean():.2%}")
    print(f"  Range: {ensemble_vals.min():.2%} → {ensemble_vals.max():.2%}")
    print(f"  Daily Variation: {ensemble_vals.max() - ensemble_vals.min():.2%}")
    print(f"  Trading Days: {len(stock_data)}")
