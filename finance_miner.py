import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Configuration
COMPANIES = [
    {"name": "Microsoft", "ticker": "MSFT", "repo": "microsoft/vscode"},
    {"name": "Google", "ticker": "GOOGL", "repo": "tensorflow/tensorflow"},
    {"name": "Meta", "ticker": "META", "repo": "facebook/react"},
    {"name": "Amazon", "ticker": "AMZN", "repo": "aws/aws-cdk"},
    {"name": "Apple", "ticker": "AAPL", "repo": "apple/swift"},
    {"name": "NVIDIA", "ticker": "NVDA", "repo": "NVIDIA/NeMo"},
    {"name": "Netflix", "ticker": "NFLX", "repo": "Netflix/conductor"},
    {"name": "Alibaba", "ticker": "BABA", "repo": "ant-design/ant-design"},
    {"name": "Airbnb", "ticker": "ABNB", "repo": "airbnb/lottie-android"},
    {"name": "Uber", "ticker": "UBER", "repo": "uber/kepler.gl"},
    {"name": "Shopify", "ticker": "SHOP", "repo": "Shopify/hydrogen"},
]

# 5 Years ago
START_DATE = (datetime.now() - timedelta(days=1825)).strftime('%Y-%m-%d')
# Tomorrow (to ensure we get today's data if market is closed)
END_DATE = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

def fetch_stock_data(ticker):
    print(f"Fetching stock data for {ticker} from {START_DATE} to {END_DATE}...")
    try:
        # Use explicit dates
        data = yf.download(ticker, start=START_DATE, end=END_DATE)
        
        # HACK: Fetch last 60 days separately to ensure recent data is complete
        # (Fixes mysterious gap in Nov/Dec)
        recent_start = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        print(f"Fetching recent data from {recent_start}...")
        recent_data = yf.download(ticker, start=recent_start, end=END_DATE)
        print(f"Recent data rows: {len(recent_data)}")
        if not recent_data.empty:
            print(recent_data.tail())
        
        if not data.empty:
            # Handle Multi-Level Columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            if not recent_data.empty:
                if isinstance(recent_data.columns, pd.MultiIndex):
                    recent_data.columns = recent_data.columns.get_level_values(0)
                
                # Merge
                data = pd.concat([data, recent_data])
                data = data[~data.index.duplicated(keep='last')]
                data.sort_index(inplace=True)

            # 3. Clean and Save
            data.reset_index(inplace=True)
            
            filename = f"{ticker}_stock_data.csv"
            data.to_csv(filename, index=False)
            print(f"Success! Stock data saved to {filename}")
            print(data.tail(20)) # DEBUG
        else:
            print(f"Warning: No data found for {ticker}")
            
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")

if __name__ == "__main__":
    for company in COMPANIES:
        fetch_stock_data(company['ticker'])