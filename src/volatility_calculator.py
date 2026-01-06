"""
Volatility Calculator Module
Calculates various volatility metrics from stock price data
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


class VolatilityCalculator:
    """Calculate different types of volatility from stock price data"""
    
    def __init__(self, window_size: int = 20):
        """
        Initialize volatility calculator
        
        Args:
            window_size: Rolling window for volatility calculations
        """
        self.window_size = window_size
    
    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate log returns from prices"""
        return np.log(prices / prices.shift(1))
    
    def close_to_close_volatility(self, df: pd.DataFrame) -> pd.Series:
        """
        Standard volatility using close prices
        
        Args:
            df: DataFrame with 'Close' column
            
        Returns:
            Annualized volatility series
        """
        returns = df['Close'].pct_change()
        volatility = returns.rolling(window=self.window_size).std() * np.sqrt(252)
        return volatility
    
    def parkinson_volatility(self, df: pd.DataFrame) -> pd.Series:
        """
        Parkinson's range-based volatility estimator
        More efficient than close-to-close for intraday data
        
        Formula: sqrt((1/(4*ln(2))) * ln(High/Low)^2)
        
        Args:
            df: DataFrame with 'High' and 'Low' columns
            
        Returns:
            Annualized volatility series
        """
        high_low_ratio = np.log(df['High'] / df['Low'])
        parkinson_vol = high_low_ratio ** 2
        parkinson_vol = parkinson_vol.rolling(window=self.window_size).mean()
        parkinson_vol = np.sqrt(parkinson_vol / (4 * np.log(2))) * np.sqrt(252)
        return parkinson_vol
    
    def garman_klass_volatility(self, df: pd.DataFrame) -> pd.Series:
        """
        Garman-Klass volatility estimator
        Uses OHLC data for better estimates
        
        Args:
            df: DataFrame with OHLC columns
            
        Returns:
            Annualized volatility series
        """
        high_low = np.log(df['High'] / df['Low']) ** 2
        close_open = np.log(df['Close'] / df['Open']) ** 2
        
        gk_vol = 0.5 * high_low - (2 * np.log(2) - 1) * close_open
        gk_vol = gk_vol.rolling(window=self.window_size).mean()
        gk_vol = np.sqrt(gk_vol) * np.sqrt(252)
        return gk_vol
    
    def realized_volatility(self, df: pd.DataFrame, method: str = 'garman_klass') -> pd.Series:
        """
        Calculate realized volatility (what actually happened)
        This is the TARGET variable for our prediction models
        
        Args:
            df: Stock price DataFrame
            method: 'close', 'parkinson', or 'garman_klass'
            
        Returns:
            Realized volatility series
        """
        if method == 'close':
            return self.close_to_close_volatility(df)
        elif method == 'parkinson':
            return self.parkinson_volatility(df)
        elif method == 'garman_klass':
            return self.garman_klass_volatility(df)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def intraday_range(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate intraday price range as percentage
        Simple volatility proxy
        
        Args:
            df: DataFrame with 'High', 'Low', 'Open' columns
            
        Returns:
            Percentage range series
        """
        return ((df['High'] - df['Low']) / df['Open']) * 100
    
    def calculate_all_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all volatility metrics at once
        
        Args:
            df: Stock price DataFrame with OHLC columns
            
        Returns:
            DataFrame with all volatility metrics
        """
        result = pd.DataFrame(index=df.index)
        
        # Returns
        result['Returns'] = df['Close'].pct_change()
        result['Log_Returns'] = self.calculate_returns(df['Close'])
        
        # Volatility measures
        result['Vol_Close'] = self.close_to_close_volatility(df)
        result['Vol_Parkinson'] = self.parkinson_volatility(df)
        result['Vol_GK'] = self.garman_klass_volatility(df)
        
        # Range-based proxy
        result['Intraday_Range'] = self.intraday_range(df)
        
        # Rolling statistics
        result['Vol_7D'] = result['Vol_GK'].rolling(7).mean()
        result['Vol_30D'] = result['Vol_GK'].rolling(30).mean()
        
        # Volatility of volatility
        result['Vol_of_Vol'] = result['Vol_GK'].rolling(20).std()
        
        return result


def prepare_stock_data(csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare stock data with volatility calculations
    
    Args:
        csv_path: Path to stock CSV file
        
    Returns:
        Tuple of (original_df, volatility_df)
    """
    # Load data
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    
    # Calculate volatility
    calculator = VolatilityCalculator(window_size=20)
    vol_df = calculator.calculate_all_metrics(df)
    
    # Merge
    result = pd.concat([df, vol_df], axis=1)
    
    return df, result


if __name__ == "__main__":
    # Test with one stock
    import os
    
    test_file = "AAPL_stock_data.csv"
    if os.path.exists(test_file):
        print(f"Testing volatility calculator with {test_file}")
        _, df = prepare_stock_data(test_file)
        print(f"\nCalculated {len(df)} days of data")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nLatest volatility metrics:")
        print(df[['Vol_Close', 'Vol_Parkinson', 'Vol_GK', 'Intraday_Range']].tail())
    else:
        print(f"File {test_file} not found")
