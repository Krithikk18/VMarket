"""
GARCH Volatility Model
Implements GARCH(1,1) for baseline volatility prediction
"""

import pandas as pd
import numpy as np
from arch import arch_model
from typing import Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')


class GARCHPredictor:
    """GARCH(1,1) model for volatility prediction"""
    
    def __init__(self, p: int = 1, q: int = 1):
        """
        Initialize GARCH model
        
        Args:
            p: GARCH lag order
            q: ARCH lag order
        """
        self.p = p
        self.q = q
        self.model = None
        self.fitted_model = None
    
    def fit(self, returns: pd.Series, verbose: bool = False) -> None:
        """
        Fit GARCH model to returns data
        
        Args:
            returns: Return series (percentage returns * 100)
            verbose: Print fitting output
        """
        # Scale returns to percentage
        returns_scaled = returns * 100
        returns_scaled = returns_scaled.dropna()
        
        # Create and fit model
        self.model = arch_model(
            returns_scaled,
            vol='Garch',
            p=self.p,
            q=self.q,
            dist='normal'
        )
        
        disp = 'iter' if verbose else 'off'
        self.fitted_model = self.model.fit(disp=disp, show_warning=False)
    
    def predict(self, horizon: int = 1) -> Tuple[float, float, float]:
        """
        Predict volatility for next N days
        
        Args:
            horizon: Prediction horizon in days
            
        Returns:
            Tuple of (mean_forecast, lower_bound, upper_bound)
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get forecast
        forecast = self.fitted_model.forecast(horizon=horizon)
        
        # Extract variance forecast and convert to volatility
        variance = forecast.variance.values[-1, :]
        volatility = np.sqrt(variance) * np.sqrt(252) / 100  # Annualized
        
        mean_vol = volatility[0] if horizon == 1 else volatility.mean()
        
        # Confidence bounds (simple approach)
        std_err = volatility.std() if horizon > 1 else mean_vol * 0.1
        lower = max(0, mean_vol - 1.96 * std_err)
        upper = mean_vol + 1.96 * std_err
        
        return mean_vol, lower, upper
    
    def get_conditional_volatility(self) -> pd.Series:
        """
        Get fitted conditional volatility series
        
        Returns:
            Conditional volatility series
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted.")
        
        cond_vol = self.fitted_model.conditional_volatility
        # Convert to annualized volatility
        return cond_vol * np.sqrt(252) / 100


class VolatilityForecaster:
    """Complete volatility forecasting system"""
    
    def __init__(self):
        self.models = {}  # Store models per ticker
        self.predictions = {}
    
    def train_model(self, ticker: str, stock_df: pd.DataFrame) -> None:
        """
        Train GARCH model for a specific ticker
        
        Args:
            ticker: Stock ticker symbol
            stock_df: DataFrame with 'Close' prices
        """
        print(f"Training GARCH model for {ticker}...")
        
        # Calculate returns
        returns = stock_df['Close'].pct_change().dropna()
        
        # Fit model
        garch = GARCHPredictor(p=1, q=1)
        garch.fit(returns, verbose=False)
        
        self.models[ticker] = garch
        print(f"  Model trained on {len(returns)} observations")
    
    def predict_volatility(self, ticker: str, horizon: int = 1) -> Dict:
        """
        Predict volatility for a ticker
        
        Args:
            ticker: Stock ticker
            horizon: Days ahead to predict
            
        Returns:
            Dictionary with prediction results
        """
        if ticker not in self.models:
            raise ValueError(f"No model found for {ticker}")
        
        garch = self.models[ticker]
        mean_vol, lower, upper = garch.predict(horizon)
        
        result = {
            'ticker': ticker,
            'horizon': horizon,
            'predicted_volatility': mean_vol,
            'lower_bound': lower,
            'upper_bound': upper,
            'confidence_interval': upper - lower
        }
        
        self.predictions[ticker] = result
        return result
    
    def get_historical_fit(self, ticker: str) -> pd.Series:
        """
        Get historical fitted volatility
        
        Args:
            ticker: Stock ticker
            
        Returns:
            Historical volatility series
        """
        if ticker not in self.models:
            raise ValueError(f"No model found for {ticker}")
        
        return self.models[ticker].get_conditional_volatility()


if __name__ == "__main__":
    # Test GARCH model
    import os
    
    test_file = "AAPL_stock_data.csv"
    if os.path.exists(test_file):
        print(f"Testing GARCH model with {test_file}\n")
        
        # Load data
        df = pd.read_csv(test_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # Train model
        forecaster = VolatilityForecaster()
        forecaster.train_model('AAPL', df)
        
        # Make prediction
        prediction = forecaster.predict_volatility('AAPL', horizon=1)
        
        print(f"\n{'='*50}")
        print(f"VOLATILITY PREDICTION FOR AAPL")
        print(f"{'='*50}")
        print(f"Predicted Volatility: {prediction['predicted_volatility']:.2%}")
        print(f"95% Confidence Interval: [{prediction['lower_bound']:.2%}, {prediction['upper_bound']:.2%}]")
        print(f"{'='*50}")
        
        # Get historical fit
        hist_vol = forecaster.get_historical_fit('AAPL')
        print(f"\nHistorical volatility (last 10 days):")
        print(hist_vol.tail(10))
    else:
        print(f"File {test_file} not found")
