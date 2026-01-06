"""
LSTM Volatility Predictor
Deep learning model for volatility prediction using news sentiment and price data
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


class VolatilityLSTM(nn.Module):
    """LSTM neural network for volatility prediction"""
    
    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        """
        Initialize LSTM model
        
        Args:
            input_dim: Number of input features
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(VolatilityLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        """Forward pass"""
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Take last timestep output
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        x = self.fc1(last_output)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Ensure positive output (volatility can't be negative)
        x = torch.abs(x)
        
        return x


class LSTMVolatilityPredictor:
    """Complete LSTM volatility prediction system"""
    
    def __init__(
        self,
        sequence_length: int = 60,
        hidden_dim: int = 128,
        num_layers: int = 2,
        learning_rate: float = 0.001
    ):
        """
        Initialize predictor
        
        Args:
            sequence_length: Days of history to use
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            learning_rate: Learning rate
        """
        self.sequence_length = sequence_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.feature_names = []
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'Vol_GK'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM training
        
        Args:
            df: DataFrame with features and target
            target_col: Name of target volatility column
            
        Returns:
            Tuple of (X, y) arrays
        """
        # Select features - MUST match training! (Vol_GK is target, not feature)
        feature_cols = [
            'Open', 'High', 'Returns', 'Vol_Close', 'Intraday_Range',
            'sentiment_mean', 'sentiment_std', 'news_count',
            'sentiment_intensity', 'sentiment_ma_3'
        ]
        
        # Filter available columns
        feature_cols = [col for col in feature_cols if col in df.columns]
        self.feature_names = feature_cols
        
        # Fill NaNs
        df_clean = df[feature_cols + [target_col]].fillna(0)
        
        # Scale features
        X_scaled = self.scaler_X.fit_transform(df_clean[feature_cols].values)
        y_scaled = self.scaler_y.fit_transform(df_clean[[target_col]].values)
        
        # Create sequences
        X, y = [], []
        for i in range(len(X_scaled) - self.sequence_length):
            X.append(X_scaled[i:i + self.sequence_length])
            y.append(y_scaled[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2
    ) -> List[float]:
        """
        Train LSTM model
        
        Args:
            X: Input sequences
            y: Target values
            epochs: Training epochs
            batch_size: Batch size
            validation_split: Validation data ratio
            
        Returns:
            List of training losses
        """
        # Train/val split
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).to(self.device)
        
        # Initialize model
        input_dim = X.shape[2]
        self.model = VolatilityLSTM(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers
        ).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        losses = []
        
        print(f"Training LSTM on {self.device}...")
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        for epoch in range(epochs):
            self.model.train()
            
            # Mini-batch training
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i + batch_size]
                batch_y = y_train[i:i + batch_size]
                
                # Forward pass
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val)
                val_loss = criterion(val_pred, y_val)
            
            losses.append(val_loss.item())
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Val Loss: {val_loss.item():.6f}")
        
        print("Training complete!")
        return losses
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input sequences
            
        Returns:
            Predicted volatilities (inverse scaled)
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions_scaled = self.model(X_tensor).cpu().numpy()
        
        # Inverse scale
        predictions = self.scaler_y.inverse_transform(predictions_scaled)
        return predictions.flatten()
    
    def save_model(self, filepath: str):
        """Save model to disk"""
        if self.model is None:
            raise ValueError("No model to save")
        
        torch.save({
            'model_state': self.model.state_dict(),
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'feature_names': self.feature_names,
            'sequence_length': self.sequence_length
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from disk"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.scaler_X = checkpoint['scaler_X']
        self.scaler_y = checkpoint['scaler_y']
        self.feature_names = checkpoint['feature_names']
        self.sequence_length = checkpoint['sequence_length']
        
        # Use actual number of features from checkpoint
        input_dim = len(self.feature_names)
        self.model = VolatilityLSTM(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    print("LSTM Volatility Predictor module loaded")
    print(f"CUDA available: {torch.cuda.is_available()}")
