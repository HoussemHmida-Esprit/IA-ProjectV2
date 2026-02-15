"""
Forecasting Engine - Separate Time-Series Risk Prediction
Decoupled from accident classification models
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime, timedelta
import pickle


class ForecastingEngine:
    """
    Time-series forecasting engine for accident risk prediction
    
    Purpose: Predict daily/hourly accident counts and risk levels
    Separate from: Individual accident classification (RF, XGB, TabT)
    
    Use Cases:
    - Daily accident count forecasting
    - Risk level prediction for time periods
    - Temporal pattern analysis
    - Resource allocation planning
    """
    
    def __init__(self, model_path='lstm_forecaster.pth'):
        self.model_path = Path(model_path)
        self.model = None
        self.scaler = None
        self.sequence_length = 7  # Use 7 days to predict next day
        
    def load_model(self):
        """Load trained LSTM model"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        # Reconstruct model architecture
        from lstm_forecasting import AccidentLSTM
        
        self.model = AccidentLSTM(
            input_size=checkpoint.get('input_size', 1),
            hidden_size=checkpoint.get('hidden_size', 64),
            num_layers=checkpoint.get('num_layers', 2),
            output_size=checkpoint.get('output_size', 1)
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load scaler if available
        if 'scaler' in checkpoint:
            self.scaler = checkpoint['scaler']
        
        print(f"✓ Forecasting model loaded from: {self.model_path}")
        
        return self
    
    def prepare_sequence(self, data: np.ndarray, sequence_length: int = None):
        """
        Prepare sequences for LSTM prediction
        
        Args:
            data: Time series data (daily counts)
            sequence_length: Length of input sequence
            
        Returns:
            Sequences ready for prediction
        """
        if sequence_length is None:
            sequence_length = self.sequence_length
        
        # Scale data if scaler available
        if self.scaler is not None:
            data_scaled = self.scaler.transform(data.reshape(-1, 1))
        else:
            data_scaled = data.reshape(-1, 1)
        
        # Create sequences
        sequences = []
        for i in range(len(data_scaled) - sequence_length):
            seq = data_scaled[i:i + sequence_length]
            sequences.append(seq)
        
        return np.array(sequences)
    
    def predict_next_day(self, historical_data: np.ndarray) -> float:
        """
        Predict accident count for next day
        
        Args:
            historical_data: Last 7 days of accident counts
            
        Returns:
            Predicted accident count for next day
        """
        if self.model is None:
            self.load_model()
        
        # Prepare sequence
        if len(historical_data) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} days of data")
        
        # Take last sequence_length days
        sequence = historical_data[-self.sequence_length:]
        
        # Scale if scaler available
        if self.scaler is not None:
            sequence_scaled = self.scaler.transform(sequence.reshape(-1, 1))
        else:
            sequence_scaled = sequence.reshape(-1, 1)
        
        # Convert to tensor
        sequence_tensor = torch.FloatTensor(sequence_scaled).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            prediction_scaled = self.model(sequence_tensor).item()
        
        # Inverse scale
        if self.scaler is not None:
            prediction = self.scaler.inverse_transform([[prediction_scaled]])[0][0]
        else:
            prediction = prediction_scaled
        
        return max(0, prediction)  # Ensure non-negative
    
    def predict_next_n_days(self, historical_data: np.ndarray, n_days: int = 7) -> np.ndarray:
        """
        Predict accident counts for next N days
        
        Args:
            historical_data: Historical accident counts
            n_days: Number of days to forecast
            
        Returns:
            Array of predicted counts
        """
        if self.model is None:
            self.load_model()
        
        predictions = []
        current_sequence = historical_data[-self.sequence_length:].copy()
        
        for _ in range(n_days):
            # Predict next day
            next_day = self.predict_next_day(current_sequence)
            predictions.append(next_day)
            
            # Update sequence (rolling window)
            current_sequence = np.append(current_sequence[1:], next_day)
        
        return np.array(predictions)
    
    def calculate_risk_level(self, predicted_count: float, 
                           historical_mean: float = None,
                           historical_std: float = None) -> str:
        """
        Calculate risk level based on predicted count
        
        Args:
            predicted_count: Predicted accident count
            historical_mean: Historical average
            historical_std: Historical standard deviation
            
        Returns:
            Risk level: 'Low', 'Medium', 'High', 'Very High'
        """
        if historical_mean is None or historical_std is None:
            # Use simple thresholds
            if predicted_count < 80:
                return 'Low'
            elif predicted_count < 100:
                return 'Medium'
            elif predicted_count < 120:
                return 'High'
            else:
                return 'Very High'
        
        # Use statistical thresholds
        z_score = (predicted_count - historical_mean) / historical_std
        
        if z_score < -0.5:
            return 'Low'
        elif z_score < 0.5:
            return 'Medium'
        elif z_score < 1.5:
            return 'High'
        else:
            return 'Very High'
    
    def forecast_with_risk(self, historical_data: np.ndarray, 
                          n_days: int = 7) -> pd.DataFrame:
        """
        Generate forecast with risk levels
        
        Args:
            historical_data: Historical accident counts
            n_days: Number of days to forecast
            
        Returns:
            DataFrame with predictions and risk levels
        """
        # Calculate historical statistics
        hist_mean = np.mean(historical_data)
        hist_std = np.std(historical_data)
        
        # Generate predictions
        predictions = self.predict_next_n_days(historical_data, n_days)
        
        # Calculate risk levels
        risk_levels = [
            self.calculate_risk_level(pred, hist_mean, hist_std)
            for pred in predictions
        ]
        
        # Create forecast DataFrame
        today = datetime.now().date()
        dates = [today + timedelta(days=i+1) for i in range(n_days)]
        
        forecast_df = pd.DataFrame({
            'date': dates,
            'predicted_count': predictions,
            'risk_level': risk_levels,
            'historical_mean': hist_mean,
            'historical_std': hist_std
        })
        
        return forecast_df
    
    def get_hourly_risk_pattern(self, date: datetime = None) -> pd.DataFrame:
        """
        Get hourly risk pattern for a specific date
        
        Args:
            date: Date to analyze (default: today)
            
        Returns:
            DataFrame with hourly risk levels
        """
        if date is None:
            date = datetime.now().date()
        
        # Simple hourly risk pattern based on historical data
        # In production, this would use actual hourly LSTM predictions
        hourly_risks = []
        
        for hour in range(24):
            # Rush hours (7-9, 17-19) are high risk
            if 7 <= hour <= 9 or 17 <= hour <= 19:
                risk = 'High'
                count = 8
            # Night hours (22-6) are medium risk
            elif 22 <= hour or hour <= 6:
                risk = 'Medium'
                count = 5
            # Other hours are low risk
            else:
                risk = 'Low'
                count = 3
            
            hourly_risks.append({
                'hour': hour,
                'risk_level': risk,
                'estimated_count': count
            })
        
        return pd.DataFrame(hourly_risks)


def demonstrate_forecasting():
    """Demonstrate forecasting engine usage"""
    print("="*60)
    print("FORECASTING ENGINE DEMONSTRATION")
    print("="*60)
    
    # Create engine
    engine = ForecastingEngine(model_path='lstm_forecaster.pth')
    
    # Check if model exists
    if not engine.model_path.exists():
        print("⚠️ LSTM model not found. Train it first:")
        print("   python models/lstm_forecasting.py")
        return
    
    # Load model
    engine.load_model()
    
    # Simulate historical data (last 30 days)
    print("\n1. Simulating historical data...")
    np.random.seed(42)
    historical_data = np.random.normal(100, 15, 30)  # Mean 100, std 15
    
    print(f"   Last 7 days: {historical_data[-7:].round(1)}")
    print(f"   Mean: {historical_data.mean():.1f}")
    print(f"   Std: {historical_data.std():.1f}")
    
    # Predict next day
    print("\n2. Predicting next day...")
    next_day_pred = engine.predict_next_day(historical_data)
    print(f"   Predicted count: {next_day_pred:.1f} accidents")
    
    # Predict next 7 days
    print("\n3. Predicting next 7 days...")
    next_week = engine.predict_next_n_days(historical_data, n_days=7)
    print(f"   Predictions: {next_week.round(1)}")
    
    # Forecast with risk levels
    print("\n4. Generating forecast with risk levels...")
    forecast = engine.forecast_with_risk(historical_data, n_days=7)
    print(forecast.to_string(index=False))
    
    # Hourly risk pattern
    print("\n5. Hourly risk pattern for today...")
    hourly = engine.get_hourly_risk_pattern()
    print(hourly.head(10).to_string(index=False))
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nKey Points:")
    print("  • Forecasting is SEPARATE from accident classification")
    print("  • Use for: daily counts, risk levels, resource planning")
    print("  • NOT for: individual accident severity prediction")


if __name__ == '__main__':
    demonstrate_forecasting()
