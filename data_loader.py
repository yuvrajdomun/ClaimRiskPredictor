"""
Data loader module for insurance claims dataset.
Downloads and loads insurance claims data for fraud prediction.
"""

import pandas as pd
import numpy as np
import requests
from pathlib import Path
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InsuranceDataLoader:
    """Handles loading and basic preprocessing of insurance claims data."""
    
    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path or "data/insurance_claims.csv"
        self.data = None
        
    def load_sample_data(self) -> pd.DataFrame:
        """Generate sample insurance claims data if no external source available."""
        np.random.seed(42)
        n_samples = 10000
        
        # Generate synthetic insurance claims data
        data = {
            'age': np.random.normal(45, 15, n_samples).astype(int).clip(18, 85),
            'gender': np.random.choice(['M', 'F'], n_samples),
            'vehicle_age': np.random.poisson(5, n_samples).clip(0, 20),
            'vehicle_type': np.random.choice(['saloon', 'suv', 'van', 'sports', 'luxury'], n_samples),
            'annual_mileage': np.random.normal(10000, 4000, n_samples).astype(int).clip(1000, 40000),
            'driving_violations': np.random.poisson(0.5, n_samples).clip(0, 10),
            'claim_amount': np.random.lognormal(7.8, 1, n_samples).astype(int).clip(400, 80000),
            'previous_claims': np.random.poisson(0.8, n_samples).clip(0, 10),
            'credit_score': np.random.normal(650, 100, n_samples).astype(int).clip(300, 850),
            'region': np.random.choice(['london', 'midlands', 'north', 'south', 'scotland'], n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Create fraud probability based on realistic factors
        fraud_prob = (
            0.1 +
            0.15 * (df['driving_violations'] > 2) +
            0.1 * (df['previous_claims'] > 2) +
            0.1 * (df['claim_amount'] > 50000) +
            0.05 * (df['credit_score'] < 500) +
            0.05 * (df['age'] < 25) +
            0.05 * (df['vehicle_type'] == 'sports')
        ).clip(0, 1)
        
        df['is_fraud'] = np.random.binomial(1, fraud_prob, n_samples)
        
        logger.info(f"Generated {len(df)} synthetic insurance claims records")
        logger.info(f"Fraud rate: {df['is_fraud'].mean():.3f}")
        
        return df
    
    def load_data(self) -> pd.DataFrame:
        """Load insurance claims data from file or generate sample data."""
        try:
            # Try to load from file first
            if Path(self.data_path).exists():
                self.data = pd.read_csv(self.data_path)
                logger.info(f"Loaded data from {self.data_path}")
            else:
                # Generate sample data
                logger.info("Data file not found, generating sample data")
                self.data = self.load_sample_data()
                
                # Save generated data
                Path(self.data_path).parent.mkdir(parents=True, exist_ok=True)
                self.data.to_csv(self.data_path, index=False)
                logger.info(f"Saved generated data to {self.data_path}")
                
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            logger.info("Falling back to sample data generation")
            self.data = self.load_sample_data()
            
        return self.data
    
    def get_basic_info(self) -> dict:
        """Get basic information about the dataset."""
        if self.data is None:
            self.load_data()
            
        return {
            'shape': self.data.shape,
            'fraud_rate': self.data['is_fraud'].mean(),
            'columns': list(self.data.columns),
            'missing_values': self.data.isnull().sum().to_dict(),
            'data_types': self.data.dtypes.to_dict()
        }

if __name__ == "__main__":
    loader = InsuranceDataLoader()
    data = loader.load_data()
    info = loader.get_basic_info()
    
    print("Dataset Info:")
    for key, value in info.items():
        print(f"{key}: {value}")