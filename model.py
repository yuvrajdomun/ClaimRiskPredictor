"""
ML model for insurance claim fraud prediction.
Includes preprocessing, training, and prediction functionality.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class FraudPredictor:
    """Insurance claim fraud prediction model."""
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.is_trained = False
        
    def create_preprocessor(self, df: pd.DataFrame) -> ColumnTransformer:
        """Create preprocessing pipeline."""
        
        # Identify feature types
        numeric_features = ['age', 'vehicle_age', 'annual_mileage', 'driving_violations', 
                          'claim_amount', 'previous_claims', 'credit_score']
        categorical_features = ['gender', 'vehicle_type', 'region']
        
        # Create preprocessing steps
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        return preprocessor
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional features."""
        df = df.copy()
        
        # Feature engineering
        df['claim_per_mile'] = df['claim_amount'] / (df['annual_mileage'] + 1)
        df['violations_per_year'] = df['driving_violations'] / (df['age'] - 17)  # Assume driving since 18
        df['high_risk_vehicle'] = (df['vehicle_type'].isin(['sports', 'luxury'])).astype(int)
        df['young_driver'] = (df['age'] < 25).astype(int)
        df['high_claim'] = (df['claim_amount'] > df['claim_amount'].quantile(0.8)).astype(int)
        df['credit_risk'] = (df['credit_score'] < 600).astype(int)
        
        return df
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, Any]:
        """Train the fraud prediction model."""
        
        # Prepare features
        df_features = self.prepare_features(df)
        
        # Separate features and target
        X = df_features.drop(['is_fraud'], axis=1)
        y = df_features['is_fraud']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Create preprocessor and model pipeline
        self.preprocessor = self.create_preprocessor(X_train)
        
        # Try multiple models and select best
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=42,
                class_weight='balanced'
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            )
        }
        
        best_score = 0
        best_model_name = None
        
        for name, model in models.items():
            pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('classifier', model)
            ])
            
            # Cross-validation
            cv_scores = cross_val_score(pipeline, X_train, y_train, 
                                      cv=5, scoring='roc_auc')
            avg_score = cv_scores.mean()
            
            logger.info(f"{name}: CV AUC = {avg_score:.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            if avg_score > best_score:
                best_score = avg_score
                best_model_name = name
                self.model = pipeline
        
        # Train final model
        self.model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        test_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Store feature names for later use
        if hasattr(self.preprocessor, 'get_feature_names_out'):
            try:
                self.feature_names = self.preprocessor.get_feature_names_out()
            except:
                self.feature_names = X.columns.tolist()
        else:
            self.feature_names = X.columns.tolist()
        
        self.is_trained = True
        
        results = {
            'best_model': best_model_name,
            'cv_auc': best_score,
            'test_auc': test_auc,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'feature_importance': self.get_feature_importance()
        }
        
        logger.info(f"Model trained successfully. Best model: {best_model_name}, Test AUC: {test_auc:.4f}")
        
        return results
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict fraud probability."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        df_features = self.prepare_features(df)
        X = df_features.drop(['is_fraud'], axis=1, errors='ignore')
        
        return self.model.predict_proba(X)[:, 1]
    
    def predict_single(self, **kwargs) -> float:
        """Predict fraud probability for a single claim."""
        # Create DataFrame from input
        df = pd.DataFrame([kwargs])
        
        # Ensure correct data types for numeric columns
        numeric_columns = ['age', 'vehicle_age', 'annual_mileage', 'driving_violations', 
                          'claim_amount', 'previous_claims', 'credit_score']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col])
        
        # Add is_fraud column as placeholder (will be ignored)
        df['is_fraud'] = 0
        
        return float(self.predict(df)[0])
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model."""
        if not self.is_trained:
            return {}
        
        try:
            # Get the classifier from the pipeline
            classifier = self.model.named_steps['classifier']
            
            if hasattr(classifier, 'feature_importances_'):
                importances = classifier.feature_importances_
                
                # Try to get feature names from preprocessor
                try:
                    feature_names = self.preprocessor.get_feature_names_out()
                except:
                    feature_names = [f'feature_{i}' for i in range(len(importances))]
                
                return dict(zip(feature_names, importances))
            
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
            
        return {}
    
    def save_model(self, filepath: str):
        """Save trained model to file."""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from file."""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.feature_names = model_data.get('feature_names')
        self.is_trained = model_data.get('is_trained', True)
        
        logger.info(f"Model loaded from {filepath}")

if __name__ == "__main__":
    from data_loader import InsuranceDataLoader
    
    # Load data
    loader = InsuranceDataLoader()
    data = loader.load_data()
    
    # Train model
    predictor = FraudPredictor()
    results = predictor.train(data)
    
    print("Training Results:")
    for key, value in results.items():
        if key not in ['classification_report']:
            print(f"{key}: {value}")
    
    print("\nClassification Report:")
    print(results['classification_report'])
    
    # Test single prediction
    test_claim = {
        'age': 23,
        'gender': 'M',
        'vehicle_age': 1,
        'vehicle_type': 'sports',
        'annual_mileage': 25000,
        'driving_violations': 3,
        'claim_amount': 75000,
        'previous_claims': 1,
        'credit_score': 450,
        'region': 'urban'
    }
    
    fraud_prob = predictor.predict_single(**test_claim)
    print(f"\nTest claim fraud probability: {fraud_prob:.4f}")