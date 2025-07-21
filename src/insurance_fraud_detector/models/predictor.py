"""
Fraud Prediction Model Module

This module provides the main fraud prediction functionality for the insurance
fraud detection system. It includes ensemble prediction, feature engineering,
and comprehensive prediction explanations.

Classes:
    FraudPredictor: Main prediction class with ensemble capabilities
    PredictionResult: Container for prediction results and explanations
    
Functions:
    load_predictor: Load a trained predictor from disk
    create_predictor: Factory function to create predictor instance
    
Example:
    >>> from insurance_fraud_detector.models.predictor import FraudPredictor
    >>> predictor = FraudPredictor(config)
    >>> result = predictor.predict(claim_data)
    >>> print(f"Fraud probability: {result.fraud_probability:.3f}")
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
import joblib
from pathlib import Path
import time

# Scikit-learn imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Local imports
from ..utils.config import Config
from ..utils.logger import LoggerMixin, log_execution_time
from ..data.preprocessor import DataPreprocessor
from ..training.trainer import ModelTrainer, ModelRegistry

@dataclass
class FeatureContribution:
    """
    Container for individual feature contributions to prediction.
    
    Attributes:
        feature_name: Name of the feature
        value: Value of the feature for this prediction
        contribution: Contribution to the prediction score
        importance: Overall importance of this feature
    """
    feature_name: str
    value: Any
    contribution: float
    importance: float = 0.0
    
    @property
    def contribution_percentage(self) -> float:
        """Get contribution as percentage."""
        return self.contribution * 100


@dataclass
class ModelContribution:
    """
    Container for individual model contributions to ensemble prediction.
    
    Attributes:
        model_name: Name of the model
        prediction: Individual model prediction
        confidence: Model confidence score
        weight: Weight in ensemble
    """
    model_name: str
    prediction: float
    confidence: float
    weight: float
    
    @property
    def weighted_prediction(self) -> float:
        """Get weighted prediction contribution."""
        return self.prediction * self.weight


@dataclass
class PredictionResult:
    """
    Container for prediction results and explanations.
    
    This class holds all information about a fraud prediction including
    the probability score, risk level, feature contributions, model
    contributions, and human-readable explanations.
    
    Attributes:
        fraud_probability: Probability of fraud (0-1)
        risk_level: Risk level (Low, Medium, High)
        confidence_score: Prediction confidence (0-1)
        processing_time_ms: Time taken for prediction in milliseconds
        feature_contributions: List of feature contributions
        model_contributions: List of model contributions
        risk_factors: List of factors that increase risk
        protective_factors: List of factors that decrease risk
        explanation: Human-readable explanation
        recommendation: Recommended action
    """
    fraud_probability: float
    risk_level: str
    confidence_score: float = 0.0
    processing_time_ms: float = 0.0
    feature_contributions: List[FeatureContribution] = field(default_factory=list)
    model_contributions: List[ModelContribution] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    protective_factors: List[str] = field(default_factory=list)
    explanation: str = ""
    recommendation: str = ""
    
    @property
    def risk_percentage(self) -> float:
        """Get risk as percentage."""
        return self.fraud_probability * 100
    
    @property
    def top_risk_factors(self) -> List[FeatureContribution]:
        """Get top 5 risk-increasing features."""
        positive_contributions = [
            fc for fc in self.feature_contributions if fc.contribution > 0
        ]
        return sorted(positive_contributions, key=lambda x: x.contribution, reverse=True)[:5]
    
    @property
    def top_protective_factors(self) -> List[FeatureContribution]:
        """Get top 5 risk-reducing features."""
        negative_contributions = [
            fc for fc in self.feature_contributions if fc.contribution < 0
        ]
        return sorted(negative_contributions, key=lambda x: abs(x.contribution), reverse=True)[:5]


class FraudPredictor(LoggerMixin):
    """
    Main fraud prediction class with ensemble capabilities.
    
    This class provides comprehensive fraud prediction functionality including:
    - Ensemble prediction using multiple models
    - Feature importance analysis
    - Prediction explanations and interpretability
    - Risk level classification
    - Confidence scoring
    
    The predictor can operate in two modes:
    1. Training mode: Trains new models on provided data
    2. Inference mode: Uses pre-trained models for predictions
    
    Attributes:
        config: Configuration object
        models: Dictionary of trained models
        preprocessor: Data preprocessor instance
        trainer: Model trainer instance (for training mode)
        feature_names: Names of input features
        is_trained: Whether models have been trained
    
    Example:
        >>> predictor = FraudPredictor(config)
        >>> predictor.train(training_data)
        >>> result = predictor.predict(claim_data)
        >>> print(f"Risk: {result.risk_level} ({result.fraud_probability:.1%})")
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize fraud predictor.
        
        Args:
            config: Configuration object. If None, loads default config.
        """
        self.config = config or Config()
        self.models: Dict[str, Any] = {}
        self.model_weights: Dict[str, float] = {}
        self.preprocessor = DataPreprocessor(self.config)
        self.trainer: Optional[ModelTrainer] = None
        self.feature_names: List[str] = []
        self.is_trained = False
        
        # Get risk thresholds from config
        self.risk_thresholds = self.config.get_risk_thresholds()
        
        # Get enabled models and weights
        self.enabled_models = self.config.get_enabled_models()
        ensemble_config = self.config.get('models.ensemble', {})
        self.model_weights = ensemble_config.get('weights', {})
        
        self.logger.info("FraudPredictor initialized")
    
    @log_execution_time
    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Train fraud detection models on provided data.
        
        This method trains multiple machine learning models on the provided
        insurance claims data and prepares the ensemble for prediction.
        
        Args:
            data: Training data with features and 'is_fraud' target column
            **kwargs: Additional arguments passed to trainer
            
        Returns:
            Dictionary with training results and metrics
            
        Raises:
            ValueError: If data doesn't contain required columns
            
        Example:
            >>> results = predictor.train(training_data)
            >>> print(f"Best model: {results['best_model']}")
        """
        self.logger.info("Starting model training")
        
        # Validate data
        if 'is_fraud' not in data.columns:
            raise ValueError("Training data must contain 'is_fraud' column")
        
        # Initialize trainer
        self.trainer = ModelTrainer(self.config)
        
        # Train models
        training_results = self.trainer.train(
            data, 
            model_names=self.enabled_models,
            **kwargs
        )
        
        # Extract trained models
        for model_name, result in training_results.items():
            self.models[model_name] = result.pipeline
        
        # Store feature names
        feature_cols = [col for col in data.columns if col != 'is_fraud']
        self.feature_names = feature_cols
        
        # Mark as trained
        self.is_trained = True
        
        # Get best model info
        best_result = self.trainer.get_best_model()
        best_model_name = best_result.model_name if best_result else None
        
        self.logger.info(f"Training completed. Best model: {best_model_name}")
        
        return {
            'training_results': training_results,
            'best_model': best_model_name,
            'model_comparison': self.trainer.compare_models(),
            'feature_names': self.feature_names
        }
    
    @log_execution_time
    def predict(self, claim_data: Union[Dict[str, Any], pd.DataFrame]) -> PredictionResult:
        """
        Predict fraud probability for a single claim or batch of claims.
        
        This method performs ensemble prediction using all trained models
        and provides comprehensive explanations for the prediction.
        
        Args:
            claim_data: Single claim as dict or DataFrame with claim features
            
        Returns:
            PredictionResult with fraud probability and explanations
            
        Raises:
            RuntimeError: If predictor hasn't been trained
            ValueError: If claim_data has invalid format
            
        Example:
            >>> claim = {'age': 35, 'claim_amount': 25000, ...}
            >>> result = predictor.predict(claim)
            >>> print(f"Fraud probability: {result.fraud_probability:.3f}")
        """
        if not self.is_trained:
            raise RuntimeError("Predictor must be trained before making predictions")
        
        start_time = time.time()
        
        # Convert to DataFrame if needed
        if isinstance(claim_data, dict):
            claim_df = pd.DataFrame([claim_data])
        else:
            claim_df = claim_data.copy()
        
        # Validate input features
        self._validate_input(claim_df)
        
        # Get ensemble prediction
        fraud_probability = self._predict_ensemble(claim_df.iloc[0])
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence(claim_df.iloc[0])
        
        # Determine risk level
        risk_level = self._get_risk_level(fraud_probability)
        
        # Get feature contributions (if available)
        feature_contributions = self._get_feature_contributions(claim_df.iloc[0])
        
        # Get model contributions
        model_contributions = self._get_model_contributions(claim_df.iloc[0])
        
        # Generate explanations
        risk_factors, protective_factors = self._analyze_risk_factors(
            claim_df.iloc[0], feature_contributions
        )
        
        explanation = self._generate_explanation(
            fraud_probability, risk_factors, protective_factors
        )
        
        recommendation = self._generate_recommendation(fraud_probability, risk_level)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        result = PredictionResult(
            fraud_probability=fraud_probability,
            risk_level=risk_level,
            confidence_score=confidence_score,
            processing_time_ms=processing_time,
            feature_contributions=feature_contributions,
            model_contributions=model_contributions,
            risk_factors=risk_factors,
            protective_factors=protective_factors,
            explanation=explanation,
            recommendation=recommendation
        )
        
        self.logger.info(
            f"Prediction completed: {risk_level} risk ({fraud_probability:.3f}) "
            f"in {processing_time:.1f}ms"
        )
        
        return result
    
    def predict_batch(self, claims_data: pd.DataFrame) -> List[PredictionResult]:
        """
        Predict fraud probability for multiple claims.
        
        Args:
            claims_data: DataFrame with multiple claims
            
        Returns:
            List of PredictionResult objects
        """
        results = []
        
        for idx, row in claims_data.iterrows():
            try:
                result = self.predict(row.to_dict())
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error predicting claim {idx}: {e}")
                # Create error result
                error_result = PredictionResult(
                    fraud_probability=0.5,  # Neutral
                    risk_level="Unknown",
                    explanation=f"Error in prediction: {str(e)}"
                )
                results.append(error_result)
        
        return results
    
    def predict_single(
        self,
        age: int,
        gender: str,
        vehicle_age: int,
        vehicle_type: str,
        annual_mileage: int,
        driving_violations: int,
        claim_amount: float,
        previous_claims: int,
        credit_score: int,
        region: str
    ) -> float:
        """
        Convenience method for single claim prediction.
        
        This method provides a simple interface for making predictions
        with individual parameter values.
        
        Args:
            age: Driver age
            gender: Driver gender ('M' or 'F')
            vehicle_age: Vehicle age in years
            vehicle_type: Type of vehicle
            annual_mileage: Annual mileage in miles
            driving_violations: Number of driving violations
            claim_amount: Claim amount in pounds
            previous_claims: Number of previous claims
            credit_score: Credit score
            region: Region type ('urban', 'suburban', 'rural')
            
        Returns:
            Fraud probability as float (0-1)
        """
        claim_data = {
            'age': age,
            'gender': gender,
            'vehicle_age': vehicle_age,
            'vehicle_type': vehicle_type,
            'annual_mileage': annual_mileage,
            'driving_violations': driving_violations,
            'claim_amount': claim_amount,
            'previous_claims': previous_claims,
            'credit_score': credit_score,
            'region': region
        }
        
        result = self.predict(claim_data)
        return result.fraud_probability
    
    def _validate_input(self, claim_df: pd.DataFrame) -> None:
        """
        Validate input claim data.
        
        Args:
            claim_df: Claim data to validate
            
        Raises:
            ValueError: If data is invalid
        """
        required_features = self.feature_names
        missing_features = [f for f in required_features if f not in claim_df.columns]
        
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Additional validation can be added here
        # (e.g., value ranges, data types)
    
    def _predict_ensemble(self, claim: pd.Series) -> float:
        """
        Make ensemble prediction using all trained models.
        
        Args:
            claim: Single claim data
            
        Returns:
            Ensemble fraud probability
        """
        predictions = []
        weights = []
        
        claim_df = claim.to_frame().T
        
        for model_name, model in self.models.items():
            try:
                # Get prediction probability
                pred_proba = model.predict_proba(claim_df)[0, 1]
                predictions.append(pred_proba)
                
                # Get model weight
                weight = self.model_weights.get(model_name, 1.0)
                weights.append(weight)
                
            except Exception as e:
                self.logger.warning(f"Error in model {model_name}: {e}")
                continue
        
        if not predictions:
            self.logger.error("No models produced valid predictions")
            return 0.5  # Neutral prediction
        
        # Calculate weighted average
        predictions = np.array(predictions)
        weights = np.array(weights)
        
        # Normalize weights
        weights = weights / weights.sum()
        
        # Weighted ensemble prediction
        ensemble_pred = np.average(predictions, weights=weights)
        
        return float(ensemble_pred)
    
    def _calculate_confidence(self, claim: pd.Series) -> float:
        """
        Calculate prediction confidence score.
        
        Args:
            claim: Single claim data
            
        Returns:
            Confidence score (0-1)
        """
        if not self.models:
            return 0.0
        
        predictions = []
        claim_df = claim.to_frame().T
        
        for model in self.models.values():
            try:
                pred_proba = model.predict_proba(claim_df)[0, 1]
                predictions.append(pred_proba)
            except Exception:
                continue
        
        if len(predictions) < 2:
            return 0.5
        
        # Confidence based on agreement between models
        predictions = np.array(predictions)
        std_dev = np.std(predictions)
        
        # Lower standard deviation = higher confidence
        confidence = max(0.0, 1.0 - (std_dev * 2))
        
        return confidence
    
    def _get_risk_level(self, fraud_probability: float) -> str:
        """
        Determine risk level based on fraud probability.
        
        Args:
            fraud_probability: Fraud probability (0-1)
            
        Returns:
            Risk level string
        """
        low_threshold = self.risk_thresholds.get('low_risk', 0.3)
        high_threshold = self.risk_thresholds.get('high_risk', 0.7)
        
        if fraud_probability < low_threshold:
            return "Low"
        elif fraud_probability < high_threshold:
            return "Medium"
        else:
            return "High"
    
    def _get_feature_contributions(self, claim: pd.Series) -> List[FeatureContribution]:
        """
        Get feature contributions to prediction (simplified).
        
        Args:
            claim: Single claim data
            
        Returns:
            List of feature contributions
        """
        contributions = []
        
        # This is a simplified approach - in practice, you might use
        # SHAP, LIME, or other explainability tools
        
        try:
            # Get feature importance from first available tree-based model
            tree_model = None
            for model in self.models.values():
                classifier = model.named_steps.get('classifier')
                if hasattr(classifier, 'feature_importances_'):
                    tree_model = classifier
                    break
            
            if tree_model is None:
                return contributions
            
            # Get feature names after preprocessing
            preprocessor = list(self.models.values())[0].named_steps.get('preprocessor')
            if hasattr(preprocessor, 'get_feature_names_out'):
                processed_feature_names = preprocessor.get_feature_names_out()
            else:
                processed_feature_names = self.feature_names
            
            # Create contributions based on feature importance and values
            claim_df = claim.to_frame().T
            processed_data = preprocessor.transform(claim_df)[0]
            
            for i, (feature_name, importance) in enumerate(zip(processed_feature_names, tree_model.feature_importances_)):
                if i < len(processed_data):
                    # Simplified contribution calculation
                    contribution = processed_data[i] * importance * 0.1
                    
                    # Map back to original feature if possible
                    original_feature = feature_name
                    for orig_name in self.feature_names:
                        if orig_name in feature_name:
                            original_feature = orig_name
                            break
                    
                    feature_contrib = FeatureContribution(
                        feature_name=original_feature,
                        value=claim.get(original_feature, 'N/A'),
                        contribution=contribution,
                        importance=importance
                    )
                    contributions.append(feature_contrib)
            
            # Sort by absolute contribution
            contributions.sort(key=lambda x: abs(x.contribution), reverse=True)
            
        except Exception as e:
            self.logger.warning(f"Could not calculate feature contributions: {e}")
        
        return contributions[:10]  # Top 10 features
    
    def _get_model_contributions(self, claim: pd.Series) -> List[ModelContribution]:
        """
        Get individual model contributions to ensemble prediction.
        
        Args:
            claim: Single claim data
            
        Returns:
            List of model contributions
        """
        contributions = []
        claim_df = claim.to_frame().T
        
        for model_name, model in self.models.items():
            try:
                pred_proba = model.predict_proba(claim_df)[0, 1]
                weight = self.model_weights.get(model_name, 1.0)
                
                # Simple confidence calculation
                confidence = min(pred_proba, 1 - pred_proba) * 2
                
                contribution = ModelContribution(
                    model_name=model_name,
                    prediction=pred_proba,
                    confidence=confidence,
                    weight=weight
                )
                contributions.append(contribution)
                
            except Exception as e:
                self.logger.warning(f"Error getting contribution from {model_name}: {e}")
        
        return contributions
    
    def _analyze_risk_factors(
        self, 
        claim: pd.Series, 
        feature_contributions: List[FeatureContribution]
    ) -> Tuple[List[str], List[str]]:
        """
        Analyze claim to identify risk and protective factors.
        
        Args:
            claim: Single claim data
            feature_contributions: List of feature contributions
            
        Returns:
            Tuple of (risk_factors, protective_factors)
        """
        risk_factors = []
        protective_factors = []
        
        # Rule-based risk factor analysis
        age = claim.get('age', 0)
        driving_violations = claim.get('driving_violations', 0)
        claim_amount = claim.get('claim_amount', 0)
        credit_score = claim.get('credit_score', 700)
        annual_mileage = claim.get('annual_mileage', 0)
        previous_claims = claim.get('previous_claims', 0)
        
        # Risk factors
        if driving_violations >= 2:
            risk_factors.append(f"Multiple driving violations ({driving_violations})")
        
        if credit_score < 650:
            risk_factors.append(f"Lower credit score ({credit_score})")
        
        if annual_mileage > 20000:
            risk_factors.append(f"High annual mileage ({annual_mileage:,} miles)")
        
        if claim_amount > 50000:
            risk_factors.append(f"High claim amount (£{claim_amount:,})")
        
        if previous_claims >= 3:
            risk_factors.append(f"Multiple previous claims ({previous_claims})")
        
        if age < 25:
            risk_factors.append(f"Young driver (age {age})")
        
        # Protective factors
        if driving_violations == 0:
            protective_factors.append("Clean driving record")
        
        if credit_score > 750:
            protective_factors.append(f"Good credit score ({credit_score})")
        
        if annual_mileage < 8000:
            protective_factors.append(f"Low annual mileage ({annual_mileage:,} miles)")
        
        if previous_claims == 0:
            protective_factors.append("No previous claims")
        
        if age > 50:
            protective_factors.append(f"Experienced driver (age {age})")
        
        return risk_factors, protective_factors
    
    def _generate_explanation(
        self, 
        fraud_probability: float,
        risk_factors: List[str],
        protective_factors: List[str]
    ) -> str:
        """
        Generate human-readable explanation for prediction.
        
        Args:
            fraud_probability: Fraud probability
            risk_factors: List of risk factors
            protective_factors: List of protective factors
            
        Returns:
            Human-readable explanation
        """
        risk_level = self._get_risk_level(fraud_probability)
        
        explanation = f"This claim is classified as {risk_level} risk with a {fraud_probability:.1%} probability of being fraudulent. "
        
        if risk_factors:
            explanation += f"Key risk factors include: {', '.join(risk_factors[:3])}. "
        
        if protective_factors:
            explanation += f"Mitigating factors include: {', '.join(protective_factors[:3])}. "
        
        explanation += "This assessment is based on analysis of historical claim patterns and multiple machine learning models."
        
        return explanation
    
    def _generate_recommendation(self, fraud_probability: float, risk_level: str) -> str:
        """
        Generate recommended action based on prediction.
        
        Args:
            fraud_probability: Fraud probability
            risk_level: Risk level
            
        Returns:
            Recommended action
        """
        if risk_level == "Low":
            return "Process claim automatically. Standard verification procedures apply."
        elif risk_level == "Medium":
            return "Flag for enhanced review. Additional documentation may be required."
        else:
            return "Refer to fraud investigation team. Conduct thorough investigation before processing."
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get overall feature importance across all models.
        
        Returns:
            Dictionary of feature importances or None
        """
        if not self.trainer:
            return None
        
        # Get feature importance from best model
        best_result = self.trainer.get_best_model()
        if best_result and best_result.feature_importance:
            return best_result.feature_importance
        
        return None
    
    def save_model(self, file_path: Union[str, Path]) -> None:
        """
        Save trained predictor to disk.
        
        Args:
            file_path: Path to save the model
        """
        if not self.is_trained:
            raise RuntimeError("No trained model to save")
        
        model_data = {
            'models': self.models,
            'model_weights': self.model_weights,
            'feature_names': self.feature_names,
            'config': self.config.config,
            'risk_thresholds': self.risk_thresholds
        }
        
        joblib.dump(model_data, file_path)
        self.logger.info(f"Model saved to {file_path}")
    
    @classmethod
    def load_model(cls, file_path: Union[str, Path]) -> 'FraudPredictor':
        """
        Load trained predictor from disk.
        
        Args:
            file_path: Path to saved model
            
        Returns:
            Loaded FraudPredictor instance
        """
        model_data = joblib.load(file_path)
        
        # Create new predictor instance
        config = Config()
        config.config = model_data['config']
        
        predictor = cls(config)
        predictor.models = model_data['models']
        predictor.model_weights = model_data['model_weights']
        predictor.feature_names = model_data['feature_names']
        predictor.risk_thresholds = model_data['risk_thresholds']
        predictor.is_trained = True
        
        predictor.logger.info(f"Model loaded from {file_path}")
        return predictor


def load_predictor(file_path: Union[str, Path]) -> FraudPredictor:
    """
    Load a trained predictor from disk.
    
    Args:
        file_path: Path to saved predictor
        
    Returns:
        Loaded FraudPredictor instance
    """
    return FraudPredictor.load_model(file_path)


def create_predictor(config_path: Optional[str] = None) -> FraudPredictor:
    """
    Factory function to create FraudPredictor instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured FraudPredictor instance
    """
    config = Config(config_path) if config_path else Config()
    return FraudPredictor(config)