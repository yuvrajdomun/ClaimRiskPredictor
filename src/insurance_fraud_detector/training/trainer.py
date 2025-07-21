"""
Model Training Pipeline Module

This module provides comprehensive model training capabilities for the insurance
fraud detection system. It supports multiple machine learning algorithms,
hyperparameter optimization, cross-validation, and ensemble methods.

Classes:
    ModelTrainer: Main training pipeline class
    ModelRegistry: Registry for available models
    
Functions:
    create_trainer: Factory function to create trainer instance
    train_multiple_models: Train and compare multiple models
    
Example:
    >>> from insurance_fraud_detector.training.trainer import ModelTrainer
    >>> trainer = ModelTrainer(config)
    >>> results = trainer.train(['random_forest', 'xgboost'])
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import joblib
from pathlib import Path
import time
from datetime import datetime

# Scikit-learn imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    RandomizedSearchCV, StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, precision_recall_curve, auc,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Third-party ML libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Imbalanced learning
try:
    from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBALANCED_LEARN_AVAILABLE = True
except ImportError:
    IMBALANCED_LEARN_AVAILABLE = False

# Local imports
from ..utils.config import Config
from ..utils.logger import LoggerMixin, log_execution_time
from ..data.preprocessor import DataPreprocessor

@dataclass
class TrainingResult:
    """
    Container for model training results.
    
    Attributes:
        model_name: Name of the trained model
        model: Trained model instance
        metrics: Dictionary of evaluation metrics
        training_time: Time taken to train the model (seconds)
        cross_val_scores: Cross-validation scores
        feature_importance: Feature importance scores (if available)
        hyperparams: Hyperparameters used for training
        pipeline: Complete preprocessing + model pipeline
    """
    model_name: str
    model: Any
    metrics: Dict[str, float] = field(default_factory=dict)
    training_time: float = 0.0
    cross_val_scores: List[float] = field(default_factory=list)
    feature_importance: Optional[Dict[str, float]] = None
    hyperparams: Dict[str, Any] = field(default_factory=dict)
    pipeline: Optional[Pipeline] = None
    
    @property
    def cv_mean(self) -> float:
        """Mean cross-validation score."""
        return np.mean(self.cross_val_scores) if self.cross_val_scores else 0.0
    
    @property
    def cv_std(self) -> float:
        """Standard deviation of cross-validation scores."""
        return np.std(self.cross_val_scores) if self.cross_val_scores else 0.0


class BaseModelFactory(ABC):
    """
    Abstract base class for model factories.
    
    Each model type should implement this interface to provide
    consistent model creation and parameter handling.
    """
    
    @abstractmethod
    def create_model(self, **hyperparams) -> Any:
        """Create model instance with given hyperparameters."""
        pass
    
    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """Get default hyperparameters for this model."""
        pass
    
    @abstractmethod
    def get_param_space(self) -> Dict[str, List[Any]]:
        """Get hyperparameter search space for optimization."""
        pass


class RandomForestFactory(BaseModelFactory):
    """Factory for Random Forest models."""
    
    def create_model(self, **hyperparams) -> RandomForestClassifier:
        """Create Random Forest classifier."""
        return RandomForestClassifier(**hyperparams)
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default hyperparameters."""
        return {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        }
    
    def get_param_space(self) -> Dict[str, List[Any]]:
        """Get hyperparameter search space."""
        return {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }


class XGBoostFactory(BaseModelFactory):
    """Factory for XGBoost models."""
    
    def create_model(self, **hyperparams) -> Union[xgb.XGBClassifier, None]:
        """Create XGBoost classifier."""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available. Install with: pip install xgboost")
        return xgb.XGBClassifier(**hyperparams)
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default hyperparameters."""
        return {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'eval_metric': 'logloss'
        }
    
    def get_param_space(self) -> Dict[str, List[Any]]:
        """Get hyperparameter search space."""
        return {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }


class LightGBMFactory(BaseModelFactory):
    """Factory for LightGBM models."""
    
    def create_model(self, **hyperparams) -> Union[lgb.LGBMClassifier, None]:
        """Create LightGBM classifier."""
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not available. Install with: pip install lightgbm")
        return lgb.LGBMClassifier(**hyperparams)
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default hyperparameters."""
        return {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
    
    def get_param_space(self) -> Dict[str, List[Any]]:
        """Get hyperparameter search space."""
        return {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'num_leaves': [15, 31, 63],
            'subsample': [0.8, 0.9, 1.0]
        }


class LogisticRegressionFactory(BaseModelFactory):
    """Factory for Logistic Regression models."""
    
    def create_model(self, **hyperparams) -> LogisticRegression:
        """Create Logistic Regression classifier."""
        return LogisticRegression(**hyperparams)
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default hyperparameters."""
        return {
            'C': 1.0,
            'penalty': 'l2',
            'max_iter': 1000,
            'random_state': 42
        }
    
    def get_param_space(self) -> Dict[str, List[Any]]:
        """Get hyperparameter search space."""
        return {
            'C': [0.1, 1.0, 10.0, 100.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }


class NeuralNetworkFactory(BaseModelFactory):
    """Factory for Neural Network models."""
    
    def create_model(self, **hyperparams) -> MLPClassifier:
        """Create MLP classifier."""
        return MLPClassifier(**hyperparams)
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default hyperparameters."""
        return {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.001,
            'max_iter': 1000,
            'random_state': 42
        }
    
    def get_param_space(self) -> Dict[str, List[Any]]:
        """Get hyperparameter search space."""
        return {
            'hidden_layer_sizes': [(50,), (100,), (100, 50), (200, 100)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }


class SVMFactory(BaseModelFactory):
    """Factory for Support Vector Machine models."""
    
    def create_model(self, **hyperparams) -> SVC:
        """Create SVM classifier."""
        return SVC(**hyperparams)
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default hyperparameters."""
        return {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'probability': True,
            'random_state': 42
        }
    
    def get_param_space(self) -> Dict[str, List[Any]]:
        """Get hyperparameter search space."""
        return {
            'C': [0.1, 1.0, 10.0, 100.0],
            'kernel': ['rbf', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto']
        }


class GradientBoostingFactory(BaseModelFactory):
    """Factory for Gradient Boosting models."""
    
    def create_model(self, **hyperparams) -> GradientBoostingClassifier:
        """Create Gradient Boosting classifier."""
        return GradientBoostingClassifier(**hyperparams)
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default hyperparameters."""
        return {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
    
    def get_param_space(self) -> Dict[str, List[Any]]:
        """Get hyperparameter search space."""
        return {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }


class ModelRegistry:
    """
    Registry for available model factories.
    
    This class maintains a registry of all available model types
    and provides methods to create and configure models.
    """
    
    _factories = {
        'random_forest': RandomForestFactory(),
        'xgboost': XGBoostFactory(),
        'lightgbm': LightGBMFactory(),
        'logistic_regression': LogisticRegressionFactory(),
        'neural_network': NeuralNetworkFactory(),
        'svm': SVMFactory(),
        'gradient_boosting': GradientBoostingFactory()
    }
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of available model names."""
        available = []
        for name, factory in cls._factories.items():
            try:
                # Test if the model can be created
                factory.create_model()
                available.append(name)
            except ImportError:
                # Skip models with missing dependencies
                continue
        return available
    
    @classmethod
    def create_model(cls, model_name: str, **hyperparams) -> Any:
        """Create model instance."""
        if model_name not in cls._factories:
            raise ValueError(f"Unknown model: {model_name}")
        
        factory = cls._factories[model_name]
        params = factory.get_default_params()
        params.update(hyperparams)
        
        return factory.create_model(**params)
    
    @classmethod
    def get_param_space(cls, model_name: str) -> Dict[str, List[Any]]:
        """Get hyperparameter search space for model."""
        if model_name not in cls._factories:
            raise ValueError(f"Unknown model: {model_name}")
        
        return cls._factories[model_name].get_param_space()


class ModelTrainer(LoggerMixin):
    """
    Main model training pipeline class.
    
    This class provides comprehensive model training capabilities including:
    - Multiple algorithm support
    - Hyperparameter optimization
    - Cross-validation
    - Class imbalance handling
    - Feature preprocessing
    - Model evaluation and comparison
    
    Attributes:
        config: Configuration object
        preprocessor: Data preprocessor instance
        models: Dictionary of trained models
        results: Dictionary of training results
    
    Example:
        >>> trainer = ModelTrainer(config)
        >>> results = trainer.train(['random_forest', 'xgboost'])
        >>> best_model = trainer.get_best_model()
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize model trainer.
        
        Args:
            config: Configuration object. If None, loads default config.
        """
        self.config = config or Config()
        self.preprocessor = DataPreprocessor(self.config)
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, TrainingResult] = {}
        self.registry = ModelRegistry()
        
        # Get training configuration
        self.training_config = self.config.get_training_config()
        
        self.logger.info("ModelTrainer initialized")
    
    @log_execution_time
    def train(
        self, 
        data: pd.DataFrame,
        model_names: Optional[List[str]] = None,
        optimize_hyperparams: bool = False,
        n_trials: int = 50
    ) -> Dict[str, TrainingResult]:
        """
        Train multiple models on the given data.
        
        Args:
            data: Training data with features and target
            model_names: List of model names to train. If None, uses config
            optimize_hyperparams: Whether to perform hyperparameter optimization
            n_trials: Number of trials for hyperparameter optimization
            
        Returns:
            Dictionary of training results
            
        Example:
            >>> results = trainer.train(data, ['random_forest', 'xgboost'])
        """
        self.logger.info("Starting model training pipeline")
        
        # Use configured models if not specified
        if model_names is None:
            model_names = self.config.get_enabled_models()
        
        # Validate model names
        available_models = self.registry.get_available_models()
        invalid_models = [name for name in model_names if name not in available_models]
        if invalid_models:
            raise ValueError(f"Invalid models: {invalid_models}. Available: {available_models}")
        
        # Preprocess data
        X_train, X_val, X_test, y_train, y_val, y_test = self._prepare_data(data)
        
        self.logger.info(f"Training {len(model_names)} models: {model_names}")
        
        # Train each model
        for model_name in model_names:
            self.logger.info(f"Training {model_name}")
            
            result = self._train_single_model(
                model_name, X_train, X_val, X_test, 
                y_train, y_val, y_test,
                optimize_hyperparams, n_trials
            )
            
            self.results[model_name] = result
            self.models[model_name] = result.model
            
            self.logger.info(
                f"Completed {model_name}: "
                f"AUC={result.metrics.get('roc_auc', 0):.4f}, "
                f"Time={result.training_time:.2f}s"
            )
        
        self.logger.info(f"Training completed for all {len(model_names)} models")
        return self.results
    
    def _prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, ...]:
        """
        Prepare and split data for training.
        
        Args:
            data: Raw data with features and target
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        self.logger.info("Preparing training data")
        
        # Separate features and target
        target_col = 'is_fraud'
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        X = data.drop(target_col, axis=1)
        y = data[target_col]
        
        # Split data
        test_size = self.training_config.test_size
        val_size = self.training_config.validation_size
        random_state = self.training_config.random_state
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=y_temp
        )
        
        self.logger.info(
            f"Data split - Train: {len(X_train)}, "
            f"Val: {len(X_val)}, Test: {len(X_test)}"
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _train_single_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame, 
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        y_test: pd.Series,
        optimize_hyperparams: bool = False,
        n_trials: int = 50
    ) -> TrainingResult:
        """
        Train a single model.
        
        Args:
            model_name: Name of the model to train
            X_train, X_val, X_test: Feature data splits
            y_train, y_val, y_test: Target data splits
            optimize_hyperparams: Whether to optimize hyperparameters
            n_trials: Number of optimization trials
            
        Returns:
            TrainingResult object
        """
        start_time = time.time()
        
        # Get model hyperparameters
        if optimize_hyperparams:
            hyperparams = self._optimize_hyperparams(
                model_name, X_train, y_train, n_trials
            )
        else:
            hyperparams = self.config.get_model_hyperparams(model_name)
        
        # Create model
        model = self.registry.create_model(model_name, **hyperparams)
        
        # Create preprocessing pipeline
        pipeline = self._create_pipeline(model)
        
        # Handle class imbalance if configured
        if self.training_config.class_balance_method != 'none':
            pipeline = self._add_resampling(pipeline)
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Calculate training time
        training_time = time.time() - start_time
        
        # Evaluate model
        metrics = self._evaluate_model(pipeline, X_val, y_val, X_test, y_test)
        
        # Cross-validation scores
        cv_scores = self._cross_validate(pipeline, X_train, y_train)
        
        # Feature importance (if available)
        feature_importance = self._get_feature_importance(pipeline, X_train.columns)
        
        return TrainingResult(
            model_name=model_name,
            model=pipeline.named_steps.get('classifier', model),
            metrics=metrics,
            training_time=training_time,
            cross_val_scores=cv_scores,
            feature_importance=feature_importance,
            hyperparams=hyperparams,
            pipeline=pipeline
        )
    
    def _create_pipeline(self, model: Any) -> Pipeline:
        """
        Create preprocessing + model pipeline.
        
        Args:
            model: Model instance
            
        Returns:
            Complete pipeline
        """
        # Get data configuration
        data_config = self.config.get_data_config()
        
        # Create preprocessor
        preprocessor = self.preprocessor.create_preprocessor()
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        return pipeline
    
    def _add_resampling(self, pipeline: Pipeline) -> Pipeline:
        """
        Add resampling step to pipeline for class imbalance.
        
        Args:
            pipeline: Existing pipeline
            
        Returns:
            Pipeline with resampling
        """
        if not IMBALANCED_LEARN_AVAILABLE:
            self.logger.warning("Imbalanced-learn not available, skipping resampling")
            return pipeline
        
        method = self.training_config.class_balance_method
        random_state = self.training_config.random_state
        
        # Choose resampling method
        if method == 'smote':
            sampler = SMOTE(random_state=random_state)
        elif method == 'adasyn':
            sampler = ADASYN(random_state=random_state)
        elif method == 'random_oversample':
            sampler = RandomOverSampler(random_state=random_state)
        else:
            self.logger.warning(f"Unknown resampling method: {method}")
            return pipeline
        
        # Create imbalanced pipeline
        steps = pipeline.steps[:]
        steps.insert(-1, ('resampler', sampler))
        
        return ImbPipeline(steps)
    
    def _optimize_hyperparams(
        self, 
        model_name: str, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        n_trials: int = 50
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using randomized search.
        
        Args:
            model_name: Name of the model
            X_train: Training features
            y_train: Training target
            n_trials: Number of search trials
            
        Returns:
            Best hyperparameters
        """
        self.logger.info(f"Optimizing hyperparameters for {model_name}")
        
        # Get parameter space
        param_space = self.registry.get_param_space(model_name)
        
        # Create model with default params
        model = self.registry.create_model(model_name)
        pipeline = self._create_pipeline(model)
        
        # Add classifier prefix to parameter names
        param_space = {f'classifier__{k}': v for k, v in param_space.items()}
        
        # Perform randomized search
        search = RandomizedSearchCV(
            pipeline,
            param_space,
            n_iter=n_trials,
            cv=self.training_config.cv_folds,
            scoring='roc_auc',
            random_state=self.training_config.random_state,
            n_jobs=-1
        )
        
        search.fit(X_train, y_train)
        
        # Extract best parameters (remove classifier prefix)
        best_params = {}
        for key, value in search.best_params_.items():
            if key.startswith('classifier__'):
                clean_key = key.replace('classifier__', '')
                best_params[clean_key] = value
        
        self.logger.info(
            f"Hyperparameter optimization completed. "
            f"Best score: {search.best_score_:.4f}"
        )
        
        return best_params
    
    def _evaluate_model(
        self,
        model: Pipeline,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate model on validation and test sets.
        
        Args:
            model: Trained model pipeline
            X_val, y_val: Validation data
            X_test, y_test: Test data
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Validation predictions
        y_val_pred = model.predict(X_val)
        y_val_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Test predictions
        y_test_pred = model.predict(X_test)
        y_test_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics for validation set
        metrics['val_accuracy'] = accuracy_score(y_val, y_val_pred)
        metrics['val_precision'] = precision_score(y_val, y_val_pred)
        metrics['val_recall'] = recall_score(y_val, y_val_pred)
        metrics['val_f1'] = f1_score(y_val, y_val_pred)
        metrics['val_roc_auc'] = roc_auc_score(y_val, y_val_pred_proba)
        
        # Calculate PR AUC
        precision, recall, _ = precision_recall_curve(y_val, y_val_pred_proba)
        metrics['val_pr_auc'] = auc(recall, precision)
        
        # Log loss
        metrics['val_log_loss'] = log_loss(y_val, y_val_pred_proba)
        
        # Calculate metrics for test set
        metrics['test_accuracy'] = accuracy_score(y_test, y_test_pred)
        metrics['test_precision'] = precision_score(y_test, y_test_pred)
        metrics['test_recall'] = recall_score(y_test, y_test_pred)
        metrics['test_f1'] = f1_score(y_test, y_test_pred)
        metrics['test_roc_auc'] = roc_auc_score(y_test, y_test_pred_proba)
        
        # Use validation ROC AUC as primary metric for model selection
        metrics['roc_auc'] = metrics['val_roc_auc']
        
        return metrics
    
    def _cross_validate(
        self,
        pipeline: Pipeline,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> List[float]:
        """
        Perform cross-validation.
        
        Args:
            pipeline: Model pipeline
            X_train: Training features
            y_train: Training target
            
        Returns:
            List of cross-validation scores
        """
        cv = StratifiedKFold(
            n_splits=self.training_config.cv_folds,
            shuffle=True,
            random_state=self.training_config.random_state
        )
        
        scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=cv, scoring='roc_auc'
        )
        
        return scores.tolist()
    
    def _get_feature_importance(
        self,
        pipeline: Pipeline,
        feature_names: pd.Index
    ) -> Optional[Dict[str, float]]:
        """
        Extract feature importance from trained model.
        
        Args:
            pipeline: Trained pipeline
            feature_names: Names of input features
            
        Returns:
            Dictionary of feature importances or None
        """
        try:
            # Get the classifier from the pipeline
            classifier = pipeline.named_steps.get('classifier')
            
            if hasattr(classifier, 'feature_importances_'):
                # Tree-based models
                importances = classifier.feature_importances_
            elif hasattr(classifier, 'coef_'):
                # Linear models - use absolute values
                importances = np.abs(classifier.coef_[0])
            else:
                return None
            
            # Get feature names after preprocessing
            preprocessor = pipeline.named_steps.get('preprocessor')
            if hasattr(preprocessor, 'get_feature_names_out'):
                feature_names = preprocessor.get_feature_names_out()
            
            # Create importance dictionary
            importance_dict = dict(zip(feature_names, importances))
            
            # Sort by importance
            importance_dict = dict(
                sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            )
            
            return importance_dict
            
        except Exception as e:
            self.logger.warning(f"Could not extract feature importance: {e}")
            return None
    
    def get_best_model(self, metric: str = 'roc_auc') -> Optional[TrainingResult]:
        """
        Get the best performing model based on specified metric.
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            TrainingResult of best model or None if no models trained
        """
        if not self.results:
            return None
        
        best_result = max(
            self.results.values(),
            key=lambda x: x.metrics.get(metric, 0)
        )
        
        self.logger.info(
            f"Best model: {best_result.model_name} "
            f"({metric}={best_result.metrics.get(metric, 0):.4f})"
        )
        
        return best_result
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare all trained models.
        
        Returns:
            DataFrame with model comparison metrics
        """
        if not self.results:
            return pd.DataFrame()
        
        comparison_data = []
        
        for name, result in self.results.items():
            row = {
                'Model': name,
                'Training Time (s)': result.training_time,
                'CV Mean': result.cv_mean,
                'CV Std': result.cv_std,
                **result.metrics
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('roc_auc', ascending=False)
        
        return df
    
    def save_models(self, output_dir: Union[str, Path]) -> Dict[str, str]:
        """
        Save all trained models to disk.
        
        Args:
            output_dir: Directory to save models
            
        Returns:
            Dictionary mapping model names to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = {}
        
        for name, result in self.results.items():
            # Save the complete pipeline
            file_path = output_dir / f"{name}_pipeline.joblib"
            joblib.dump(result.pipeline, file_path)
            saved_paths[name] = str(file_path)
            
            self.logger.info(f"Saved {name} pipeline to {file_path}")
        
        # Save training results metadata
        results_path = output_dir / "training_results.joblib"
        joblib.dump(self.results, results_path)
        
        self.logger.info(f"Saved training results to {results_path}")
        
        return saved_paths
    
    def load_models(self, model_dir: Union[str, Path]) -> None:
        """
        Load previously trained models from disk.
        
        Args:
            model_dir: Directory containing saved models
        """
        model_dir = Path(model_dir)
        
        # Load training results
        results_path = model_dir / "training_results.joblib"
        if results_path.exists():
            self.results = joblib.load(results_path)
            
            # Load individual model pipelines
            for name in self.results.keys():
                pipeline_path = model_dir / f"{name}_pipeline.joblib"
                if pipeline_path.exists():
                    pipeline = joblib.load(pipeline_path)
                    self.results[name].pipeline = pipeline
                    self.models[name] = pipeline.named_steps.get('classifier')
            
            self.logger.info(f"Loaded {len(self.results)} models from {model_dir}")
        else:
            self.logger.warning(f"No training results found in {model_dir}")


def create_trainer(config_path: Optional[str] = None) -> ModelTrainer:
    """
    Factory function to create ModelTrainer instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured ModelTrainer instance
    """
    config = Config(config_path) if config_path else Config()
    return ModelTrainer(config)


def train_multiple_models(
    data: pd.DataFrame,
    model_names: List[str],
    config_path: Optional[str] = None,
    optimize_hyperparams: bool = False
) -> Dict[str, TrainingResult]:
    """
    Convenience function to train multiple models.
    
    Args:
        data: Training data
        model_names: List of models to train
        config_path: Path to configuration file
        optimize_hyperparams: Whether to optimize hyperparameters
        
    Returns:
        Dictionary of training results
    """
    trainer = create_trainer(config_path)
    return trainer.train(data, model_names, optimize_hyperparams)