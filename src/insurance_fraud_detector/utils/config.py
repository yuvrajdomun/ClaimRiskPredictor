"""
Configuration Management Module

This module provides centralized configuration management for the insurance
fraud detection system. It loads configuration from YAML files and provides
type-safe access to configuration parameters.

Classes:
    Config: Main configuration management class
    
Functions:
    load_config: Load configuration from file
    validate_config: Validate configuration parameters
    
Example:
    >>> from insurance_fraud_detector.utils.config import Config
    >>> config = Config()
    >>> models = config.get_enabled_models()
    >>> hyperparams = config.get_model_hyperparams('random_forest')
"""

import os
import yaml
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for individual models."""
    name: str
    hyperparams: Dict[str, Any]
    enabled: bool = True
    weight: float = 1.0

@dataclass
class DataConfig:
    """Configuration for data processing."""
    numerical_features: List[str] = field(default_factory=list)
    categorical_features: List[str] = field(default_factory=list)
    scaling_method: str = "standard"
    create_interactions: bool = True
    validation_rules: Dict[str, List[float]] = field(default_factory=dict)

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    test_size: float = 0.2
    validation_size: float = 0.15
    cv_folds: int = 5
    random_state: int = 42
    class_balance_method: str = "smote"
    early_stopping_enabled: bool = True

class Config:
    """
    Main configuration management class.
    
    This class loads configuration from YAML files and provides methods
    to access configuration parameters in a type-safe manner.
    
    Attributes:
        config (Dict): Raw configuration dictionary
        project_root (Path): Project root directory
        config_path (Path): Path to configuration file
    
    Example:
        >>> config = Config()
        >>> model_names = config.get_enabled_models()
        >>> rf_params = config.get_model_hyperparams('random_forest')
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        self.project_root = Path(__file__).parent.parent.parent.parent
        
        if config_path is None:
            self.config_path = self.project_root / "config" / "config.yaml"
        else:
            self.config_path = Path(config_path)
        
        self.config = self._load_config()
        self._validate_config()
        
        logger.info(f"Configuration loaded from {self.config_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Dict containing configuration parameters
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            yaml.YAMLError: If configuration file is invalid YAML
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            # Resolve relative paths
            config = self._resolve_paths(config)
            
            return config
            
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in configuration file: {e}")
            raise
    
    def _resolve_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve relative paths in configuration to absolute paths.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configuration with resolved paths
        """
        if 'paths' in config:
            for key, path in config['paths'].items():
                if not os.path.isabs(path):
                    config['paths'][key] = str(self.project_root / path)
        
        return config
    
    def _validate_config(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If configuration parameters are invalid
        """
        # Validate model ensemble weights sum to 1.0
        if 'models' in self.config and 'ensemble' in self.config['models']:
            weights = self.config['models']['ensemble'].get('weights', {})
            if weights and abs(sum(weights.values()) - 1.0) > 0.01:
                raise ValueError("Model ensemble weights must sum to 1.0")
        
        # Validate risk thresholds
        if 'evaluation' in self.config and 'risk_thresholds' in self.config['evaluation']:
            thresholds = self.config['evaluation']['risk_thresholds']
            low = thresholds.get('low_risk', 0.3)
            high = thresholds.get('high_risk', 0.7)
            
            if low >= high:
                raise ValueError("Low risk threshold must be less than high risk threshold")
            if not (0 <= low <= 1) or not (0 <= high <= 1):
                raise ValueError("Risk thresholds must be between 0 and 1")
        
        logger.info("Configuration validation passed")
    
    def get_enabled_models(self) -> List[str]:
        """
        Get list of enabled model names.
        
        Returns:
            List of enabled model names
        """
        return self.config.get('models', {}).get('ensemble', {}).get('enabled_models', [])
    
    def get_model_config(self, model_name: str) -> ModelConfig:
        """
        Get configuration for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            ModelConfig object with model parameters
            
        Raises:
            KeyError: If model not found in configuration
        """
        models_config = self.config.get('models', {})
        
        if model_name not in models_config:
            raise KeyError(f"Model '{model_name}' not found in configuration")
        
        hyperparams = models_config[model_name]
        enabled_models = self.get_enabled_models()
        weights = models_config.get('ensemble', {}).get('weights', {})
        
        return ModelConfig(
            name=model_name,
            hyperparams=hyperparams,
            enabled=model_name in enabled_models,
            weight=weights.get(model_name, 1.0)
        )
    
    def get_model_hyperparams(self, model_name: str) -> Dict[str, Any]:
        """
        Get hyperparameters for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary of hyperparameters
        """
        return self.get_model_config(model_name).hyperparams
    
    def get_data_config(self) -> DataConfig:
        """
        Get data processing configuration.
        
        Returns:
            DataConfig object with data processing parameters
        """
        data_config = self.config.get('data', {})
        features = data_config.get('features', {})
        
        return DataConfig(
            numerical_features=features.get('numerical', []),
            categorical_features=features.get('categorical', []),
            scaling_method=features.get('scaling_method', 'standard'),
            create_interactions=features.get('create_interactions', True),
            validation_rules=data_config.get('validation', {})
        )
    
    def get_training_config(self) -> TrainingConfig:
        """
        Get training configuration.
        
        Returns:
            TrainingConfig object with training parameters
        """
        training_config = self.config.get('training', {})
        data_split = self.config.get('data', {}).get('train_test_split', {})
        cv_config = training_config.get('cross_validation', {})
        balance_config = training_config.get('class_balance', {})
        early_stop_config = training_config.get('early_stopping', {})
        
        return TrainingConfig(
            test_size=data_split.get('test_size', 0.2),
            validation_size=data_split.get('validation_size', 0.15),
            cv_folds=cv_config.get('cv_folds', 5),
            random_state=data_split.get('random_state', 42),
            class_balance_method=balance_config.get('method', 'smote'),
            early_stopping_enabled=early_stop_config.get('enabled', True)
        )
    
    def get_risk_thresholds(self) -> Dict[str, float]:
        """
        Get risk classification thresholds.
        
        Returns:
            Dictionary with low_risk and high_risk thresholds
        """
        return self.config.get('evaluation', {}).get('risk_thresholds', {
            'low_risk': 0.3,
            'high_risk': 0.7
        })
    
    def get_metrics(self) -> List[str]:
        """
        Get list of metrics to calculate during evaluation.
        
        Returns:
            List of metric names
        """
        return self.config.get('evaluation', {}).get('metrics', [
            'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'
        ])
    
    def get_bias_detection_config(self) -> Dict[str, Any]:
        """
        Get bias detection configuration.
        
        Returns:
            Dictionary with bias detection parameters
        """
        return self.config.get('bias_detection', {})
    
    def get_api_config(self) -> Dict[str, Any]:
        """
        Get API configuration.
        
        Returns:
            Dictionary with API parameters
        """
        return self.config.get('api', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """
        Get logging configuration.
        
        Returns:
            Dictionary with logging parameters
        """
        return self.config.get('logging', {})
    
    def get_paths(self) -> Dict[str, str]:
        """
        Get configured paths.
        
        Returns:
            Dictionary with path configurations
        """
        return self.config.get('paths', {})
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key with dot notation support.
        
        Args:
            key: Configuration key (supports dot notation, e.g., 'models.ensemble.weights')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key with dot notation support.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        # Navigate to parent dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
        logger.info(f"Configuration updated: {key} = {value}")
    
    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            path: Path to save configuration. If None, uses original path.
        """
        save_path = Path(path) if path else self.config_path
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to {save_path}")


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config object
    """
    return Config(config_path)


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    temp_config = Config.__new__(Config)
    temp_config.config = config
    temp_config._validate_config()
    return True