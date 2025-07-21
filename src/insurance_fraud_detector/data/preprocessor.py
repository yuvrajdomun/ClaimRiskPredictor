"""
Data Preprocessing Module

This module provides comprehensive data preprocessing capabilities for the
insurance fraud detection system. It handles feature engineering, data
validation, scaling, and encoding operations.

Classes:
    DataPreprocessor: Main preprocessing pipeline class
    FeatureEngineer: Feature engineering utilities
    DataValidator: Data validation and cleaning
    
Functions:
    create_preprocessor: Factory function to create preprocessor
    validate_data: Validate data quality and completeness
    
Example:
    >>> from insurance_fraud_detector.data.preprocessor import DataPreprocessor
    >>> preprocessor = DataPreprocessor(config)
    >>> processed_data = preprocessor.fit_transform(raw_data)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
import warnings
from pathlib import Path

# Scikit-learn imports
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, 
    OneHotEncoder, LabelEncoder, OrdinalEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif

# Local imports
from ..utils.config import Config
from ..utils.logger import LoggerMixin, log_execution_time

class FeatureValidator:
    """
    Utility class for validating feature values against business rules.
    
    This class contains validation rules specific to insurance claims
    to ensure data quality and catch obvious errors or outliers.
    """
    
    @staticmethod
    def validate_age(age: Union[int, float]) -> bool:
        """Validate driver age."""
        return 16 <= age <= 100
    
    @staticmethod
    def validate_vehicle_age(vehicle_age: Union[int, float]) -> bool:
        """Validate vehicle age."""
        return 0 <= vehicle_age <= 50
    
    @staticmethod
    def validate_annual_mileage(mileage: Union[int, float]) -> bool:
        """Validate annual mileage."""
        return 0 <= mileage <= 100000
    
    @staticmethod
    def validate_claim_amount(amount: Union[int, float]) -> bool:
        """Validate claim amount."""
        return 0 <= amount <= 1000000
    
    @staticmethod
    def validate_credit_score(score: Union[int, float]) -> bool:
        """Validate credit score."""
        return 300 <= score <= 850
    
    @staticmethod
    def validate_driving_violations(violations: Union[int, float]) -> bool:
        """Validate driving violations count."""
        return 0 <= violations <= 20
    
    @staticmethod
    def validate_previous_claims(claims: Union[int, float]) -> bool:
        """Validate previous claims count."""
        return 0 <= claims <= 50
    
    @staticmethod
    def validate_gender(gender: str) -> bool:
        """Validate gender value."""
        return gender.upper() in ['M', 'F', 'MALE', 'FEMALE']
    
    @staticmethod
    def validate_vehicle_type(vehicle_type: str) -> bool:
        """Validate vehicle type."""
        valid_types = ['sedan', 'suv', 'truck', 'sports', 'luxury', 'hatchback', 'convertible']
        return vehicle_type.lower() in valid_types
    
    @staticmethod
    def validate_region(region: str) -> bool:
        """Validate region type."""
        valid_regions = ['urban', 'suburban', 'rural']
        return region.lower() in valid_regions


class FeatureEngineer(BaseEstimator, TransformerMixin, LoggerMixin):
    """
    Feature engineering transformer.
    
    This class creates new features from existing ones to improve model
    performance. It includes domain-specific feature engineering for
    insurance fraud detection.
    
    Features created:
    - claim_per_mile: claim_amount / annual_mileage
    - age_vehicle_interaction: age * vehicle_age
    - risk_score: composite risk score
    - claim_amount_bins: binned claim amounts
    - age_groups: age group categories
    """
    
    def __init__(self, create_interactions: bool = True, create_bins: bool = True):
        """
        Initialize feature engineer.
        
        Args:
            create_interactions: Whether to create interaction features
            create_bins: Whether to create binned features
        """
        self.create_interactions = create_interactions
        self.create_bins = create_bins
        self.fitted = False
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureEngineer':
        """
        Fit the feature engineer (calculate bin edges, etc.).
        
        Args:
            X: Input features
            y: Target variable (unused)
            
        Returns:
            Self
        """
        self.logger.info("Fitting feature engineer")
        
        # Calculate bin edges for continuous variables
        if self.create_bins and 'claim_amount' in X.columns:
            self.claim_amount_bins = np.percentile(
                X['claim_amount'].dropna(), 
                [0, 25, 50, 75, 90, 100]
            )
        
        if self.create_bins and 'age' in X.columns:
            self.age_bins = [0, 25, 35, 45, 55, 100]
        
        self.fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering transformations.
        
        Args:
            X: Input features
            
        Returns:
            Transformed features with engineered features added
        """
        if not self.fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")
        
        X_transformed = X.copy()
        
        # Claim per mile ratio
        if 'claim_amount' in X.columns and 'annual_mileage' in X.columns:
            X_transformed['claim_per_mile'] = (
                X_transformed['claim_amount'] / (X_transformed['annual_mileage'] + 1)
            )
            
        # Risk indicators
        if 'driving_violations' in X.columns:
            X_transformed['has_violations'] = (X_transformed['driving_violations'] > 0).astype(int)
            X_transformed['multiple_violations'] = (X_transformed['driving_violations'] > 1).astype(int)
        
        if 'previous_claims' in X.columns:
            X_transformed['has_previous_claims'] = (X_transformed['previous_claims'] > 0).astype(int)
            X_transformed['frequent_claimer'] = (X_transformed['previous_claims'] > 2).astype(int)
        
        # Credit risk indicators
        if 'credit_score' in X.columns:
            X_transformed['low_credit'] = (X_transformed['credit_score'] < 650).astype(int)
            X_transformed['excellent_credit'] = (X_transformed['credit_score'] > 750).astype(int)
        
        # Vehicle risk indicators
        if 'vehicle_age' in X.columns:
            X_transformed['old_vehicle'] = (X_transformed['vehicle_age'] > 10).astype(int)
            X_transformed['new_vehicle'] = (X_transformed['vehicle_age'] < 3).astype(int)
        
        # High mileage indicator
        if 'annual_mileage' in X.columns:
            X_transformed['high_mileage'] = (X_transformed['annual_mileage'] > 15000).astype(int)
        
        # Interaction features
        if self.create_interactions:
            if 'age' in X.columns and 'driving_violations' in X.columns:
                X_transformed['age_violations_interaction'] = (
                    X_transformed['age'] * X_transformed['driving_violations']
                )
            
            if 'claim_amount' in X.columns and 'previous_claims' in X.columns:
                X_transformed['claim_history_interaction'] = (
                    np.log1p(X_transformed['claim_amount']) * X_transformed['previous_claims']
                )
        
        # Binned features
        if self.create_bins:
            if hasattr(self, 'claim_amount_bins') and 'claim_amount' in X.columns:
                X_transformed['claim_amount_bin'] = pd.cut(
                    X_transformed['claim_amount'], 
                    bins=self.claim_amount_bins,
                    labels=['very_low', 'low', 'medium', 'high', 'very_high'],
                    include_lowest=True
                ).astype(str)
            
            if hasattr(self, 'age_bins') and 'age' in X.columns:
                X_transformed['age_group'] = pd.cut(
                    X_transformed['age'],
                    bins=self.age_bins,
                    labels=['young', 'middle_young', 'middle', 'middle_old', 'old'],
                    include_lowest=True
                ).astype(str)
        
        # Composite risk score
        risk_features = []
        weights = []
        
        if 'driving_violations' in X_transformed.columns:
            risk_features.append(X_transformed['driving_violations'])
            weights.append(0.3)
        
        if 'previous_claims' in X_transformed.columns:
            risk_features.append(X_transformed['previous_claims'])
            weights.append(0.25)
        
        if 'low_credit' in X_transformed.columns:
            risk_features.append(X_transformed['low_credit'])
            weights.append(0.2)
        
        if 'high_mileage' in X_transformed.columns:
            risk_features.append(X_transformed['high_mileage'])
            weights.append(0.15)
        
        if 'old_vehicle' in X_transformed.columns:
            risk_features.append(X_transformed['old_vehicle'])
            weights.append(0.1)
        
        if risk_features:
            risk_matrix = np.column_stack(risk_features)
            weights = np.array(weights)
            X_transformed['composite_risk_score'] = np.average(risk_matrix, axis=1, weights=weights)
        
        self.logger.info(f"Feature engineering completed. Added {len(X_transformed.columns) - len(X.columns)} features")
        
        return X_transformed


class DataValidator(LoggerMixin):
    """
    Data validation and cleaning utility.
    
    This class provides comprehensive data validation for insurance
    claims data, including outlier detection, missing value analysis,
    and business rule validation.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize data validator.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.validation_rules = self.config.get('data.validation', {})
        
    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate entire dataframe and return validation report.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        self.logger.info("Starting data validation")
        
        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': {},
            'outliers': {},
            'invalid_values': {},
            'duplicates': 0,
            'warnings': [],
            'errors': []
        }
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        report['missing_values'] = missing_counts[missing_counts > 0].to_dict()
        
        # Check for duplicates
        report['duplicates'] = df.duplicated().sum()
        
        # Validate individual columns
        for column in df.columns:
            if column in df.columns:
                validation_result = self._validate_column(df[column], column)
                if validation_result['outliers'] > 0:
                    report['outliers'][column] = validation_result['outliers']
                if validation_result['invalid_values'] > 0:
                    report['invalid_values'][column] = validation_result['invalid_values']
                
                report['warnings'].extend(validation_result['warnings'])
                report['errors'].extend(validation_result['errors'])
        
        # Overall data quality score
        total_issues = (
            sum(report['missing_values'].values()) +
            sum(report['outliers'].values()) +
            sum(report['invalid_values'].values()) +
            report['duplicates']
        )
        
        report['quality_score'] = max(0, 1 - (total_issues / len(df)))
        
        self.logger.info(f"Data validation completed. Quality score: {report['quality_score']:.3f}")
        
        return report
    
    def _validate_column(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """
        Validate individual column.
        
        Args:
            series: Column data
            column_name: Name of the column
            
        Returns:
            Validation results for the column
        """
        result = {
            'outliers': 0,
            'invalid_values': 0,
            'warnings': [],
            'errors': []
        }
        
        # Skip validation for columns with all missing values
        if series.isnull().all():
            result['errors'].append(f"Column '{column_name}' has all missing values")
            return result
        
        # Numerical validation
        if series.dtype in ['int64', 'float64']:
            result.update(self._validate_numerical_column(series, column_name))
        
        # Categorical validation
        elif series.dtype == 'object':
            result.update(self._validate_categorical_column(series, column_name))
        
        return result
    
    def _validate_numerical_column(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """Validate numerical column."""
        result = {'outliers': 0, 'invalid_values': 0, 'warnings': [], 'errors': []}
        
        # Get validation rules for this column
        column_rules = self.validation_rules.get(f'{column_name}_range')
        
        if column_rules:
            min_val, max_val = column_rules
            
            # Count values outside valid range
            invalid_mask = (series < min_val) | (series > max_val)
            result['invalid_values'] = invalid_mask.sum()
            
            if result['invalid_values'] > 0:
                result['warnings'].append(
                    f"Column '{column_name}' has {result['invalid_values']} values outside valid range [{min_val}, {max_val}]"
                )
        
        # Outlier detection using IQR method
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_mask = (series < lower_bound) | (series > upper_bound)
        result['outliers'] = outlier_mask.sum()
        
        if result['outliers'] > len(series) * 0.05:  # More than 5% outliers
            result['warnings'].append(
                f"Column '{column_name}' has {result['outliers']} outliers ({result['outliers']/len(series)*100:.1f}%)"
            )
        
        return result
    
    def _validate_categorical_column(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """Validate categorical column."""
        result = {'outliers': 0, 'invalid_values': 0, 'warnings': [], 'errors': []}
        
        # Check for unexpected categories
        if column_name == 'gender':
            valid_values = ['M', 'F', 'Male', 'Female', 'male', 'female']
            invalid_mask = ~series.str.upper().isin([v.upper() for v in valid_values])
            result['invalid_values'] = invalid_mask.sum()
        
        elif column_name == 'vehicle_type':
            valid_values = ['sedan', 'suv', 'truck', 'sports', 'luxury', 'hatchback', 'convertible']
            invalid_mask = ~series.str.lower().isin(valid_values)
            result['invalid_values'] = invalid_mask.sum()
        
        elif column_name == 'region':
            valid_values = ['urban', 'suburban', 'rural']
            invalid_mask = ~series.str.lower().isin(valid_values)
            result['invalid_values'] = invalid_mask.sum()
        
        # Check for too many unique values (possible data quality issue)
        unique_ratio = series.nunique() / len(series)
        if unique_ratio > 0.9 and len(series) > 100:
            result['warnings'].append(
                f"Column '{column_name}' has high cardinality ({series.nunique()} unique values)"
            )
        
        return result
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data based on validation results.
        
        Args:
            df: DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        df_cleaned = df.copy()
        
        self.logger.info("Starting data cleaning")
        
        # Remove duplicates
        initial_rows = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates()
        removed_duplicates = initial_rows - len(df_cleaned)
        
        if removed_duplicates > 0:
            self.logger.info(f"Removed {removed_duplicates} duplicate rows")
        
        # Clean specific columns
        if 'gender' in df_cleaned.columns:
            df_cleaned['gender'] = df_cleaned['gender'].str.upper().map({'M': 'M', 'MALE': 'M', 'F': 'F', 'FEMALE': 'F'})
        
        if 'vehicle_type' in df_cleaned.columns:
            df_cleaned['vehicle_type'] = df_cleaned['vehicle_type'].str.lower()
        
        if 'region' in df_cleaned.columns:
            df_cleaned['region'] = df_cleaned['region'].str.lower()
        
        # Cap outliers for numerical columns
        numerical_columns = df_cleaned.select_dtypes(include=[np.number]).columns
        
        for col in numerical_columns:
            if col in self.validation_rules:
                min_val, max_val = self.validation_rules[f'{col}_range']
                df_cleaned[col] = df_cleaned[col].clip(lower=min_val, upper=max_val)
        
        self.logger.info("Data cleaning completed")
        
        return df_cleaned


class DataPreprocessor(LoggerMixin):
    """
    Main data preprocessing pipeline.
    
    This class orchestrates the complete data preprocessing pipeline including
    feature engineering, validation, scaling, and encoding. It provides a
    scikit-learn compatible interface for easy integration with ML pipelines.
    
    Attributes:
        config: Configuration object
        feature_engineer: Feature engineering component
        validator: Data validation component
        fitted: Whether the preprocessor has been fitted
        
    Example:
        >>> preprocessor = DataPreprocessor(config)
        >>> pipeline = preprocessor.create_preprocessor()
        >>> processed_data = pipeline.fit_transform(raw_data)
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize data preprocessor.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.data_config = self.config.get_data_config()
        
        # Initialize components
        self.feature_engineer = FeatureEngineer(
            create_interactions=self.data_config.create_interactions,
            create_bins=True
        )
        
        self.validator = DataValidator(self.config)
        self.fitted = False
        
        self.logger.info("DataPreprocessor initialized")
    
    def create_preprocessor(self) -> ColumnTransformer:
        """
        Create scikit-learn preprocessing pipeline.
        
        Returns:
            ColumnTransformer with preprocessing steps
        """
        # Get feature lists
        numerical_features = self.data_config.numerical_features
        categorical_features = self.data_config.categorical_features
        
        # Numerical preprocessing pipeline
        numerical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', self._get_scaler())
        ])
        
        # Categorical preprocessing pipeline
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine transformers
        preprocessor = ColumnTransformer([
            ('numerical', numerical_transformer, numerical_features),
            ('categorical', categorical_transformer, categorical_features)
        ])
        
        return preprocessor
    
    def _get_scaler(self):
        """Get configured scaler."""
        scaling_method = self.data_config.scaling_method
        
        if scaling_method == 'standard':
            return StandardScaler()
        elif scaling_method == 'minmax':
            return MinMaxScaler()
        elif scaling_method == 'robust':
            return RobustScaler()
        else:
            return StandardScaler()  # Default
    
    @log_execution_time
    def preprocess_data(
        self, 
        data: pd.DataFrame,
        validate: bool = True,
        clean: bool = True,
        engineer_features: bool = True
    ) -> pd.DataFrame:
        """
        Complete data preprocessing pipeline.
        
        Args:
            data: Raw input data
            validate: Whether to validate data quality
            clean: Whether to clean data
            engineer_features: Whether to create engineered features
            
        Returns:
            Preprocessed data
        """
        self.logger.info("Starting complete data preprocessing")
        
        processed_data = data.copy()
        
        # Data validation
        if validate:
            validation_report = self.validator.validate_dataframe(processed_data)
            
            if validation_report['quality_score'] < 0.8:
                self.logger.warning(f"Data quality score is low: {validation_report['quality_score']:.3f}")
            
            # Log validation issues
            if validation_report['errors']:
                for error in validation_report['errors']:
                    self.logger.error(error)
            
            if validation_report['warnings']:
                for warning in validation_report['warnings']:
                    self.logger.warning(warning)
        
        # Data cleaning
        if clean:
            processed_data = self.validator.clean_data(processed_data)
        
        # Feature engineering
        if engineer_features:
            if not self.feature_engineer.fitted:
                self.feature_engineer.fit(processed_data)
            
            processed_data = self.feature_engineer.transform(processed_data)
        
        self.logger.info(f"Data preprocessing completed. Shape: {processed_data.shape}")
        
        return processed_data
    
    def get_feature_names(self, preprocessor: ColumnTransformer) -> List[str]:
        """
        Get feature names after preprocessing.
        
        Args:
            preprocessor: Fitted preprocessor
            
        Returns:
            List of feature names
        """
        try:
            return preprocessor.get_feature_names_out().tolist()
        except AttributeError:
            # Fallback for older scikit-learn versions
            feature_names = []
            
            for name, transformer, features in preprocessor.transformers_:
                if name == 'numerical':
                    feature_names.extend(features)
                elif name == 'categorical':
                    if hasattr(transformer.named_steps['encoder'], 'get_feature_names_out'):
                        cat_features = transformer.named_steps['encoder'].get_feature_names_out(features)
                        feature_names.extend(cat_features)
                    else:
                        feature_names.extend(features)
            
            return feature_names


def create_preprocessor(config_path: Optional[str] = None) -> DataPreprocessor:
    """
    Factory function to create DataPreprocessor instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured DataPreprocessor instance
    """
    config = Config(config_path) if config_path else Config()
    return DataPreprocessor(config)


def validate_data(data: pd.DataFrame, config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Validate data quality and return report.
    
    Args:
        data: DataFrame to validate
        config_path: Path to configuration file
        
    Returns:
        Validation report
    """
    config = Config(config_path) if config_path else Config()
    validator = DataValidator(config)
    return validator.validate_dataframe(data)