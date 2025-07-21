"""
Causal inference module for what-if analysis on insurance claims.
Uses DoWhy library for causal effect estimation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging

try:
    import dowhy
    from dowhy import CausalModel
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("DoWhy not available. Using simplified causal analysis.")

logger = logging.getLogger(__name__)

class CausalAnalyzer:
    """Performs causal inference for insurance claim risk factors."""
    
    def __init__(self):
        self.causal_model = None
        self.data = None
        self.treatment_vars = ['driving_violations', 'vehicle_age', 'annual_mileage', 'previous_claims']
        self.outcome_var = 'is_fraud'
        self.confounders = ['age', 'gender', 'credit_score', 'region', 'vehicle_type']
        
    def prepare_causal_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for causal analysis."""
        df_causal = df.copy()
        
        # Encode categorical variables for causal analysis
        categorical_cols = ['gender', 'vehicle_type', 'region']
        for col in categorical_cols:
            if col in df_causal.columns:
                df_causal[col] = pd.Categorical(df_causal[col]).codes
        
        return df_causal
    
    def build_causal_graph(self) -> str:
        """Define causal graph structure."""
        # Simple causal graph for insurance claims
        graph = """
        digraph {
            age -> driving_violations;
            age -> is_fraud;
            gender -> driving_violations;
            gender -> is_fraud;
            credit_score -> is_fraud;
            vehicle_type -> vehicle_age;
            vehicle_type -> is_fraud;
            region -> annual_mileage;
            region -> is_fraud;
            driving_violations -> is_fraud;
            vehicle_age -> is_fraud;
            annual_mileage -> is_fraud;
            previous_claims -> is_fraud;
        }
        """
        return graph
    
    def analyze_treatment_effect(self, df: pd.DataFrame, treatment: str, 
                                treatment_value: float) -> Dict[str, Any]:
        """Analyze causal effect of a treatment on fraud probability."""
        
        if not DOWHY_AVAILABLE:
            return self._simple_causal_analysis(df, treatment, treatment_value)
        
        try:
            df_causal = self.prepare_causal_data(df)
            
            # Create binary treatment variable
            treatment_median = df_causal[treatment].median()
            df_causal[f'{treatment}_high'] = (df_causal[treatment] > treatment_median).astype(int)
            
            # Build causal model
            graph = self.build_causal_graph().replace(treatment, f'{treatment}_high')
            
            model = CausalModel(
                data=df_causal,
                treatment=f'{treatment}_high',
                outcome=self.outcome_var,
                graph=graph,
                common_causes=self.confounders
            )
            
            # Identify causal effect
            identified_estimand = model.identify_effect()
            
            # Estimate causal effect
            causal_estimate = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.linear_regression"
            )
            
            # Refute the estimate
            refutation = model.refute_estimate(
                identified_estimand, 
                causal_estimate,
                method_name="random_common_cause"
            )
            
            return {
                'treatment': treatment,
                'treatment_threshold': treatment_median,
                'causal_effect': float(causal_estimate.value),
                'confidence_interval': [float(causal_estimate.value - 1.96 * causal_estimate.stderr),
                                      float(causal_estimate.value + 1.96 * causal_estimate.stderr)],
                'p_value': float(causal_estimate.p_value) if hasattr(causal_estimate, 'p_value') else None,
                'refutation_successful': abs(refutation.new_effect - causal_estimate.value) < 0.05
            }
            
        except Exception as e:
            logger.error(f"DoWhy causal analysis failed: {e}")
            return self._simple_causal_analysis(df, treatment, treatment_value)
    
    def _simple_causal_analysis(self, df: pd.DataFrame, treatment: str, 
                               treatment_value: float) -> Dict[str, Any]:
        """Simple correlation-based causal analysis fallback."""
        
        # Split data into treatment groups
        treatment_median = df[treatment].median()
        high_treatment = df[df[treatment] > treatment_median]
        low_treatment = df[df[treatment] <= treatment_median]
        
        # Calculate fraud rates
        high_fraud_rate = high_treatment[self.outcome_var].mean()
        low_fraud_rate = low_treatment[self.outcome_var].mean()
        
        # Calculate effect size
        effect_size = high_fraud_rate - low_fraud_rate
        
        # Simple statistical test
        from scipy import stats
        try:
            _, p_value = stats.ttest_ind(
                high_treatment[self.outcome_var],
                low_treatment[self.outcome_var]
            )
        except:
            p_value = None
        
        return {
            'treatment': treatment,
            'treatment_threshold': treatment_median,
            'causal_effect': effect_size,
            'high_treatment_fraud_rate': high_fraud_rate,
            'low_treatment_fraud_rate': low_fraud_rate,
            'p_value': p_value,
            'method': 'simple_comparison'
        }
    
    def what_if_analysis(self, df: pd.DataFrame, base_claim: Dict[str, Any], 
                        interventions: Dict[str, Any]) -> Dict[str, Any]:
        """Perform what-if analysis by changing specific variables."""
        
        results = {}
        
        # Baseline prediction (assuming we have access to a predictor)
        baseline_data = pd.DataFrame([base_claim])
        
        for intervention_var, new_value in interventions.items():
            if intervention_var not in base_claim:
                continue
                
            # Create intervention data
            intervention_data = baseline_data.copy()
            intervention_data[intervention_var] = new_value
            
            # Calculate causal effect for this intervention
            causal_result = self.analyze_treatment_effect(df, intervention_var, new_value)
            
            # Calculate expected change in fraud probability
            if intervention_var in self.treatment_vars:
                treatment_median = df[intervention_var].median()
                if new_value > treatment_median and base_claim[intervention_var] <= treatment_median:
                    expected_change = causal_result['causal_effect']
                elif new_value <= treatment_median and base_claim[intervention_var] > treatment_median:
                    expected_change = -causal_result['causal_effect']
                else:
                    expected_change = 0
            else:
                expected_change = 0
            
            results[intervention_var] = {
                'original_value': base_claim[intervention_var],
                'new_value': new_value,
                'expected_fraud_change': expected_change,
                'causal_analysis': causal_result
            }
        
        return results
    
    def get_policy_recommendations(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate policy recommendations based on causal analysis."""
        
        recommendations = []
        
        for treatment in self.treatment_vars:
            result = self.analyze_treatment_effect(df, treatment, df[treatment].median())
            
            if result['causal_effect'] > 0.05:  # Significant positive effect on fraud
                recommendations.append({
                    'factor': treatment,
                    'recommendation': f'Implement stricter policies for high {treatment}',
                    'expected_fraud_reduction': result['causal_effect'],
                    'confidence': 'high' if result.get('p_value', 1) < 0.05 else 'medium'
                })
        
        # Sort by impact
        recommendations.sort(key=lambda x: x['expected_fraud_reduction'], reverse=True)
        
        return recommendations

class BiasDetector:
    """Detects potential bias in fraud prediction models."""
    
    def __init__(self):
        self.protected_attributes = ['age', 'gender', 'region']
        
    def detect_disparate_impact(self, df: pd.DataFrame, predictions: np.ndarray, 
                               threshold: float = 0.8) -> Dict[str, Any]:
        """Detect disparate impact across protected groups."""
        
        results = {}
        df_analysis = df.copy()
        df_analysis['prediction'] = predictions
        df_analysis['high_risk'] = predictions > 0.5
        
        for attr in self.protected_attributes:
            if attr not in df.columns:
                continue
                
            group_rates = df_analysis.groupby(attr)['high_risk'].mean()
            
            # Calculate disparate impact ratios
            max_rate = group_rates.max()
            min_rate = group_rates.min()
            
            disparate_impact_ratio = min_rate / max_rate if max_rate > 0 else 1.0
            
            results[attr] = {
                'group_rates': group_rates.to_dict(),
                'disparate_impact_ratio': disparate_impact_ratio,
                'passes_80_rule': disparate_impact_ratio >= threshold,
                'max_rate': max_rate,
                'min_rate': min_rate
            }
        
        return results
    
    def fairness_report(self, df: pd.DataFrame, predictions: np.ndarray) -> Dict[str, Any]:
        """Generate comprehensive fairness report."""
        
        disparate_impact = self.detect_disparate_impact(df, predictions)
        
        # Overall fairness score
        fairness_scores = [result['disparate_impact_ratio'] for result in disparate_impact.values()]
        overall_fairness = np.mean(fairness_scores) if fairness_scores else 1.0
        
        return {
            'disparate_impact': disparate_impact,
            'overall_fairness_score': overall_fairness,
            'bias_detected': overall_fairness < 0.8,
            'recommendations': self._generate_bias_recommendations(disparate_impact)
        }
    
    def _generate_bias_recommendations(self, disparate_impact: Dict) -> List[str]:
        """Generate recommendations to address bias."""
        
        recommendations = []
        
        for attr, results in disparate_impact.items():
            if not results['passes_80_rule']:
                recommendations.append(
                    f"Address bias in {attr}: disparate impact ratio is {results['disparate_impact_ratio']:.3f}"
                )
        
        if not recommendations:
            recommendations.append("No significant bias detected across protected attributes")
        
        return recommendations

if __name__ == "__main__":
    from data_loader import InsuranceDataLoader
    from model import FraudPredictor
    
    # Load data and train model
    loader = InsuranceDataLoader()
    data = loader.load_data()
    
    predictor = FraudPredictor()
    predictor.train(data)
    
    # Causal analysis
    analyzer = CausalAnalyzer()
    
    # Test what-if analysis
    base_claim = {
        'age': 35,
        'gender': 'M',
        'vehicle_age': 3,
        'vehicle_type': 'sedan',
        'annual_mileage': 15000,
        'driving_violations': 1,
        'claim_amount': 25000,
        'previous_claims': 0,
        'credit_score': 700,
        'region': 'suburban'
    }
    
    interventions = {
        'driving_violations': 5,
        'previous_claims': 3,
        'annual_mileage': 30000
    }
    
    what_if_results = analyzer.what_if_analysis(data, base_claim, interventions)
    
    print("What-if Analysis Results:")
    for var, result in what_if_results.items():
        print(f"{var}: {result['original_value']} -> {result['new_value']}")
        print(f"  Expected fraud probability change: {result['expected_fraud_change']:.4f}")
    
    # Bias detection
    predictions = predictor.predict(data)
    bias_detector = BiasDetector()
    fairness_report = bias_detector.fairness_report(data, predictions)
    
    print(f"\nFairness Report:")
    print(f"Overall fairness score: {fairness_report['overall_fairness_score']:.3f}")
    print(f"Bias detected: {fairness_report['bias_detected']}")
    print("Recommendations:")
    for rec in fairness_report['recommendations']:
        print(f"  - {rec}")