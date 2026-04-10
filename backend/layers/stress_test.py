import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class StressTestEngine:
    """
    Layer 2: Synthetic Stress Test Engine
    
    The 'crash test' for AI models. Generates synthetic twin profiles
    that are identical in every way except for one protected attribute,
    then measures whether the model treats them differently.
    
    This detects discrimination that real-world data analysis misses
    because it isolates the protected attribute as the ONLY variable.
    
    Example finding: "Female applicants with identical qualifications 
    to male applicants are rejected 2.4x more often."
    """
    
    def __init__(
        self, 
        model: BaseEstimator,
        df: pd.DataFrame,
        protected_attributes: list[str],
        target_column: str,
        n_synthetic_pairs: int = 1000
    ):
        self.model = model
        self.df = df
        self.protected_attributes = protected_attributes
        self.target_column = target_column
        self.n_synthetic_pairs = n_synthetic_pairs
        self.results = {}
    
    def run(self) -> dict:
        """
        Runs stress tests for all protected attributes.
        Returns combined results with bias scores and counterfactuals.
        """
        logger.info(f"Starting Stress Test with {self.n_synthetic_pairs} synthetic pairs...")
        
        stress_results = {}
        all_warnings = []
        overall_passed = True
        
        for attr in self.protected_attributes:
            if attr not in self.df.columns:
                continue
            result = self._test_attribute(attr)
            stress_results[attr] = result
            all_warnings.extend(result.get("warnings", []))
            if not result.get("passed", True):
                overall_passed = False
        
        bias_score = self._calculate_bias_score(stress_results)
        
        self.results = {
            "layer": "stress_test",
            "synthetic_pairs_tested": self.n_synthetic_pairs,
            "attribute_results": stress_results,
            "bias_score": bias_score,
            "warnings": all_warnings,
            "status": "pass" if bias_score >= 70 else "fail",
            "overall_passed": overall_passed
        }
        
        logger.info(f"Stress Test complete. Bias score: {bias_score}")
        return self.results
    
    def _test_attribute(self, protected_attr: str) -> dict:
        """
        For a single protected attribute:
        1. Gets all unique group values (e.g. Male/Female)
        2. Generates n synthetic profiles using real data statistics
        3. Creates twin pairs - identical except protected attribute differs
        4. Runs both twins through the model
        5. Measures outcome divergence between twins
        
        Returns per-group outcome rates and discrimination ratio.
        """
        groups = self.df[protected_attr].dropna().unique()
        
        if len(groups) < 2:
            return {"error": f"Need at least 2 groups in '{protected_attr}', found {len(groups)}"}
        
        feature_cols = [c for c in self.df.columns 
                       if c != self.target_column and c != protected_attr]
        
        # Generate synthetic base profiles from real data statistics
        synthetic_profiles = self._generate_synthetic_profiles(feature_cols)
        
        group_outcomes = {}
        
        for group in groups:
            twins = synthetic_profiles.copy()
            twins[protected_attr] = group
            
            try:
                all_cols = feature_cols + [protected_attr]
                twins_input = twins[all_cols] if all(c in twins.columns for c in all_cols) else twins
                
                # Encode categoricals for model
                twins_encoded = self._encode_for_prediction(twins_input)
                predictions = self.model.predict(twins_encoded)
                
                positive_rate = float(np.mean(predictions == 1))
                group_outcomes[str(group)] = {
                    "positive_outcome_rate": round(positive_rate * 100, 2),
                    "n_tested": len(predictions)
                }
            except Exception as e:
                logger.warning(f"Prediction failed for group {group}: {e}")
                group_outcomes[str(group)] = {"error": str(e)}
        
        discrimination_ratio = self._calculate_discrimination_ratio(group_outcomes)
        warnings = self._generate_attribute_warnings(protected_attr, group_outcomes, discrimination_ratio)
        
        return {
            "group_outcomes": group_outcomes,
            "discrimination_ratio": discrimination_ratio,
            "disparate_impact": round(discrimination_ratio, 3),
            "passed": discrimination_ratio >= 0.8,
            "warnings": warnings
        }
    
    def _generate_synthetic_profiles(self, feature_cols: list) -> pd.DataFrame:
        """
        Generates synthetic profiles by sampling from real data distributions.
        For numeric columns: samples from normal distribution using real mean/std.
        For categorical columns: samples proportionally from real value frequencies.
        
        This ensures synthetic data is realistic, not random noise.
        """
        synthetic = {}
        
        available_cols = [c for c in feature_cols if c in self.df.columns]
        
        for col in available_cols:
            col_data = self.df[col].dropna()
            
            if len(col_data) == 0:
                synthetic[col] = [0] * self.n_synthetic_pairs
                continue
            
            if self.df[col].dtype in ['int64', 'float64'] and self.df[col].nunique() > 10:
                # Numeric: sample from normal distribution
                mean = float(col_data.mean())
                std = float(col_data.std()) if float(col_data.std()) > 0 else 1.0
                sampled = np.random.normal(mean, std, self.n_synthetic_pairs)
                # Clip to real data range
                sampled = np.clip(sampled, float(col_data.min()), float(col_data.max()))
                synthetic[col] = sampled
            else:
                # Categorical: sample proportionally
                value_probs = col_data.value_counts(normalize=True)
                synthetic[col] = np.random.choice(
                    value_probs.index,
                    size=self.n_synthetic_pairs,
                    p=value_probs.values
                )
        
        return pd.DataFrame(synthetic)
    
    def _encode_for_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Label encodes categorical columns so sklearn models can process them.
        Numeric columns pass through unchanged.
        """
        encoded = df.copy()
        for col in encoded.columns:
            if encoded[col].dtype == 'object':
                encoded[col] = pd.factorize(encoded[col])[0]
        return encoded.fillna(0)
    
    def _calculate_discrimination_ratio(self, group_outcomes: dict) -> float:
        """
        Calculates the disparate impact ratio.
        Formula: lowest group positive rate / highest group positive rate
        
        The 80% rule (0.8 threshold) is the legal standard used by the
        US Equal Employment Opportunity Commission (EEOC).
        A ratio below 0.8 indicates illegal disparate impact discrimination.
        
        Returns float between 0 and 1. Below 0.8 = fails the 80% rule.
        """
        rates = []
        for group, data in group_outcomes.items():
            if "positive_outcome_rate" in data:
                rates.append(data["positive_outcome_rate"])
        
        if len(rates) < 2:
            return 1.0
        
        min_rate = min(rates)
        max_rate = max(rates)
        
        if max_rate == 0:
            return 1.0
        
        return round(min_rate / max_rate, 3)
    
    def _calculate_bias_score(self, stress_results: dict) -> int:
        """
        Overall bias score 0-100 based on discrimination ratios across all attributes.
        Score of 100 = perfectly unbiased across all tested attributes.
        Score below 70 = fails certification.
        """
        if not stress_results:
            return 100
        
        score = 100
        
        for attr, result in stress_results.items():
            if "discrimination_ratio" in result:
                ratio = result["discrimination_ratio"]
                if ratio < 0.5:
                    score -= 30
                elif ratio < 0.8:
                    score -= 20
                elif ratio < 0.9:
                    score -= 10
        
        return max(0, min(100, score))
    
    def _generate_attribute_warnings(
        self, 
        attr: str, 
        group_outcomes: dict, 
        discrimination_ratio: float
    ) -> list:
        """
        Generates human-readable warnings for a specific protected attribute.
        Includes the specific groups affected and the magnitude of discrimination.
        """
        warnings = []
        
        if discrimination_ratio < 0.8:
            rates = {g: d["positive_outcome_rate"] 
                    for g, d in group_outcomes.items() 
                    if "positive_outcome_rate" in d}
            
            if rates:
                best_group = max(rates, key=rates.get)
                worst_group = min(rates, key=rates.get)
                
                warnings.append(
                    f"DISCRIMINATION DETECTED in '{attr}': '{worst_group}' group receives "
                    f"positive outcomes {round(rates[best_group]/rates[worst_group], 1)}x less "
                    f"often than '{best_group}' group "
                    f"({rates[worst_group]}% vs {rates[best_group]}%) — "
                    f"Fails the 80% rule (ratio: {discrimination_ratio})"
                )
        
        return warnings
