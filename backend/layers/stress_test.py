import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional


class StressTestEngine:
    """
    StressTestEngine generates synthetic twin profiles based on real dataset statistics
    to stress test specific machine learning models for bias and fairness.
    It focuses on ensuring counterfactual fairness through twin profile evaluation
    and calculating disparate impact ratios based on the 80% rule.
    """

    def __init__(
        self, 
        model: Any, 
        dataset: pd.DataFrame, 
        protected_attributes: List[str], 
        target_column: Optional[str] = None,
        positive_outcome: Any = 1
    ):
        """
        Initializes the Stress Test Engine.

        Args:
            model: A trained sklearn-compatible model with a .predict() method.
            dataset: A pandas DataFrame containing reference real data.
                     Used to calculate statistics to formulate synthetic twins.
            protected_attributes: List of columns representing protected attributes (e.g., 'Age', 'Gender').
            target_column: Optional name of the target/label column to exclude from profile generation.
            positive_outcome: The value in the model output that signifies a positive/favorable outcome.
        """
        self.model = model
        self.dataset = dataset.copy()
        
        # Drop the target column if provided, to ensure we only process feature variables
        if target_column and target_column in self.dataset.columns:
            self.dataset = self.dataset.drop(columns=[target_column])
            
        self.protected_attributes = protected_attributes
        self.positive_outcome = positive_outcome

        # Verify all protected attributes exist in the feature set dataframe
        missing = [attr for attr in self.protected_attributes if attr not in self.dataset.columns]
        if missing:
            raise ValueError(f"Protected attributes not found in dataset: {missing}")

    def _generate_synthetic_base(self, num_samples: int) -> pd.DataFrame:
        """
        Generates base synthetic profiles using the statistical distribution of the original dataset.
        Protected attributes are NOT generated here; they are injected later to form specific twins.
        """
        synthetic_data = {}
        for col in self.dataset.columns:
            if col in self.protected_attributes:
                continue
                
            col_data = self.dataset[col].dropna()
            if len(col_data) == 0:
                synthetic_data[col] = np.zeros(num_samples)
                continue
                
            if pd.api.types.is_numeric_dtype(col_data):
                mean = col_data.mean()
                std = col_data.std()
                if std == 0 or pd.isna(std):
                     vals = np.full(num_samples, mean)
                else:
                    # Generate normal distribution mapped around mean and bounded by min/max
                    vals = np.random.normal(loc=mean, scale=std, size=num_samples)
                    vals = np.clip(vals, col_data.min(), col_data.max())
                
                # Maintain data type context natively (e.g. integer fields generate integer profiles)
                if pd.api.types.is_integer_dtype(col_data):
                    vals = np.round(vals).astype(int)
                synthetic_data[col] = vals
            else:
                # Random sampling based on native dataset frequencies for categorical features
                value_counts = col_data.value_counts(normalize=True)
                vals = np.random.choice(value_counts.index, p=value_counts.values, size=num_samples)
                synthetic_data[col] = vals
                
        return pd.DataFrame(synthetic_data)

    def run(self, num_samples: int = 1000) -> Dict[str, Any]:
        """
        Executes the stress test workflow framework:
        - Scaffolds synthetic twin profile groups.
        - Isolates variations to solely protected attributes.
        - Models outputs and benchmarks the inference divergences.
        - Computes the 80% (Four-Fifths) Disparate Impact rule compliance.

        Args:
            num_samples: Number of synthetic structural profiles to generate.
            
        Returns:
            A structured dict tracking metrics, sub-status checks, and generated warnings.
        """
        results = {
            "status": "Pass",
            "overall_warnings": [],
            "protected_attributes": {}
        }
        
        # 1. Spawn core structural profiles independent of protective classes
        base_profiles = self._generate_synthetic_base(num_samples)
        expected_columns = self.dataset.columns.tolist() 
        
        for attr in self.protected_attributes:
            unique_values = self.dataset[attr].dropna().unique()
            if len(unique_values) < 2:
                results["protected_attributes"][attr] = {
                    "error": "Insufficient unique attribute variations to generate counterfactual twins."
                }
                continue

            attr_results = {
                "values": [
                    int(v) if isinstance(v, (np.integer, int)) else float(v) if isinstance(v, (np.floating, float)) else str(v)
                    for v in unique_values
                ],
                "positive_rates": {},
                "disparate_impact_ratios": {},
                "flipping_rate": 0.0,
                "warnings": []
            }
            
            predictions_by_value = {}
            
            # 2. Generate systematic comparative twins and evaluate metrics footprint
            for val in unique_values:
                twin_df = base_profiles.copy()
                twin_df[attr] = val
                
                # Overwrite orthogonal protective attributes cleanly with their primary modes
                for other_attr in self.protected_attributes:
                    if other_attr != attr:
                        mode_val = self.dataset[other_attr].mode()[0]
                        twin_df[other_attr] = mode_val
                        
                # Preserve precise dataset layout architecture for model predict boundaries
                twin_df = twin_df[expected_columns]
                
                # Perform native inference request
                preds = self.model.predict(twin_df)
                predictions_by_value[val] = preds
                
                # Audit and register predicted selection rate for specific demographic
                positive_rate = np.mean(preds == self.positive_outcome)
                attr_results["positive_rates"][str(val)] = float(positive_rate)
            
            # 3. Assess Counterfactual Flipping Event Rate
            # Represents the % of twins whose prediction result flipped strictly off demographic shift
            preds_matrix = np.column_stack(list(predictions_by_value.values()))
            flips = np.std(preds_matrix, axis=1) > 0
            attr_results["flipping_rate"] = float(np.mean(flips))
            
            if attr_results["flipping_rate"] > 0.05:
                warning_msg = (
                    f"Counterfactual vulnerability detected on '{attr}': "
                    f"{attr_results['flipping_rate'] * 100:.1f}% of evaluated profiles reversed outcomes "
                    f"solely due to a shift in this protected attribute."
                )
                attr_results["warnings"].append(warning_msg)
                results["overall_warnings"].append(warning_msg)

            # 4. Enforce strict 80% Disparate Impact (Four-Fifths) rule mappings
            pos_rates = list(attr_results["positive_rates"].values())
            max_rate = max(pos_rates)
            min_rate = min(pos_rates)
            
            if max_rate > 0:
                di_ratio = min_rate / max_rate
                attr_results["disparate_impact_ratio_min_max"] = float(di_ratio)
                
                if di_ratio < 0.8:
                    warning_msg = (
                        f"Four-Fifths (80%) Rule violation on '{attr}'. "
                        f"Disparate impact ratio is {di_ratio:.3f} (< 0.8), "
                        f"indicating a significantly disadvantaged unprivileged sub-cohort."
                    )
                    attr_results["warnings"].append(warning_msg)
                    results["overall_warnings"].append(warning_msg)
            else:
                attr_results["disparate_impact_ratio_min_max"] = 0.0
                warning_msg = f"Zero absolute favorable outcomes aggregated across all classes for '{attr}'."
                attr_results["warnings"].append(warning_msg)

            # Analyze deep pairwise comparative distributions against inferred favored demographic
            sorted_rates = sorted(attr_results["positive_rates"].items(), key=lambda x: x[1], reverse=True)
            if len(sorted_rates) > 0:
                privileged_val, priv_rate = sorted_rates[0]
                attr_results["inferred_privileged_group"] = privileged_val
                
                for group, pop_rate in sorted_rates[1:]:
                    ratio = pop_rate / priv_rate if priv_rate > 0 else 0.0
                    attr_results["disparate_impact_ratios"][f"{group}_vs_{privileged_val}"] = float(ratio)
                    
            results["protected_attributes"][attr] = attr_results

        # Aggregate pass/fail flags if platform constraints breach boundaries
        if len(results["overall_warnings"]) > 0:
            results["status"] = "Fail"
            
        return results
