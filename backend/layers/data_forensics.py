import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class DataForensicsEngine:
    """
    Layer 1: Data Forensics Engine
    
    Analyzes training datasets for bias BEFORE a model is trained.
    Answers: Is the data itself fair and representative?
    
    Four checks:
    1. Representation audit - demographic distribution analysis
    2. Proxy variable detection - finds columns that correlate with protected attributes
    3. Class imbalance check - outcome label distribution
    4. Overall data health score (0-100)
    """
    
    def __init__(self, df: pd.DataFrame, protected_attributes: list[str], target_column: str):
        self.df = df
        self.protected_attributes = protected_attributes
        self.target_column = target_column
        self.results = {}
    
    def run(self) -> dict:
        """
        Runs all four forensics checks and returns combined results dict.
        """
        logger.info("Starting Data Forensics analysis...")
        
        self.results = {
            "layer": "data_forensics",
            "total_records": len(self.df),
            "total_features": len(self.df.columns),
            "representation": self.check_representation(),
            "proxy_variables": self.detect_proxy_variables(),
            "class_imbalance": self.check_class_imbalance(),
            "health_score": 0,
            "warnings": [],
            "status": "pass"
        }
        
        self.results["health_score"] = self._calculate_health_score()
        self.results["warnings"] = self._generate_warnings()
        self.results["status"] = "pass" if self.results["health_score"] >= 70 else "fail"
        
        logger.info(f"Data Forensics complete. Health score: {self.results['health_score']}")
        return self.results
    
    def check_representation(self) -> dict:
        """
        For each protected attribute, calculates the distribution of groups.
        Flags any group with less than 20% representation as underrepresented.
        
        Returns dict with per-attribute distributions and underrepresentation flags.
        """
        representation = {}
        
        for attr in self.protected_attributes:
            if attr not in self.df.columns:
                representation[attr] = {"error": f"Column '{attr}' not found in dataset"}
                continue
            
            value_counts = self.df[attr].value_counts(normalize=True)
            groups = {}
            underrepresented = []
            
            for group, proportion in value_counts.items():
                groups[str(group)] = round(float(proportion * 100), 2)
                if proportion < 0.20:
                    underrepresented.append(str(group))
            
            representation[attr] = {
                "distribution": groups,
                "underrepresented_groups": underrepresented,
                "is_balanced": len(underrepresented) == 0
            }
        
        return representation
    
    def detect_proxy_variables(self) -> dict:
        """
        Detects columns that correlate strongly with protected attributes.
        These are 'proxy variables' — seemingly neutral columns that encode
        protected attribute information and can cause indirect discrimination.
        
        Method: Cramers V for categorical, point-biserial for mixed types.
        Flags correlation above 0.3 as a proxy risk.
        
        Returns dict of suspicious columns with correlation scores.
        """
        proxies = {}
        
        for attr in self.protected_attributes:
            if attr not in self.df.columns:
                continue
            
            attr_proxies = []
            
            for col in self.df.columns:
                if col == attr or col == self.target_column:
                    continue
                
                try:
                    # Drop nulls for this pair
                    pair = self.df[[attr, col]].dropna()
                    
                    if len(pair) < 10:
                        continue
                    
                    # Use Cramers V for categorical columns
                    if self.df[col].dtype == 'object' or self.df[col].nunique() < 10:
                        correlation = self._cramers_v(pair[attr], pair[col])
                    else:
                        # Use absolute Pearson correlation for numeric
                        # Encode protected attribute if categorical
                        attr_encoded = pd.factorize(pair[attr])[0] if pair[attr].dtype == 'object' else pair[attr]
                        correlation = abs(attr_encoded.corr(pair[col]))
                    
                    if correlation > 0.3 and not np.isnan(correlation):
                        attr_proxies.append({
                            "column": col,
                            "correlation": round(float(correlation), 3),
                            "risk_level": "high" if correlation > 0.6 else "medium"
                        })
                except Exception:
                    continue
            
            proxies[attr] = sorted(attr_proxies, key=lambda x: x["correlation"], reverse=True)
        
        return proxies
    
    def check_class_imbalance(self) -> dict:
        """
        Checks if the target/outcome column is imbalanced.
        Severe imbalance (minority class < 20%) causes models to be biased
        toward the majority class.
        
        Returns imbalance ratio and severity flag.
        """
        if self.target_column not in self.df.columns:
            return {"error": f"Target column '{self.target_column}' not found"}
        
        value_counts = self.df[self.target_column].value_counts(normalize=True)
        min_class = float(value_counts.min())
        max_class = float(value_counts.max())
        
        imbalance_ratio = round(max_class / min_class, 2) if min_class > 0 else 999
        
        return {
            "distribution": {str(k): round(float(v * 100), 2) for k, v in value_counts.items()},
            "imbalance_ratio": imbalance_ratio,
            "minority_class_pct": round(min_class * 100, 2),
            "severity": "severe" if imbalance_ratio > 5 else "moderate" if imbalance_ratio > 2 else "balanced",
            "is_balanced": imbalance_ratio <= 2
        }
    
    def _cramers_v(self, x: pd.Series, y: pd.Series) -> float:
        """
        Calculates Cramer's V statistic for categorical-categorical correlation.
        Returns value between 0 (no correlation) and 1 (perfect correlation).
        """
        try:
            confusion_matrix = pd.crosstab(x, y)
            chi2 = stats.chi2_contingency(confusion_matrix)[0]
            n = len(x)
            phi2 = chi2 / n
            r, k = confusion_matrix.shape
            phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
            rcorr = r - ((r-1)**2)/(n-1)
            kcorr = k - ((k-1)**2)/(n-1)
            denominator = min((kcorr-1), (rcorr-1))
            if denominator <= 0:
                return 0.0
            return float(np.sqrt(phi2corr / denominator))
        except Exception:
            return 0.0
    
    def _calculate_health_score(self) -> int:
        """
        Calculates overall data health score 0-100.
        
        Scoring:
        - Representation: 40 points (deduct for each underrepresented group)
        - Proxy variables: 30 points (deduct for each high-risk proxy found)
        - Class imbalance: 30 points (deduct based on severity)
        """
        score = 100
        
        # Representation penalty
        for attr, data in self.results["representation"].items():
            if "underrepresented_groups" in data:
                penalty = len(data["underrepresented_groups"]) * 10
                score -= min(penalty, 40)
        
        # Proxy variable penalty
        for attr, proxies in self.results["proxy_variables"].items():
            high_risk = [p for p in proxies if p["risk_level"] == "high"]
            medium_risk = [p for p in proxies if p["risk_level"] == "medium"]
            score -= min(len(high_risk) * 10 + len(medium_risk) * 5, 30)
        
        # Class imbalance penalty
        imbalance = self.results["class_imbalance"]
        if "severity" in imbalance:
            if imbalance["severity"] == "severe":
                score -= 30
            elif imbalance["severity"] == "moderate":
                score -= 15
        
        return max(0, min(100, score))
    
    def _generate_warnings(self) -> list:
        """
        Generates human-readable warning messages based on findings.
        These feed into the Gemini governance layer for plain-language reporting.
        """
        warnings = []
        
        for attr, data in self.results["representation"].items():
            if "underrepresented_groups" in data and data["underrepresented_groups"]:
                warnings.append(
                    f"Underrepresentation detected: {', '.join(data['underrepresented_groups'])} "
                    f"in '{attr}' column represent less than 20% of dataset"
                )
        
        for attr, proxies in self.results["proxy_variables"].items():
            for proxy in proxies:
                if proxy["risk_level"] == "high":
                    warnings.append(
                        f"High-risk proxy variable: '{proxy['column']}' has {proxy['correlation']} "
                        f"correlation with protected attribute '{attr}'"
                    )
        
        if "severity" in self.results["class_imbalance"]:
            if self.results["class_imbalance"]["severity"] == "severe":
                warnings.append(
                    f"Severe class imbalance: minority class represents only "
                    f"{self.results['class_imbalance']['minority_class_pct']}% of target variable"
                )
        
        return warnings
