import pandas as pd
import numpy as np
from typing import Dict, List

class FeatureEngineer:
    def __init__(self):
        pass

    def generate_feature_suggestions(self, col_info: List[Dict]) -> Dict:
        """Generate basic rule-based suggestions (can be replaced with LLM)."""
        suggestions = []
        for col in col_info:
            if col["dtype"] in ["int64", "float64"]:
                # For numeric columns
                suggestions.extend([
                    {
                        "column": col["name"],
                        "technique": "standardization",
                        "rationale": "Normalize numeric values",
                        "priority": "high"
                    },
                    {
                        "column": col["name"],
                        "technique": "binning",
                        "rationale": "Convert continuous into categories",
                        "priority": "medium"
                    }
                ])
            elif col["dtype"] == "object":
                suggestions.append({
                    "column": col["name"],
                    "technique": "one_hot_encoding",
                    "rationale": "Convert categorical to dummy variables",
                    "priority": "high"
                })
        return {"suggestions": suggestions}

    def analyze_dataset(self, df: pd.DataFrame) -> Dict:
        """Collect column info & produce suggestions."""
        column_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            unique_count = df[col].nunique()
            missing_count = df[col].isnull().sum()

            column_info.append({
                "name": col,
                "dtype": dtype,
                "unique_values": unique_count,
                "missing_values": missing_count
            })

        return self.generate_feature_suggestions(column_info)

    def apply_transformations(self, df: pd.DataFrame, transformations: List[Dict]) -> pd.DataFrame:
        """Apply transformations like standardization, binning, one-hot, etc."""
        df_transformed = df.copy()
        for transform in transformations:
            col = transform["column"]
            technique = transform["technique"].lower()

            if col not in df_transformed.columns:
                continue

            if technique == "standardization":
                mean_val = df_transformed[col].mean()
                std_val = df_transformed[col].std() or 1e-8
                df_transformed[f"{col}_standardized"] = (df_transformed[col] - mean_val) / std_val

            elif technique == "binning":
                try:
                    n_unique = df_transformed[col].nunique()
                    if n_unique <= 10:
                        # fallback to one-hot
                        encoded = pd.get_dummies(df_transformed[col], prefix=col)
                        df_transformed = pd.concat([df_transformed, encoded], axis=1)
                    else:
                        df_transformed[f"{col}_binned"] = pd.qcut(
                            df_transformed[col],
                            q=5,
                            labels=['very_low', 'low', 'medium', 'high', 'very_high'],
                            duplicates='drop'
                        )
                except:
                    # fallback
                    encoded = pd.get_dummies(df_transformed[col], prefix=col)
                    df_transformed = pd.concat([df_transformed, encoded], axis=1)

            elif technique == "one_hot_encoding":
                encoded = pd.get_dummies(df_transformed[col], prefix=col)
                df_transformed = pd.concat([df_transformed, encoded], axis=1)

        return df_transformed
