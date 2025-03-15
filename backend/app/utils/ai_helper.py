"""
AI Helper module for dataset analysis and chat. Works offline without requiring external APIs.
"""
import os
import json
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

# Load environment variables
load_dotenv()

class AIHelper:
    """
    AI Helper class for dataset analysis and chat.
    """
    def __init__(self, model_name: str = "local"):
        """
        Initialize the AI Helper.
        
        Args:
            model_name: Not used, kept for compatibility
        """
        self.model_name = model_name
        print(f"Initialized AIHelper with local model (no external API required)")
    
    def analyze_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze a dataset and provide insights and suggestions.
        
        Args:
            df: The pandas DataFrame to analyze
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Basic dataset summary
            summary = {
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": df.columns.tolist(),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "missing_values": {col: int(df[col].isnull().sum()) for col in df.columns},
                "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
                "categorical_columns": df.select_dtypes(include=["object", "category"]).columns.tolist()
            }
            
            # Use the default analysis implementation
            print("Using default analysis")
            return self._generate_default_analysis(summary, df)
                
        except Exception as e:
            print(f"Error analyzing dataset: {str(e)}")
            # Return a minimal analysis
            return {
                "dataset_overview": "Error analyzing dataset",
                "data_quality_issues": [f"Error: {str(e)}"],
                "feature_engineering": []
            }
    
    def _generate_default_analysis(self, summary: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a default analysis.
        
        Args:
            summary: Dataset summary dictionary
            df: The pandas DataFrame
            
        Returns:
            Dictionary with default analysis
        """
        # Create a basic analysis
        analysis = {
            "dataset_overview": f"Dataset with {summary['row_count']} rows and {summary['column_count']} columns. " +
                               f"Contains {len(summary['numeric_columns'])} numeric and {len(summary['categorical_columns'])} categorical columns.",
            "data_quality_issues": [],
            "feature_engineering": []
        }
        
        # Check for missing values
        missing_cols = [col for col, count in summary["missing_values"].items() if count > 0]
        if missing_cols:
            analysis["data_quality_issues"].append(f"Missing values in columns: {', '.join(missing_cols)}")
            
            # Add imputation suggestions
            for col in missing_cols:
                if col in summary["numeric_columns"]:
                    analysis["feature_engineering"].append({
                        "column": col,
                        "transformation": "impute",
                        "reason": "Fill missing values with mean to preserve distribution"
                    })
                else:
                    analysis["feature_engineering"].append({
                        "column": col,
                        "transformation": "impute",
                        "reason": "Fill missing values with most frequent value"
                    })
        
        # Add scaling suggestions for numeric columns
        for col in summary["numeric_columns"]:
            analysis["feature_engineering"].append({
                "column": col,
                "transformation": "standardize",
                "reason": "Standardize numeric features to improve model performance"
            })
        
        # Add encoding suggestions for categorical columns
        for col in summary["categorical_columns"]:
            analysis["feature_engineering"].append({
                "column": col,
                "transformation": "one_hot_encode",
                "reason": "Convert categorical features to numeric representation"
            })
            
        return analysis
    
    def chat_with_data(self, df: pd.DataFrame, question: str) -> str:
        """
        Chat with a dataset by answering questions about it.
        
        Args:
            df: The pandas DataFrame to analyze
            question: The question to answer
            
        Returns:
            A response to the question
        """
        try:
            # Basic dataset summary
            summary = {
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": df.columns.tolist(),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "missing_values": {col: int(df[col].isnull().sum()) for col in df.columns},
                "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
                "categorical_columns": df.select_dtypes(include=["object", "category"]).columns.tolist()
            }
            
            # Add basic statistics for numeric columns
            if summary["numeric_columns"]:
                summary["numeric_stats"] = df[summary["numeric_columns"]].describe().to_dict()
            
            # Add value counts for categorical columns (top 5 values)
            if summary["categorical_columns"]:
                summary["categorical_stats"] = {}
                for col in summary["categorical_columns"]:
                    summary["categorical_stats"][col] = df[col].value_counts().head(5).to_dict()
            
            # Skip Hugging Face API entirely - default to our own implementation
            return self._generate_default_chat_response(df, question, summary)
        except Exception as e:
            print(f"Error in chat_with_data: {str(e)}")
            return f"I'm sorry, I encountered an error while analyzing your dataset: {str(e)}"
    
    def _generate_default_chat_response(self, df: pd.DataFrame, question: str, summary: Dict[str, Any]) -> str:
        """
        Generate a default response for chat.
        
        Args:
            df: The pandas DataFrame
            question: The user's question
            summary: The dataset summary
            
        Returns:
            A default response
        """
        question_lower = question.lower()
        
        # Basic dataset information
        if any(keyword in question_lower for keyword in ["overview", "summary", "describe", "what is this dataset", "tell me about"]):
            numeric_stats = ""
            if summary["numeric_columns"]:
                numeric_stats = "\n\nNumeric columns summary:\n"
                for col in summary["numeric_columns"][:5]:  # Limit to first 5 numeric columns
                    if col in df.columns:
                        col_data = df[col].dropna()
                        if len(col_data) > 0:
                            numeric_stats += f"- {col}: min={col_data.min():.2f}, max={col_data.max():.2f}, mean={col_data.mean():.2f}, median={col_data.median():.2f}\n"
            
            return f"""
            This dataset contains {summary['row_count']} rows and {summary['column_count']} columns.
            It has {len(summary['numeric_columns'])} numeric columns and {len(summary['categorical_columns'])} categorical columns.
            
            The columns are: {', '.join(summary['columns'])}.
            {numeric_stats}
            """
        
        # Missing values
        if any(keyword in question_lower for keyword in ["missing", "null", "na", "empty"]):
            missing_cols = [col for col, count in summary['missing_values'].items() if count > 0]
            if missing_cols:
                missing_details = "\n".join([f"- {col}: {summary['missing_values'][col]} missing values ({summary['missing_values'][col]/summary['row_count']*100:.1f}%)" 
                                           for col in missing_cols[:10]])  # Limit to first 10 columns with missing values
                
                return f"""
                There are missing values in {len(missing_cols)} columns:
                
                {missing_details}
                
                You might want to consider imputing these missing values before analysis.
                For numeric columns, you could use mean, median, or mode imputation.
                For categorical columns, you could use mode imputation or create a new 'Unknown' category.
                """
            else:
                return "Good news! There are no missing values in this dataset."
        
        # Correlations
        if any(keyword in question_lower for keyword in ["correlation", "correlate", "relationship", "related"]):
            if summary['numeric_columns']:
                # Check if specific columns are mentioned
                specific_cols = []
                for col in summary['numeric_columns']:
                    if col.lower() in question_lower:
                        specific_cols.append(col)
                
                if specific_cols:
                    # Calculate correlations for specific columns
                    corr_results = []
                    for col in specific_cols:
                        other_cols = [c for c in summary['numeric_columns'] if c != col]
                        for other_col in other_cols:
                            if col in df.columns and other_col in df.columns:
                                corr = df[col].corr(df[other_col])
                                corr_results.append((col, other_col, corr))
                    
                    # Sort by absolute correlation value
                    corr_results.sort(key=lambda x: abs(x[2]), reverse=True)
                    
                    if corr_results:
                        corr_text = "\n".join([f"- {col1} and {col2}: {corr:.2f}" for col1, col2, corr in corr_results[:5]])
                        return f"""
                        Here are the correlations for the columns you mentioned:
                        
                        {corr_text}
                        
                        A value close to 1 indicates a strong positive correlation, while a value close to -1 indicates a strong negative correlation.
                        """
                
                # General correlation analysis
                corr_matrix = df[summary['numeric_columns']].corr()
                high_corr_pairs = []
                
                for i, col1 in enumerate(corr_matrix.columns):
                    for col2 in corr_matrix.columns[i+1:]:
                        corr_value = corr_matrix.loc[col1, col2]
                        if abs(corr_value) > 0.7:  # Threshold for high correlation
                            high_corr_pairs.append((col1, col2, corr_value))
                
                if high_corr_pairs:
                    corr_text = "\n".join([f"- {col1} and {col2}: {corr:.2f}" for col1, col2, corr in high_corr_pairs[:5]])
                    return f"""
                    I found some strong correlations in your dataset:
                    
                    {corr_text}
                    
                    These correlations might be worth investigating further.
                    A value close to 1 indicates a strong positive correlation, while a value close to -1 indicates a strong negative correlation.
                    """
                else:
                    return "I didn't find any particularly strong correlations between the numeric columns in your dataset."
            else:
                return "There are no numeric columns in this dataset to calculate correlations."
        
        # Feature engineering suggestions
        if any(keyword in question_lower for keyword in ["feature engineering", "transform", "improve", "suggestion"]):
            suggestions = []
            
            # Missing value imputation
            missing_cols = [col for col, count in summary['missing_values'].items() if count > 0]
            if missing_cols:
                suggestions.append(f"Impute missing values in columns: {', '.join(missing_cols[:3])}")
            
            # Scaling numeric features
            if summary['numeric_columns']:
                suggestions.append(f"Normalize or standardize numeric columns: {', '.join(summary['numeric_columns'][:3])}")
                
                # Check for skewed distributions
                skewed_cols = []
                for col in summary['numeric_columns']:
                    if col in df.columns:
                        col_data = df[col].dropna()
                        if len(col_data) > 0 and abs(col_data.skew()) > 1:
                            skewed_cols.append(col)
                
                if skewed_cols:
                    suggestions.append(f"Apply log or power transformation to skewed columns: {', '.join(skewed_cols[:3])}")
            
            # Encoding categorical features
            if summary['categorical_columns']:
                suggestions.append(f"One-hot encode categorical columns: {', '.join(summary['categorical_columns'][:3])}")
                
                # Check for high cardinality
                high_card_cols = []
                for col in summary['categorical_columns']:
                    if col in df.columns and df[col].nunique() > 10:
                        high_card_cols.append(col)
                
                if high_card_cols:
                    suggestions.append(f"Consider target encoding or frequency encoding for high-cardinality columns: {', '.join(high_card_cols[:3])}")
            
            # Feature creation
            if len(summary['numeric_columns']) >= 2:
                suggestions.append("Create interaction features between numeric columns")
            
            if suggestions:
                return "Here are some feature engineering suggestions:\n\n- " + "\n- ".join(suggestions)
            else:
                return "Your dataset looks good as is. I don't have any specific feature engineering suggestions."
        
        # Statistical analysis
        if any(keyword in question_lower for keyword in ["statistics", "statistical", "stats"]):
            if summary['numeric_columns']:
                stats_text = ""
                for col in summary['numeric_columns'][:5]:  # Limit to first 5 numeric columns
                    if col in df.columns:
                        col_data = df[col].dropna()
                        if len(col_data) > 0:
                            stats_text += f"\n{col}:\n"
                            stats_text += f"- Count: {len(col_data)}\n"
                            stats_text += f"- Mean: {col_data.mean():.2f}\n"
                            stats_text += f"- Median: {col_data.median():.2f}\n"
                            stats_text += f"- Std Dev: {col_data.std():.2f}\n"
                            stats_text += f"- Min: {col_data.min():.2f}\n"
                            stats_text += f"- Max: {col_data.max():.2f}\n"
                            stats_text += f"- 25th Percentile: {col_data.quantile(0.25):.2f}\n"
                            stats_text += f"- 75th Percentile: {col_data.quantile(0.75):.2f}\n"
                
                return f"""
                Here are the statistical summaries for the numeric columns in your dataset:
                {stats_text}
                """
            else:
                return "There are no numeric columns in this dataset to provide statistical summaries."
        
        # Default response
        return f"""
        I'm analyzing your dataset with {summary['row_count']} rows and {summary['column_count']} columns.
        
        Your dataset contains columns: {', '.join(summary['columns'][:5])}{' and more' if len(summary['columns']) > 5 else ''}.
        
        Try asking specific questions about:
        - Dataset overview or summary
        - Missing values analysis
        - Correlation analysis between numeric columns
        - Statistical summaries of columns
        - Feature engineering suggestions
        
        For example:
        - "What are the correlations between numeric columns?"
        - "Tell me about missing values in this dataset"
        - "What feature engineering would you suggest?"
        - "Give me a statistical summary of the numeric columns"
        """
    
    def generate_feature_engineering_code(self, df: pd.DataFrame, transformations: List[Dict[str, Any]]) -> str:
        """
        Generate Python code for feature engineering based on the given transformations.
        
        Args:
            df: The pandas DataFrame
            transformations: List of transformation specifications
            
        Returns:
            Python code as a string
        """
        try:
            # Use a more robust implementation
            print("Using robust feature engineering implementation")
            
            # Generate code using the default implementation
            return self._generate_default_code(df, transformations)
        except Exception as e:
            print(f"Error generating feature engineering code: {str(e)}")
            return "# Error generating feature engineering code"
    
    def _generate_default_code(self, df: pd.DataFrame, transformations: List[Dict[str, Any]]) -> str:
        """
        Generate default feature engineering code.
        
        Args:
            df: The pandas DataFrame
            transformations: List of transformation specifications
            
        Returns:
            Python code as a string
        """
        code_lines = [
            "import pandas as pd",
            "import numpy as np",
            "from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler",
            "from sklearn.impute import SimpleImputer",
            "from sklearn.preprocessing import OneHotEncoder",
            "",
            "def apply_feature_engineering(df):",
            "    # Make a copy of the DataFrame to avoid modifying the original",
            "    df_transformed = df.copy()",
            ""
        ]
        
        # Process each transformation
        for t in transformations:
            transformation_type = str(t.get("transformation", "")).lower()
            column = str(t.get("column", ""))
            reason = str(t.get("reason", "No reason provided"))
            
            if not column or not transformation_type:
                continue
                
            if "standard" in transformation_type:
                code_lines.append(f"    # Standardize {column}: {reason}")
                code_lines.append(f"    scaler = StandardScaler()")
                code_lines.append(f"    df_transformed['{column}'] = scaler.fit_transform(df_transformed[['{column}']].fillna(0).values)")
                code_lines.append(f"    print(\"Standardized '{column}'\")")
                code_lines.append("")
                
            elif "normaliz" in transformation_type or "minmax" in transformation_type:
                code_lines.append(f"    # Normalize {column}: {reason}")
                code_lines.append(f"    scaler = MinMaxScaler()")
                code_lines.append(f"    df_transformed['{column}'] = scaler.fit_transform(df_transformed[['{column}']].fillna(0).values)")
                code_lines.append(f"    print(\"Normalized '{column}'\")")
                code_lines.append("")
                
            elif "robust" in transformation_type:
                code_lines.append(f"    # Robust scale {column}: {reason}")
                code_lines.append(f"    scaler = RobustScaler()")
                code_lines.append(f"    df_transformed['{column}'] = scaler.fit_transform(df_transformed[['{column}']].fillna(0).values)")
                code_lines.append(f"    print(\"Robust scaled '{column}'\")")
                code_lines.append("")
                
            elif "imput" in transformation_type or "miss" in transformation_type:
                code_lines.append(f"    # Impute missing values in {column}: {reason}")
                code_lines.append(f"    imputer = SimpleImputer(strategy='mean')")
                code_lines.append(f"    df_transformed['{column}'] = imputer.fit_transform(df_transformed[['{column}']].values)")
                code_lines.append(f"    print(\"Imputed missing values in '{column}'\")")
                code_lines.append("")
                
            elif "log" in transformation_type:
                code_lines.append(f"    # Log transform {column}: {reason}")
                code_lines.append(f"    # Add a small constant to avoid log(0)")
                code_lines.append(f"    if (df_transformed['{column}'] <= 0).any():")
                code_lines.append(f"        min_val = df_transformed['{column}'][df_transformed['{column}'] > 0].min() if (df_transformed['{column}'] > 0).any() else 1")
                code_lines.append(f"        df_transformed['{column}_log'] = np.log(df_transformed['{column}'] + min_val)")
                code_lines.append(f"    else:")
                code_lines.append(f"        df_transformed['{column}_log'] = np.log(df_transformed['{column}'])")
                code_lines.append(f"    print(\"Log transformed '{column}'\")")
                code_lines.append("")
                
            elif "one_hot" in transformation_type or "onehot" in transformation_type or "encod" in transformation_type:
                code_lines.append(f"    # One-hot encode {column}: {reason}")
                code_lines.append(f"    dummies = pd.get_dummies(df_transformed['{column}'], prefix='{column}')")
                code_lines.append(f"    df_transformed = pd.concat([df_transformed, dummies], axis=1)")
                code_lines.append(f"    print(\"One-hot encoded '{column}' into {{len(dummies.columns)}} columns\")")
                code_lines.append("")
        
        # Return the transformed dataframe
        code_lines.append("    return df_transformed")
        
        return "\n".join(code_lines) 