from bson import ObjectId
from typing import Any, Dict, List, Union
import pandas as pd
import numpy as np

def convert_objectid_to_str(obj: Any) -> Any:
    """
    Recursively convert all ObjectId instances to strings in any data structure.
    Works with dictionaries, lists, and nested combinations of these.
    
    Args:
        obj: Any Python object that might contain ObjectId instances
        
    Returns:
        The same object with all ObjectId instances converted to strings
    """
    if isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: convert_objectid_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_objectid_to_str(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_objectid_to_str(item) for item in obj)
    else:
        return obj 

def assess_training_compatibility(df, metadata=None):
    """
    Assess the compatibility of a dataset for machine learning training.
    
    Args:
        df: pandas DataFrame to assess
        metadata: Optional dictionary with dataset metadata including transformations
        
    Returns:
        Dictionary with compatibility metrics and recommendations
    """
    # Initialize results dictionary
    results = {
        'overall_score': 0,
        'missing_values_score': 0,
        'data_types_score': 0,
        'feature_distribution_score': 0,
        'correlation_score': 0,
        'size_score': 0,
        'preprocessing_score': 0,  # New score for preprocessing quality
        'recommendations': []
    }
    
    # Check for missing values
    missing_pct = df.isnull().mean().mean() * 100
    if missing_pct == 0:
        results['missing_values_score'] = 100
        results['missing_values_message'] = "No missing values detected. Excellent!"
    elif missing_pct < 5:
        results['missing_values_score'] = 80
        results['missing_values_message'] = f"Low percentage of missing values ({missing_pct:.2f}%). Good."
        results['recommendations'].append("Consider imputing the small amount of missing values.")
    elif missing_pct < 15:
        results['missing_values_score'] = 60
        results['missing_values_message'] = f"Moderate percentage of missing values ({missing_pct:.2f}%). Needs attention."
        results['recommendations'].append("Use imputation techniques to handle missing values.")
    else:
        results['missing_values_score'] = 30
        results['missing_values_message'] = f"High percentage of missing values ({missing_pct:.2f}%). Problematic."
        results['recommendations'].append("Significant missing data. Consider dropping columns with too many missing values or using advanced imputation techniques.")
    
    # Check data types
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    
    # Check for binned numeric columns (like '(-0.839, 0.46]')
    binned_cols = []
    for col in categorical_cols:
        # Check if values look like interval notation
        if df[col].dtype == 'object' and df[col].str.contains(r'[\(\[].*,.*[\)\]]').any():
            binned_cols.append(col)
    
    # Adjust categorical columns list to exclude binned columns
    categorical_cols = [col for col in categorical_cols if col not in binned_cols]
    
    # Check for PCA components
    pca_cols = [col for col in df.columns if col.startswith('PC') and col[2:].isdigit()]
    
    # Evaluate data types with consideration for binned and PCA columns
    if len(numeric_cols) > 0 and (len(categorical_cols) > 0 or len(binned_cols) > 0):
        results['data_types_score'] = 100
        results['data_types_message'] = f"Good mix of numeric ({len(numeric_cols)}) and categorical ({len(categorical_cols) + len(binned_cols)}) features."
    elif len(numeric_cols) > 0:
        results['data_types_score'] = 80
        results['data_types_message'] = f"Contains {len(numeric_cols)} numeric features but no categorical features."
    elif len(categorical_cols) > 0 or len(binned_cols) > 0:
        results['data_types_score'] = 70
        results['data_types_message'] = f"Contains {len(categorical_cols) + len(binned_cols)} categorical features but no numeric features."
        results['recommendations'].append("Consider encoding categorical variables appropriately for machine learning.")
    else:
        results['data_types_score'] = 50
        results['data_types_message'] = "Unusual data types. May require special preprocessing."
        results['recommendations'].append("Review data types and convert to appropriate numeric or categorical types.")
    
    if len(datetime_cols) > 0:
        results['data_types_message'] += f" Also contains {len(datetime_cols)} datetime features."
        results['recommendations'].append("Extract useful features from datetime columns (year, month, day, etc.).")
    
    if len(pca_cols) > 0:
        results['data_types_message'] += f" Includes {len(pca_cols)} PCA components."
    
    if len(binned_cols) > 0:
        results['data_types_message'] += f" Contains {len(binned_cols)} binned numeric features."
    
    # Check feature distributions
    skewed_features = 0
    for col in numeric_cols:
        if df[col].skew() > 1.5 or df[col].skew() < -1.5:
            skewed_features += 1
    
    skewed_pct = (skewed_features / len(numeric_cols)) * 100 if numeric_cols else 0
    
    if skewed_pct < 20:
        results['feature_distribution_score'] = 100
        results['feature_distribution_message'] = "Most features have good distributions."
    elif skewed_pct < 50:
        results['feature_distribution_score'] = 70
        results['feature_distribution_message'] = f"{skewed_pct:.1f}% of numeric features are skewed."
        results['recommendations'].append("Consider applying transformations (log, sqrt, etc.) to skewed features.")
    else:
        results['feature_distribution_score'] = 40
        results['feature_distribution_message'] = f"Many features ({skewed_pct:.1f}%) have skewed distributions."
        results['recommendations'].append("Apply transformations to normalize skewed distributions.")
    
    # Check correlations
    if len(numeric_cols) > 1:
        try:
            corr_matrix = df[numeric_cols].corr().abs()
            # Get upper triangle of correlation matrix
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            # Find features with correlation greater than 0.8
            high_corr = [column for column in upper.columns if any(upper[column] > 0.8)]
            
            if len(high_corr) == 0:
                results['correlation_score'] = 100
                results['correlation_message'] = "No highly correlated features detected."
            elif len(high_corr) < 3:
                results['correlation_score'] = 80
                results['correlation_message'] = f"Found {len(high_corr)} features with high correlation."
                results['recommendations'].append("Consider removing some highly correlated features.")
            else:
                results['correlation_score'] = 60
                results['correlation_message'] = f"Found {len(high_corr)} features with high correlation."
                results['recommendations'].append("Use feature selection or PCA to address multicollinearity.")
        except Exception as e:
            results['correlation_score'] = 50
            results['correlation_message'] = "Could not compute correlations."
    else:
        results['correlation_score'] = 50
        results['correlation_message'] = "Not enough numeric features to assess correlation."
    
    # Check dataset size
    if len(df) < 100:
        results['size_score'] = 30
        results['size_message'] = f"Very small dataset ({len(df)} rows). May lead to overfitting."
        results['recommendations'].append("Consider data augmentation techniques to increase dataset size.")
    elif len(df) < 1000:
        results['size_score'] = 60
        results['size_message'] = f"Small dataset ({len(df)} rows). May be sufficient for simple models."
        results['recommendations'].append("Use cross-validation to ensure model generalization.")
    elif len(df) < 10000:
        results['size_score'] = 80
        results['size_message'] = f"Moderate dataset size ({len(df)} rows). Good for most models."
    else:
        results['size_score'] = 100
        results['size_message'] = f"Large dataset ({len(df)} rows). Excellent for training complex models."
    
    # Evaluate preprocessing quality based on metadata and dataset characteristics
    results['preprocessing_score'] = 50  # Default score
    preprocessing_notes = []
    
    # Check if feature engineering or PCA has been applied
    has_feature_engineering = False
    has_pca = False
    
    if metadata:
        if 'transformations' in metadata:
            has_feature_engineering = True
            preprocessing_notes.append(f"Applied {len(metadata['transformations'])} feature transformations.")
        
        if 'pca_info' in metadata:
            has_pca = True
            preprocessing_notes.append(f"Applied PCA with {metadata['pca_info'].get('n_components', 'unknown')} components.")
    
    # Also check for evidence of preprocessing in the data itself
    if len(binned_cols) > 0:
        has_feature_engineering = True
        preprocessing_notes.append(f"Dataset contains {len(binned_cols)} binned features.")
    
    if len(pca_cols) > 0:
        has_pca = True
        preprocessing_notes.append(f"Dataset contains {len(pca_cols)} PCA components.")
    
    # Score preprocessing quality
    if has_feature_engineering and has_pca:
        results['preprocessing_score'] = 100
        results['preprocessing_message'] = "Excellent preprocessing: Both feature engineering and dimensionality reduction applied."
    elif has_feature_engineering:
        results['preprocessing_score'] = 85
        results['preprocessing_message'] = "Good preprocessing: Feature engineering applied."
    elif has_pca:
        results['preprocessing_score'] = 80
        results['preprocessing_message'] = "Good preprocessing: Dimensionality reduction applied."
    else:
        results['preprocessing_score'] = 50
        results['preprocessing_message'] = "Basic preprocessing: Consider applying feature engineering or dimensionality reduction."
        results['recommendations'].append("Apply feature engineering to improve model performance.")
    
    if preprocessing_notes:
        results['preprocessing_message'] += f" ({'; '.join(preprocessing_notes)})"
    
    # Calculate overall score including preprocessing
    results['overall_score'] = int(np.mean([
        results['missing_values_score'],
        results['data_types_score'],
        results['feature_distribution_score'],
        results['correlation_score'],
        results['size_score'],
        results['preprocessing_score']
    ]))
    
    # Set overall message
    if results['overall_score'] >= 80:
        results['overall_message'] = "This dataset is well-suited for machine learning training."
    elif results['overall_score'] >= 60:
        results['overall_message'] = "This dataset is suitable for machine learning with some preprocessing."
    else:
        results['overall_message'] = "This dataset requires significant preprocessing before machine learning training."
    
    # Add general recommendations if needed
    if not results['recommendations']:
        results['recommendations'].append("Dataset looks good for training. Consider standard preprocessing steps like scaling numeric features.")
    
    return results 