"""
Advanced feature engineering utilities for data transformation.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import logging
from ..database.mongodb import cache_result

logger = logging.getLogger(__name__)

def apply_log_transformation(df: pd.DataFrame, column: str, handle_zeros: bool = True, epsilon: float = 1e-6) -> pd.DataFrame:
    """
    Apply logarithmic transformation to a column.
    
    Args:
        df: Input DataFrame
        column: Column to transform
        handle_zeros: Whether to handle zeros by adding epsilon
        epsilon: Small value to add when handling zeros
        
    Returns:
        DataFrame with transformed column
    """
    df_result = df.copy()
    
    if handle_zeros:
        # Add small epsilon to avoid log(0)
        min_positive = df_result[column][df_result[column] > 0].min() if any(df_result[column] > 0) else epsilon
        df_result[f"{column}_log"] = np.log(df_result[column] + epsilon)
    else:
        # Only apply to positive values
        df_result[f"{column}_log"] = np.log(df_result[column])
        
    return df_result

def apply_polynomial_features(
    df: pd.DataFrame, 
    columns: List[str], 
    degree: int = 2, 
    interaction_only: bool = False
) -> pd.DataFrame:
    """
    Generate polynomial and interaction features.
    
    Args:
        df: Input DataFrame
        columns: Columns to use for generating polynomial features
        degree: Polynomial degree
        interaction_only: Whether to include only interaction features
        
    Returns:
        DataFrame with polynomial features added
    """
    df_result = df.copy()
    
    # Select only numeric columns from the specified columns
    numeric_cols = df[columns].select_dtypes(include=['number']).columns.tolist()
    
    if not numeric_cols:
        logger.warning(f"No numeric columns found among {columns}")
        return df_result
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
    poly_features = poly.fit_transform(df[numeric_cols])
    
    # Get feature names
    feature_names = poly.get_feature_names_out(numeric_cols)
    
    # Add polynomial features to the dataframe
    poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df.index)
    
    # Remove the original features to avoid duplication
    poly_df = poly_df.loc[:, ~poly_df.columns.isin(numeric_cols)]
    
    # Concatenate with original dataframe
    df_result = pd.concat([df_result, poly_df], axis=1)
    
    return df_result

def apply_binning(
    df: pd.DataFrame, 
    column: str, 
    num_bins: int = 5, 
    strategy: str = 'uniform', 
    labels: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Apply binning to a numeric column.
    
    Args:
        df: Input DataFrame
        column: Column to bin
        num_bins: Number of bins
        strategy: Binning strategy ('uniform', 'quantile', or 'kmeans')
        labels: Optional labels for the bins
        
    Returns:
        DataFrame with binned column added
    """
    df_result = df.copy()
    
    if strategy == 'uniform':
        # Uniform binning
        df_result[f"{column}_bin"] = pd.cut(
            df_result[column], 
            bins=num_bins, 
            labels=labels
        )
    elif strategy == 'quantile':
        # Quantile-based binning
        df_result[f"{column}_bin"] = pd.qcut(
            df_result[column], 
            q=num_bins, 
            labels=labels,
            duplicates='drop'
        )
    elif strategy == 'kmeans':
        # K-means binning
        from sklearn.cluster import KMeans
        
        # Reshape for KMeans
        values = df_result[column].values.reshape(-1, 1)
        
        # Apply KMeans
        kmeans = KMeans(n_clusters=num_bins, random_state=0).fit(values)
        df_result[f"{column}_bin"] = kmeans.labels_
        
        # Convert to categorical with labels if provided
        if labels:
            df_result[f"{column}_bin"] = pd.Categorical(
                df_result[f"{column}_bin"],
                categories=range(num_bins),
                labels=labels
            )
    else:
        raise ValueError(f"Unknown binning strategy: {strategy}")
    
    return df_result

def apply_time_features(
    df: pd.DataFrame, 
    date_column: str, 
    features: List[str] = ['year', 'month', 'day', 'dayofweek', 'hour']
) -> pd.DataFrame:
    """
    Extract time-based features from a date column.
    
    Args:
        df: Input DataFrame
        date_column: Date column to extract features from
        features: List of time features to extract
        
    Returns:
        DataFrame with time features added
    """
    df_result = df.copy()
    
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df_result[date_column]):
        df_result[date_column] = pd.to_datetime(df_result[date_column], errors='coerce')
    
    # Extract requested features
    for feature in features:
        if feature == 'year':
            df_result[f"{date_column}_year"] = df_result[date_column].dt.year
        elif feature == 'month':
            df_result[f"{date_column}_month"] = df_result[date_column].dt.month
        elif feature == 'day':
            df_result[f"{date_column}_day"] = df_result[date_column].dt.day
        elif feature == 'dayofweek':
            df_result[f"{date_column}_dayofweek"] = df_result[date_column].dt.dayofweek
        elif feature == 'hour':
            df_result[f"{date_column}_hour"] = df_result[date_column].dt.hour
        elif feature == 'minute':
            df_result[f"{date_column}_minute"] = df_result[date_column].dt.minute
        elif feature == 'quarter':
            df_result[f"{date_column}_quarter"] = df_result[date_column].dt.quarter
        elif feature == 'is_weekend':
            df_result[f"{date_column}_is_weekend"] = (df_result[date_column].dt.dayofweek >= 5).astype(int)
        elif feature == 'is_month_start':
            df_result[f"{date_column}_is_month_start"] = df_result[date_column].dt.is_month_start.astype(int)
        elif feature == 'is_month_end':
            df_result[f"{date_column}_is_month_end"] = df_result[date_column].dt.is_month_end.astype(int)
    
    return df_result

def apply_lag_features(
    df: pd.DataFrame, 
    column: str, 
    lags: List[int], 
    group_by: Optional[str] = None
) -> pd.DataFrame:
    """
    Create lag features for time series data.
    
    Args:
        df: Input DataFrame
        column: Column to create lags for
        lags: List of lag periods
        group_by: Optional column to group by (for panel data)
        
    Returns:
        DataFrame with lag features added
    """
    df_result = df.copy()
    
    if group_by:
        # Create lags within each group
        for lag in lags:
            df_result[f"{column}_lag_{lag}"] = df_result.groupby(group_by)[column].shift(lag)
    else:
        # Create lags for the entire series
        for lag in lags:
            df_result[f"{column}_lag_{lag}"] = df_result[column].shift(lag)
    
    return df_result

def apply_rolling_features(
    df: pd.DataFrame, 
    column: str, 
    windows: List[int], 
    functions: List[str] = ['mean', 'std', 'min', 'max'], 
    group_by: Optional[str] = None
) -> pd.DataFrame:
    """
    Create rolling window features for time series data.
    
    Args:
        df: Input DataFrame
        column: Column to create rolling features for
        windows: List of window sizes
        functions: List of aggregation functions to apply
        group_by: Optional column to group by (for panel data)
        
    Returns:
        DataFrame with rolling features added
    """
    df_result = df.copy()
    
    for window in windows:
        for func in functions:
            if group_by:
                # Apply rolling functions within each group
                if func == 'mean':
                    df_result[f"{column}_roll_{window}_{func}"] = df_result.groupby(group_by)[column].transform(
                        lambda x: x.rolling(window, min_periods=1).mean()
                    )
                elif func == 'std':
                    df_result[f"{column}_roll_{window}_{func}"] = df_result.groupby(group_by)[column].transform(
                        lambda x: x.rolling(window, min_periods=1).std()
                    )
                elif func == 'min':
                    df_result[f"{column}_roll_{window}_{func}"] = df_result.groupby(group_by)[column].transform(
                        lambda x: x.rolling(window, min_periods=1).min()
                    )
                elif func == 'max':
                    df_result[f"{column}_roll_{window}_{func}"] = df_result.groupby(group_by)[column].transform(
                        lambda x: x.rolling(window, min_periods=1).max()
                    )
            else:
                # Apply rolling functions to the entire series
                if func == 'mean':
                    df_result[f"{column}_roll_{window}_{func}"] = df_result[column].rolling(window, min_periods=1).mean()
                elif func == 'std':
                    df_result[f"{column}_roll_{window}_{func}"] = df_result[column].rolling(window, min_periods=1).std()
                elif func == 'min':
                    df_result[f"{column}_roll_{window}_{func}"] = df_result[column].rolling(window, min_periods=1).min()
                elif func == 'max':
                    df_result[f"{column}_roll_{window}_{func}"] = df_result[column].rolling(window, min_periods=1).max()
    
    return df_result

@cache_result(max_size=32)
def select_features(
    df: pd.DataFrame, 
    target_column: str, 
    method: str = 'mutual_info', 
    k: Optional[int] = None, 
    threshold: Optional[float] = None
) -> List[str]:
    """
    Perform automated feature selection.
    
    Args:
        df: Input DataFrame
        target_column: Target variable column
        method: Feature selection method ('mutual_info', 'f_regression', 'rfe', 'importance')
        k: Number of features to select (if None, use threshold)
        threshold: Threshold for feature importance (if k is None)
        
    Returns:
        List of selected feature names
    """
    # Prepare data
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Keep only numeric columns
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    X = X[numeric_cols]
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # Determine k if not provided
    if k is None:
        if threshold is None:
            # Default to half of features
            k = max(1, len(numeric_cols) // 2)
        else:
            # Will be handled by importance thresholding
            k = len(numeric_cols)
    
    selected_indices = []
    
    if method == 'mutual_info':
        # Mutual information
        selector = SelectKBest(mutual_info_regression, k=k)
        selector.fit(X_imputed, y)
        selected_indices = selector.get_support(indices=True)
        
    elif method == 'f_regression':
        # F-regression
        selector = SelectKBest(f_regression, k=k)
        selector.fit(X_imputed, y)
        selected_indices = selector.get_support(indices=True)
        
    elif method == 'rfe':
        # Recursive feature elimination
        if pd.api.types.is_numeric_dtype(y):
            estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            
        selector = RFE(estimator, n_features_to_select=k)
        selector.fit(X_imputed, y)
        selected_indices = selector.get_support(indices=True)
        
    elif method == 'importance':
        # Feature importance from random forest
        if pd.api.types.is_numeric_dtype(y):
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            
        model.fit(X_imputed, y)
        importances = model.feature_importances_
        
        if threshold is not None:
            # Select features above threshold
            selected_indices = [i for i, imp in enumerate(importances) if imp > threshold]
        else:
            # Select top k features
            selected_indices = np.argsort(importances)[-k:]
    
    # Convert indices to feature names
    selected_features = [numeric_cols[i] for i in selected_indices]
    
    return selected_features

def generate_advanced_transformation_code(
    df: pd.DataFrame, 
    transformations: List[Dict[str, Any]]
) -> str:
    """
    Generate Python code for advanced feature transformations.
    
    Args:
        df: Input DataFrame
        transformations: List of transformation specifications
        
    Returns:
        Python code as string
    """
    code_lines = [
        "import pandas as pd",
        "import numpy as np",
        "from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures",
        "from sklearn.impute import SimpleImputer",
        "from sklearn.preprocessing import OneHotEncoder",
        "",
        "def apply_feature_engineering(df):",
        "    # Make a copy of the DataFrame to avoid modifying the original",
        "    df_transformed = df.copy()",
        ""
    ]
    
    for transform in transformations:
        transform_type = transform.get("transformation", "").lower()
        column = transform.get("column", "")
        reason = transform.get("reason", "")
        
        if not column or not transform_type:
            continue
        
        if transform_type == "standardize":
            code_lines.extend([
                f"    # Standardize {column}: {reason}",
                f"    scaler = StandardScaler()",
                f"    df_transformed['{column}'] = scaler.fit_transform(df_transformed[['{column}']].fillna(0).values)",
                f"    print(\"Standardized '{column}'\")",
                ""
            ])
        
        elif transform_type == "normalize":
            code_lines.extend([
                f"    # Normalize {column}: {reason}",
                f"    scaler = MinMaxScaler()",
                f"    df_transformed['{column}'] = scaler.fit_transform(df_transformed[['{column}']].fillna(0).values)",
                f"    print(\"Normalized '{column}'\")",
                ""
            ])
        
        elif transform_type == "robust_scale":
            code_lines.extend([
                f"    # Robust scale {column}: {reason}",
                f"    scaler = RobustScaler()",
                f"    df_transformed['{column}'] = scaler.fit_transform(df_transformed[['{column}']].fillna(0).values)",
                f"    print(\"Robust scaled '{column}'\")",
                ""
            ])
        
        elif transform_type == "log_transform":
            code_lines.extend([
                f"    # Log transform {column}: {reason}",
                f"    # Add small epsilon to avoid log(0)",
                f"    epsilon = 1e-6",
                f"    df_transformed['{column}_log'] = np.log(df_transformed['{column}'] + epsilon)",
                f"    print(\"Log transformed '{column}'\")",
                ""
            ])
        
        elif transform_type == "sqrt_transform":
            code_lines.extend([
                f"    # Square root transform {column}: {reason}",
                f"    df_transformed['{column}_sqrt'] = np.sqrt(np.abs(df_transformed['{column}']))",
                f"    print(\"Square root transformed '{column}'\")",
                ""
            ])
        
        elif transform_type == "one_hot_encode":
            code_lines.extend([
                f"    # One-hot encode {column}: {reason}",
                f"    dummies = pd.get_dummies(df_transformed['{column}'], prefix='{column}')",
                f"    df_transformed = pd.concat([df_transformed, dummies], axis=1)",
                f"    print(\"One-hot encoded '{column}' into {{len(dummies.columns)}} columns\")",
                ""
            ])
        
        elif transform_type == "binning":
            num_bins = transform.get("num_bins", 5)
            code_lines.extend([
                f"    # Bin {column} into {num_bins} categories: {reason}",
                f"    df_transformed['{column}_bin'] = pd.cut(df_transformed['{column}'], bins={num_bins})",
                f"    print(\"Binned '{column}' into {num_bins} categories\")",
                ""
            ])
        
        elif transform_type == "polynomial":
            degree = transform.get("degree", 2)
            code_lines.extend([
                f"    # Create polynomial features for {column}: {reason}",
                f"    poly = PolynomialFeatures(degree={degree}, include_bias=False)",
                f"    poly_features = poly.fit_transform(df_transformed[['{column}']].fillna(0))",
                f"    feature_names = ['{column}'] + ['{column}^' + str(i+2) for i in range({degree-1})]",
                f"    for i, name in enumerate(feature_names[1:], 1):",
                f"        df_transformed[name] = poly_features[:, i]",
                f"    print(\"Created polynomial features for '{column}' with degree {degree}\")",
                ""
            ])
        
        elif transform_type == "interaction":
            interact_with = transform.get("interact_with", [])
            if interact_with:
                interact_cols = ", ".join([f"'{col}'" for col in interact_with])
                code_lines.extend([
                    f"    # Create interaction features for {column}: {reason}",
                    f"    interact_cols = [{interact_cols}]",
                    f"    for col in interact_cols:",
                    f"        df_transformed['{column}_x_' + col] = df_transformed['{column}'] * df_transformed[col]",
                    f"    print(\"Created interaction features for '{column}' with {{len(interact_cols)}} columns\")",
                    ""
                ])
    
    # Add return statement
    code_lines.append("    return df_transformed")
    
    return "\n".join(code_lines) 