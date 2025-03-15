import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from sklearn.preprocessing import LabelEncoder

class DataProcessor:
    def __init__(self):
        self.df = None
        self.label_encoders = {}

    def load_data(self, file) -> pd.DataFrame:
        """Load CSV into a DataFrame."""
        try:
            self.df = pd.read_csv(file)
            return self.df
        except Exception as e:
            raise Exception(f"Error loading CSV file: {e}")

    def get_basic_stats(self) -> Dict:
        """Get basic dataset statistics."""
        if self.df is None:
            raise Exception("No dataset loaded")

        stats = {
            "rows": len(self.df),
            "columns": len(self.df.columns),
            "numeric_columns": len(self.df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": len(self.df.select_dtypes(include=['object']).columns),
            "missing_values": str(self.df.isnull().sum().sum()),
            "memory_usage": self.df.memory_usage(deep=True).sum() / 1024**2
        }
        return stats

    def get_column_info(self) -> List[Dict]:
        """Get metadata for each column."""
        if self.df is None:
            raise Exception("No dataset loaded")

        column_info = []
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            unique_count = self.df[col].nunique()
            missing_count = self.df[col].isnull().sum()
            column_info.append({
                "name": col,
                "dtype": dtype,
                "unique_values": unique_count,
                "missing_values": missing_count
            })
        return column_info

    def export_data(self) -> Tuple[pd.DataFrame, Dict]:
        """Export processed DataFrame and transformations metadata."""
        if self.df is None:
            raise Exception("No dataset loaded")

        transformations = {
            "label_encodings": {
                col: list(le.classes_) 
                for col, le in self.label_encoders.items()
            }
        }
        return self.df, transformations
