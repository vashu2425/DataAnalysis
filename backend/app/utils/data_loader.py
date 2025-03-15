import pandas as pd
import os
import logging
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)

def load_dataset(file_path: str, chunk_size: Optional[int] = None) -> pd.DataFrame:
    """
    Load a dataset from a file path with optional chunked processing.
    
    Args:
        file_path: Path to the dataset file
        chunk_size: Number of rows to process at a time (None for loading entire file)
        
    Returns:
        Loaded DataFrame
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if chunk_size is not None and chunk_size > 0:
            logger.info(f"Loading dataset in chunks of {chunk_size} rows")
            # Process large files in chunks
            if file_extension == '.csv':
                chunks = []
                for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                    # Process each chunk (can add specific processing here)
                    chunks.append(chunk)
                return pd.concat(chunks, ignore_index=True)
            elif file_extension in ['.xlsx', '.xls']:
                # Excel files don't support native chunking, but we can implement row limits
                return pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file extension for chunked processing: {file_extension}")
        else:
            # Load entire file at once for smaller datasets
            if file_extension == '.csv':
                return pd.read_csv(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                return pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file extension: {file_extension}")
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

def save_dataset(df: pd.DataFrame, file_path: str, chunk_size: Optional[int] = None) -> None:
    """
    Save a dataset to a file path with optional chunked processing.
    
    Args:
        df: DataFrame to save
        file_path: Path to save the dataset
        chunk_size: Number of rows to write at a time (None for saving entire file at once)
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if chunk_size is not None and chunk_size > 0 and file_extension == '.csv':
            logger.info(f"Saving dataset in chunks of {chunk_size} rows")
            # Write in chunks to reduce memory usage
            for i in range(0, len(df), chunk_size):
                mode = 'w' if i == 0 else 'a'
                header = i == 0  # Only include header in first chunk
                df.iloc[i:i+chunk_size].to_csv(file_path, mode=mode, header=header, index=False)
        else:
            # Save entire file at once
            if file_extension == '.csv':
                df.to_csv(file_path, index=False)
            elif file_extension in ['.xlsx', '.xls']:
                df.to_excel(file_path, index=False)
            else:
                raise ValueError(f"Unsupported file extension: {file_extension}")
        
        logger.info(f"Successfully saved dataset to: {file_path}")
    except Exception as e:
        logger.error(f"Error saving dataset: {str(e)}")
        raise

def get_appropriate_chunk_size(file_size_bytes: int) -> Optional[int]:
    """
    Determine appropriate chunk size based on file size.
    
    Args:
        file_size_bytes: Size of the file in bytes
        
    Returns:
        Recommended chunk size (rows) or None for small files
    """
    # For files smaller than 50MB, don't use chunking
    if file_size_bytes < 50 * 1024 * 1024:
        return None
    
    # For files between 50MB and 500MB, use moderate chunk size
    elif file_size_bytes < 500 * 1024 * 1024:
        return 10000
    
    # For very large files, use smaller chunks
    else:
        return 5000 