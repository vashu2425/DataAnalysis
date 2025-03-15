from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class Dataset(BaseModel):
    name: str
    columns: List[str]
    row_count: int
    metadata: Dict[str, Any]

class DatasetResponse(BaseModel):
    id: str
    name: str
    columns: List[str]
    row_count: int
    metadata: Dict[str, Any]

class AnalysisResult(BaseModel):
    id: Optional[str] = None
    dataset_id: str
    analysis_type: str
    results: Dict[str, Any]
    created_at: datetime = Field(default_factory=datetime.now)

class FeatureEngineeringResult(BaseModel):
    dataset_id: str
    original_features: List[str]
    new_features: List[str]
    transformations: List[Dict[str, Any]]
    created_at: datetime = Field(default_factory=datetime.utcnow) 