from bson import ObjectId
from typing import Any, Dict, List, Union

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