"""
MongoDB connection and in-memory fallback database module.
"""
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv
import os
import logging
from bson import ObjectId
import time
from functools import lru_cache
from typing import Dict, Any, Optional, List, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get configuration from environment
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "data_analysis_db")
USE_IN_MEMORY = os.getenv("USE_IN_MEMORY", "false").lower() == "true"

# Connection pool settings
MAX_POOL_SIZE = int(os.getenv("MONGODB_MAX_POOL_SIZE", "100"))
MIN_POOL_SIZE = int(os.getenv("MONGODB_MIN_POOL_SIZE", "10"))
MAX_IDLE_TIME_MS = int(os.getenv("MONGODB_MAX_IDLE_TIME_MS", "60000"))  # 1 minute

# Cache settings
CACHE_TTL = int(os.getenv("CACHE_TTL_SECONDS", "300"))  # 5 minutes cache TTL
CACHE_SIZE = int(os.getenv("CACHE_SIZE", "128"))  # Maximum number of cached items

# In-memory storage
memory_datasets = {}
memory_analyses = {}

# Simple time-based cache
class SimpleCache:
    def __init__(self, ttl_seconds: int = 300, max_size: int = 100):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            item = self.cache[key]
            if time.time() - item["timestamp"] < self.ttl_seconds:
                return item["value"]
            else:
                # Remove expired item
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any) -> None:
        # If cache is full, remove oldest item
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]["timestamp"])
            del self.cache[oldest_key]
        
        self.cache[key] = {
            "value": value,
            "timestamp": time.time()
        }
    
    def invalidate(self, key: str) -> None:
        if key in self.cache:
            del self.cache[key]
    
    def clear(self) -> None:
        self.cache.clear()

# Initialize cache
dataset_cache = SimpleCache(ttl_seconds=CACHE_TTL, max_size=CACHE_SIZE)
analysis_cache = SimpleCache(ttl_seconds=CACHE_TTL, max_size=CACHE_SIZE)

# This is a simple mock of a MongoDB collection for in-memory use
class InMemoryCollection:
    def __init__(self, name):
        self.name = name
        self.storage = memory_datasets if name == "datasets" else memory_analyses
        self.cache = dataset_cache if name == "datasets" else analysis_cache
        logger.info(f"Using in-memory collection: {name}")
    
    def insert_one(self, document):
        if "_id" not in document:
            document["_id"] = ObjectId()
        
        doc_id = str(document["_id"])
        self.storage[doc_id] = document
        
        # Invalidate cache for this ID
        self.cache.invalidate(doc_id)
        
        return type('obj', (object,), {'inserted_id': document["_id"]})
    
    def find_one(self, query):
        if "_id" in query:
            doc_id = str(query["_id"])
            
            # Try to get from cache first
            cached_doc = self.cache.get(doc_id)
            if cached_doc:
                return cached_doc
            
            # Get from storage and cache it
            doc = self.storage.get(doc_id)
            if doc:
                self.cache.set(doc_id, doc)
            return doc
        return None
    
    def find(self, query=None):
        # For simplicity, we don't cache find() results
        return list(self.storage.values())
    
    def update_one(self, query, update, upsert=False):
        if "_id" in query:
            doc_id = str(query["_id"])
            if doc_id in self.storage:
                # Apply updates
                if "$set" in update:
                    for key, value in update["$set"].items():
                        self.storage[doc_id][key] = value
                
                # Invalidate cache
                self.cache.invalidate(doc_id)
                
                return type('obj', (object,), {'modified_count': 1})
            elif upsert:
                # Create new document
                new_doc = {"_id": query["_id"]}
                if "$set" in update:
                    for key, value in update["$set"].items():
                        new_doc[key] = value
                self.storage[doc_id] = new_doc
                return type('obj', (object,), {'upserted_id': query["_id"]})
        return type('obj', (object,), {'modified_count': 0})

# MongoDB collection wrapper with caching
class CachedCollection:
    def __init__(self, collection, cache):
        self.collection = collection
        self.cache = cache
    
    def insert_one(self, document):
        result = self.collection.insert_one(document)
        # Invalidate cache for this ID
        self.cache.invalidate(str(result.inserted_id))
        return result
    
    def find_one(self, query):
        if "_id" in query and isinstance(query["_id"], ObjectId):
            doc_id = str(query["_id"])
            
            # Try to get from cache first
            cached_doc = self.cache.get(doc_id)
            if cached_doc:
                logger.debug(f"Cache hit for document {doc_id}")
                return cached_doc
            
            # Get from database and cache it
            doc = self.collection.find_one(query)
            if doc:
                self.cache.set(doc_id, doc)
            return doc
        return self.collection.find_one(query)
    
    def find(self, query=None, *args, **kwargs):
        # We don't cache find() results for simplicity
        return self.collection.find(query, *args, **kwargs)
    
    def update_one(self, query, update, upsert=False):
        result = self.collection.update_one(query, update, upsert=upsert)
        
        # Invalidate cache if document was updated
        if "_id" in query and isinstance(query["_id"], ObjectId):
            self.cache.invalidate(str(query["_id"]))
        
        return result

# Determine if we should use MongoDB or in-memory storage
if USE_IN_MEMORY:
    logger.info("Using in-memory database (MongoDB connection disabled)")
    db = type('obj', (object,), {})
    datasets_collection = InMemoryCollection("datasets")
    analysis_collection = InMemoryCollection("analysis_results")
    
    # Define connect/close methods for consistency
    def connect():
        logger.info("In-memory database ready")
    
    def close():
        logger.info("In-memory database closed")
    
    # Attach methods to db object
    db.connect = connect
    db.close = close
else:
    try:
        logger.info(f"Connecting to MongoDB at {MONGODB_URI}")
        client = MongoClient(
            MONGODB_URI, 
            serverSelectionTimeoutMS=5000,
            maxPoolSize=MAX_POOL_SIZE,
            minPoolSize=MIN_POOL_SIZE,
            maxIdleTimeMS=MAX_IDLE_TIME_MS
        )
        
        # Verify connection
        client.admin.command('ping')
        
        # Set up database
        db = client[DATABASE_NAME]
        
        # Wrap collections with cache
        datasets_collection = CachedCollection(db.datasets, dataset_cache)
        analysis_collection = CachedCollection(db.analysis_results, analysis_cache)
        
        logger.info(f"Connected to MongoDB successfully with connection pool (min={MIN_POOL_SIZE}, max={MAX_POOL_SIZE})")
        
        # Define connect/close methods for consistency
        def connect():
            logger.info("MongoDB already connected")
        
        def close():
            client.close()
            logger.info("MongoDB connection closed")
        
        # Attach methods to db object
        db.connect = connect
        db.close = close
        
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {str(e)}")
        logger.info("Falling back to in-memory database")
        
        # Fall back to in-memory storage
        db = type('obj', (object,), {})
        datasets_collection = InMemoryCollection("datasets")
        analysis_collection = InMemoryCollection("analysis_results")
        
        # Define connect/close methods for consistency
        def connect():
            logger.info("In-memory database ready")
        
        def close():
            logger.info("In-memory database closed")
        
        # Attach methods to db object
        db.connect = connect
        db.close = close

# Decorator for caching expensive function results
def cache_result(max_size=128):
    """Decorator to cache function results using LRU cache"""
    def decorator(func):
        @lru_cache(maxsize=max_size)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator 