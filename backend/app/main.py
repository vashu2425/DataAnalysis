from fastapi import FastAPI, Request, HTTPException, Depends, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
from pathlib import Path
import pandas as pd
import json
from bson import ObjectId
from .database.mongodb import db, datasets_collection, analysis_collection
from .routes import dataset_routes
from .utils.helpers import convert_objectid_to_str

# Get the current directory
BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Data Analysis Platform")

# Custom JSON encoder for MongoDB ObjectId
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        return super().default(obj)

# Override FastAPI's default JSON encoder
app.json_encoder = CustomJSONEncoder

# Mount static files
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Templates
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Include API routes
app.include_router(dataset_routes.router, prefix="/api/v1", tags=["api"])

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page"""
    try:
        return templates.TemplateResponse(
            "index.html",
            {"request": request}
        )
    except Exception as e:
        print(f"Error rendering home page: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/datasets", response_class=HTMLResponse)
async def datasets_page(request: Request):
    """Render the datasets page"""
    try:
        return templates.TemplateResponse(
            "index.html",
            {"request": request}
        )
    except Exception as e:
        print(f"Error rendering datasets page: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analysis/{dataset_id}", response_class=HTMLResponse)
async def analysis_page(request: Request, dataset_id: str):
    """Render the analysis page"""
    try:
        # Check for invalid dataset ID
        if not dataset_id or dataset_id == "0":
            return templates.TemplateResponse(
                "error.html",
                {"request": request, "error": "Invalid dataset ID provided"}
            )
            
        # Try to convert the dataset_id to ObjectId
        try:
            obj_id = ObjectId(dataset_id)
        except Exception as e:
            print(f"Invalid dataset ID format: {dataset_id}")
            return templates.TemplateResponse(
                "error.html",
                {"request": request, "error": "Invalid dataset ID format"}
            )
            
        dataset = datasets_collection.find_one({"_id": obj_id})
        if not dataset:
            print(f"Dataset not found with ID: {dataset_id}")
            return templates.TemplateResponse(
                "error.html",
                {"request": request, "error": "Dataset not found"}
            )
        
        # Convert ObjectId to string for JSON serialization
        dataset['_id'] = str(dataset['_id'])
        
        return templates.TemplateResponse(
            "analysis.html",
            {"request": request, "dataset": dataset}
        )
    except Exception as e:
        print(f"Error rendering analysis page: {str(e)}")
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "error": str(e)}
        )

@app.get("/chat/{dataset_id}", response_class=HTMLResponse)
async def chat_page(request: Request, dataset_id: str):
    """Render the chat page"""
    try:
        # Check for invalid dataset ID
        if not dataset_id or dataset_id == "0":
            return templates.TemplateResponse(
                "error.html",
                {"request": request, "error": "Invalid dataset ID provided"}
            )
            
        # Try to convert the dataset_id to ObjectId
        try:
            obj_id = ObjectId(dataset_id)
        except Exception as e:
            print(f"Invalid dataset ID format: {dataset_id}")
            return templates.TemplateResponse(
                "error.html",
                {"request": request, "error": "Invalid dataset ID format"}
            )
            
        dataset = datasets_collection.find_one({"_id": obj_id})
        if not dataset:
            print(f"Dataset not found with ID: {dataset_id}")
            return templates.TemplateResponse(
                "error.html",
                {"request": request, "error": "Dataset not found"}
            )
        
        # Convert ObjectId to string for JSON serialization
        dataset['_id'] = str(dataset['_id'])
        
        return templates.TemplateResponse(
            "chat.html",
            {"request": request, "dataset": dataset}
        )
    except Exception as e:
        print(f"Error rendering chat page: {str(e)}")
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "error": str(e)}
        )

@app.on_event("startup")
async def startup_event():
    """Initialize database connection on startup"""
    try:
        # The connect method is already attached to the db object
        # in the mongodb.py module, so we don't need to check if it exists
        if hasattr(db, 'connect'):
            db.connect()
        else:
            print("Database object does not have connect method")
    except Exception as e:
        print(f"Error connecting to database: {str(e)}")
        # Don't raise the exception, let the app continue with in-memory DB

@app.on_event("shutdown")
async def shutdown_event():
    """Close database connection on shutdown"""
    try:
        if hasattr(db, 'close'):
            db.close()
        else:
            print("Database object does not have close method")
    except Exception as e:
        print(f"Error closing database connection: {str(e)}") 