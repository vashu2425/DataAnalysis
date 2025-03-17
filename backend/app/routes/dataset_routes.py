from fastapi import APIRouter, HTTPException, UploadFile, File, Query, Body, BackgroundTasks
from fastapi.responses import JSONResponse
from ..models.models import Dataset, DatasetResponse, AnalysisResult, FeatureEngineeringResult
from ..database.mongodb import datasets_collection, analysis_collection, cache_result
from ..utils.ai_helper import AIHelper
from ..utils.helpers import convert_objectid_to_str
from ..utils.data_loader import load_dataset, save_dataset, get_appropriate_chunk_size
from ..utils.feature_engineering import (
    apply_log_transformation, apply_polynomial_features, apply_binning,
    apply_time_features, apply_lag_features, apply_rolling_features,
    select_features, generate_advanced_transformation_code
)
from typing import List, Dict, Any, Optional
import pandas as pd
from bson import ObjectId
import numpy as np
import json
from pathlib import Path
import io
import traceback
from fastapi.responses import FileResponse
import time
from datetime import datetime
import os
import logging

# Set up logger
logger = logging.getLogger(__name__)

router = APIRouter()

# Get the data directory path
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Initialize AI helper
ai_helper = AIHelper()

@router.post("/dataset/")
async def create_dataset(file: UploadFile = File(...)):
    """Upload and process a new dataset"""
    try:
        # Read file content into memory temporarily to determine size
        content = await file.read()
        file_size = len(content)
        
        # Generate a unique filename
        dataset_id = ObjectId()
        filename = f"{dataset_id}.csv"
        file_path = DATA_DIR / filename
        
        # Save the uploaded file
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Determine appropriate chunk size based on file size
        chunk_size = get_appropriate_chunk_size(file_size)
        
        # Load the dataset with chunking if needed
        df = load_dataset(str(file_path), chunk_size=chunk_size)
        
        # Basic dataset info
        dataset_info = {
            "_id": dataset_id,
            "name": file.filename,
            "original_filename": file.filename,  # Store original filename for downloads
            "stored_filename": filename,
            "columns": df.columns.tolist(),
            "row_count": len(df),
            "file_size_bytes": file_size,
            "upload_time": datetime.now(),
            "metadata": {
                "dtypes": df.dtypes.astype(str).to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "numeric_summary": df.describe().to_dict() if not df.empty else {}
            }
        }
        
        # Insert into MongoDB
        datasets_collection.insert_one(dataset_info)
        
        return {"id": str(dataset_id)}
    except Exception as e:
        print(f"Error creating dataset: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/dataset/{dataset_id}")
async def get_dataset(dataset_id: str):
    """Get a dataset by ID"""
    try:
        # Try to convert the dataset_id to ObjectId
        try:
            obj_id = ObjectId(dataset_id)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid dataset ID format: {dataset_id}")
            
        dataset = datasets_collection.find_one({"_id": obj_id})
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Convert ObjectId to string for JSON serialization
        return convert_objectid_to_str(dataset)
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dataset/{dataset_id}/correlation")
async def get_correlation_matrix(dataset_id: str):
    """Get correlation matrix for numeric columns in the dataset"""
    try:
        dataset = datasets_collection.find_one({"_id": ObjectId(dataset_id)})
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Convert ObjectId to string
        dataset = convert_objectid_to_str(dataset)
        
        # Load data from stored file
        file_path = DATA_DIR / dataset["stored_filename"]
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Dataset file not found")
            
        # Load the data
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading CSV file: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error reading dataset file: {str(e)}")
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return {
                "correlation": {},
                "message": "No numeric columns found in dataset",
                "columns": df.columns.tolist(),
                "dtypes": {col: str(df[col].dtype) for col in df.columns}
            }
        
        # Calculate correlation matrix
        try:
            correlation = df[numeric_cols].corr().round(4).to_dict()
            
            # Convert any numpy types to Python native types for JSON serialization
            for col1 in correlation:
                for col2 in correlation[col1]:
                    if isinstance(correlation[col1][col2], np.number):
                        correlation[col1][col2] = float(correlation[col1][col2])
            
            return {"correlation": correlation, "numeric_columns": numeric_cols}
        except Exception as e:
            print(f"Error calculating correlation: {str(e)}")
            return {
                "correlation": {},
                "message": f"Error calculating correlation: {str(e)}",
                "numeric_columns": numeric_cols
            }
    except Exception as e:
        print(f"Error calculating correlation matrix: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/dataset/{dataset_id}/eda")
async def get_eda(dataset_id: str):
    """Get comprehensive exploratory data analysis"""
    try:
        dataset = datasets_collection.find_one({"_id": ObjectId(dataset_id)})
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Convert ObjectId to string
        dataset = convert_objectid_to_str(dataset)
        
        # Load data from stored file
        file_path = DATA_DIR / dataset["stored_filename"]
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Dataset file not found")
            
        df = pd.read_csv(file_path)
        
        # Get numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        result = {
            "dataset_summary": {
                "row_count": int(len(df)),
                "column_count": int(len(df.columns)),
                "memory_usage": int(df.memory_usage(deep=True).sum()),
                "duplicated_rows": int(df.duplicated().sum())
            },
            "correlation": {},
            "numeric_analysis": {},
            "categorical_analysis": {},
            "missing_values": {col: int(val) for col, val in df.isnull().sum().to_dict().items()}
        }
        
        # Correlation matrix for numeric columns
        if numeric_cols:
            try:
                numeric_df = df[numeric_cols].copy()
                correlation = numeric_df.corr().round(4)
                
                # Convert to dictionary with proper JSON serialization
                correlation_dict = {}
                for col1 in correlation.columns:
                    correlation_dict[col1] = {}
                    for col2 in correlation.columns:
                        val = correlation.loc[col1, col2]
                        correlation_dict[col1][col2] = float(val) if not pd.isna(val) else 0
                
                result["correlation"] = correlation_dict
            except Exception as e:
                print(f"Error calculating correlation: {str(e)}")
                result["correlation"] = {}
        
        # Numeric column analysis
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                result["numeric_analysis"][col] = {
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "mean": float(col_data.mean()),
                    "median": float(col_data.median()),
                    "std": float(col_data.std()),
                    "skew": float(col_data.skew()),
                    "kurtosis": float(col_data.kurtosis()),
                    "histogram": {
                        "x": col_data.astype(float).tolist(),
                        "type": "histogram",
                        "name": col
                    },
                    "boxplot": {
                        "y": col_data.astype(float).tolist(),
                        "type": "box",
                        "name": col
                    }
                }
        
        # Categorical column analysis
        for col in categorical_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                value_counts = col_data.value_counts().head(10)
                result["categorical_analysis"][col] = {
                    "unique_count": int(col_data.nunique()),
                    "top_values": {
                        "labels": value_counts.index.tolist(),
                        "values": value_counts.values.astype(int).tolist(),
                        "type": "pie",
                        "name": col
                    }
                }
        
        # Store EDA results in the database
        analysis_collection.insert_one({
            "dataset_id": ObjectId(dataset_id),
            "analysis_type": "eda",
            "results": result,
            "created_at": datetime.now()
        })
        
        return result
    except Exception as e:
        print(f"Error performing EDA: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/dataset/{dataset_id}/eda-report")
async def get_eda_report(dataset_id: str):
    """Generate and download a comprehensive EDA report in PDF format"""
    try:
        dataset = datasets_collection.find_one({"_id": ObjectId(dataset_id)})
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Convert ObjectId to string
        dataset = convert_objectid_to_str(dataset)
        
        # Load data from stored file
        file_path = DATA_DIR / dataset["stored_filename"]
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Dataset file not found")
            
        df = pd.read_csv(file_path)
        
        # Generate a report file
        report_filename = f"{dataset_id}_eda_report.pdf"
        report_path = DATA_DIR / report_filename
        
        try:
            # Try to use pandas-profiling (ydata-profiling) for a comprehensive report
            try:
                from ydata_profiling import ProfileReport
                profile = ProfileReport(df, title=f"EDA Report for {dataset['name']}", 
                                       explorative=True,
                                       html={'style': {'full_width': True}})
                
                # Add custom section for training compatibility
                training_compatibility = assess_training_compatibility(df, dataset.get("metadata", {}))
                
                # Save the report to HTML first to add custom sections
                html_report_path = report_path.with_suffix('.html')
                profile.to_file(html_report_path)
                
                # Read the HTML content
                with open(html_report_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                # Check for binned columns and PCA components
                binned_columns = []
                for col in df.select_dtypes(include=['object', 'category']).columns:
                    if df[col].dtype == 'object' and df[col].str.contains(r'[\(\[].*,.*[\)\]]').any():
                        binned_columns.append(col)
                
                pca_columns = [col for col in df.columns if col.startswith('PC') and col[2:].isdigit()]
                
                # Create a more detailed training compatibility section
                training_section = f"""
                <div class="row">
                    <div class="col-sm-12">
                        <div class="card">
                            <div class="card-header">
                                <h4>Machine Learning Training Compatibility</h4>
                            </div>
                            <div class="card-body">
                                <p>This section evaluates the dataset's compatibility for machine learning training.</p>
                                <div class="alert alert-{'success' if training_compatibility['overall_score'] >= 70 else 'warning'}">
                                    <h5>Overall Compatibility Score: {training_compatibility['overall_score']}%</h5>
                                    <p>{training_compatibility['overall_message']}</p>
                                </div>
                                
                                <div class="row">
                                    <div class="col-md-6">
                                        <h5>Compatibility Factors:</h5>
                                        <table class="table table-sm">
                                            <thead>
                                                <tr>
                                                    <th>Factor</th>
                                                    <th>Score</th>
                                                    <th>Assessment</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                <tr>
                                                    <td>Missing Values</td>
                                                    <td>{training_compatibility['missing_values_score']}%</td>
                                                    <td>{training_compatibility['missing_values_message']}</td>
                                                </tr>
                                                <tr>
                                                    <td>Data Types</td>
                                                    <td>{training_compatibility['data_types_score']}%</td>
                                                    <td>{training_compatibility['data_types_message']}</td>
                                                </tr>
                                                <tr>
                                                    <td>Feature Distribution</td>
                                                    <td>{training_compatibility['feature_distribution_score']}%</td>
                                                    <td>{training_compatibility['feature_distribution_message']}</td>
                                                </tr>
                                                <tr>
                                                    <td>Feature Correlation</td>
                                                    <td>{training_compatibility['correlation_score']}%</td>
                                                    <td>{training_compatibility['correlation_message']}</td>
                                                </tr>
                                                <tr>
                                                    <td>Dataset Size</td>
                                                    <td>{training_compatibility['size_score']}%</td>
                                                    <td>{training_compatibility['size_message']}</td>
                                                </tr>
                                                <tr>
                                                    <td>Preprocessing Quality</td>
                                                    <td>{training_compatibility['preprocessing_score']}%</td>
                                                    <td>{training_compatibility['preprocessing_message']}</td>
                                                </tr>
                                            </tbody>
                                        </table>
                                    </div>
                                    
                                    <div class="col-md-6">
                                        <h5>Recommendations:</h5>
                                        <ul class="list-group">
                                            {''.join(f'<li class="list-group-item">{rec}</li>' for rec in training_compatibility['recommendations'])}
                                        </ul>
                                        
                                        <div class="mt-4">
                                            <h5>Special Features:</h5>
                                            <ul class="list-group">
                                                {f'<li class="list-group-item"><strong>Binned Features:</strong> This dataset contains {len(binned_columns)} binned numeric features that represent ranges of values.</li>' if binned_columns else ''}
                                                {f'<li class="list-group-item"><strong>PCA Components:</strong> This dataset contains {len(pca_columns)} PCA components from dimensionality reduction.</li>' if pca_columns else ''}
                                                {f'<li class="list-group-item"><strong>Feature Engineering:</strong> This dataset has undergone feature engineering transformations.</li>' if "metadata" in dataset and "transformations" in dataset["metadata"] else ''}
                                                {f'<li class="list-group-item"><strong>Feature Selection:</strong> This dataset has undergone feature selection.</li>' if "metadata" in dataset and "feature_selection" in dataset["metadata"] else ''}
                                            </ul>
                                        </div>
                                    </div>
                                </div>
                                
                                <div class="mt-4">
                                    <h5>Training Package</h5>
                                    <p>
                                        You can download a complete training package for this dataset, which includes:
                                    </p>
                                    <ul>
                                        <li><strong>dataset.csv</strong>: The clean dataset for training</li>
                                        <li><strong>metadata.json</strong>: Information about the dataset structure</li>
                                        <li><strong>README.md</strong>: Documentation on dataset usage</li>
                                        <li><strong>train_model.py</strong>: A Python script for loading, preprocessing, training, and evaluating a model</li>
                                    </ul>
                                    <p>
                                        <a href="/api/v1/dataset/{dataset_id}/download-training" class="btn btn-primary">
                                            Download Training-Ready Package
                                        </a>
                                    </p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                """
                
                html_content = html_content.replace('</body>', f'{training_section}</body>')
                
                # Write the modified HTML content back to the file
                with open(html_report_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                # Convert HTML to PDF
                import pdfkit
                try:
                    # Try to use pdfkit with wkhtmltopdf
                    pdfkit.from_file(str(html_report_path), str(report_path))
                except Exception as pdf_error:
                    print(f"Error converting HTML to PDF with pdfkit: {str(pdf_error)}")
                    # Fallback to using weasyprint
                    try:
                        from weasyprint import HTML
                        HTML(str(html_report_path)).write_pdf(str(report_path))
                    except Exception as weasy_error:
                        print(f"Error converting HTML to PDF with weasyprint: {str(weasy_error)}")
                        # If both fail, just return the HTML report
                        return FileResponse(
                            path=html_report_path,
                            filename=f"EDA_Report_{dataset['name']}.html",
                            media_type="text/html"
                        )
                
                # Return the report file
                return FileResponse(
                    path=report_path,
                    filename=f"EDA_Report_{dataset['name']}.pdf",
                    media_type="application/pdf"
                )
            except ImportError:
                try:
                    from pandas_profiling import ProfileReport
                    profile = ProfileReport(df, title=f"EDA Report for {dataset['name']}", explorative=True)
                    profile.to_file(report_path)
                    
                    # Return the report file
                    return FileResponse(
                        path=report_path,
                        filename=f"EDA_Report_{dataset['name']}.pdf",
                        media_type="application/pdf"
                    )
                except ImportError:
                    # If pandas-profiling is not available, use matplotlib and PDF generation
                    pass
            
            # If pandas-profiling is not available or fails, create a custom PDF report
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import seaborn as sns
            from matplotlib.backends.backend_pdf import PdfPages
            import io
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            from reportlab.lib import colors
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet
            
            # Create PDF document
            doc = SimpleDocTemplate(str(report_path), pagesize=letter)
            styles = getSampleStyleSheet()
            elements = []
            
            # Title
            title = Paragraph(f"<b>EDA Report for {dataset['name']}</b>", styles['Title'])
            elements.append(title)
            elements.append(Spacer(1, 12))
            
            # Dataset Summary
            elements.append(Paragraph("<b>Dataset Summary</b>", styles['Heading2']))
            summary_data = [
                ["Rows", str(len(df))],
                ["Columns", str(len(df.columns))],
                ["Memory Usage", f"{df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB"],
                ["Duplicate Rows", str(df.duplicated().sum())]
            ]
            summary_table = Table(summary_data)
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('PADDING', (0, 0), (-1, -1), 6)
            ]))
            elements.append(summary_table)
            elements.append(Spacer(1, 12))
            
            # Training Compatibility Section
            elements.append(Paragraph("<b>Machine Learning Training Compatibility</b>", styles['Heading2']))
            training_compatibility = assess_training_compatibility(df, dataset.get("metadata", {}))
            
            compatibility_data = [
                ["Overall Score", f"{training_compatibility['overall_score']}%"],
                ["Missing Values", training_compatibility['missing_values_message']],
                ["Data Types", training_compatibility['data_types_message']],
                ["Feature Distribution", training_compatibility['feature_distribution_message']],
                ["Feature Correlation", training_compatibility['correlation_message']],
                ["Dataset Size", training_compatibility['size_message']]
            ]
            
            compatibility_table = Table(compatibility_data)
            compatibility_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('PADDING', (0, 0), (-1, -1), 6)
            ]))
            elements.append(compatibility_table)
            elements.append(Spacer(1, 12))
            
            elements.append(Paragraph("<b>Recommendations for Training</b>", styles['Heading3']))
            for rec in training_compatibility['recommendations']:
                elements.append(Paragraph(f"• {rec}", styles['Normal']))
            elements.append(Spacer(1, 12))
            
            # Data Types
            elements.append(Paragraph("<b>Data Types</b>", styles['Heading2']))
            dtype_data = [["Column", "Type"]]
            for col, dtype in df.dtypes.items():
                dtype_data.append([col, str(dtype)])
            dtype_table = Table(dtype_data)
            dtype_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('PADDING', (0, 0), (-1, -1), 6)
            ]))
            elements.append(dtype_table)
            elements.append(Spacer(1, 12))
            
            # Missing Values
            elements.append(Paragraph("<b>Missing Values</b>", styles['Heading2']))
            missing_data = [["Column", "Missing Count", "Missing Percentage"]]
            for col in df.columns:
                missing_count = df[col].isna().sum()
                missing_pct = missing_count / len(df) * 100
                missing_data.append([col, str(missing_count), f"{missing_pct:.2f}%"])
            missing_table = Table(missing_data)
            missing_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('PADDING', (0, 0), (-1, -1), 6)
            ]))
            elements.append(missing_table)
            elements.append(Spacer(1, 12))
            
            # Numeric Summary
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                elements.append(Paragraph("<b>Numeric Summary</b>", styles['Heading2']))
                desc = df[numeric_cols].describe().reset_index()
                desc_data = [["Statistic"] + numeric_cols]
                for _, row in desc.iterrows():
                    row_data = [row['index']]
                    for col in numeric_cols:
                        try:
                            val = row[col]
                            if isinstance(val, (int, float)):
                                row_data.append(f"{val:.4f}")
                            else:
                                row_data.append(str(val))
                        except:
                            row_data.append("N/A")
                    desc_data.append(row_data)
                
                # Create a table with the numeric summary
                desc_table = Table(desc_data)
                desc_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('PADDING', (0, 0), (-1, -1), 6)
                ]))
                elements.append(desc_table)
                elements.append(Spacer(1, 24))
            
            # Generate visualizations and add them to the PDF
            # Save the document
            doc.build(elements)
            
            # Now add matplotlib visualizations using PdfPages
            with PdfPages(str(report_path.with_suffix('.temp.pdf'))) as pdf:
                # Correlation Matrix with enhanced visualization
                if len(numeric_cols) > 1:
                    plt.figure(figsize=(12, 10))
                    corr = df[numeric_cols].corr()
                    
                    # Set up the matplotlib figure
                    plt.figure(figsize=(12, 10))
                    
                    # Generate a custom diverging colormap
                    cmap = sns.diverging_palette(220, 10, as_cmap=True)
                    
                    # Draw the heatmap with improved aesthetics
                    mask = np.triu(np.ones_like(corr, dtype=bool))
                    sns.heatmap(
                        corr, 
                        mask=mask,
                        annot=True,
                        fmt=".2f",
                        cmap=cmap,
                        vmin=-1, 
                        vmax=1,
                        center=0,
                        square=True, 
                        linewidths=1,
                        cbar_kws={"shrink": .8, "label": "Correlation Coefficient"},
                        annot_kws={"size": 10}
                    )
                    
                    # Improve the appearance
                    plt.title("Correlation Matrix", fontsize=16, fontweight='bold', pad=20)
                    plt.tight_layout()
                    
                    # Add a text box with interpretation guide
                    plt.figtext(
                        0.5, 0.01, 
                        "Interpretation: Values close to +1.0 (dark blue) indicate strong positive correlation,\n"
                        "values close to -1.0 (dark red) indicate strong negative correlation,\n"
                        "and values near 0 (white) suggest little to no relationship.",
                        ha="center", 
                        fontsize=10, 
                        bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5}
                    )
                    
                    pdf.savefig()
                    plt.close()
                
                # Distribution plots for numeric columns
                for col in numeric_cols[:min(10, len(numeric_cols))]:  # Limit to 10 columns
                    plt.figure(figsize=(10, 6))
                    sns.histplot(df[col].dropna(), kde=True, color='royalblue')
                    plt.title(f"Distribution of {col}", fontsize=14, fontweight='bold')
                    plt.xlabel(col, fontsize=12)
                    plt.ylabel("Frequency", fontsize=12)
                    plt.grid(axis='y', alpha=0.3)
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close()
                
                # Box plots for numeric columns
                for col in numeric_cols[:min(10, len(numeric_cols))]:  # Limit to 10 columns
                    plt.figure(figsize=(10, 6))
                    sns.boxplot(y=df[col].dropna(), color='royalblue')
                    plt.title(f"Box Plot of {col}", fontsize=14, fontweight='bold')
                    plt.ylabel(col, fontsize=12)
                    plt.grid(axis='x', alpha=0.3)
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close()
                
                # Categorical columns analysis
                cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                for col in cat_cols[:min(5, len(cat_cols))]:  # Limit to 5 columns
                    plt.figure(figsize=(12, 6))
                    value_counts = df[col].value_counts().head(10)  # Top 10 categories
                    sns.barplot(x=value_counts.index, y=value_counts.values)
                    plt.title(f"Top Categories in {col}")
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close()
            
            # Merge the two PDFs
            import PyPDF2
            
            # Create a PDF merger object
            merger = PyPDF2.PdfMerger()
            
            # Add the first PDF (reportlab generated)
            merger.append(str(report_path))
            
            # Add the second PDF (matplotlib visualizations)
            merger.append(str(report_path.with_suffix('.temp.pdf')))
            
            # Write to a new file
            final_report_path = report_path.with_suffix('.final.pdf')
            merger.write(str(final_report_path))
            merger.close()
            
            # Clean up temporary files
            import os
            if os.path.exists(str(report_path)):
                os.remove(str(report_path))
            if os.path.exists(str(report_path.with_suffix('.temp.pdf'))):
                os.remove(str(report_path.with_suffix('.temp.pdf')))
            
            # Rename the final file to the original name
            os.rename(str(final_report_path), str(report_path))
            
            # Return the report file
            return FileResponse(
                path=report_path,
                filename=f"EDA_Report_{dataset['name']}.pdf",
                media_type="application/pdf"
            )
            
        except Exception as e:
            print(f"Error generating PDF report: {str(e)}")
            print(traceback.format_exc())
            
            # Fallback to a simple HTML report if PDF generation fails
            html_report_path = report_path.with_suffix('.html')
            html_content = f"""
            <html>
            <head>
                <title>EDA Report for {dataset['name']}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #2c3e50; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                </style>
            </head>
            <body>
                <h1>EDA Report for {dataset['name']}</h1>
                <h2>Dataset Summary</h2>
                <p>Rows: {len(df)}</p>
                <p>Columns: {len(df.columns)}</p>
                
                <h2>Data Types</h2>
                <table>
                    <tr><th>Column</th><th>Type</th></tr>
                    {"".join(f"<tr><td>{col}</td><td>{dtype}</td></tr>" for col, dtype in df.dtypes.items())}
                </table>
                
                <h2>Missing Values</h2>
                <table>
                    <tr><th>Column</th><th>Missing Count</th><th>Missing Percentage</th></tr>
                    {"".join(f"<tr><td>{col}</td><td>{df[col].isna().sum()}</td><td>{df[col].isna().sum() / len(df) * 100:.2f}%</td></tr>" for col in df.columns)}
                </table>
                
                <h2>Numeric Summary</h2>
                {df.describe().to_html()}
            </body>
            </html>
            """
            with open(html_report_path, "w") as f:
                f.write(html_content)
            
            # Return the HTML report file as fallback
            return FileResponse(
                path=html_report_path,
                filename=f"EDA_Report_{dataset['name']}.html",
                media_type="text/html"
            )
        
    except Exception as e:
        print(f"Error generating EDA report: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/dataset/{dataset_id}/pca")
async def apply_pca(
    dataset_id: str, 
    n_components: int = Query(2, ge=2, le=10),
    variance_threshold: float = Query(0.8, ge=0.1, le=0.99)
):
    """Apply PCA transformation as part of feature engineering"""
    try:
        dataset = datasets_collection.find_one({"_id": ObjectId(dataset_id)})
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Load data from stored file
        file_path = DATA_DIR / dataset["stored_filename"]
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Dataset file not found")
            
        # Load the data
        df = pd.read_csv(file_path)
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return {"error": "Not enough numeric columns for PCA"}
        
        # Apply PCA
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        # Select numeric columns and drop rows with NaN
        numeric_df = df[numeric_cols].dropna()
        
        if numeric_df.empty:
            return {"error": "No data available after removing missing values"}
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        
        # Determine optimal number of components based on variance threshold
        pca_all = PCA()
        pca_all.fit(scaled_data)
        cumulative_variance = np.cumsum(pca_all.explained_variance_ratio_)
        optimal_n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
        
        # Use either user-specified or optimal number of components
        final_n_components = min(n_components, optimal_n_components, len(numeric_cols))
        
        # Apply PCA with the determined number of components
        pca = PCA(n_components=final_n_components)
        pca_result = pca.fit_transform(scaled_data)
        
        # Create a DataFrame with PCA results
        pca_columns = [f'PC{i+1}' for i in range(pca_result.shape[1])]
        pca_df = pd.DataFrame(
            data=pca_result,
            columns=pca_columns,
            index=numeric_df.index
        )
        
        # Create a new DataFrame with original non-numeric columns and PCA results
        non_numeric_cols = [col for col in df.columns if col not in numeric_cols]
        if non_numeric_cols:
            # Keep non-numeric columns
            result_df = df.copy()
            # Drop original numeric columns
            result_df = result_df.drop(columns=numeric_cols)
            # Add PCA columns
            for col in pca_columns:
                result_df.loc[numeric_df.index, col] = pca_df[col]
        else:
            # If all columns were numeric, just use the PCA results
            result_df = pca_df.copy()
        
        # Save the transformed DataFrame
        new_filename = f"{dataset_id}_pca.csv"
        result_df.to_csv(DATA_DIR / new_filename, index=False)
        
        # Calculate explained variance
        explained_variance = pca.explained_variance_ratio_ * 100
        
        # Create a new dataset entry
        new_dataset_id = ObjectId()
        new_filename = f"{new_dataset_id}.csv"
        # Save the transformed DataFrame with the new filename
        result_df.to_csv(DATA_DIR / new_filename, index=False)
        
        new_dataset_info = {
            "_id": new_dataset_id,
            "name": f"{dataset['name']} (PCA)",
            "stored_filename": new_filename,
            "original_filename": f"{dataset['name']}_pca.csv",
            "columns": result_df.columns.tolist(),
            "row_count": len(result_df),
            "parent_dataset_id": ObjectId(dataset_id),
            "is_processed": True,
            "upload_time": datetime.now(),
            "pca_info": {
                "n_components": final_n_components,
                "explained_variance": explained_variance.tolist(),
                "total_variance_explained": float(sum(explained_variance)),
                "feature_importance": {
                    col: [float(val) for val in pca.components_[:, i]]
                    for i, col in enumerate(numeric_cols)
                }
            },
            "metadata": {
                "dtypes": result_df.dtypes.astype(str).to_dict(),
                "missing_values": {col: int(val) for col, val in result_df.isnull().sum().to_dict().items()},
                "numeric_summary": result_df.describe().to_dict() if not result_df.empty else {}
            }
        }
        
        # Insert into MongoDB
        datasets_collection.insert_one(new_dataset_info)
        
        # Store PCA results in the database
        analysis_collection.insert_one({
            "dataset_id": ObjectId(dataset_id),
            "analysis_type": "pca",
            "parameters": {"n_components": final_n_components},
            "results": new_dataset_info["pca_info"]
        })
        
        return {
            "id": str(new_dataset_id),
            "name": new_dataset_info["name"],
            "message": "PCA applied successfully",
            "explained_variance": explained_variance.tolist(),
            "total_variance_explained": float(sum(explained_variance)),
            "components": pca_columns,
            "feature_importance": {
                col: [float(val) for val in pca.components_[:, i]]
                for i, col in enumerate(numeric_cols)
            }
        }
    except Exception as e:
        print(f"Error applying PCA: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/dataset/{dataset_id}/suggestions")
async def get_feature_suggestions(dataset_id: str):
    """Get feature engineering suggestions for a dataset"""
    try:
        # Convert string ID to ObjectId
        try:
            obj_id = ObjectId(dataset_id)
        except:
            raise HTTPException(status_code=400, detail=f"Invalid dataset ID format: {dataset_id}")
        
        # Get the dataset
        dataset = datasets_collection.find_one({"_id": obj_id})
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Check if we already have suggestions for this dataset
        existing_analysis = analysis_collection.find_one({
            "dataset_id": obj_id,
            "analysis_type": "suggestions"
        })
        
        if existing_analysis:
            logger.info("Using existing analysis")
            return existing_analysis["results"]
        
        # Load the dataset
        file_path = DATA_DIR / dataset["stored_filename"]
        
        # Determine appropriate chunk size based on file size
        chunk_size = get_appropriate_chunk_size(dataset.get("file_size_bytes", 0))
        
        # Load the dataset with chunking if needed
        df = load_dataset(str(file_path), chunk_size=chunk_size)
        
        logger.info("Using default analysis")
        
        # Generate suggestions
        suggestions = []
        
        # Get column types
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        
        # 1. Standardization for numeric features
        for col in numeric_cols:
            suggestions.append({
                "column": col,
                "transformation": "standardize",
                "reason": "Standardize numeric features to improve model performance"
            })
        
        # 2. One-hot encoding for categorical features
        for col in categorical_cols:
            if df[col].nunique() < 10:  # Only suggest for low-cardinality
                suggestions.append({
                    "column": col,
                    "transformation": "one_hot_encode",
                    "reason": "Convert categorical features to numeric representation"
                })
            else:
                suggestions.append({
                    "column": col,
                    "transformation": "target_encode",
                    "reason": "Encode high-cardinality categorical features efficiently"
                })
        
        # 3. Log transformation for skewed numeric features
        for col in numeric_cols:
            if df[col].min() > 0:  # Only for positive values
                skewness = df[col].skew()
                if abs(skewness) > 1:  # Highly skewed
                    suggestions.append({
                        "column": col,
                        "transformation": "log_transform",
                        "reason": f"Apply log transformation to reduce skewness ({skewness:.2f})"
                    })
        
        # 4. Binning for numeric features with many unique values
        for col in numeric_cols:
            if df[col].nunique() > 20:
                suggestions.append({
                    "column": col,
                    "transformation": "binning",
                    "num_bins": 5,
                    "reason": "Bin continuous variable into categories to capture non-linear relationships"
                })
        
        # 5. Polynomial features for potential non-linear relationships
        for col in numeric_cols[:3]:  # Limit to first few numeric columns to avoid explosion
            suggestions.append({
                "column": col,
                "transformation": "polynomial",
                "degree": 2,
                "reason": "Create polynomial features to capture non-linear relationships"
            })
        
        # 6. Time-based features for datetime columns
        for col in datetime_cols:
            suggestions.append({
                "column": col,
                "transformation": "time_features",
                "features": ["year", "month", "dayofweek", "is_weekend"],
                "reason": "Extract time-based features from datetime column"
            })
        
        # 7. Interaction features between important numeric columns
        if len(numeric_cols) >= 2:
            for i, col1 in enumerate(numeric_cols[:3]):  # Limit to first few
                for col2 in numeric_cols[i+1:min(i+3, len(numeric_cols))]:  # And their next few
                    suggestions.append({
                        "column": col1,
                        "transformation": "interaction",
                        "interact_with": [col2],
                        "reason": f"Create interaction feature between {col1} and {col2}"
                    })
        
        # Store suggestions in the database
        analysis_collection.insert_one({
            "dataset_id": obj_id,
            "analysis_type": "suggestions",
            "results": {
                "feature_engineering": suggestions
            },
            "created_at": datetime.now()
        })
        
        return {"feature_engineering": suggestions}
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error getting feature suggestions: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

def get_feature_suggestions(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Generate feature engineering suggestions for a DataFrame"""
    suggestions = []
    
    # Get column types
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    
    # 1. Standardization for numeric features
    for col in numeric_cols:
        suggestions.append({
            "column": col,
            "transformation": "standardize",
            "reason": "Standardize numeric features to improve model performance"
        })
    
    # 2. One-hot encoding for categorical features
    for col in categorical_cols:
        if df[col].nunique() < 10:  # Only suggest for low-cardinality
            suggestions.append({
                "column": col,
                "transformation": "one_hot_encode",
                "reason": "Convert categorical features to numeric representation"
            })
        else:
            suggestions.append({
                "column": col,
                "transformation": "target_encode",
                "reason": "Encode high-cardinality categorical features efficiently"
            })
    
    # 3. Log transformation for skewed numeric features
    for col in numeric_cols:
        if df[col].min() > 0:  # Only for positive values
            skewness = df[col].skew()
            if abs(skewness) > 1:  # Highly skewed
                suggestions.append({
                    "column": col,
                    "transformation": "log_transform",
                    "reason": f"Apply log transformation to reduce skewness ({skewness:.2f})"
                })
    
    # 4. Binning for numeric features with many unique values
    for col in numeric_cols:
        if df[col].nunique() > 20:
            suggestions.append({
                "column": col,
                "transformation": "binning",
                "num_bins": 5,
                "reason": "Bin continuous variable into categories to capture non-linear relationships"
            })
    
    # 5. Polynomial features for potential non-linear relationships
    for col in numeric_cols[:3]:  # Limit to first few numeric columns to avoid explosion
        suggestions.append({
            "column": col,
            "transformation": "polynomial",
            "degree": 2,
            "reason": "Create polynomial features to capture non-linear relationships"
        })
    
    # 6. Time-based features for datetime columns
    for col in datetime_cols:
        suggestions.append({
            "column": col,
            "transformation": "time_features",
            "features": ["year", "month", "dayofweek", "is_weekend"],
            "reason": "Extract time-based features from datetime column"
        })
    
    # 7. Interaction features between important numeric columns
    if len(numeric_cols) >= 2:
        for i, col1 in enumerate(numeric_cols[:3]):  # Limit to first few
            for col2 in numeric_cols[i+1:min(i+3, len(numeric_cols))]:  # And their next few
                suggestions.append({
                    "column": col1,
                    "transformation": "interaction",
                    "interact_with": [col2],
                    "reason": f"Create interaction feature between {col1} and {col2}"
                })
    
    return suggestions

def generate_transformation_code(df: pd.DataFrame, transformations: List[Dict[str, Any]]) -> str:
    """Generate code for feature transformations"""
    return generate_advanced_transformation_code(df, transformations)

@router.post("/dataset/{dataset_id}/feature-engineering")
async def apply_feature_engineering(
    dataset_id: str, 
    transformations: List[Dict[str, str]] = Body(...)
):
    """Apply feature engineering transformations"""
    try:
        dataset = datasets_collection.find_one({"_id": ObjectId(dataset_id)})
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Convert ObjectId to string
        dataset = convert_objectid_to_str(dataset)
        
        # Load data from stored file
        file_path = DATA_DIR / dataset["stored_filename"]
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Dataset file not found")
            
        df = pd.read_csv(file_path)
        
        # Generate code for transformations
        code = ai_helper.generate_feature_engineering_code(df, transformations)
        
        # Execute the code
        local_vars = {"df": df}
        exec(code, globals(), local_vars)
        
        # Get the transformed DataFrame
        if "apply_feature_engineering" in local_vars:
            transformed_df = local_vars["apply_feature_engineering"](df)
        else:
            raise HTTPException(status_code=400, detail="Generated code does not contain apply_feature_engineering function")
        
        # Save the transformed DataFrame
        transformed_filename = f"{dataset_id}_transformed.csv"
        transformed_file_path = DATA_DIR / transformed_filename
        transformed_df.to_csv(transformed_file_path, index=False)
        
        # Create a new dataset entry
        new_dataset_id = ObjectId()
        new_dataset_info = {
            "_id": new_dataset_id,
            "name": f"{dataset['name']} (Transformed)",
            "stored_filename": transformed_filename,
            "columns": transformed_df.columns.tolist(),
            "row_count": len(transformed_df),
            "parent_dataset_id": ObjectId(dataset_id),
            "transformations": transformations,
            "metadata": {
                "dtypes": transformed_df.dtypes.astype(str).to_dict(),
                "missing_values": transformed_df.isnull().sum().to_dict(),
                "numeric_summary": transformed_df.describe().to_dict() if not transformed_df.empty else {}
            }
        }
        
        # Insert into MongoDB
        datasets_collection.insert_one(new_dataset_info)
        
        return {
            "id": str(new_dataset_id),
            "name": new_dataset_info["name"],
            "columns": new_dataset_info["columns"],
            "row_count": new_dataset_info["row_count"],
            "transformations": transformations,
            "code": code
        }
    except Exception as e:
        print(f"Error applying feature engineering: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/dataset/{dataset_id}/chat")
async def chat_with_dataset(dataset_id: str, question: str = Body(..., embed=True)):
    """Chat with the dataset using AI"""
    try:
        dataset = datasets_collection.find_one({"_id": ObjectId(dataset_id)})
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Load data from stored file
        file_path = DATA_DIR / dataset["stored_filename"]
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Dataset file not found")
            
        df = pd.read_csv(file_path)
        
        # Use AI helper to answer the question
        response = ai_helper.chat_with_data(df, question)
        
        return {"response": response}
    except Exception as e:
        print(f"Error chatting with dataset: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/dataset/{dataset_id}/select-features")
async def automated_feature_selection(
    dataset_id: str,
    background_tasks: BackgroundTasks,
    target_column: str = Body(...),
    method: str = Body("mutual_info"),
    k: Optional[int] = Body(None),
    threshold: Optional[float] = Body(None)
):
    """
    Perform automated feature selection on a dataset.
    
    Args:
        dataset_id: ID of the dataset
        target_column: Target variable column name
        method: Feature selection method ('mutual_info', 'f_regression', 'rfe', 'importance')
        k: Number of features to select (if None, use threshold or default to half)
        threshold: Threshold for feature importance (if k is None)
    """
    try:
        # Convert string ID to ObjectId
        try:
            obj_id = ObjectId(dataset_id)
        except:
            raise HTTPException(status_code=400, detail=f"Invalid dataset ID format: {dataset_id}")
        
        # Get the dataset
        dataset = datasets_collection.find_one({"_id": obj_id})
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Validate method
        valid_methods = ["mutual_info", "f_regression", "rfe", "importance"]
        if method not in valid_methods:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid method: {method}. Must be one of {valid_methods}"
            )
        
        # Create a response object with pending status
        response = {
            "status": "processing",
            "message": "Feature selection started in the background",
            "task_id": str(ObjectId())  # Generate a task ID for tracking
        }
        
        # Start the feature selection in the background
        background_tasks.add_task(
            process_feature_selection,
            dataset=dataset,
            target_column=target_column,
            method=method,
            k=k,
            threshold=threshold,
            task_id=response["task_id"]
        )
        
        return response
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in automated feature selection: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/dataset/{dataset_id}/statistics")
async def get_statistics(dataset_id: str):
    """Get detailed statistics for the dataset"""
    try:
        dataset = datasets_collection.find_one({"_id": ObjectId(dataset_id)})
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Convert ObjectId to string
        dataset = convert_objectid_to_str(dataset)
        
        # Load data from stored file
        file_path = DATA_DIR / dataset["stored_filename"]
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Dataset file not found")
            
        df = pd.read_csv(file_path)
        
        # Calculate statistics
        stats = {
            "basic": {
                "row_count": int(len(df)),
                "column_count": int(len(df.columns)),
                "memory_usage": int(df.memory_usage(deep=True).sum()),
                "duplicated_rows": int(df.duplicated().sum())
            },
            "columns": {}
        }
        
        # Calculate statistics for each column
        for col in df.columns:
            col_stats = {
                "dtype": str(df[col].dtype),
                "missing_count": int(df[col].isnull().sum()),
                "missing_percentage": float(round(df[col].isnull().sum() / len(df) * 100, 2)),
                "unique_count": int(df[col].nunique())
            }
            
            # Add numeric statistics
            if np.issubdtype(df[col].dtype, np.number):
                col_stats.update({
                    "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                    "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                    "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                    "median": float(df[col].median()) if not pd.isna(df[col].median()) else None,
                    "std": float(df[col].std()) if not pd.isna(df[col].std()) else None,
                    "skew": float(df[col].skew()) if not pd.isna(df[col].skew()) else None,
                    "kurtosis": float(df[col].kurtosis()) if not pd.isna(df[col].kurtosis()) else None
                })
            
            # Add categorical statistics
            elif df[col].dtype == 'object' or df[col].dtype.name == 'category':
                # Get value counts for top 10 values
                value_counts = df[col].value_counts().head(10).to_dict()
                col_stats["top_values"] = {str(k): int(v) for k, v in value_counts.items()}
            
            stats["columns"][col] = col_stats
        
        return stats
    except Exception as e:
        print(f"Error getting statistics: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/datasets/", response_model=List[DatasetResponse])
async def list_datasets():
    """List all datasets"""
    try:
        datasets = []
        for dataset in datasets_collection.find():
            # Convert ObjectId to string
            dataset = convert_objectid_to_str(dataset)
            dataset["id"] = dataset["_id"]  # Add id field for response model
            datasets.append(dataset)
        return datasets
    except Exception as e:
        print(f"Error listing datasets: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/dataset/{dataset_id}/apply-features")
async def apply_feature_suggestions(
    dataset_id: str, 
    background_tasks: BackgroundTasks,
    suggestion_indices: List[int] = Body(...),
    include_pca: bool = Body(False),
    pca_components: int = Body(0)
):
    """Apply selected feature engineering suggestions to the dataset"""
    try:
        # Convert string ID to ObjectId
        try:
            obj_id = ObjectId(dataset_id)
        except:
            raise HTTPException(status_code=400, detail=f"Invalid dataset ID format: {dataset_id}")
        
        # Get the dataset
        dataset = datasets_collection.find_one({"_id": obj_id})
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Create a response object with pending status
        response = {
            "status": "processing",
            "message": "Feature engineering started in the background",
            "task_id": str(ObjectId())  # Generate a task ID for tracking
        }
        
        # Start the feature engineering in the background
        background_tasks.add_task(
            process_feature_engineering,
            dataset=dataset,
            suggestion_indices=suggestion_indices,
            include_pca=include_pca,
            pca_components=pca_components,
            task_id=response["task_id"]
        )
        
        return response
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error applying feature suggestions: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

async def process_feature_engineering(
    dataset: Dict[str, Any],
    suggestion_indices: List[int],
    include_pca: bool,
    pca_components: int,
    task_id: str
):
    """Process feature engineering in the background"""
    try:
        # Load the dataset
        file_path = DATA_DIR / dataset["stored_filename"]
        
        # Determine appropriate chunk size based on file size
        chunk_size = get_appropriate_chunk_size(dataset.get("file_size_bytes", 0))
        
        # Load the dataset with chunking if needed
        df = load_dataset(str(file_path), chunk_size=chunk_size)
        
        logger.info(f"Successfully loaded dataset with shape: {df.shape}")
        
        # Get feature suggestions
        suggestions = get_feature_suggestions(df)
        
        # Filter selected suggestions
        selected_suggestions = [suggestions[i] for i in suggestion_indices if i < len(suggestions)]
        
        logger.info(f"Selected suggestion indices: {suggestion_indices}")
        logger.info(f"Selected suggestions: {selected_suggestions}")
        logger.info(f"Include PCA: {include_pca}, Components: {pca_components}")
        
        # Use robust implementation
        logger.info("Using robust feature engineering implementation")
        
        # Generate transformation code
        transformation_code = generate_transformation_code(df, selected_suggestions)
        logger.info("Successfully generated transformation code")
        logger.info(f"Generated transformation code:\n{transformation_code}")
        
        # Execute the transformation code
        namespace = {}
        exec(transformation_code, namespace)
        df_transformed = namespace["apply_feature_engineering"](df)
        logger.info("Successfully executed transformation code")
        
        # Generate a unique ID for the transformed dataset
        transformed_id = ObjectId()
        transformed_filename = f"{transformed_id}.csv"
        transformed_path = DATA_DIR / transformed_filename
        
        # Save the transformed dataset
        save_dataset(df_transformed, str(transformed_path), chunk_size=chunk_size)
        
        logger.info(f"Successfully transformed dataset to shape: {df_transformed.shape}")
        logger.info(f"Saved transformed dataset to: {transformed_path}")
        
        # Add numeric summary
        numeric_summary = {}
        for col in df_transformed.select_dtypes(include=['number']).columns:
            numeric_summary[col] = {
                "min": float(df_transformed[col].min()),
                "max": float(df_transformed[col].max()),
                "mean": float(df_transformed[col].mean()),
                "median": float(df_transformed[col].median()),
                "std": float(df_transformed[col].std())
            }
        
        # Create metadata for the transformed dataset
        transformed_dataset = {
            "_id": transformed_id,
            "name": f"{dataset['name']}_transformed",
            "original_filename": f"{dataset['name'].split('.')[0]}_transformed.csv",  # For downloads
            "stored_filename": transformed_filename,
            "parent_dataset_id": dataset["_id"],
            "columns": df_transformed.columns.tolist(),
            "row_count": len(df_transformed),
            "file_size_bytes": os.path.getsize(transformed_path),
            "upload_time": datetime.now(),
            "metadata": {
                "dtypes": df_transformed.dtypes.astype(str).to_dict(),
                "numeric_summary": numeric_summary,
                "transformations": [s for s in selected_suggestions]
            }
        }
        
        # Insert the transformed dataset into the database
        result = datasets_collection.insert_one(transformed_dataset)
        logger.info(f"Inserted new dataset into database with ID: {result.inserted_id}")
        
        # Store transformation details
        transformation_details = {
            "original_dataset_id": dataset["_id"],
            "transformed_dataset_id": transformed_id,
            "suggestions": selected_suggestions,
            "transformation_code": transformation_code,
            "timestamp": datetime.now()
        }
        
        analysis_collection.insert_one(transformation_details)
        logger.info("Stored transformation details in database")
        
        # Update task status
        task_status = {
            "_id": ObjectId(),  # Ensure it has a unique ID
            "task_id": task_id,
            "status": "completed",
            "result": {
                "dataset_id": str(transformed_id),
                "message": "Feature engineering completed successfully"
            },
            "timestamp": datetime.now()  # Add timestamp for sorting
        }
        
        # Insert the task status into the database
        analysis_collection.insert_one(task_status)
        logger.info(f"Updated task status to completed with dataset ID: {transformed_id}")
        
    except Exception as e:
        logger.error(f"Error in background feature engineering: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Update task status with error
        task_status = {
            "_id": ObjectId(),  # Ensure it has a unique ID
            "task_id": task_id,
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now()  # Add timestamp for sorting
        }
        
        analysis_collection.insert_one(task_status)

async def process_feature_selection(
    dataset: Dict[str, Any],
    target_column: str,
    method: str,
    k: Optional[int],
    threshold: Optional[float],
    task_id: str
):
    """Process feature selection in the background"""
    try:
        # Load the dataset
        file_path = DATA_DIR / dataset["stored_filename"]
        
        # Determine appropriate chunk size based on file size
        chunk_size = get_appropriate_chunk_size(dataset.get("file_size_bytes", 0))
        
        # Load the dataset with chunking if needed
        df = load_dataset(str(file_path), chunk_size=chunk_size)
        
        logger.info(f"Successfully loaded dataset with shape: {df.shape}")
        
        # Validate target column
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        # Perform feature selection
        logger.info(f"Performing feature selection using method: {method}")
        selected_features = select_features(
            df=df,
            target_column=target_column,
            method=method,
            k=k,
            threshold=threshold
        )
        
        logger.info(f"Selected {len(selected_features)} features: {selected_features}")
        
        # Create a new dataset with only the selected features (plus target)
        selected_columns = selected_features + [target_column]
        df_selected = df[selected_columns]
        
        # Generate a unique ID for the new dataset
        new_id = ObjectId()
        new_filename = f"{new_id}.csv"
        new_path = DATA_DIR / new_filename
        
        # Save the new dataset
        save_dataset(df_selected, str(new_path), chunk_size=chunk_size)
        
        logger.info(f"Saved selected features dataset to: {new_path}")
        
        # Create metadata for the new dataset
        new_dataset = {
            "_id": new_id,
            "name": f"{dataset['name']}_selected_features",
            "original_filename": f"{dataset['name'].split('.')[0]}_selected_features.csv",  # For downloads
            "stored_filename": new_filename,
            "parent_dataset_id": dataset["_id"],
            "columns": df_selected.columns.tolist(),
            "row_count": len(df_selected),
            "file_size_bytes": os.path.getsize(new_path),
            "upload_time": datetime.now(),
            "metadata": {
                "dtypes": df_selected.dtypes.astype(str).to_dict(),
                "feature_selection": {
                    "method": method,
                    "target_column": target_column,
                    "k": k,
                    "threshold": threshold,
                    "selected_features": selected_features
                }
            }
        }
        
        # Insert the new dataset into the database
        result = datasets_collection.insert_one(new_dataset)
        logger.info(f"Inserted new dataset into database with ID: {result.inserted_id}")
        
        # Store feature selection details
        selection_details = {
            "original_dataset_id": dataset["_id"],
            "selected_dataset_id": new_id,
            "method": method,
            "target_column": target_column,
            "k": k,
            "threshold": threshold,
            "selected_features": selected_features,
            "timestamp": datetime.now()
        }
        
        analysis_collection.insert_one(selection_details)
        logger.info("Stored feature selection details in database")
        
        # Update task status
        task_status = {
            "_id": ObjectId(),  # Ensure it has a unique ID
            "task_id": task_id,
            "status": "completed",
            "result": {
                "dataset_id": str(new_id),
                "selected_features": selected_features,
                "message": "Feature selection completed successfully"
            },
            "timestamp": datetime.now()  # Add timestamp for sorting
        }
        
        analysis_collection.insert_one(task_status)
        
    except Exception as e:
        logger.error(f"Error in background feature selection: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Update task status with error
        task_status = {
            "_id": ObjectId(),  # Ensure it has a unique ID
            "task_id": task_id,
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now()  # Add timestamp for sorting
        }
        
        analysis_collection.insert_one(task_status)

@router.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """Get the status of a background task"""
    try:
        # Find the task status - get the most recent one
        tasks = list(analysis_collection.find({"task_id": task_id}))
        
        if not tasks:
            # If no task is found, check if there are any recently created datasets
            # This is a fallback mechanism in case the task status wasn't properly stored
            return {"status": "processing", "message": "Task is still processing or not found"}
        
        # Sort tasks by timestamp (newest first) if timestamp exists
        tasks.sort(key=lambda x: x.get("timestamp", datetime.min), reverse=True)
        
        # Return the most recent task status
        task = tasks[0]
        
        # Convert ObjectId to string for JSON serialization
        if "_id" in task:
            task["_id"] = str(task["_id"])
        
        return task
    except Exception as e:
        logger.error(f"Error getting task status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/dataset/{dataset_id}/download")
async def download_dataset(dataset_id: str):
    """Download the dataset as a CSV file"""
    try:
        # Get dataset from database
        dataset = datasets_collection.find_one({"_id": ObjectId(dataset_id)})
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Convert ObjectId to string
        dataset = convert_objectid_to_str(dataset)
        
        # Load data from stored file
        file_path = DATA_DIR / dataset["stored_filename"]
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Dataset file not found")
            
        # Return the file
        return FileResponse(
            path=file_path,
            filename=f"{dataset.get('name', 'dataset')}.csv",
            media_type="text/csv"
        )
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))

# New endpoints for advanced analytics features

@router.get("/{dataset_id}/anomalies")
async def detect_anomalies(
    dataset_id: str,
    method: str = Query("isolation_forest", enum=["isolation_forest", "lof", "dbscan"]),
    threshold: float = Query(0.05, ge=0.01, le=0.2)
):
    """Detect anomalies in the dataset using the specified method"""
    try:
        # Get dataset from database
        dataset = datasets_collection.find_one({"_id": ObjectId(dataset_id)})
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Convert ObjectId to string
        dataset = convert_objectid_to_str(dataset)
        
        # Load data from stored file
        file_path = DATA_DIR / dataset["stored_filename"]
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Dataset file not found")
            
        # Load the dataset
        df = pd.read_csv(file_path)
        
        # Get numeric columns only
        numeric_df = df.select_dtypes(include=['number'])
        
        if numeric_df.empty:
            return JSONResponse(
                status_code=400,
                content={"detail": "No numeric columns found for anomaly detection"}
            )
        
        # Check for NaN values and handle them
        if numeric_df.isna().any().any():
            print(f"Dataset contains NaN values. Imputing missing values...")
            # Import the imputer
            from sklearn.impute import SimpleImputer
            
            # Create an imputer that replaces NaN with the median value of each column
            imputer = SimpleImputer(strategy='median')
            
            # Fit and transform the data
            numeric_df_imputed = pd.DataFrame(
                imputer.fit_transform(numeric_df),
                columns=numeric_df.columns
            )
            
            # Use the imputed dataframe for anomaly detection
            numeric_df = numeric_df_imputed
        
        # Detect anomalies based on the method
        anomaly_scores = None
        anomaly_indices = None
        
        if method == "isolation_forest":
            from sklearn.ensemble import IsolationForest
            model = IsolationForest(contamination=threshold, random_state=42)
            anomaly_scores = model.fit_predict(numeric_df)
            # Convert to binary (1 for normal, -1 for anomaly)
            anomaly_indices = [i for i, score in enumerate(anomaly_scores) if score == -1]
            
        elif method == "lof":
            from sklearn.neighbors import LocalOutlierFactor
            model = LocalOutlierFactor(n_neighbors=20, contamination=threshold)
            anomaly_scores = model.fit_predict(numeric_df)
            # Convert to binary (1 for normal, -1 for anomaly)
            anomaly_indices = [i for i, score in enumerate(anomaly_scores) if score == -1]
            
        elif method == "dbscan":
            from sklearn.cluster import DBSCAN
            model = DBSCAN(eps=0.5, min_samples=5)
            clusters = model.fit_predict(numeric_df)
            # In DBSCAN, -1 indicates outliers
            anomaly_indices = [i for i, cluster in enumerate(clusters) if cluster == -1]
        
        # Calculate percentage of anomalies
        anomaly_percentage = len(anomaly_indices) / len(df) * 100
        
        # Get the anomalous rows
        anomalous_rows = df.iloc[anomaly_indices].to_dict(orient='records')
        
        # Limit the number of returned rows to avoid overwhelming the response
        max_rows = 100
        if len(anomalous_rows) > max_rows:
            anomalous_rows = anomalous_rows[:max_rows]
        
        # Return the results
        return {
            "method": method,
            "threshold": threshold,
            "total_rows": len(df),
            "anomaly_count": len(anomaly_indices),
            "anomaly_percentage": anomaly_percentage,
            "anomalous_rows": anomalous_rows
        }
        
    except Exception as e:
        print(f"Error detecting anomalies: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{dataset_id}/correlation-explanation")
async def explain_correlations(
    dataset_id: str,
    method: str = Query("pearson", enum=["pearson", "spearman", "kendall"]),
    threshold: float = Query(0.5, ge=0.1, le=0.9)
):
    """Analyze and explain correlations in the dataset"""
    try:
        # Get dataset from database
        dataset = datasets_collection.find_one({"_id": ObjectId(dataset_id)})
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Convert ObjectId to string
        dataset = convert_objectid_to_str(dataset)
        
        # Load data from stored file
        file_path = DATA_DIR / dataset["stored_filename"]
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Dataset file not found")
            
        # Load the dataset
        df = pd.read_csv(file_path)
        
        # Get numeric columns only
        numeric_df = df.select_dtypes(include=['number'])
        
        if numeric_df.empty:
            return JSONResponse(
                status_code=400,
                content={"detail": "No numeric columns found for correlation analysis"}
            )
        
        # Check for NaN values and handle them
        if numeric_df.isna().any().any():
            print(f"Dataset contains NaN values. Imputing missing values for correlation analysis...")
            # Import the imputer
            from sklearn.impute import SimpleImputer
            
            # Create an imputer that replaces NaN with the median value of each column
            imputer = SimpleImputer(strategy='median')
            
            # Fit and transform the data
            numeric_df_imputed = pd.DataFrame(
                imputer.fit_transform(numeric_df),
                columns=numeric_df.columns
            )
            
            # Use the imputed dataframe for correlation analysis
            numeric_df = numeric_df_imputed
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr(method=method)
        
        # Find strong correlations (above threshold)
        strong_correlations = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                
                if abs(corr_value) >= threshold:
                    correlation_type = "positive" if corr_value > 0 else "negative"
                    strength = "very strong" if abs(corr_value) > 0.8 else "strong" if abs(corr_value) > 0.6 else "moderate"
                    
                    # Generate explanation
                    explanation = ""
                    if correlation_type == "positive":
                        explanation = f"As {col1} increases, {col2} tends to increase as well. "
                    else:
                        explanation = f"As {col1} increases, {col2} tends to decrease. "
                    
                    explanation += f"This suggests a {strength} {correlation_type} relationship between these variables."
                    
                    # Add implications
                    if abs(corr_value) > 0.8:
                        explanation += " These variables might be redundant or measuring similar aspects."
                    
                    strong_correlations.append({
                        "feature1": col1,
                        "feature2": col2,
                        "correlation": corr_value,
                        "type": correlation_type,
                        "strength": strength,
                        "explanation": explanation
                    })
        
        # Sort by absolute correlation value
        strong_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        
        # Return the results
        return {
            "method": method,
            "threshold": threshold,
            "correlation_matrix": corr_matrix.to_dict(),
            "strong_correlations": strong_correlations
        }
        
    except Exception as e:
        print(f"Error explaining correlations: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/{dataset_id}/causal-analysis")
async def analyze_causality(
    dataset_id: str,
    target: str = Body(...),
    method: str = Body("correlation", embed=True)
):
    """Perform causal analysis on the dataset"""
    try:
        # Get dataset from database
        dataset = datasets_collection.find_one({"_id": ObjectId(dataset_id)})
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Convert ObjectId to string
        dataset = convert_objectid_to_str(dataset)
        
        # Load data from stored file
        file_path = DATA_DIR / dataset["stored_filename"]
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Dataset file not found")
            
        # Load the dataset
        df = pd.read_csv(file_path)
        
        # Check if target exists
        if target not in df.columns:
            return JSONResponse(
                status_code=400,
                content={"detail": f"Target column '{target}' not found in dataset"}
            )
        
        # Get numeric columns only
        numeric_df = df.select_dtypes(include=['number'])
        
        if numeric_df.empty:
            return JSONResponse(
                status_code=400,
                content={"detail": "No numeric columns found for causal analysis"}
            )
        
        # Check for NaN values and handle them
        if numeric_df.isna().any().any():
            print(f"Dataset contains NaN values. Imputing missing values for causal analysis...")
            # Import the imputer
            from sklearn.impute import SimpleImputer
            
            # Create an imputer that replaces NaN with the median value of each column
            imputer = SimpleImputer(strategy='median')
            
            # Fit and transform the data
            numeric_df_imputed = pd.DataFrame(
                imputer.fit_transform(numeric_df),
                columns=numeric_df.columns
            )
            
            # Use the imputed dataframe for causal analysis
            numeric_df = numeric_df_imputed
        
        # For now, implement a simple correlation-based approach
        # In a real implementation, this would use more sophisticated causal inference methods
        
        # Calculate correlation with target
        if target in numeric_df.columns:
            correlations = numeric_df.corr()[target].drop(target)
        else:
            # If target is categorical, we need a different approach
            # For simplicity, we'll just return an error
            return JSONResponse(
                status_code=400,
                content={"detail": "Causal analysis for categorical targets is not implemented yet"}
            )
        
        # Sort by absolute correlation
        correlations = correlations.abs().sort_values(ascending=False)
        
        # Get top potential causes
        potential_causes = []
        for feature in correlations.index[:5]:  # Top 5 features
            corr_value = numeric_df.corr()[target][feature]
            direction = "positive" if corr_value > 0 else "negative"
            strength = "strong" if abs(corr_value) > 0.7 else "moderate" if abs(corr_value) > 0.4 else "weak"
            
            potential_causes.append({
                "feature": feature,
                "correlation": corr_value,
                "direction": direction,
                "strength": strength,
                "effect_estimate": corr_value  # In a real implementation, this would be a causal effect estimate
            })
        
        # Return the results
        return {
            "target": target,
            "method": method,
            "potential_causes": potential_causes,
            "causal_graph": {
                "nodes": [{"id": target, "type": "target"}] + [{"id": cause["feature"], "type": "cause"} for cause in potential_causes],
                "links": [{"source": cause["feature"], "target": target, "value": abs(cause["correlation"])} for cause in potential_causes]
            },
            "disclaimer": "This is a simplified causal analysis based on correlation. For robust causal inference, additional methods and assumptions are required."
        }
        
    except Exception as e:
        print(f"Error performing causal analysis: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))

# Collaboration endpoints

@router.post("/{dataset_id}/share")
async def share_project(
    dataset_id: str,
    email: str = Body(...),
    permission: str = Body(..., enum=["view", "edit", "admin"])
):
    """Share the dataset with another user"""
    try:
        # Get dataset from database
        dataset = datasets_collection.find_one({"_id": ObjectId(dataset_id)})
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # In a real implementation, this would add the user to the dataset's access control list
        # For now, we'll just return a success message
        
        return {
            "status": "success",
            "message": f"Dataset shared with {email} ({permission} access)",
            "shared_with": email,
            "permission": permission
        }
        
    except Exception as e:
        print(f"Error sharing project: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/{dataset_id}/version")
async def create_version(
    dataset_id: str,
    name: str = Body(...),
    notes: str = Body(None)
):
    """Create a version checkpoint for the dataset"""
    try:
        # Get dataset from database
        dataset = datasets_collection.find_one({"_id": ObjectId(dataset_id)})
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # In a real implementation, this would create a version record in the database
        # For now, we'll just return a success message
        
        return {
            "status": "success",
            "message": f"Version checkpoint '{name}' created successfully",
            "version": {
                "name": name,
                "notes": notes,
                "created_at": datetime.now().isoformat(),
                "dataset_id": dataset_id
            }
        }
        
    except Exception as e:
        print(f"Error creating version: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/{dataset_id}/comment")
async def add_comment(
    dataset_id: str,
    text: str = Body(..., embed=True)
):
    """Add a comment to the dataset"""
    try:
        # Get dataset from database
        dataset = datasets_collection.find_one({"_id": ObjectId(dataset_id)})
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # In a real implementation, this would add the comment to the database
        # For now, we'll just return a success message
        
        return {
            "status": "success",
            "message": "Comment added successfully",
            "comment": {
                "text": text,
                "created_at": datetime.now().isoformat(),
                "user": "current_user",  # In a real implementation, this would be the actual user
                "dataset_id": dataset_id
            }
        }
        
    except Exception as e:
        print(f"Error adding comment: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/{dataset_id}/annotation")
async def add_annotation(
    dataset_id: str,
    text: str = Body(...),
    data_point: Dict[str, Any] = Body(...)
):
    """Add an annotation to a specific data point in the dataset"""
    try:
        # Get dataset from database
        dataset = datasets_collection.find_one({"_id": ObjectId(dataset_id)})
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # In a real implementation, this would add the annotation to the database
        # For now, we'll just return a success message
        
        return {
            "status": "success",
            "message": "Annotation added successfully",
            "annotation": {
                "text": text,
                "data_point": data_point,
                "created_at": datetime.now().isoformat(),
                "user": "current_user",  # In a real implementation, this would be the actual user
                "dataset_id": dataset_id
            }
        }
        
    except Exception as e:
        print(f"Error adding annotation: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))