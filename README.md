# Data Analysis Platform

A comprehensive platform for data analysis, feature engineering, and exploratory data analysis.

## Features

- Dataset upload and management
- Automated exploratory data analysis (EDA)
- AI-powered feature engineering suggestions
- Dimensionality reduction with PCA
- Interactive visualizations
- Dataset transformation tracking

## Setup Instructions

### Option 1: Using Conda Environment (Recommended)

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd your_project
   ```

2. **Create and activate the conda environment**:
   ```bash
   conda env create -f environment.yml
   conda activate data_analysis_platform
   ```

3. **Start the server**:
   ```bash
   ./start_server.sh
   ```
   Or manually:
   ```bash
   python -m uvicorn backend.app.main:app --reload --port 8000
   ```

4. **Access the application**:
   Open your browser and navigate to `http://localhost:8000`

### Option 2: Using pip

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd your_project
   ```

2. **Create and activate a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the server**:
   ```bash
   ./start_server.sh
   ```
   Or manually:
   ```bash
   python -m uvicorn backend.app.main:app --reload --port 8000
   ```

5. **Access the application**:
   Open your browser and navigate to `http://localhost:8000`

## Configuration

### Environment Variables

Create a `.env` file in the project root with the following variables:

```
# MongoDB Configuration (optional)
MONGODB_URI=mongodb://localhost:27017
USE_IN_MEMORY=true  # Set to false to use MongoDB

# Application Settings
DEBUG=true
```

## Usage

1. **Upload a dataset**: From the home page, upload a CSV file.
2. **Explore the dataset**: Navigate through the tabs to view dataset statistics, visualizations, and more.
3. **Apply feature engineering**: Use the AI-powered suggestions to transform your data.
4. **Reduce dimensionality**: Apply PCA to your dataset for visualization or further analysis.
5. **Download transformed datasets**: Download any transformed version of your dataset.

## Project Structure

- `backend/`: Backend server code
  - `app/`: Main application code
    - `routes/`: API endpoints
    - `utils/`: Utility functions
    - `templates/`: HTML templates
    - `static/`: Static assets
- `data/`: Data storage directory
- `requirements.txt`: Python dependencies
- `environment.yml`: Conda environment specification
- `start_server.sh`: Server startup script 