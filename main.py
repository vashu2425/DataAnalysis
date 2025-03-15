import streamlit as st
import pandas as pd
import json
import numpy as np
from sklearn.decomposition import PCA
import requests
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get backend API URL
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000/api/v1")

# Project utilities
from utils.data_processor import DataProcessor
from utils.feature_engineer import FeatureEngineer
from utils.visualizer import Visualizer
from utils.llm_helper import LLMHelper

# Set Streamlit page config
st.set_page_config(
    page_title="Data Analysis & Feature Engineering Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()
if 'feature_engineer' not in st.session_state:
    st.session_state.feature_engineer = FeatureEngineer()
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = Visualizer()
if 'current_df' not in st.session_state:
    st.session_state.current_df = None
if 'dataset_id' not in st.session_state:
    st.session_state.dataset_id = None
if 'suggestions' not in st.session_state:
    st.session_state.suggestions = None
if 'llm_helper' not in st.session_state:
    try:
        st.session_state.llm_helper = LLMHelper(model_name="tiiuae/falcon-7b-instruct")
    except Exception as e:
        st.error(f"Failed to initialize LLM: {e}")
        st.stop()
if 'analysis_mode' not in st.session_state:
    st.session_state.analysis_mode = 'analysis'

def upload_dataset_to_backend(file):
    """Upload dataset to backend and return dataset ID"""
    try:
        files = {'file': file}
        response = requests.post(f"{BACKEND_API_URL}/dataset/", files=files)
        response.raise_for_status()
        return response.json()['id']
    except requests.exceptions.RequestException as e:
        st.error(f"Error uploading dataset to backend: {str(e)}")
        return None

def get_dataset_from_backend(dataset_id):
    """Get dataset details from backend"""
    try:
        # Check if dataset_id is valid
        if not dataset_id or dataset_id == "0":
            st.error("Invalid dataset ID")
            return None
            
        response = requests.get(f"{BACKEND_API_URL}/dataset/{dataset_id}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching dataset from backend: {str(e)}")
        return None

def create_analysis_in_backend(dataset_id, analysis_type):
    """Create analysis in backend"""
    try:
        response = requests.post(
            f"{BACKEND_API_URL}/dataset/{dataset_id}/analysis",
            params={"analysis_type": analysis_type}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error creating analysis in backend: {str(e)}")
        return None

def update_correlation_matrix():
    """
    Show data preview & correlation matrix side by side.
    """
    if st.session_state.current_df is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Data Preview")
            st.dataframe(st.session_state.current_df.head(), use_container_width=True)
        with col2:
            st.subheader("Correlation Matrix")
            fig = st.session_state.visualizer.create_correlation_matrix(st.session_state.current_df)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No DataFrame loaded to show correlation.")

# Apply PCA
def apply_pca(df: pd.DataFrame, n_components: int) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        st.warning("No numeric columns for PCA.")
        return df

    df_pca = df.dropna(subset=numeric_cols).copy()
    if df_pca.empty:
        st.warning("All numeric columns contain NaNs. PCA skipped.")
        return df

    pca = PCA(n_components=n_components)
    pca_transformed = pca.fit_transform(df_pca[numeric_cols])

    pca_cols = [f"PCA_{i+1}" for i in range(n_components)]
    df_pca_pca = pd.DataFrame(pca_transformed, columns=pca_cols, index=df_pca.index)

    df_pca.drop(columns=numeric_cols, inplace=True)
    df_pca = pd.concat([df_pca, df_pca_pca], axis=1)

    df_copy = df.copy()
    df_copy.loc[df_pca.index, df_pca.columns] = df_pca
    return df_copy

# Main Title
st.title("Data Analysis & Feature Engineering Platform")
st.markdown("Upload your dataset and get AI-powered feature engineering suggestions using open-source models")

# File Upload
uploaded_file = st.file_uploader("**1. Upload CSV file**", type=['csv'])

if uploaded_file is not None:
    try:
        with st.spinner('Loading and analyzing data...'):
            # Load data locally
            df = st.session_state.data_processor.load_data(uploaded_file)
            st.session_state.current_df = df
            st.session_state.data_processor.df = df

            # Upload to backend
            dataset_id = upload_dataset_to_backend(uploaded_file)
            if dataset_id:
                st.session_state.dataset_id = dataset_id
                st.success("Dataset uploaded successfully!")
            
            basic_stats = st.session_state.data_processor.get_basic_stats()
            column_info = st.session_state.data_processor.get_column_info()

        st.subheader("Choose Your Analysis Mode")
        colA, colB = st.columns(2)
        with colA:
            if st.button("Data Analysis & Feature Engineering"):
                st.session_state.analysis_mode = 'analysis'
        with colB:
            if st.button("Chat with Your Dataset"):
                st.session_state.analysis_mode = 'chat'

        if st.session_state.analysis_mode == 'analysis':
            st.subheader("Dataset Overview")
            st.write(json.dumps(basic_stats, indent=2))  # Fix: Convert to string before displaying
            update_correlation_matrix()

            st.markdown("---")
            st.subheader("Feature Engineering")

            if st.button("Get AI Suggestions"):
                with st.spinner('Analyzing dataset...'):
                    try:
                        st.session_state.suggestions = st.session_state.feature_engineer.analyze_dataset(
                            st.session_state.current_df
                        )
                    except Exception as e:
                        st.error(f"Error generating suggestions: {str(e)}")

            if st.session_state.suggestions is not None:
                if st.button("Apply All Feature Engineering", type="primary"):
                    with st.spinner('Applying all transformations...'):
                        st.session_state.current_df = st.session_state.feature_engineer.apply_transformations(
                            st.session_state.current_df,
                            st.session_state.suggestions['suggestions']
                        )
                        st.session_state.data_processor.df = st.session_state.current_df
                    st.success("Applied all suggested transformations!")
                    st.session_state.suggestions = None
                    update_correlation_matrix()

            st.markdown("---")
            st.subheader("Dimensionality Reduction (PCA)")
            pca_components = st.slider("Number of PCA Components", min_value=2, max_value=20, value=5)
            if st.button("Apply PCA"):
                with st.spinner("Applying PCA..."):
                    st.session_state.current_df = apply_pca(st.session_state.current_df, pca_components)
                    st.session_state.data_processor.df = st.session_state.current_df
                st.success("PCA applied!")
                update_correlation_matrix()

        else:
            st.subheader("Chat with Your Dataset")

            if st.session_state.current_df is None:
                st.warning("Upload a dataset first to chat with it.")
            else:
                # Get more detailed dataset information
                numeric_cols = st.session_state.current_df.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = st.session_state.current_df.select_dtypes(include=['object', 'category']).columns.tolist()
                
                dataset_info = {
                    "shape": st.session_state.current_df.shape,
                    "numeric_columns": numeric_cols,
                    "categorical_columns": categorical_cols,
                    "missing_values": st.session_state.current_df.isnull().sum().to_dict(),
                    "basic_stats": basic_stats
                }

                prebuilt_questions = [
                    "What are the key insights from this dataset?",
                    "Which features are most important for analysis?",
                    "Are there any data quality issues I should address?",
                    "What kind of visualizations would be helpful for this dataset?",
                    "Suggest some feature engineering steps for this dataset"
                ]
                
                st.write("**Dataset Overview:**")
                st.write(f"- Shape: {dataset_info['shape']}")
                st.write(f"- Numeric columns: {len(numeric_cols)}")
                st.write(f"- Categorical columns: {len(categorical_cols)}")
                
                chosen_question = st.selectbox("Pick a prebuilt question:", prebuilt_questions)
                custom_question = st.text_input("Or ask your own question:", "")

                if st.button("Ask Question"):
                    question = custom_question if custom_question else chosen_question
                    
                    prompt = f"""
                    Dataset Information:
                    - Shape: {dataset_info['shape']}
                    - Numeric columns: {numeric_cols}
                    - Categorical columns: {categorical_cols}
                    - Missing values: {dataset_info['missing_values']}
                    - Basic statistics: {json.dumps(dataset_info['basic_stats'], indent=2)}
                    
                    Question: {question}
                    
                    Please provide a clear and concise answer focusing on the specific question asked.
                    """
                    
                    with st.spinner("Analyzing your dataset..."):
                        try:
                            response = st.session_state.llm_helper.generate_text(prompt, max_new_tokens=300)
                            st.write("**AI Analysis:**")
                            st.write(response)
                        except Exception as e:
                            st.error(f"Error generating response: {str(e)}")

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a CSV file to begin analysis.")
