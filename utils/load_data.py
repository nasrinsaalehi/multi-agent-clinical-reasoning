"""
Data Loading Module - Part of Perception Agent
Extracts and loads patient data from CSV files.
"""

import pandas as pd
import os
from typing import Optional, Dict, Any


def load_csv_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a CSV dataset file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame containing the loaded data
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise ValueError(f"Error loading CSV file: {str(e)}")


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get summary statistics for a dataset.
    
    Args:
        df: DataFrame to summarize
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'row_count': len(df),
        'column_count': len(df.columns),
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_summary': None,
        'sample_rows': None,
    }
    
    # Get numeric column summary
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        summary['numeric_summary'] = df[numeric_cols].describe().to_dict()
    
    # Get sample rows (first 5)
    summary['sample_rows'] = df.head(5).to_dict('records')
    
    return summary


def create_synthetic_dataset(n_samples: int = 100) -> pd.DataFrame:
    """
    Create a synthetic patient dataset for demo purposes.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        DataFrame with synthetic patient data
    """
    import numpy as np
    
    np.random.seed(42)
    
    data = {
        'gender': np.random.choice(['M', 'F'], n_samples),
        'anchor_age': np.random.choice(['NEWBORN', 'YOUNG_ADULT', 'MIDDLE_ADULT', 'SENIOR'], n_samples),
        'admission_type': np.random.choice(['EMERGENCY', 'ELECTIVE', 'URGENT'], n_samples),
        'insurance': np.random.choice(['Medicare', 'Medicaid', 'Other'], n_samples),
        'infectious': np.random.randint(0, 2, n_samples),
        'neoplasms': np.random.randint(0, 2, n_samples),
        'endocrine': np.random.randint(0, 2, n_samples),
        'blood': np.random.randint(0, 2, n_samples),
        'mental': np.random.randint(0, 2, n_samples),
        'nervous': np.random.randint(0, 2, n_samples),
        'circulatory': np.random.randint(0, 2, n_samples),
        'respiratory': np.random.randint(0, 2, n_samples),
        'digestive': np.random.randint(0, 2, n_samples),
        'genitourinary': np.random.randint(0, 2, n_samples),
        'skin': np.random.randint(0, 2, n_samples),
        'muscular': np.random.randint(0, 2, n_samples),
        'congenital': np.random.randint(0, 2, n_samples),
        'prenatal': np.random.randint(0, 2, n_samples),
        'injury': np.random.randint(0, 2, n_samples),
        'misc': np.random.randint(0, 2, n_samples),
        'Neuro Intermediate': np.random.randint(0, 2, n_samples),
        'Neuro Surgical Intensive Care Unit (Neuro SICU)': np.random.randint(0, 2, n_samples),
        'Other-ICU': np.random.randint(0, 2, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Generate synthetic LOS (Length of Stay) based on features
    base_los = 3.0
    los = base_los + np.random.normal(0, 2, n_samples)
    los += df['circulatory'] * 2.5
    los += df['respiratory'] * 1.8
    los += df['infectious'] * 1.5
    los = np.maximum(los, 0.5)  # Minimum LOS of 0.5 days
    df['los'] = los
    
    return df

