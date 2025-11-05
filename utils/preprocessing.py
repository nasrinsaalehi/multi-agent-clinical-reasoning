"""
Data Preprocessing Module - Part of Perception Agent
Normalizes and preprocesses patient data for model training.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any

def preprocess_data(df: pd.DataFrame, target_col: str = 'los') -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    df = df.copy()

    # If target col missing, compute LOS from timestamps if possible
    if target_col not in df.columns:
        if target_col == 'los':
            if 'admittime' in df.columns and 'dischtime' in df.columns:
                try:
                    df['admittime'] = pd.to_datetime(df['admittime'])
                    df['dischtime'] = pd.to_datetime(df['dischtime'])
                    df['los'] = (df['dischtime'] - df['admittime']).dt.total_seconds() / 86400.0
                    df = df[df['los'] > 0].copy()
                except Exception as e:
                    raise ValueError(f"Could not compute '{target_col}': {e}")
            else:
                raise ValueError(f"Target column '{target_col}' not found and admittime/dischtime missing")
        else:
            raise ValueError(f"Target column '{target_col}' not found")

    # Extract target
    target = df[target_col].copy()
    features_df = df.drop(columns=[target_col])

    # Drop identifiers/timestamps
    cols_to_drop = ['subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime',
                    'died_at_the_hospital', 'dod', 'Unnamed: 0']
    for c in cols_to_drop:
        if c in features_df.columns:
            features_df = features_df.drop(columns=[c])

    # Encode gender
    if 'gender' in features_df.columns:
        features_df['gender'] = features_df['gender'].replace({'M': 0, 'F': 1})

    # Categorical -> one-hot with notebook-style prefixes
    categorical_cols = ['admission_type', 'insurance', 'anchor_age']
    prefix_mapping = {'admission_type': 'ADM', 'insurance': 'INS', 'anchor_age': 'AGE'}

    # Ensure categorical cols exist and fill NAs
    for col in categorical_cols:
        if col in features_df.columns:
            if features_df[col].isnull().any():
                # fill with mode if available, else 'Unknown'
                try:
                    mode = features_df[col].mode()
                    fill_val = mode.iloc[0] if len(mode) > 0 else 'Unknown'
                except Exception:
                    fill_val = 'Unknown'
                features_df[col] = features_df[col].fillna(fill_val)
        else:
            # add as Unknown column (so get_dummies creates AGE_Unknown etc.)
            features_df[col] = 'Unknown'

    # One-hot encode with explicit prefixes to avoid numeric-only labels
    try:
        features_df = pd.get_dummies(features_df, columns=categorical_cols, prefix=prefix_mapping, dummy_na=False)
    except Exception:
        # fallback per-column
        for col in categorical_cols:
            if col in features_df.columns:
                pref = prefix_mapping.get(col, col.upper())
                dummies = pd.get_dummies(features_df[col], prefix=pref)
                features_df = features_df.drop(columns=[col])
                features_df = pd.concat([features_df, dummies], axis=1)

    # Remove any columns that have purely-numeric string names (artifact columns like '56')
    numeric_name_cols = [c for c in features_df.columns if isinstance(c, str) and c.isdigit()]
    if numeric_name_cols:
        features_df = features_df.drop(columns=numeric_name_cols)

    # Drop remaining object columns (free text) - e.g., notes_text
    obj_cols = features_df.select_dtypes(include=['object']).columns.tolist()
    if obj_cols:
        features_df = features_df.drop(columns=obj_cols)

    # Fill remaining NaNs and convert bools
    features_df = features_df.fillna(0)
    bool_cols = features_df.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        features_df[col] = features_df[col].astype(int)

    preprocessing_info = {
        'feature_columns': list(features_df.columns),
        'categorical_columns': categorical_cols,
        'dummy_columns': [c for c in features_df.columns if any(c.startswith(p + '_') for p in prefix_mapping.values())],
        'n_samples': len(features_df),
        'n_features': len(features_df.columns),
    }

    return features_df, target, preprocessing_info


def prepare_input_for_inference(input_dict: dict, feature_columns: list) -> pd.DataFrame:
    """
    Prepare a single patient input for model inference.

    Args:
        input_dict: Dictionary with patient features
        feature_columns: List of expected feature column names

    Returns:
        DataFrame with single row ready for inference
    """
    # Create a base DataFrame with all feature columns set to 0
    data = {col: 0 for col in feature_columns}

    # Map input values
    if 'gender' in feature_columns:
        data['gender'] = 0 if input_dict.get('gender') == 'M' else 1

    age_cat = input_dict.get('anchor_age', 'MIDDLE_ADULT')
    age_col = f'AGE_{age_cat}'
    if age_col in feature_columns:
        data[age_col] = 1

    adm_type = input_dict.get('admission_type', 'EMERGENCY')
    adm_col = f'ADM_{adm_type}'
    if adm_col in feature_columns:
        data[adm_col] = 1

    insurance = input_dict.get('insurance', 'Other')
    ins_col = f'INS_{insurance}'
    if ins_col in feature_columns:
        data[ins_col] = 1

    diag_categories = ['infectious', 'neoplasms', 'endocrine', 'blood', 'mental', 'nervous',
                       'circulatory', 'respiratory', 'digestive', 'genitourinary', 'skin',
                       'muscular', 'congenital', 'prenatal', 'injury', 'misc']
    for cat in diag_categories:
        if cat in feature_columns:
            data[cat] = 1 if input_dict.get(cat, False) else 0

    if 'Neuro Intermediate' in feature_columns:
        data['Neuro Intermediate'] = 1 if input_dict.get('neuro_intermediate', False) else 0
    if 'Neuro Surgical Intensive Care Unit (Neuro SICU)' in feature_columns:
        data['Neuro Surgical Intensive Care Unit (Neuro SICU)'] = 1 if input_dict.get('neuro_sicu', False) else 0
    if 'Other-ICU' in feature_columns:
        data['Other-ICU'] = 1 if input_dict.get('other_icu', False) else 0

    df = pd.DataFrame([data])

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]

    return df
