"""
Text feature extraction utilities.
This deployment intentionally omits generating textual features from clinical notes
and demo JSON files. The functions below keep the same API but return empty
DataFrames / None vectorizers so downstream code can safely concat without
reading or producing any text-derived features.
"""

from typing import Optional, Tuple
import pandas as pd


def vectorize_text(series: pd.Series, max_features: int = 200):
    """Return an empty DataFrame and None vectorizer.

    We intentionally do NOT perform any TF-IDF or embedding extraction here to
    avoid consuming unstructured/demo text data. Callers can still accept the
    returned empty DataFrame and a `None` vectorizer and continue processing.
    """
    # Ensure we return a DataFrame aligned with the input index to allow safe
    # concatenation with feature matrices. No columns are created.
    empty_df = pd.DataFrame(index=series.index)
    return empty_df, None


def transform_text(series: pd.Series, vectorizer) -> pd.DataFrame:
    """Transform text using a fitted vectorizer — here returns empty DataFrame.

    If a vectorizer is provided but text transformation is intentionally
    disabled, return an empty DataFrame aligned with the input index. This
    keeps downstream code stable while ensuring no text features are produced.
    """
    empty_df = pd.DataFrame(index=series.index)
    return empty_df
