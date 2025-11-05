"""
Extracts textual descriptions from MIMIC demo CSVs by joining diagnosis codes
with their long titles and optional patient metadata to form a simple note per admission.

NOTE: Text extraction from the MIMIC demo files is intentionally disabled in this
deployment. The project policy is to not consume the demo JSON/text files for
feature extraction. This module therefore provides a no-op shim that returns an
empty notes DataFrame. If you later want to re-enable text extraction, restore
an implementation that reads the appropriate CSVs and returns `hadm_id, notes_text`.
"""

import pandas as pd


def build_admission_texts(mimic_path: str) -> pd.DataFrame:
    """Return an empty notes DataFrame (text extraction disabled).

    The function keeps the same signature so callers won't break, but it will not
    read or produce any textual features. It returns an empty DataFrame with the
    expected columns so merges are safe but no text features will be produced.

    Args:
        mimic_path: Path to the MIMIC demo folder (ignored)

    Returns:
        pd.DataFrame with columns ['hadm_id', 'notes_text'] and no rows.
    """
    # Return empty DataFrame with expected columns and no rows. This prevents
    # the rest of the application from attempting to vectorize or otherwise
    # process textual notes from the demo data.
    return pd.DataFrame(columns=['hadm_id', 'notes_text'])
