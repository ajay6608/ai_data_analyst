"""
profiler.py
-----------
Generates a quick statistical overview of an uploaded dataset:
shape, column types, missing values, and summary statistics.

This is the "auto EDA" step that runs automatically the moment
a file is uploaded, before the user asks any question.
"""

import pandas as pd


def get_data_overview(df: pd.DataFrame) -> dict:
    """Return a dictionary summarizing the dataframe's structure and quality."""

    numeric_cols = df.select_dtypes(include="number").columns

    overview = {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().mean() * 100).round(2).to_dict(),
        "numeric_summary": df[numeric_cols].describe().round(2) if len(numeric_cols) > 0 else None,
        "head": df.head(5),
    }
    return overview


def get_dataframe_context_for_llm(df: pd.DataFrame, sample_rows: int = 3) -> str:
    """
    Build a compact text description of the dataframe to send to the LLM
    as context, so it knows the column names, types, and sample values
    without needing the full dataset.
    """
    dtypes_str = "\n".join(f"- {col}: {dtype}" for col, dtype in df.dtypes.astype(str).items())
    sample_str = df.head(sample_rows).to_string(index=False)

    context = (
        f"Dataframe shape: {df.shape[0]} rows, {df.shape[1]} columns\n\n"
        f"Columns and types:\n{dtypes_str}\n\n"
        f"Sample rows:\n{sample_str}"
    )
    return context
