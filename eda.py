"""
eda.py
------
Automatically generates common exploratory charts using matplotlib
and seaborn: missing value bar chart, numeric distributions, and a
correlation heatmap. These run without the user asking anything.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_style("whitegrid")


def plot_missing_values(df: pd.DataFrame):
    """Bar chart of columns that have missing values. Returns None if there are none."""
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    missing.sort_values().plot(kind="barh", ax=ax, color="#D85A30")
    ax.set_title("Missing values per column")
    ax.set_xlabel("Count")
    fig.tight_layout()
    return fig


def plot_numeric_distributions(df: pd.DataFrame, max_cols: int = 4):
    """Histogram + KDE for up to max_cols numeric columns."""
    numeric_cols = df.select_dtypes(include="number").columns[:max_cols]
    if len(numeric_cols) == 0:
        return None

    fig, axes = plt.subplots(1, len(numeric_cols), figsize=(4 * len(numeric_cols), 4))
    if len(numeric_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, numeric_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color="#1D9E75")
        ax.set_title(col)

    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame):
    """Correlation heatmap across numeric columns. Returns None if fewer than 2 numeric cols."""
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        return None

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation heatmap")
    fig.tight_layout()
    return fig
