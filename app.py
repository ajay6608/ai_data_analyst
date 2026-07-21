"""
app.py
------
AI Data Analyst - main Streamlit app.

Flow:
  1. User uploads a CSV (or loads the sample dataset)
  2. Tab 1 shows an auto-generated data overview
  3. Tab 2 shows auto-generated EDA charts (matplotlib/seaborn)
  4. Tab 3 lets the user ask questions in plain English; Gemini writes
     pandas code, we validate + run it safely, then show the result,
     chart, and a plain-English insight.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

from profiler import get_data_overview
from eda import plot_missing_values, plot_numeric_distributions, plot_correlation_heatmap
from gemini_handler import generate_pandas_code, generate_insight
from safe_executor import run_safe_code, UnsafeCodeError

load_dotenv()

st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")
st.title("📊 AI Data Analyst")
st.caption("Upload a dataset and ask questions about it in plain English.")

# --- File upload / sample data ---
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    st.session_state["df"] = pd.read_csv(uploaded_file)

if "df" not in st.session_state:
    st.info("👆 Upload a CSV file to get started, or try the sample dataset below.")
    if st.button("Load sample sales data"):
        st.session_state["df"] = pd.read_csv("sample_data/sample_sales.csv")
        st.rerun()

# --- Main app (only once data is loaded) ---
if "df" in st.session_state:
    df = st.session_state["df"]

    tab1, tab2, tab3 = st.tabs(["📋 Data overview", "🔍 Auto EDA", "💬 Ask AI"])

    # --- Tab 1: overview ---
    with tab1:
        st.subheader("Preview")
        st.dataframe(df.head(10), use_container_width=True)

        overview = get_data_overview(df)
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Shape:** {overview['shape'][0]} rows x {overview['shape'][1]} columns")
            st.write("**Column types:**")
            st.json(overview["dtypes"])
        with col2:
            st.write("**Missing values (%):**")
            st.json(overview["missing_pct"])

        if overview["numeric_summary"] is not None:
            st.write("**Summary statistics:**")
            st.dataframe(overview["numeric_summary"], use_container_width=True)

    # --- Tab 2: auto EDA ---
    with tab2:
        st.subheader("Automatic exploratory analysis")

        fig1 = plot_missing_values(df)
        if fig1:
            st.pyplot(fig1)
        else:
            st.success("No missing values found.")

        fig2 = plot_numeric_distributions(df)
        if fig2:
            st.pyplot(fig2)

        fig3 = plot_correlation_heatmap(df)
        if fig3:
            st.pyplot(fig3)
        else:
            st.info("Not enough numeric columns for a correlation heatmap.")

    # --- Tab 3: ask AI ---
    with tab3:
        st.subheader("Ask a question about your data")
        st.caption("Example: \"What are the top 5 products by revenue?\" or \"Show revenue trend over time\"")

        question = st.text_input("Your question", key="question_input")

        if st.button("Analyze", type="primary") and question:
            with st.spinner("Thinking..."):
                try:
                    code = generate_pandas_code(question, df)

                    with st.expander("View generated code"):
                        st.code(code, language="python")

                    result, fig = run_safe_code(code, df, pd, np, plt, sns)

                    if result is not None:
                        st.write("**Result:**")
                        if isinstance(result, (pd.DataFrame, pd.Series)):
                            st.dataframe(result, use_container_width=True)
                        else:
                            st.write(result)

                    if fig is not None:
                        st.pyplot(fig)

                    insight = generate_insight(question, result)
                    st.write("**Insight:**")
                    st.success(insight)

                except UnsafeCodeError as e:
                    st.error(f"Blocked unsafe code: {e}")
                except Exception as e:
                    st.error(f"Something went wrong: {e}")

    st.divider()
    if st.button("Clear data and start over"):
        del st.session_state["df"]
        st.rerun()
