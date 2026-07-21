"""
gemini_handler.py
------------------
Talks to Google's Gemini API to:
1. Translate a natural-language question into pandas code
2. Turn a raw result into a short plain-English insight

Get a free Gemini API key at https://aistudio.google.com/apikey
and put it in a .env file as GEMINI_API_KEY=your_key_here
"""

import os
import google.generativeai as genai
import pandas as pd
from profiler import get_dataframe_context_for_llm

# Google retires/renames Gemini model IDs fairly often. Rather than hardcode
# one, we read it from .env so a future deprecation only needs a one-line
# change there instead of editing this file. Defaults to the current stable
# model as of July 2026.
DEFAULT_MODEL = "gemini-3.5-flash"


def _get_model():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Add it to a .env file (see .env.example)."
        )
    genai.configure(api_key=api_key)
    model_name = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    return genai.GenerativeModel(model_name)


def generate_pandas_code(question: str, df: pd.DataFrame) -> str:
    """Ask Gemini to write pandas code that answers `question` using dataframe `df`."""
    model = _get_model()
    context = get_dataframe_context_for_llm(df)

    prompt = f"""You are a data analyst assistant. You are given a pandas dataframe called `df`.

{context}

The user asks: "{question}"

Write ONLY python pandas code (no explanation, no markdown fences, no comments) that:
1. Computes the answer and stores it in a variable named `result`.
2. If a chart would help answer the question, create a matplotlib figure using
   plt.subplots() and store the figure object in a variable named `fig`
   (matplotlib.pyplot is available as `plt`, seaborn as `sns`). If no chart is
   needed, do not create a `fig` variable at all.
3. Only use the variables `df`, `pd`, `np`, `plt`, `sns` which already exist.
   Do NOT import anything. Do NOT read or write files. Do NOT use exec, eval,
   or open.

Return only the raw python code."""

    response = model.generate_content(prompt)
    code = response.text.strip()

    # Gemini sometimes wraps code in markdown fences even when asked not to
    code = code.replace("```python", "").replace("```", "").strip()
    return code


def generate_insight(question: str, result) -> str:
    """Ask Gemini for a short plain-English interpretation of a computed result."""
    model = _get_model()

    # Keep the result representation short so we don't blow up the prompt
    result_str = str(result)
    if len(result_str) > 1500:
        result_str = result_str[:1500] + "... (truncated)"

    prompt = f"""The user asked: "{question}"
The computed result is:
{result_str}

Write a short, 1-2 sentence, plain-English business insight about what this
result means. Don't just repeat the numbers, interpret them."""

    response = model.generate_content(prompt)
    return response.text.strip()
