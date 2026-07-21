# AI Data Analyst

Upload a CSV, and ask questions about it in plain English. The app uses the
Gemini API to translate your question into pandas code, runs that code in a
sandboxed environment, and shows you a table, a chart (matplotlib/seaborn),
and a plain-English insight.

## Features

- **Auto data overview** - shape, column types, missing values, summary stats, generated instantly on upload
- **Auto EDA** - missing value chart, numeric distributions, correlation heatmap, no user input needed
- **Ask AI** - natural language question -> Gemini generates pandas code -> code is validated and run safely -> result + chart + insight
- **Transparent & safe** - the generated code is always shown to you, and is checked against a blocklist (no imports, no file access, no `exec`/`eval`) before running

## Project structure

```
ai-data-analyst/
├── app.py               # Streamlit UI - main entry point
├── profiler.py          # data overview / profiling logic
├── eda.py               # automatic matplotlib/seaborn charts
├── gemini_handler.py     # Gemini API calls (NL -> pandas code, insights)
├── safe_executor.py      # AST-based safety check + sandboxed exec
├── requirements.txt
├── .env.example
└── sample_data/
    └── sample_sales.csv  # synthetic dataset to demo the app with
```

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Get a free Gemini API key**
   Go to https://aistudio.google.com/apikey, sign in, and create a key.

3. **Add your key**
   Copy `.env.example` to `.env` and paste your key in:
   ```bash
   cp .env.example .env
   ```
   ```
   GEMINI_API_KEY=your_actual_key_here
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

5. Open the local URL Streamlit gives you, upload a CSV (or click "Load sample
   sales data"), and start asking questions like:
   - "What are the top 5 products by revenue?"
   - "Show me revenue by region as a bar chart"
   - "Is there a trend in units sold over time?"

## How the "safe execution" works

LLMs can occasionally generate code that isn't what you want, or in the worst
case, code that tries to touch the file system or environment. Before any
generated code runs:

1. It's parsed into an AST (abstract syntax tree).
2. Imports, `exec`/`eval`/`open`, dunder attribute access, and OS-related
   names are rejected outright.
3. It's executed with a restricted set of builtins - only safe things like
   `len()`, `sum()`, `range()` are available, and only `df`, `pandas`,
   `numpy`, `matplotlib.pyplot`, and `seaborn` are exposed to it.

## Ideas to extend this further

- Add a MySQL/PostgreSQL connection mode and let Gemini generate SQL instead of pandas (great for showing off SQL skills alongside this project)
- Cache repeated questions to reduce API calls
- Add a "confidence check" - ask Gemini to also explain its code in one line, so users can sanity-check it before trusting the result
- Export the full analysis session as a PDF report

## Deploying (free)

Push this to a public GitHub repo, then deploy for free on
[Streamlit Community Cloud](https://streamlit.io/cloud) - point it at
`app.py`, add `GEMINI_API_KEY` under "Secrets", and you'll have a live demo
link to put on your resume.
