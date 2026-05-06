# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements_rag.txt

# Run all tests
pytest tests/

# Run a single test file
pytest tests/rag/test_retriever.py

# Run a single test
pytest tests/rag/test_retriever.py::test_extract_year_with_marker -v

# Start the chatbot UI
streamlit run rag_chatbot/app.py

# Build the vector index manually (first run: tens of minutes)
python -c "from rag_chatbot.indexer import load_or_build_index; load_or_build_index()"
```

## Environment Setup

Copy `.env.template` to `.env` and fill in:
```
NVIDIA_NIM_API_KEY=...
NVIDIA_NIM_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_NIM_MODEL=moonshotai/kimi-k2.6
```

**Local-only data** (gitignored, must exist for the RAG pipeline to work):
- `preprocessed_data/청년_동아일보_정제완료_EA13796.csv`
- `preprocessed_data/청년_한겨레_정제완료_EA10690.csv`
- `preprocessed_data/청년_한국일보_정제완료_EA10813.csv`
- `output/chroma_db/` — auto-created on first index build

## Architecture

### RAG Pipeline (`rag_chatbot/`)

The pipeline is split across four modules with strict dependency order:

```
config.py → indexer.py → retriever.py → app.py
```

**`config.py`** — Single source of truth for all constants. `BASE_DIR` is resolved from `__file__`, so all paths work regardless of CWD.

**`indexer.py`** — Two-stage index lifecycle:
1. `load_or_build_index()` checks `collection.count() > 0` to decide load vs. build
2. Build path: CSV → kss sentence splitting → `"\n".join()` → `Document` → `SentenceSplitter(chunk_size=512, overlap=20)` → ChromaDB via LlamaIndex

The `tokens` column in the CSVs (Kiwi morphological analysis output) is intentionally unused. `text_cleaned` goes directly to the granite embedding model's own tokenizer; kss is only used to mark sentence boundaries.

**`retriever.py`** — Per-query engine construction:
- Year extracted via `(?<!\d)(20[12]\d)(?!\d)` — `\b` doesn't work here because Python's `re` treats Korean `년` as `\w`
- Newspaper filter applied only when exactly 1 newspaper is selected; OR-filter is unsupported by `MetadataFilters`, so 2-paper selection falls back to no newspaper filter
- LLM is passed as `llm=` kwarg to `index.as_query_engine()`, **not** via `Settings.llm` — avoids global state mutation between Streamlit requests

**`app.py`** — Streamlit UI. Index loaded once via `@st.cache_resource` (survives reruns). Sidebar year/newspaper selections are passed to `retriever.query()` as explicit arguments, not embedded in the query string.

### Test Suite (`tests/rag/`)

Tests cover `config`, `indexer`, and `retriever` with no live LLM or real ChromaDB calls. The `mock_csv` fixture in `conftest.py` provides 3 rows × 3 newspapers = 9 documents. The kss test asserts exactly 2 `"\n"` characters in the 3-sentence "청년 취업 현황" article.

### Analysis Notebooks (`01_crawl/`, `02_network_analysis/`)

Notebooks use `../` relative paths to reach project-root data directories (`news_main/`, `preprocessed_data/`, `output/`). In `text_mining.ipynb`, `_project_root = _cwd.parent` is hardcoded (not auto-detected) because `synonym_stopword.py` is in the same directory as the notebook.
