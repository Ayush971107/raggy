# PDF RAG App

## Setup

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -U streamlit llama-index llama-index-readers llama-index-llms-openai llama-index-embeddings-openai pypdf

# 3. Launch app
streamlit run app.py
