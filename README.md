# RAG over Western Alliance Bank Filings

A retrieval-augmented generation (RAG) app that answers natural-language
questions about Western Alliance Bancorporation (WAB) using its public
SEC Form 10-K, with every answer grounded in the source text.

## What it does
Ask about WAB's financials, risk factors, or strategy and get an answer
drawn only from the filing, with the source passages and their similarity
scores shown so you can verify it.

## Stack
- LlamaIndex - RAG pipeline (load, chunk, embed, retrieve, generate)
- Local embeddings: BAAI/bge-small-en-v1.5 (runs on device, no API cost)
- LlamaIndex built-in vector store (SimpleVectorStore, persisted to ./storage)
- Claude (Sonnet or Haiku) - answer generation
- Streamlit - UI, with sidebar controls for model and retrieval depth

## How it works
Indexing: load -> chunk -> embed -> store.
Querying: embed the question -> retrieve top-k passages -> Claude answers
from that context only. Embeddings run locally; only generation calls the
Claude API.

## Run it
1. pip install -r requirements.txt
2. Add your key to a .env file:  ANTHROPIC_API_KEY=...
3. Make sure data/ contains the filing text (see Data note)
4. streamlit run app.py

First launch builds the index into ./storage (one-time). Later launches
load it instantly. Delete ./storage to force a rebuild after changing data.

## Data
Source: WAB FY2025 Form 10-K, filed with the SEC (EDGAR), public.
Extracted to plain text in data/wab-10k-full.txt.

## Design notes / limitations
- The model is instructed to answer only from the filing and to say
  "I don't know" when the context doesn't cover the question (grounding).
- The sidebar exposes model choice (Sonnet vs Haiku) and top-k, surfacing
  the cost / quality / latency tradeoffs as product decisions.
- Currently indexes the 10-K only; an FDIC Call Report can be added to data.
