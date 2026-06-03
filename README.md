# RAG over Western Alliance Bank Filings

**Grounded Q&A over a bank's SEC 10-K — built so every answer is traceable to source.** Ask a natural-language question about Western Alliance Bancorporation (WAB) and get an answer drawn *only* from its public Form 10-K, with the exact source passages and similarity scores shown so a user can verify it.

> **What this demonstrates as an AI PM:** scoping a retrieval product around the constraint that actually matters in finance — *trust* — and making deliberate product calls on grounding, source attribution, cost, and the quality/latency tradeoffs that decide whether a tool is usable in a regulated environment.

## The problem

Analysts, risk teams, and relationship managers spend hours digging through dense regulatory filings to answer specific questions — *What are WAB's largest risk factors? How is the deposit base concentrated? What's their CRE exposure?* A general chatbot can answer fluently and **be wrong**, which in financial analysis is worse than useless: an unsourced, confidently incorrect figure can drive a bad credit or investment call.

The product requirement, therefore, isn't "answer questions about the filing." It's "answer questions about the filing *in a way the user can trust and check.*" Retrieval-augmented generation is the means; verifiable grounding is the actual product.

## What it does

Ask about WAB's financials, risk factors, or strategy and get an answer synthesized only from the 10-K. Alongside each answer, the app surfaces the retrieved source passages and their similarity scores, so the user can confirm the basis for the response rather than taking it on faith. When the filing doesn't cover a question, the model is instructed to say **"I don't know"** rather than guess.

## Product decisions (the interesting part)

This is where the project reads as PM work, not just a RAG demo:

- **Grounding as a hard requirement.** The model answers strictly from retrieved context and abstains when coverage is missing. In banking, a defensible "I don't know" beats a plausible hallucination every time.
- **Source citations exposed in the UI.** Showing passages *and* similarity scores turns the model from a black box into an auditable tool — the difference between something compliance will tolerate and something it will block.
- **Cost / quality / latency surfaced as user controls.** The sidebar exposes model choice (Sonnet vs Haiku) and retrieval depth (top-k), making the core tradeoffs explicit and tunable instead of hidden engineering defaults.
- **Local embeddings to control cost and data exposure.** Embeddings run on-device (no per-token API cost, no document text leaving for embedding); only final answer generation calls the Claude API — a sensible posture when the source material is sensitive financial data.

## How it works

**Indexing:** load → chunk → embed → store.
**Querying:** embed the question → retrieve top-k passages → Claude answers from that context only.
Embeddings run locally; only generation calls the Claude API. First launch builds the index into `./storage` (one-time); later launches load it instantly.

## Stack

- **LlamaIndex** — RAG pipeline (load, chunk, embed, retrieve, generate)
- **BAAI/bge-small-en-v1.5** — local embeddings (on-device, no API cost)
- **SimpleVectorStore** — LlamaIndex built-in vector store, persisted to `./storage`
- **Claude (Sonnet or Haiku)** — answer generation
- **Streamlit** — UI with sidebar controls for model and retrieval depth

## Run it

1. `pip install -r requirements.txt`
2. Add your key to a `.env` file: `ANTHROPIC_API_KEY=...`
3. Make sure `data/` contains the filing text (see Data note)
4. `streamlit run app.py`

Delete `./storage` to force a rebuild after changing data.

## Data

Source: WAB FY2025 Form 10-K, filed with the SEC (EDGAR), public. Extracted to plain text in `data/wab-10k-full.txt`.

## Limitations and roadmap

- Currently indexes the 10-K only. The natural next step is adding the **FDIC Call Report** so the same interface answers regulatory and financial-condition questions side by side.
- No evaluation harness yet — a production version would add a graded question set scoring **answer faithfulness** and **retrieval precision**, so quality regressions are caught before release.
- Retrieval is dense-only; hybrid (keyword + dense) retrieval would improve recall on exact figures and defined terms common in filings.

---

*Part of a portfolio demonstrating AI product management in banking and payments: [capability evaluation](https://github.com/Xander-Le/capability-comparator) → RAG over bank filings (this project) → [tool-calling payments agent](https://github.com/Xander-Le/payment-rail-agent).*
