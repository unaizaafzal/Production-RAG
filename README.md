# Production RAG Engine with Monitoring & Observability

> A domain-specific document Q&A system with hybrid retrieval, cross-encoder reranking, citation enforcement, and a fully instrumented observability & Monitering layer.

Built with **Python · LangChain · Pinecone · Cohere · Groq · Streamlit · Langfuse · HuggingFace**

---

## Project Structure

This repository covers two engineering layers built on top of each other:

**[v1.0-rag-baseline](../../tree/v1.0-rag-baseline)** — Production RAG Application
The core document Q&A system: PDF ingestion, hybrid retrieval, reranking, citation enforcement, multi-turn chat.

**[main](../../tree/main)** — Monitoring & Observability
Full-stack observability added on top: distributed tracing, p50/p95 latency tracking, LLM-as-a-judge evaluation, CI regression gating.

---

# Part 1: Production RAG Application

## What This Does

Upload any technical PDF. Ask questions about it in natural language. Get answers with inline citations showing exactly which page and document each claim came from.

The system doesn't just do keyword search or pure semantic search — it runs both simultaneously, fuses the results, then uses a cloud reranker to pick the most relevant chunks before the LLM ever sees them. This is what separates a production RAG system from a tutorial chatbot.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│     Query Contextualization         │
│  Rewrites follow-up questions into  │
│  standalone searchable queries      │
│  using conversation history         │
└──────────────────┬──────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
┌──────────────┐     ┌─────────────────┐
│  BM25 Sparse │     │ Pinecone Dense  │
│  Keyword     │     │ Vector Search   │
│  Search      │     │ (all-MiniLM-L6) │
└──────┬───────┘     └────────┬────────┘
        └──────────┬──────────┘
                   ▼
        ┌──────────────────────┐
        │  Reciprocal Rank     │
        │  Fusion (RRF)        │
        │  Top 15 candidates   │
        └──────────┬───────────┘
                   ▼
        ┌──────────────────────┐
        │  Cohere Rerank v3.5  │
        │  Cross-encoder       │
        │  Narrows to top 4    │
        └──────────┬───────────┘
                   ▼
        ┌──────────────────────┐
        │  Groq LLM            │
        │  Llama 3.3 70B       │
        │  Citation-enforced   │
        │  answer generation   │
        └──────────┬───────────┘
                   ▼
        Answer with [Source: file, Page: X] citations
```

---

## Why Hybrid Retrieval

**Vector search alone** converts your query to an embedding and finds semantically similar chunks. It understands meaning but can miss exact keyword matches.

**BM25 alone** is pure keyword matching — fast and precise for exact terms, but blind to meaning. It does not know that "attention mechanism" and "self-attention" refer to the same concept.

**Hybrid with RRF** combines both ranked lists. Documents appearing high in both lists get a strong combined score. The c=60 constant smooths the fusion and prevents a single top-ranked result from dominating.

**Cohere cross-encoder reranking** takes the top 15 fused candidates and rescores them by reading the query and each chunk together as a pair — not as separate embeddings. This is significantly more accurate but too slow to run on all chunks, which is why it only runs on the already-narrowed pool of 15.

---

## Citation Enforcement

Every chunk passed to the LLM is prefixed with its source:

```
[Source: attention_is_all_you_need.pdf, Page: 3]
The attention mechanism allows the model to...
```

The system prompt explicitly requires the LLM to cite sources inline after every claim. Without this instruction, the model sees the source tags but has no obligation to include them in the response, leading to unverifiable answers.

---

## Tech Stack

| Component | Tool | Purpose |
|---|---|---|
| PDF parsing | PyPDFium2 via LangChain | Handles multi-column academic paper layouts |
| Text chunking | RecursiveCharacterTextSplitter | 1200 char chunks, 300 char overlap |
| Embeddings | all-MiniLM-L6-v2 HuggingFace | Local dense vector generation |
| Vector database | Pinecone | Persistent vector storage with namespace isolation |
| Sparse search | BM25Okapi rank-bm25 | In-memory keyword ranking |
| Reranker | Cohere rerank-v3.5 | Cross-encoder reranking |
| LLM | Llama 3.3 70B via Groq | Answer generation |
| Frontend | Streamlit | Chat UI with PDF uploader |

---

## File Structure

```
├── ingest.py          # One-time: load PDFs, chunk, embed, upsert to Pinecone
├── query.py           # CLI chat session — standalone RAG pipeline
├── app.py             # Streamlit UI — full monitored pipeline
├── eval_test.py       # Automated evaluation suite with CI gate
├── data/              # Place your PDF documents here
└── .github/           # GitHub Actions CI workflow
```

---

## Setup

**Prerequisites:** Python 3.10+, API keys for Pinecone, Cohere, Groq, Langfuse

### 1. Clone and install
```bash
git clone <your-repo-url>
cd <your-repo>
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment variables
Create a `.env` file:
```
PINECONE_API_KEY=your_key
COHERE_API_KEY=your_key
GROQ_API_KEY=your_key
LANGFUSE_PUBLIC_KEY=your_key
LANGFUSE_SECRET_KEY=your_key
```

### 3. Add your documents
Drop PDF files into the `./data` directory.

### 4. Index your documents (run once)
```bash
python ingest.py
```

### 5. Run the app
```bash
streamlit run app.py
```

Upload your PDF via the sidebar, click Ingest, then start asking questions.

---

## How Chunking Works

Documents are split into 1200-character chunks with 300-character overlap. The overlap prevents information loss at chunk boundaries — if a key sentence falls at the end of one chunk, the next chunk starts 300 characters earlier and captures it again.

PyPDFium2 is used instead of standard PDF loaders because it correctly handles multi-column layouts common in academic papers. Standard loaders read columns left-to-right across the page, mixing content from separate columns.

---

# Part 2: Monitoring & Observability

## What This Adds

The base RAG system works. But without observability you cannot answer: which step is slow, which queries fail, whether quality is degrading over time, or whether a code change broke retrieval accuracy. This layer adds full visibility into every part of the pipeline.

---

## Observability Architecture

```
User Query
    │
    ▼
@observe(name="rag-chat-transaction")   ← root trace in Langfuse
    │
    ├── span: bm25-sparse-retrieval     ← timed individually
    ├── span: pinecone-dense-retrieval  ← timed individually
    ├── generation: cohere-cloud-rerank ← timed + model logged
    └── LangChain callbacks             ← LLM call traced
    │
    ▼
Langfuse Dashboard
    Session ID · User ID · Tags · Per-step latency · Token counts
```

Every query produces one trace in Langfuse with child spans for each retrieval step. You can see exactly how long BM25 took vs Pinecone vs Cohere vs the LLM for every single query.

---

## p50 / p95 Latency Tracking

After every query the response time is appended to an in-session history array. Two percentile metrics are calculated in real time and displayed in the sidebar:

**p50 (median)** — half of queries are faster than this. Your typical user experience.

**p95 (tail latency)** — 95% of queries are faster than this. Your worst-case experience for 1 in 20 users. Production SLAs are almost always defined on p95, not average, because averages hide outliers.

---

## LLM-as-a-Judge Evaluation

Instead of a fixed metric library, the evaluation suite uses the LLM itself as a quality judge. For each question in the golden dataset:

1. Context is retrieved from Pinecone
2. The LLM generates an answer
3. The same LLM scores how faithful the answer is to the context — returning a float from 0.0 to 1.0

The judge prompt specifically flags hallucinations — claims in the answer not supported by the retrieved context.

**Why LLM-as-a-judge instead of RAGAS?**
RAGAS requires manually written reference answers for every test question. LLM-as-a-judge only needs the retrieved context and generated answer, making it cheaper and easier to scale to new document domains without ground truth authoring.

---

## CI Regression Gate

`eval_test.py` runs in GitHub Actions on every push:

```
Push to main
    │
    ▼
GitHub Actions triggers eval_test.py
    │
    ├── Faithfulness score ≥ 0.8 → exit(0) → deployment proceeds
    └── Faithfulness score < 0.8 → exit(1) → deployment blocked
```

Exit codes are how CI systems communicate pass/fail. Scores are logged to Langfuse with create_score() so quality trends are visible across every eval run over time.

---

## Monitoring Tech Stack

| Component | Tool | Purpose |
|---|---|---|
| Distributed tracing | Langfuse | Per-step span visibility across the full pipeline |
| Latency percentiles | NumPy percentile | p50/p95 calculated in-session |
| Evaluation | LLM-as-a-judge | Faithfulness scoring without reference answers |
| CI gating | GitHub Actions + exit codes | Block deployment on quality regression |
| Score persistence | Langfuse create_score | Track quality trends over time |
