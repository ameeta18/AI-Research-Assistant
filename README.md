# AI Research Assistant

[![Continuous Integration](https://github.com/ameeta18/Ai_Researcher/actions/workflows/ci.yml/badge.svg)](https://github.com/ameeta18/Ai_Researcher/actions/workflows/ci.yml)

A production-oriented AI research assistant that discovers academic papers,
validates and reads open-access PDFs, builds a session-isolated RAG corpus, and
generates citation-checked research PDFs.

The project demonstrates more than an LLM chat interface: it includes durable
workflow state, caching, reliability controls, structured observability,
offline evaluation gates, database migrations, tests, and a containerized local
deployment.

## What it does

1. Searches Semantic Scholar and arXiv for relevant papers.
2. Validates candidate links before treating them as PDFs.
3. Extracts, chunks, and embeds selected papers.
4. Retrieves relevant passages from a session-isolated FAISS index.
5. Uses a LangGraph tool-calling workflow to analyze or write from that context.
6. Validates LaTeX citations and compiles the final document with Tectonic.

## Architecture

```mermaid
flowchart LR
    UI[Streamlit / FastAPI] --> AGENT[LangGraph agent]

    AGENT --> SEARCH[Semantic Scholar / arXiv]
    SEARCH <--> REDIS[(Redis search cache)]

    AGENT --> PDF[PDF validation and extraction]
    PDF --> CHUNKS[Chunking]
    CHUNKS --> EMB[Gemini embeddings]
    EMB --> FAISS[(Session-isolated FAISS)]
    CHUNKS --> PG[(PostgreSQL)]

    AGENT <--> FAISS
    AGENT <--> CHECKPOINTS[PostgreSQL checkpoints]
    UI <--> PG

    AGENT --> LATEX[Citation validator and Tectonic]
    LATEX --> OUTPUT[Generated PDF]
```

### Storage responsibilities

| Component | Responsibility |
|---|---|
| PostgreSQL | Durable sessions, messages, paper metadata, extracted chunks, feedback records, and LangGraph checkpoints |
| FAISS | Fast semantic similarity search over the current session's embedded corpus |
| Redis | Short-lived cache for public paper-search results |
| Streamlit memory | User-supplied Gemini key and active UI state |

PostgreSQL does not replace FAISS. PostgreSQL is the durable system of record;
FAISS remains the retrieval index. If an in-memory FAISS index is lost, the
application can rebuild it from the session's PostgreSQL chunks.

## Production-focused features

- **Session isolation:** PDF state, embedding clients, and FAISS indexes are
  isolated by random conversation UUID.
- **Durable state:** PostgreSQL persists chat history, source chunks, metadata,
  feedback-ready records, and LangGraph execution checkpoints.
- **Caching:** Redis caches repeat paper searches with configurable TTLs and
  degrades safely to a cache miss when unavailable.
- **Reliability:** External requests use bounded timeouts, retryable-status
  rules, exponential backoff, jitter, `Retry-After`, and safe user-facing errors.
- **Input boundaries:** Limits cover message length, query length, result count,
  PDF size/pages/text, retrieval `k`, and LaTeX compilation time.
- **Observability:** Structured JSON logs include request/session correlation,
  latency, status, retry, and tool metadata without prompt or API-key content.
- **Optional tracing:** LangSmith tracing is configurable and hides inputs and
  outputs by default.
- **Citation safety:** Generated LaTeX must have matching `\cite{}` and
  `\bibitem{}` keys, complete reference metadata, URLs, and no unresolved
  compiler citation warnings.
- **Reproducible delivery:** Locked dependencies, non-root container execution,
  service health checks, versioned Alembic migrations, and GitHub Actions gates.

## Security and identity boundaries

- Every Streamlit user must provide their own Gemini API key.
- The Streamlit application never falls back to the deployer's
  `GOOGLE_API_KEY`.
- Gemini keys are not stored in PostgreSQL, graph state, logs, traces, or
  generated reports.
- PostgreSQL sessions use random UUIDs, never API keys, as identifiers.
- The database is authentication-ready: `users.id` is an internal UUID and
  `sessions.user_id` is nullable until Google login/logout is implemented.
- PostgreSQL and Redis are private inside the Docker Compose network.

Google authentication is intentionally future work; the current version does
not claim multi-user authorization based on login identity.

## Evaluation

The repository contains a deterministic offline challenge set with relevant
documents, paraphrases, lexical distractors, ambiguous tool routing, unsupported
claims, missing citations, and representative tool failures. Known failures
remain in the dataset so the report does not present artificial `1.0` scores.

Current checked-in baseline:

| Metric | Score |
|---|---:|
| Retrieval hit rate | 0.8333 |
| Retrieval mean reciprocal rank | 0.8333 |
| Retrieval concept coverage | 0.8611 |
| Tool-selection accuracy | 0.9000 |
| Groundedness | 0.8333 |
| Citation completeness | 0.8333 |
| Tool success rate | 0.9000 |

The suite contains 12 retrieval cases, 10 tool-selection cases, 6 generation
cases, and 10 tool events. It enforces both absolute minimums and maximum
allowed drops from the reviewed baseline.

This is a repeatable regression gate, not a replacement for live-model
evaluation or production monitoring.

## Technology stack

| Area | Technology |
|---|---|
| Agent workflow | LangGraph and LangChain |
| Model and embeddings | Google Gemini |
| Vector retrieval | FAISS |
| Durable data | PostgreSQL and SQLAlchemy |
| Schema migrations | Alembic |
| Workflow checkpoints | LangGraph PostgreSQL checkpointer |
| Cache | Redis |
| Interactive UI | Streamlit |
| API | FastAPI |
| Academic sources | Semantic Scholar and arXiv |
| PDF processing | PyPDF2 and Tectonic |
| Observability | Structured JSON logging and optional LangSmith tracing |
| Packaging | uv with a committed lockfile |
| Delivery | Docker, Docker Compose, and GitHub Actions |
| Tool interoperability | Model Context Protocol (MCP) |

## Project structure

```text
Ai_Researcher/
├── src/
│   ├── app.py                    # Streamlit interface and BYOK flow
│   ├── api.py                    # FastAPI interface
│   ├── graph.py                  # LangGraph agent workflow
│   ├── config.py                 # Validated environment configuration
│   ├── reliability.py            # Validation, timeout, and retry policies
│   ├── observability.py          # Structured logs and tracing
│   ├── cache.py                  # Redis cache boundary
│   ├── checkpointing.py          # Memory/PostgreSQL checkpoint selection
│   ├── session_store.py          # Session-isolated runtime resources
│   ├── evaluation_suite.py       # Deterministic evaluation runner
│   ├── migrate.py                # App and checkpoint migrations
│   ├── persistence/
│   │   ├── schema.py             # SQLAlchemy application schema
│   │   └── repository.py         # PostgreSQL persistence boundary
│   └── tools/
│       ├── semantic_scholar_tool.py
│       ├── arxiv_tool.py
│       ├── pdf_sources.py
│       ├── read_pdf.py
│       ├── vector_store.py
│       └── write_pdf.py
├── migrations/                   # Versioned Alembic migrations
├── evals/                        # Challenge set and reviewed baseline
├── scripts/run_evaluation.py     # Evaluation CLI
├── tests/                        # Unit, contract, and integration tests
├── .github/workflows/ci.yml      # Tests, eval gate, and image build
├── Dockerfile                    # Non-root production image
├── compose.yaml                  # App, PostgreSQL, migration, and Redis stack
├── mcp_server.py                 # Standalone academic-search MCP server
├── .env.example
├── pyproject.toml
└── uv.lock
```

## Quick start with Docker Compose

### Requirements

- Docker Desktop or Docker Engine with Compose
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)

Copy the example configuration and change the local PostgreSQL password:

```bash
cp .env.example .env
```

On PowerShell:

```powershell
Copy-Item .env.example .env
```

Then start the complete free local stack:

```bash
docker-compose up --build
```

Open <http://localhost:8501> and enter your Gemini API key in the password field.
The migration container upgrades both the application schema and LangGraph
checkpoint tables before Streamlit starts.

Stop the containers with:

```bash
docker-compose down
```

Named volumes preserve PostgreSQL data, generated papers, and the Tectonic
package cache. Use `docker-compose down -v` only when you intentionally want to
delete that local data.

## Lightweight local development

Python 3.11 or newer and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --locked
uv run python -m streamlit run src/app.py
```

With the defaults in `.env.example`, PostgreSQL and Redis are disabled for this
host-run mode. The app uses in-memory sessions/checkpoints, and users still
provide their own Gemini key in the UI.

To enable a host-accessible PostgreSQL instance, configure:

```dotenv
DATABASE_ENABLED=true
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST:5432/DATABASE
```

Apply migrations before starting the application:

```bash
uv run python -m src.migrate
```

## FastAPI backend

The FastAPI interface is server-controlled and requires `GOOGLE_API_KEY` in the
server environment. It does not accept API keys in request bodies.

```bash
uv run uvicorn src.api:app --reload
```

Interactive documentation is available at <http://127.0.0.1:8000/docs>.

## Tests and evaluation

Run the same quality checks used in CI:

```bash
uv sync --locked
uv run python -m unittest discover -s tests -v
uv run python scripts/run_evaluation.py
```

The evaluation command writes local reports to `output/evaluations/`. Generated
reports and PDFs are intentionally ignored by Git. See
[evals/README.md](evals/README.md) for baseline-management rules.

GitHub Actions additionally starts PostgreSQL, applies migrations, runs the
database integration test, builds the production image, and verifies its Python
and Tectonic runtimes.

## MCP server

The academic search tools are also exposed through a standalone Model Context
Protocol server:

```bash
uv run python mcp_server.py
```

Example MCP client configuration:

```json
{
  "mcpServers": {
    "academic-research": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/absolute/path/to/Ai_Researcher",
        "python",
        "mcp_server.py"
      ]
    }
  }
}
```

## Current roadmap

- Google OIDC login/logout and user-owned conversation history
- Production deployment with managed secrets and TLS
- Live-model evaluation from reviewed production traces
- Database backups, retention policy, and restore drills
