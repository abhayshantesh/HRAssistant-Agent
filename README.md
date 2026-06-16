# HRAssistant-Agent 💼

A production-style **HR copilot** that answers employee questions by combining:

- **RAG** (Retrieval-Augmented Generation) over HR policy documents using FAISS semantic search, and
- **Structured database lookups** through real LLM **tool/function calling**.

A hybrid router decides, per question, whether to use policy documents, the
employee database, or both — then synthesizes a single grounded, cited answer.

Powered by **OpenRouter** (free models) · **LangChain** · **FAISS** · **Streamlit**.

---

## How it works

```
                       ┌─────────────────────────────┐
   User question  ───▶ │   Router (LLM + fallback)   │
                       │  RAG_ONLY / DB_ONLY / HYBRID │
                       └──────────────┬──────────────┘
                          ┌───────────┴───────────┐
                          ▼                        ▼
                 RAG: FAISS retrieve       DB: LLM tool calls
                 (policy chunks)           (employee records)
                          └───────────┬───────────┘
                                      ▼
                          Synthesis (LLM) → cited answer
```

**RAG pipeline:** `Document → Chunk → Embed → FAISS → Retrieve → LLM Answer`
Embeddings are generated locally (`all-MiniLM-L6-v2`), so no embedding API key is needed.

**Tool calling:** the model is given three tools — `get_employee`,
`get_leave_balance`, `get_department` — and decides which to call. Results are
fed back for synthesis.

---

## Quick start (one click)

Get a free API key at **https://openrouter.ai/keys**, then:

- **Windows:** double-click **`run.bat`**
- **macOS/Linux:** `./run.sh`

The launcher creates a virtual environment, installs dependencies, creates a
`.env` from the template on first run (set your `OPENROUTER_API_KEY` in it),
and starts the app at http://localhost:8501.

### Manual start

```bash
git clone <your-repo-url>
cd HRAssistant-Agent

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt

copy .env.example .env         # Windows  (cp on macOS/Linux)
# edit .env and set OPENROUTER_API_KEY

streamlit run app.py
```

In the browser: **upload one or more policy PDFs** in the sidebar to enable
policy (RAG) answers, then optionally pick an employee for personalized
record lookups and start asking. Employee-record questions work without any
upload.

### Example questions

| Question | Route |
|---|---|
| "What is the maternity leave policy?" | `RAG_ONLY` |
| "What is E001's leave balance?" | `DB_ONLY` |
| "What's the maternity policy, and how many leaves does Rajesh have left?" | `HYBRID` |

---

## Project structure

```
HRAssistant-Agent/
├── app.py                  # Streamlit chat UI
├── config.py               # Settings (env-driven, no hardcoded secrets)
├── requirements.txt
├── .env.example
├── data/
│   └── employee_data.csv   # 15 sample employees (policy docs come from PDF upload)
└── src/
    ├── llm.py              # OpenRouter provider layer (chat + tool calling + fallback)
    ├── rag.py              # FAISS RAG pipeline
    ├── database.py         # Employee DB + tool definitions/executors
    └── agent.py            # Routing + tool calling + answer synthesis
```

---

## Configuration

All settings come from environment variables (see `.env.example`). The only
required one is `OPENROUTER_API_KEY`. Models are tried in order with automatic
fallback:

1. `deepseek/deepseek-chat`
2. `meta-llama/llama-3.3-70b-instruct`
3. `qwen/qwen3-32b`

Set `OPENROUTER_MODEL` to pin a single model. Other knobs: `TEMPERATURE`,
`MAX_TOKENS`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `RETRIEVAL_K`.

---

## Deploy on Streamlit Community Cloud

The repo is deployment-ready: env-driven config, no hardcoded secrets, pinned
dependencies, and a committed `.streamlit/config.toml`.

1. **Push to GitHub** — make sure `.env` and `.streamlit/secrets.toml` are NOT
   committed (they are gitignored). `requirements.txt`, `app.py`, the `data/`
   folder, and `.streamlit/config.toml` must be present.
2. Go to **https://share.streamlit.io** → **New app**, connect your GitHub repo.
3. Set:
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Open **Advanced settings → Secrets** and paste:
   ```toml
   OPENROUTER_API_KEY = "sk-or-your-key-here"
   ```
   (`config.py` reads `st.secrets` automatically when no env var is present.)
5. Click **Deploy**. First launch downloads the embedding model (~90 MB);
   subsequent loads are fast (cached).
6. In the running app, **upload a policy PDF** in the sidebar to enable RAG.

**Verify after deploy:** upload a PDF, then ask one question of each type to
confirm all three routes work:
- "What is the maternity leave policy?" → `RAG_ONLY`
- "What is E001's leave balance?" → `DB_ONLY`
- "Maternity policy and how many leaves does E001 have left?" → `HYBRID`

---

## Notes & limitations

- Employee data is a CSV and the FAISS index lives in memory — fine for demos,
  swap for a real DB / persistent vector store for production scale.
- No authentication: any user can select any employee. Add auth before real use.
- Free OpenRouter models vary in tool-calling quality; the router falls back to
  deterministic keyword routing if the model call fails.

## License

MIT — see [LICENSE](LICENSE).
