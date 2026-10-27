# Agentic RAG for Slack

A Slack app that answers questions from files shared in a channel. Claude runs as an agent:
it decides when to search, rewrites weak queries, searches again, and can fetch web pages
before replying in-thread with citations.

```
Slack (Socket Mode) -> app.py -> agent.py (Claude tool-use loop)
                                    |- search_knowledge_base -> rag.py (ChromaDB, per channel)
                                    |- list_sources
                                    |- fetch_url
                                    `- save_url_to_knowledge_base
```

## Slack setup (about 3 minutes)

1. Go to https://api.slack.com/apps -> **Create New App -> From a manifest**, pick your workspace, paste `manifest.yaml`.
2. **Basic Information -> App-Level Tokens -> Generate** with scope `connections:write` -> `SLACK_APP_TOKEN` (xapp-).
3. **Install App** to the workspace, then copy the **Bot User OAuth Token** -> `SLACK_BOT_TOKEN` (xoxb-).
4. Invite the bot to a channel: `/invite @Knowledge Agent`.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # fill in the three keys
python app.py
```

## Usage

- Upload a PDF/TXT/MD/CSV to a channel the bot is in (or DM it): it indexes the file.
- `@Knowledge Agent what does the Q3 report say about churn?` - answers in a thread; follow-ups in the same thread keep context.
- DMs work without a mention.
- `/kb list`, `/kb delete <name>`, `/kb forget`, `/kb help`

## Design notes

- **Scoping:** each channel (and each DM) has an isolated knowledge base, keyed `team:channel`. Anyone in a channel can query that channel's documents, so avoid indexing private files in broad channels.
- **Memory:** per-thread, last 8 exchanges, kept in process memory (lost on restart).
- **Embeddings:** Chroma's default local MiniLM model, downloaded on first use.
- **Access control:** set `ALLOWED_USER_IDS` to restrict who can spend your API credits.
- **Safety:** `fetch_url` rejects non-public addresses and does not follow redirects (SSRF guard).
- **Going further:** HTTP/Events API mode for production (swap the handler), pgvector/Qdrant for the store, a reranker, Redis for thread memory.
