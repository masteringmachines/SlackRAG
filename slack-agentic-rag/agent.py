"""The agent: Claude + tools, looped until it has enough to answer."""
import asyncio
import ipaddress
import json
import logging
import re
import socket
from urllib.parse import urlparse

import anthropic
import httpx
from bs4 import BeautifulSoup

import config
from rag import KnowledgeBase

log = logging.getLogger(__name__)

SYSTEM = """You are a research assistant inside Slack. You can search the knowledge base of \
documents shared in this channel and fetch public web pages.

How to work:
- For any question that might depend on the shared documents, call search_knowledge_base first.
- If results are weak or incomplete, rewrite the query (synonyms, narrower/broader terms) and \
search again. Split multi-part questions into separate searches.
- Use fetch_url only when the user gives a URL, or the knowledge base lacks the answer and a \
specific page would help. Use save_url_to_knowledge_base only when asked to save a page.
- Ground answers in retrieved text and cite sources like _[filename]_. If the knowledge base \
does not contain the answer, say so instead of guessing.
- Be concise. Format for Slack mrkdwn: *bold* uses single asterisks, no headings, no tables, \
bullets with "•"."""

TOOLS = [
    {
        "name": "search_knowledge_base",
        "description": "Semantic search over documents shared in this channel. Returns top passages with source filenames.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "description": "1-10", "default": config.TOP_K},
            },
            "required": ["query"],
        },
    },
    {
        "name": "list_sources",
        "description": "List documents currently in this channel's knowledge base.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "fetch_url",
        "description": "Download a public web page and return its readable text (truncated).",
        "input_schema": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
    },
    {
        "name": "save_url_to_knowledge_base",
        "description": "Fetch a public web page and store it in this channel's knowledge base.",
        "input_schema": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
    },
]


def _is_public_http_url(url: str) -> bool:
    """Basic SSRF guard: http(s) only; host must resolve to public IPs."""
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        return False
    try:
        return all(ipaddress.ip_address(i[4][0]).is_global for i in socket.getaddrinfo(p.hostname, None))
    except socket.gaierror:
        return False


async def _fetch_page_text(url: str) -> str:
    if not _is_public_http_url(url):
        raise ValueError("URL must be a public http(s) address")
    async with httpx.AsyncClient(follow_redirects=False, timeout=20) as client:
        r = await client.get(url, headers={"User-Agent": "slack-rag-bot/1.0"})
        r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    return " ".join(soup.get_text(" ").split())


def to_mrkdwn(text: str) -> str:
    """Convert common Markdown the model may emit into Slack mrkdwn."""
    text = re.sub(r"\*\*(.+?)\*\*", r"*\1*", text)
    text = re.sub(r"^#{1,6}\s+(.+)$", r"*\1*", text, flags=re.M)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"<\2|\1>", text)
    return text


class Agent:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.client = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
        self.history: dict[str, list[dict]] = {}  # keyed by conversation (thread) id

    def forget(self, convo_id: str) -> None:
        self.history.pop(convo_id, None)

    async def _run_tool(self, scope: str, name: str, args: dict) -> str:
        try:
            if name == "search_knowledge_base":
                k = max(1, min(int(args.get("top_k", config.TOP_K)), 10))
                hits = await asyncio.to_thread(self.kb.search, scope, args["query"], k)
                return json.dumps(hits, ensure_ascii=False) if hits else "No results. The knowledge base may be empty."
            if name == "list_sources":
                srcs = await asyncio.to_thread(self.kb.list_sources, scope)
                return json.dumps(srcs) if srcs else "Knowledge base is empty."
            if name == "fetch_url":
                text = await _fetch_page_text(args["url"])
                return text[:8000] + (" …[truncated]" if len(text) > 8000 else "")
            if name == "save_url_to_knowledge_base":
                text = await _fetch_page_text(args["url"])
                n = await asyncio.to_thread(self.kb.add_document, scope, args["url"], text)
                return f"Saved {n} chunks from {args['url']}."
            return f"Unknown tool: {name}"
        except Exception as e:  # surface to the model so it can adapt
            log.warning("tool %s failed: %s", name, e)
            return f"Error: {e}"

    async def answer(self, scope: str, convo_id: str, user_text: str) -> str:
        past = self.history.setdefault(convo_id, [])
        messages = past + [{"role": "user", "content": user_text}]
        final = ""
        for _ in range(config.MAX_AGENT_STEPS):
            resp = await self.client.messages.create(
                model=config.MODEL, max_tokens=1500, system=SYSTEM, tools=TOOLS, messages=messages
            )
            messages.append({"role": "assistant", "content": resp.content})
            final = "".join(b.text for b in resp.content if b.type == "text").strip()
            if resp.stop_reason != "tool_use":
                break
            results = []
            for block in resp.content:
                if block.type == "tool_use":
                    log.info("scope %s tool %s %s", scope, block.name, block.input)
                    out = await self._run_tool(scope, block.name, block.input)
                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": out})
            messages.append({"role": "user", "content": results})
        else:
            final = final or "I hit my step limit before finishing. Try a more specific question."

        past.append({"role": "user", "content": user_text})
        past.append({"role": "assistant", "content": final or "(no answer)"})
        del past[: -2 * config.HISTORY_TURNS]
        return to_mrkdwn(final) or "I couldn't produce an answer."
