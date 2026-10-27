import asyncio
import logging
import re
import tempfile
from pathlib import Path

import httpx
from slack_bolt.adapter.socket_mode.aiohttp import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp

import config
from agent import Agent
from rag import KnowledgeBase, extract_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("slack-rag")

app = AsyncApp(token=config.SLACK_BOT_TOKEN)
kb = KnowledgeBase()
agent = Agent(kb)

HELP = (
    "*Knowledge Agent*\n"
    "• Upload PDF/TXT/MD/CSV files to this channel and I'll index them.\n"
    "• @mention me with a question (or just message me in a DM).\n"
    "• `/kb list` – documents indexed here\n"
    "• `/kb delete <name>` – remove a document\n"
    "• `/kb forget` – clear my conversation memory in this channel\n"
    "• `/kb help` – this message"
)


def allowed(user_id: str | None) -> bool:
    return not config.ALLOWED_USER_IDS or (user_id in config.ALLOWED_USER_IDS)


def scope_for(team_id: str, channel_id: str) -> str:
    return f"{team_id}:{channel_id}"


def strip_mention(text: str) -> str:
    return re.sub(r"<@[A-Z0-9]+>", "", text or "").strip()


async def respond(client, scope: str, channel: str, thread_ts: str, text: str):
    """Post a placeholder, run the agent, then replace the placeholder with the answer."""
    placeholder = await client.chat_postMessage(channel=channel, thread_ts=thread_ts, text=":hourglass_flowing_sand: Searching…")
    try:
        answer = await agent.answer(scope, f"{channel}:{thread_ts}", text)
    except Exception:
        log.exception("agent failed")
        answer = ":warning: Something went wrong while answering. Please try again."
    await client.chat_update(channel=channel, ts=placeholder["ts"], text=answer)


@app.event("app_mention")
async def on_mention(event, client, context):
    if not allowed(event.get("user")):
        return
    text = strip_mention(event.get("text", ""))
    if not text:
        await client.chat_postMessage(channel=event["channel"], thread_ts=event.get("thread_ts") or event["ts"], text=HELP)
        return
    scope = scope_for(context.team_id, event["channel"])
    await respond(client, scope, event["channel"], event.get("thread_ts") or event["ts"], text)


@app.event("message")
async def on_message(event, client, context):
    # Only handle plain DMs; channel questions arrive via app_mention, files via file_shared.
    if event.get("channel_type") != "im" or event.get("subtype") or event.get("bot_id"):
        return
    if not allowed(event.get("user")):
        return
    text = (event.get("text") or "").strip()
    if not text:
        return
    scope = scope_for(context.team_id, event["channel"])
    await respond(client, scope, event["channel"], event.get("thread_ts") or event["ts"], text)


@app.event("file_shared")
async def on_file_shared(event, client, context):
    channel = event.get("channel_id")
    if not channel or not allowed(event.get("user_id")):
        return
    info = (await client.files_info(file=event["file_id"]))["file"]
    name = info.get("name") or "untitled.txt"
    ext = Path(name).suffix.lower()
    if ext not in config.SUPPORTED_EXTS:
        await client.chat_postMessage(channel=channel, text=f"I can't index `{name}`. Supported: {', '.join(sorted(config.SUPPORTED_EXTS))}")
        return
    if info.get("size", 0) > config.MAX_FILE_BYTES:
        await client.chat_postMessage(channel=channel, text=f"`{name}` is too large (limit {config.MAX_FILE_BYTES // 2**20} MB).")
        return
    url = info.get("url_private_download")
    if not url:
        return

    scope = scope_for(context.team_id, channel)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / Path(name).name
        async with httpx.AsyncClient(follow_redirects=True, timeout=60) as http:
            r = await http.get(url, headers={"Authorization": f"Bearer {config.SLACK_BOT_TOKEN}"})
            r.raise_for_status()
            path.write_bytes(r.content)
        try:
            text = await asyncio.to_thread(extract_text, path)
            n = await asyncio.to_thread(kb.add_document, scope, name, text)
        except Exception as e:
            log.exception("ingest failed")
            await client.chat_postMessage(channel=channel, text=f":warning: Couldn't read `{name}`: {e}")
            return
    msg = (f":white_check_mark: Indexed `{name}` ({n} chunks). @mention me to ask about it."
           if n else f":warning: No text could be extracted from `{name}` (scanned PDF?).")
    await client.chat_postMessage(channel=channel, text=msg)


@app.command("/kb")
async def kb_command(ack, command, respond):
    await ack()
    if not allowed(command.get("user_id")):
        await respond("You're not authorized to use this bot.")
        return
    scope = scope_for(command["team_id"], command["channel_id"])
    parts = (command.get("text") or "").strip().split(maxsplit=1)
    sub = parts[0].lower() if parts else "help"
    arg = parts[1].strip() if len(parts) > 1 else ""

    if sub == "list":
        srcs = await asyncio.to_thread(kb.list_sources, scope)
        await respond("\n".join(f"• `{n}` ({c} chunks)" for n, c in srcs.items()) if srcs else "Nothing indexed in this channel yet.")
    elif sub == "delete":
        if not arg:
            await respond("Usage: `/kb delete <name>`")
            return
        await asyncio.to_thread(kb.delete_source, scope, arg)
        await respond(f"Removed `{arg}` (if it existed).")
    elif sub == "forget":
        for key in [k for k in agent.history if k.startswith(f"{command['channel_id']}:")]:
            agent.forget(key)
        await respond("Conversation memory cleared for this channel. Documents are untouched.")
    else:
        await respond(HELP)


async def main():
    handler = AsyncSocketModeHandler(app, config.SLACK_APP_TOKEN)
    log.info("Starting in Socket Mode")
    await handler.start_async()


if __name__ == "__main__":
    asyncio.run(main())
