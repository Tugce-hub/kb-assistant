"""Slack bot: answers @mentions and DMs in-thread, with sources and feedback buttons.

Socket Mode is used so the bot needs no public ingress (works behind the
corporate firewall / in a private k8s namespace).

Env: SLACK_BOT_TOKEN (xoxb-...), SLACK_APP_TOKEN (xapp-..., connections:write),
     ANTHROPIC_API_KEY. App manifest: deploy/slack-manifest.yaml.

    python -m kbassist.slack_app
"""

from __future__ import annotations

import logging
import os
import re

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from kbassist.generation import Answer
from kbassist.service import KnowledgeService

log = logging.getLogger("kbassist.slack")

MENTION = re.compile(r"<@[A-Z0-9]+>\s*")


def build_blocks(svc: KnowledgeService, answer: Answer, trace_id: str) -> list[dict]:
    blocks: list[dict] = [{"type": "markdown", "text": answer.text[:11000]}]
    cited = [(i, h) for i, h in enumerate(answer.hits, start=1) if i in answer.citations]
    if cited:
        links = "  ".join(
            f"<{svc.permalink(h.chunk)}|[{i}] {h.chunk.path}:{h.chunk.start_line}>" for i, h in cited
        )
        blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": links[:3000]}]})
    blocks.append(
        {
            "type": "actions",
            "block_id": f"feedback:{trace_id}",
            "elements": [
                {"type": "button", "action_id": "feedback_up", "text": {"type": "plain_text", "text": "👍 Helpful"}, "value": "up"},
                {"type": "button", "action_id": "feedback_down", "text": {"type": "plain_text", "text": "👎 Not helpful"}, "value": "down"},
            ],
        }
    )
    return blocks


def create_app(svc: KnowledgeService) -> App:
    app = App(token=os.environ["SLACK_BOT_TOKEN"])

    def handle(event: dict, say, client) -> None:
        text = MENTION.sub("", event.get("text", "")).strip()
        if not text:
            return
        user = event["user"]
        thread_ts = event.get("thread_ts") or event["ts"]
        try:
            client.reactions_add(channel=event["channel"], timestamp=event["ts"], name="eyes")
        except Exception:  # missing reactions:write scope is not fatal
            pass
        try:
            answer, trace = svc.ask(text, user, channel=event["channel"])
            say(text=answer.text[:3000], blocks=build_blocks(svc, answer, trace.trace_id), thread_ts=thread_ts)
        except Exception:
            log.exception("failed to answer")
            say(text="Sorry, something went wrong while answering. The error has been logged.", thread_ts=thread_ts)

    @app.event("app_mention")
    def on_mention(event, say, client):
        handle(event, say, client)

    @app.event("message")
    def on_message(event, say, client):
        # Only direct messages; channel messages must @mention the bot.
        if event.get("channel_type") == "im" and not event.get("bot_id") and not event.get("subtype"):
            handle(event, say, client)

    @app.action(re.compile("feedback_(up|down)"))
    def on_feedback(ack, body, action):
        ack()
        trace_id = action["block_id"].split(":", 1)[-1]
        svc.audit.write("feedback", user=body["user"]["id"], trace_id=trace_id, rating=action["value"])

    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    svc = KnowledgeService()
    # Warm up: load index and models before accepting traffic.
    svc.search("warmup", "eval-runner", k=1)
    SocketModeHandler(create_app(svc), os.environ["SLACK_APP_TOKEN"]).start()


if __name__ == "__main__":
    main()
