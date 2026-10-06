"""Command line: build the index, search, ask.

    python -m kbassist.cli index
    python -m kbassist.cli search "how do I disable timeouts" --mode hybrid_rerank
    python -m kbassist.cli ask "httpx'te varsayılan timeout nedir?"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from kbassist.config import get_settings
from kbassist.index.store import build_index, save_index
from kbassist.retrieval import MODES
from kbassist.service import KnowledgeService


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="kbassist")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="sync sources and (re)build the index")
    p_index.add_argument("--no-sync", action="store_true", help="use existing checkouts as-is")

    p_search = sub.add_parser("search")
    p_search.add_argument("query")
    p_search.add_argument("--mode", choices=MODES, default="hybrid_rerank")
    p_search.add_argument("-k", type=int, default=5)
    p_search.add_argument("--user", default="eval-runner")

    p_ask = sub.add_parser("ask")
    p_ask.add_argument("question")
    p_ask.add_argument("--user", default="eval-runner")

    args = parser.parse_args(argv)
    settings = get_settings()

    if args.cmd == "index":
        index = build_index(settings, sync=not args.no_sync)
        save_index(index, settings.path("paths.index_dir"))
        print(json.dumps(index.manifest, indent=2))
        return 0

    svc = KnowledgeService(settings)
    if args.cmd == "search":
        hits, trace = svc.search(args.query, args.user, k=args.k, mode=args.mode)
        for i, h in enumerate(hits, 1):
            c = h.chunk
            print(f"{i}. [{h.score:.3f}] {c.path}:{c.start_line}-{c.end_line}  {c.title}  {h.ranks}")
        print(json.dumps(trace.to_dict(), indent=2))
    elif args.cmd == "ask":
        answer, trace = svc.ask(args.question, args.user)
        print(answer.text, "\n")
        for i, h in enumerate(answer.hits, 1):
            mark = "*" if i in answer.citations else " "
            print(f"{mark}[{i}] {svc.permalink(h.chunk)}")
        print(json.dumps(trace.to_dict(svc.pricing), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
