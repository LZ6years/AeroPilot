"""Manual experience-library administration; no flight lifecycle hooks.

Run with F:/conda_env/flyagent/python.exe. Model connections are constructed here,
not inside the independent library. Commands emit UTF-8-compatible JSON.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from plugins.research.flight_experience.store import Store
from plugins.research.flight_experience.maintenance import Worker, publish


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init")
    imp = commands.add_parser("import")
    imp.add_argument("paths", type=Path, nargs="+")
    imp.add_argument("--instruction-id")
    enqueue = commands.add_parser("enqueue")
    enqueue.add_argument("--ids", nargs="+")
    enqueue.add_argument("--all", action="store_true")
    enqueue.add_argument("--request-key")
    worker = commands.add_parser("worker")
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--once", action="store_true")
    worker.add_argument("--env", type=Path, default=ROOT / ".env")
    worker.add_argument("--reflection-model")
    worker.add_argument("--review-model")
    worker.add_argument('--timeout',type=float,default=180)
    search = commands.add_parser("search")
    search.add_argument("query")
    search.add_argument("--scope", choices=("production", "experiment"), default="production")
    search.add_argument("--maintenance", action="store_true")
    search.add_argument("--filters", default="{}", help="JSON tag equality filters")
    search.add_argument("--limit", type=int, default=3)
    for name in ("jobs", "proposals", "list", "audit"):
        commands.add_parser(name)
    show = commands.add_parser("show")
    show.add_argument("id")
    show.add_argument("--version", type=int)
    pub = commands.add_parser("publish")
    pub.add_argument("proposal_id")
    for name in ("withdraw", "archive"):
        cmd = commands.add_parser(name)
        cmd.add_argument("id")
        cmd.add_argument("--actor", required=True)
        cmd.add_argument("--reason", required=True)
    pub.add_argument("--actor", required=True)
    pub.add_argument("--reason", required=True)
    args = parser.parse_args()
    store = Store(args.db)
    if args.command == "init":
        store.initialize()
        result = {"database": str(store.path)}
    elif args.command == "import":
        result = [store.import_experiment(p, instruction_id=args.instruction_id) for p in args.paths]
    elif args.command == "enqueue":
        if bool(args.ids) == args.all:
            parser.error("Choose exactly one of --ids or --all")
        result = store.enqueue(store.list_ids() if args.all else args.ids, request_key=args.request_key)
    elif args.command == "worker":
        from src.llm import create_model_client
        from src.agent.config import load_model_config
        model = load_model_config(args.env)
        reflection = replace(model.to_provider_settings(), model_name=args.reflection_model or model.model_name,timeout_sec=args.timeout,provider_timeout_retries=1)
        review = replace(model.to_provider_settings(), model_name=args.review_model or model.model_name,timeout_sec=args.timeout,provider_timeout_retries=1)
        parameters = model.thinking_parameters()
        instance = Worker(store, create_model_client(reflection), create_model_client(review), args.output, parameters=parameters)
        while True:
            result = instance.run_once()
            if args.once:
                break
            if result is not None:
                print(json.dumps(result, ensure_ascii=False), flush=True)
            else:
                time.sleep(2)
    elif args.command == "search":
        result = store.search(args.query, filters=json.loads(args.filters), scope=args.scope, limit=args.limit, maintenance=args.maintenance)
    elif args.command == "show":
        result = store.get(args.id, args.version)
    elif args.command == "list":
        result = store.list_ids()
    elif args.command == "jobs":
        result = store.jobs()
    elif args.command == "proposals":
        result = store.proposals()
    elif args.command == "audit":
        result = store.audit_sources()
    elif args.command == "publish":
        result = publish(store, args.proposal_id, actor=args.actor, reason=args.reason)
    else:
        store.set_state(args.id, "withdrawn" if args.command == "withdraw" else "archived", actor=args.actor, reason=args.reason)
        result = {"id": args.id, "state": args.command}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if isinstance(result, dict) and result.get("status") == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
