"""Run the local API; managed UE scenes start only on an explicit UI request."""

import argparse
from pathlib import Path
import uvicorn
from .app import create_app
from .config import OperatorConfig


def main():
    parser = argparse.ArgumentParser(description="FlyAgent flight workbench")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--root", type=Path, default=OperatorConfig().root)
    parser.add_argument("--maps", default=OperatorConfig().maps_config)
    parser.add_argument("--env", default=".env")
    args = parser.parse_args()
    app = create_app(OperatorConfig(root=args.root.resolve(), maps_config=args.maps, env_path=args.env))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
