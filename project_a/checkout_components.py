"""Fetch pinned public components for the learning workspace; default is dry-run."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Create missing checkouts")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    components = json.loads((root / "project_a/COMPONENTS.json").read_text())["components"]
    for component in components:
        url = component["repo_url"]
        if url.endswith("/sglang"):
            continue
        if url not in {"https://github.com/donghyq/dynamo", "https://github.com/donghyq/Mooncake"}:
            parser.error("component URL is outside the declared public forks")
        target = root / ".project-a-components" / url.rsplit("/", 1)[1].lower()
        revision = component["revision"]
        if target.exists():
            result = subprocess.run(["git", "-C", str(target), "rev-parse", "HEAD"], capture_output=True, text=True)
            if result.returncode or result.stdout.strip() != revision:
                parser.error(f"existing checkout differs; inspect it manually: {target.name}")
            print(f"Already at pinned revision: {target.name} {revision}")
            continue
        commands = [
            ["git", "clone", "--filter=blob:none", "--no-checkout", "--single-branch", "--branch", component["branch"], url + ".git", str(target)],
            ["git", "-C", str(target), "checkout", "--detach", revision],
        ]
        for command in commands:
            print(shlex.join(command), flush=True)
            if args.execute:
                target.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
