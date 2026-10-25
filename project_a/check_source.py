"""Verify that study packaging preserves the frozen runtime and queue function."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent.parent
    manifest = json.loads((root / "project_a/SOURCE_MANIFEST.json").read_text())
    failures = []
    for item in manifest["runtime_files"]:
        file = root / item["path"]
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != item["sha256"]:
            failures.append(item["path"])
    client = manifest["client_extract"]
    source = (root / client["path"]).read_text()
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == client["function"])
    function = ast.get_source_segment(source, node)
    if hashlib.sha256(function.encode()).hexdigest() != client["source_function_sha256"]:
        failures.append(client["path"] + ": function changed")
    print(json.dumps({"status": "fail" if failures else "pass",
                      "runtime_revision": manifest["runtime_revision"],
                      "checked_runtime_files": len(manifest["runtime_files"]),
                      "failures": failures}, indent=2))
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
