#!/usr/bin/env python3
"""Record official baseline provenance as JSON."""

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone


def git_output(repo_dir: str, args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", "-C", repo_dir, *args], text=True).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Write official baseline provenance JSON")
    parser.add_argument("--method", required=True)
    parser.add_argument("--artist", required=True)
    parser.add_argument("--run_type", required=True, choices=["official-paper", "official-diffusers", "fair-eval"])
    parser.add_argument("--repo_url", required=True)
    parser.add_argument("--repo_dir", required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--outputs", nargs="*", default=[])
    parser.add_argument("--prompt_manifest", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": args.method,
        "artist": args.artist,
        "run_type": args.run_type,
        "official_repo_url": args.repo_url,
        "official_repo_dir": args.repo_dir,
        "official_repo_commit": git_output(args.repo_dir, ["rev-parse", "HEAD"]),
        "official_repo_dirty_status": git_output(args.repo_dir, ["status", "--short"]),
        "command": args.command,
        "outputs": args.outputs,
        "prompt_manifest": args.prompt_manifest,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote provenance: {args.out}")


if __name__ == "__main__":
    main()
