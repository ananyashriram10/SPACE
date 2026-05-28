#!/usr/bin/env python3
"""Prepare filtered prompt files for official baseline scripts."""

import argparse
import json
import os
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare official baseline prompt CSV/TXT files")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--artist", required=True)
    parser.add_argument("--artist_filter", default="")
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_txt", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    if not input_path.exists():
        cwd = Path.cwd()
        nearby = sorted(str(path) for path in (cwd / "data").glob("*.csv")) if (cwd / "data").is_dir() else []
        raise FileNotFoundError(
            f"Prompt CSV not found: {input_path}\n"
            f"Current working directory: {cwd}\n"
            f"CSV files under ./data: {nearby}\n"
            "Run: bash baselines/validate_baseline_inputs.sh"
        )

    df = pd.read_csv(input_path)
    if args.artist_filter:
        if "artist" not in df.columns:
            raise ValueError(f"{args.input_csv} has no artist column for filtering")
        wanted = args.artist_filter.strip().lower()
        df = df[df["artist"].astype(str).str.strip().str.lower() == wanted].reset_index(drop=True)

    required = {"case_number", "prompt", "evaluation_seed"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{args.input_csv} is missing required columns: {missing}")
    if df.empty:
        raise ValueError(f"No prompts left after filtering {args.input_csv} for {args.artist_filter!r}")

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_txt), exist_ok=True)
    os.makedirs(os.path.dirname(args.manifest), exist_ok=True)

    official_df = df[["case_number", "prompt", "evaluation_seed"]].copy()
    official_df.to_csv(args.out_csv, index=False)
    with open(args.out_txt, "w") as f:
        for prompt in official_df["prompt"]:
            f.write(str(prompt) + "\n")

    records = []
    for order, row in official_df.iterrows():
        case_number = int(row["case_number"])
        records.append(
            {
                "order": int(order),
                "case_number": case_number,
                "prompt": str(row["prompt"]),
                "evaluation_seed": int(row["evaluation_seed"]),
                "uce_filename": f"{case_number}_0.png",
                "concept_ablation_compvis_filename": f"{order:05d}.png",
            }
        )

    with open(args.manifest, "w") as f:
        json.dump(
            {
                "artist": args.artist,
                "artist_filter": args.artist_filter,
                "input_csv": args.input_csv,
                "out_csv": args.out_csv,
                "out_txt": args.out_txt,
                "num_prompts": len(records),
                "records": records,
            },
            f,
            indent=2,
        )

    print(f"Prepared {len(records)} prompts for {args.artist}")


if __name__ == "__main__":
    main()
