#!/usr/bin/env python3
"""Validate Van Gogh baseline and SPACE artifacts and write a run summary."""

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image


EXPECTED = 50
ARTIST = "Van Gogh"


def count_loadable_pngs(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    count = 0
    for path in sorted(directory.glob("*.png")):
        with Image.open(path) as img:
            img.verify()
        count += 1
    return count


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Van Gogh replication artifacts.")
    parser.add_argument("--write-summary", action="store_true")
    parser.add_argument("--commit", default="")
    parser.add_argument("--push-status", default="not requested")
    args = parser.parse_args()

    root = Path.cwd()
    paths = {
        "prompts": root / "data/vangogh_prompts.csv",
        "vanilla": root / "results/baseline/vangogh",
        "esd_x": root / "results/erased/vangogh",
        "uce_checkpoint": root / "baseline-models/uce/uce-Van_Gogh.safetensors",
        "uce_images": root / "results/uce/uce-Van_Gogh",
        "space_checkpoint": root / "space-models/sd/space-Van_Gogh.safetensors",
        "space_images": root / "results/space/space-Van_Gogh",
        "ca_official_delta": root / "baseline-models/concept_ablation_compvis/official_weights/concept_ablation-Van_Gogh.ckpt",
        "ca_trained_delta": root / "baseline-models/concept_ablation_compvis/deltas/concept_ablation-Van_Gogh.ckpt",
        "ca_images": root / "results/concept_ablation_compvis/concept_ablation-Van_Gogh/samples",
        "metrics_csv": root / "results/evaluation/metrics.csv",
        "ablation_table_tex": root / "results/evaluation/ablation_table.tex",
        "comparison_bars": root / "results/evaluation/comparison_bars.png",
        "comparison_png": root / "results/comparisons/van_gogh.png",
        "comparison_pdf": root / "results/comparisons/van_gogh.pdf",
    }

    errors: list[str] = []
    df = pd.read_csv(paths["prompts"]) if paths["prompts"].exists() else pd.DataFrame()
    counts = {
        "prompts": len(df),
        "vanilla_images": count_loadable_pngs(paths["vanilla"]),
        "esd_x_images": count_loadable_pngs(paths["esd_x"]),
        "uce_images": count_loadable_pngs(paths["uce_images"]),
        "space_images": count_loadable_pngs(paths["space_images"]),
        "ca_images": count_loadable_pngs(paths["ca_images"]),
    }

    require(counts["prompts"] == EXPECTED, f"Expected {EXPECTED} Van Gogh prompts, found {counts['prompts']}", errors)
    require(counts["vanilla_images"] == EXPECTED, f"Expected {EXPECTED} Vanilla images, found {counts['vanilla_images']}", errors)
    require(counts["esd_x_images"] == EXPECTED, f"Expected {EXPECTED} ESD-x images, found {counts['esd_x_images']}", errors)
    require(paths["uce_checkpoint"].exists(), f"Missing UCE checkpoint: {paths['uce_checkpoint']}", errors)
    require(counts["uce_images"] == EXPECTED, f"Expected {EXPECTED} UCE images, found {counts['uce_images']}", errors)
    require(paths["space_checkpoint"].exists(), f"Missing SPACE checkpoint: {paths['space_checkpoint']}", errors)
    require(counts["space_images"] == EXPECTED, f"Expected {EXPECTED} SPACE images, found {counts['space_images']}", errors)
    ca_delta = paths["ca_official_delta"] if paths["ca_official_delta"].exists() else paths["ca_trained_delta"]
    require(ca_delta.exists(), f"Missing CA delta: expected {paths['ca_official_delta']} or {paths['ca_trained_delta']}", errors)
    require(counts["ca_images"] == EXPECTED, f"Expected {EXPECTED} CA images, found {counts['ca_images']}", errors)
    for key in ["metrics_csv", "ablation_table_tex", "comparison_bars", "comparison_png", "comparison_pdf"]:
        require(paths[key].exists(), f"Missing report artifact: {paths[key]}", errors)

    summary = {
        "artist": ARTIST,
        "expected_count": EXPECTED,
        "counts": counts,
        "paths": {key: str(path.relative_to(root)) for key, path in paths.items()},
        "ca_delta_used": str(ca_delta.relative_to(root)) if ca_delta.exists() else "",
        "commit": args.commit,
        "push_status": args.push_status,
        "status": "failed" if errors else "ok",
        "errors": errors,
    }

    out_dir = root / "results/evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "van_gogh_run_summary.json"
    md_path = out_dir / "van_gogh_run_summary.md"

    if args.write_summary:
        json_path.write_text(json.dumps(summary, indent=2) + "\n")
        lines = [
            "# Van Gogh Baseline + SPACE Replication Summary",
            "",
            f"- Status: {summary['status']}",
            f"- Prompt count: {counts['prompts']} / {EXPECTED}",
            f"- Vanilla images: {counts['vanilla_images']} / {EXPECTED}",
            f"- ESD-x images: {counts['esd_x_images']} / {EXPECTED}",
            f"- UCE checkpoint: `{paths['uce_checkpoint'].relative_to(root)}`",
            f"- UCE images: {counts['uce_images']} / {EXPECTED}",
            f"- SPACE checkpoint: `{paths['space_checkpoint'].relative_to(root)}`",
            f"- SPACE images: {counts['space_images']} / {EXPECTED}",
            f"- CA delta: `{summary['ca_delta_used']}`",
            f"- CA images: {counts['ca_images']} / {EXPECTED}",
            f"- Comparison chart: `{paths['comparison_png'].relative_to(root)}`",
            f"- Comparison PDF: `{paths['comparison_pdf'].relative_to(root)}`",
            f"- Metrics CSV: `{paths['metrics_csv'].relative_to(root)}`",
            f"- Ablation table: `{paths['ablation_table_tex'].relative_to(root)}`",
            f"- Metric bars: `{paths['comparison_bars'].relative_to(root)}`",
            f"- Git commit: {args.commit or 'not committed yet'}",
            f"- Push status: {args.push_status}",
        ]
        if errors:
            lines.extend(["", "## Errors"])
            lines.extend(f"- {error}" for error in errors)
        md_path.write_text("\n".join(lines) + "\n")

    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
