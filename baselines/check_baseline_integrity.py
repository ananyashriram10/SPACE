#!/usr/bin/env python3
"""Lightweight integrity checks for generated official baseline outputs."""

import argparse
import glob
import json
import os
from pathlib import Path

from PIL import Image


def verify_image(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with Image.open(path) as img:
        img.verify()
    with Image.open(path) as img:
        if img.size[0] <= 0 or img.size[1] <= 0:
            raise ValueError(f"Invalid image dimensions for {path}: {img.size}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check official baseline output integrity")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--mode", required=True, choices=["uce", "concept-ablation-compvis"])
    parser.add_argument("--allow_missing", action="store_true")
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(str(image_dir))

    key = "uce_filename" if args.mode == "uce" else "concept_ablation_compvis_filename"
    missing = []
    checked = 0
    for record in manifest["records"]:
        image_path = image_dir / record[key]
        if not image_path.exists():
            missing.append(str(image_path))
            continue
        verify_image(str(image_path))
        checked += 1

    extra_pngs = sorted(glob.glob(str(image_dir / "*.png")))
    if missing and not args.allow_missing:
        raise FileNotFoundError("Missing images:\n" + "\n".join(missing[:20]))

    print(
        f"Integrity OK for {args.mode}: checked={checked}, "
        f"missing={len(missing)}, pngs_in_dir={len(extra_pngs)}, image_dir={image_dir}"
    )


if __name__ == "__main__":
    main()
