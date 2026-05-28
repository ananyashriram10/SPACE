#!/usr/bin/env python3
"""
Sort results/baseline/niche/ images into per-artist subfolders
by reading the artist column from short_niche_art_prompts.csv.

Before:
  results/baseline/niche/0_0.png   (Tyler Edlin)
  results/baseline/niche/20_0.png  (Thomas Kinkade)
  ...

After:
  results/baseline/tyler_edlin/0_0.png
  results/baseline/thomas_kinkade/20_0.png
  ...
"""
import pandas as pd
import shutil
import os
from tqdm import tqdm

SRC_DIR   = "results/baseline/niche"
CSV_PATH  = "data/short_niche_art_prompts.csv"
OUT_BASE  = "results/baseline"

def artist_to_folder(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace(":", "").replace("-", "_")

df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows from {CSV_PATH}")
print(f"Artists: {df['artist'].unique().tolist()}\n")

missing, moved = 0, 0

for _, row in tqdm(df.iterrows(), total=len(df), desc="sorting", unit="img"):
    case   = int(row["case_number"])
    artist = str(row["artist"])
    folder = artist_to_folder(artist)

    src = os.path.join(SRC_DIR, f"{case}_0.png")
    dst_dir = os.path.join(OUT_BASE, folder)
    dst = os.path.join(dst_dir, f"{case}_0.png")

    if not os.path.exists(src):
        print(f"  MISSING: {src}")
        missing += 1
        continue

    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy2(src, dst)
    moved += 1

print(f"\nDone. {moved} images copied, {missing} missing.")
print(f"\nResult folders:")
for artist in df["artist"].unique():
    folder = artist_to_folder(artist)
    path = os.path.join(OUT_BASE, folder)
    if os.path.isdir(path):
        count = len([f for f in os.listdir(path) if f.endswith(".png")])
        print(f"  {path}/ — {count} images")
