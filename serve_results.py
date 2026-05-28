#!/usr/bin/env python3
"""
Serve comparison grids in a browser via RunPod's HTTP port.
Access at: https://<pod-id>-8080.proxy.runpod.net

Usage:
  python3 serve_results.py
"""
import os
import http.server
import socketserver

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>ESD-x Replication Results</title>
<style>
  body {{ background: #0f0f0f; color: #e0e0e0; font-family: sans-serif; margin: 0; padding: 20px; }}
  h1 {{ color: #fff; margin-bottom: 4px; }}
  h2 {{ color: #ff9966; margin: 32px 0 8px; border-bottom: 1px solid #333; padding-bottom: 6px; }}
  .artist-grid {{ display: flex; flex-direction: column; gap: 16px; }}
  .row {{ display: flex; gap: 12px; align-items: flex-start; background: #1a1a1a; border-radius: 8px; padding: 12px; }}
  .prompt {{ flex: 0 0 280px; font-size: 12px; color: #bbb; line-height: 1.5; }}
  .prompt .case {{ font-size: 10px; color: #555; margin-bottom: 4px; }}
  .images {{ display: flex; gap: 10px; }}
  .img-wrap {{ text-align: center; }}
  .img-wrap img {{ width: 256px; height: 256px; object-fit: cover; border-radius: 4px; display: block; }}
  .img-wrap span {{ font-size: 10px; color: #666; margin-top: 4px; display: block; }}
  .vanilla span {{ color: #88aaff; }}
  .erased span {{ color: #ff9966; }}
  nav {{ position: sticky; top: 0; background: #0f0f0f; padding: 8px 0 12px; z-index: 10; }}
  nav a {{ color: #ff9966; margin-right: 16px; text-decoration: none; font-size: 13px; }}
  nav a:hover {{ text-decoration: underline; }}
  .missing {{ width: 256px; height: 256px; background: #222; border-radius: 4px; display:flex; align-items:center; justify-content:center; color:#444; font-size:12px; }}
</style>
</head>
<body>
<h1>ESD-x Replication — Vanilla SD v1.4 vs ESD-x</h1>
<nav>{nav}</nav>
{body}
</body></html>"""

ROW_TEMPLATE = """
<div class="row">
  <div class="prompt"><div class="case">#{case}</div>{prompt}</div>
  <div class="images">
    <div class="img-wrap vanilla">{baseline_img}<span>Vanilla SD v1.4</span></div>
    <div class="img-wrap erased">{erased_img}<span>ESD-x erased</span></div>
  </div>
</div>"""

def img_tag(path, web_path):
    if os.path.exists(path):
        return f'<img src="/{web_path}" loading="lazy">'
    return '<div class="missing">missing</div>'

ARTISTS = [
    {"name": "Kelly McKernan",  "csv": "data/kelly_prompts.csv",               "baseline": "results/baseline/kelly",          "erased": "results/erased/kelly"},
    {"name": "Van Gogh",        "csv": "data/vangogh_prompts.csv",              "baseline": "results/baseline/vangogh",        "erased": "results/erased/vangogh"},
    {"name": "Tyler Edlin",     "csv": "data/short_niche_art_prompts.csv",      "baseline": "results/baseline/tyler_edlin",    "erased": "results/erased/tyler_edlin",    "filter": "Tyler Edlin"},
    {"name": "Thomas Kinkade",  "csv": "data/short_niche_art_prompts.csv",      "baseline": "results/baseline/thomas_kinkade", "erased": "results/erased/thomas_kinkade", "filter": "Thomas Kinkade"},
    {"name": "Kilian Eng",      "csv": "data/short_niche_art_prompts.csv",      "baseline": "results/baseline/kilian_eng",     "erased": "results/erased/kilian_eng",     "filter": "Kilian Eng"},
    {"name": "Ajin: Demi Human","csv": "data/short_niche_art_prompts.csv",      "baseline": "results/baseline/ajin_demi_human","erased": "results/erased/ajin",           "filter": "Ajin: Demi Human"},
]

def build_html():
    import pandas as pd

    nav_links, sections = [], []

    for cfg in ARTISTS:
        slug = cfg["name"].lower().replace(" ", "-").replace(":", "").replace("--", "-")
        df = pd.read_csv(cfg["csv"])
        if "filter" in cfg:
            df = df[df["artist"].str.strip() == cfg["filter"]]

        rows_html = []
        for _, row in df.iterrows():
            case = int(row["case_number"])
            prompt = str(row["prompt"])
            b_path = f"{cfg['baseline']}/{case}_0.png"
            e_path = f"{cfg['erased']}/{case}_0.png"
            rows_html.append(ROW_TEMPLATE.format(
                case=case,
                prompt=prompt,
                baseline_img=img_tag(b_path, b_path),
                erased_img=img_tag(e_path, e_path),
            ))

        nav_links.append(f'<a href="#{slug}">{cfg["name"]}</a>')
        sections.append(f'<h2 id="{slug}">{cfg["name"]}</h2><div class="artist-grid">{"".join(rows_html)}</div>')

    return HTML_TEMPLATE.format(nav=" ".join(nav_links), body="\n".join(sections))

# Write static HTML file
os.makedirs("results/comparisons", exist_ok=True)
html = build_html()
html_path = "results/comparisons/index.html"
with open(html_path, "w") as f:
    f.write(html)
print(f"Generated {html_path}")

# Serve from repo root so image paths resolve
PORT = 8080
os.chdir(".")
Handler = http.server.SimpleHTTPRequestHandler
Handler.extensions_map[".html"] = "text/html"

print(f"\nServing on port {PORT}")
print(f"Open: https://<your-pod-id>-{PORT}.proxy.runpod.net/results/comparisons/index.html")
print("Ctrl+C to stop\n")

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    httpd.serve_forever()
