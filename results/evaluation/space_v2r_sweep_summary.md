# SPACE-v2R Van Gogh Sweep

Compared cached Vanilla SD v1.4 and ESD-x outputs against SPACE-v2R candidates.

Primary selection rule: first require `style_target_rate <= 0.32`, then choose the best preservation tradeoff.

## Selected Candidate

- Method: `SPACE-v2R e175`
- style_target_rate: `0.22`
- style_drop: `0.0648393253326415`
- clip_drop: `0.052369546508789`
- LPIPS: `0.6368701916933059`
- FID: `262.59571808190924`
- DINO similarity: `0.5328062808513642`

## Artifacts

- Metrics: `results/evaluation/metrics.csv`
- Table: `results/evaluation/ablation_table.tex`
- Bars: `results/evaluation/comparison_bars.png`
- Grid: `results/comparisons/van_gogh.png`
- Summary JSON: `results/evaluation/space_v2r_sweep_summary.json`
