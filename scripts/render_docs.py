"""Render numeric docs from artifacts — single source of truth (G3/G5/G6).
Reads ml/artifacts/metrics.json, writes reports/model_results.md table + README metrics block.
Usage: python scripts/render_docs.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
m = json.load(open(ROOT / "ml" / "artifacts" / "metrics.json"))
table = m.get("full_operating_metrics_table", {})

lines = [
    "# Model Results (generated — do not hand-edit)",
    "",
    "Source: `ml/artifacts/metrics.json` (nested CV, LogitEnsemble unified family).",
    "",
]

for op_key, op_title in [("balanced_05", "Balanced Operating Threshold (0.50)"), ("screening_high_sens", "Sensitivity-First Screening Threshold")]:
    lines.append(f"### {op_title}\n")
    lines.append("| Target | Threshold | ROC-AUC (95% CI) | PR-AUC (95% CI) | Accuracy (95% CI) | Sensitivity (95% CI) | Specificity (95% CI) | F1-Score | Brier |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for t in ("Cath", "LAD", "LCX", "RCA"):
        d = table.get(t, {})
        b = d.get(op_key, {})
        thresh = d.get("operating_threshold", 0.50) if op_key == "screening_high_sens" else 0.50
        def fmt(k):
            item = b.get(k, {})
            if isinstance(item, dict) and "point_estimate" in item:
                pe = item["point_estimate"]
                ci = item.get("ci_95")
                if ci:
                    return f"{pe:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
                return f"{pe:.3f}"
            elif isinstance(item, (int, float)):
                return f"{item:.3f}"
            return "—"
        
        lines.append(f"| **{t}** | {thresh:.3f} | {fmt('roc_auc')} | {fmt('pr_auc')} | {fmt('accuracy')} | {fmt('recall')} | {fmt('specificity')} | {fmt('f1')} | {fmt('brier')} |")
    lines.append("")

out = ROOT / "reports" / "model_results.md"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {out}")

# README metrics block between METRICS markers (create if missing markers -> append note)
readme = ROOT / "README.md"
txt = readme.read_text(encoding="utf-8")
block = "\n".join(lines[3:])
if "<!-- METRICS:START" in txt and "METRICS:END -->" in txt:
    import re
    txt = re.sub(r"<!-- METRICS:START.*?METRICS:END -->", f"<!-- METRICS:START -->\n{block}\n<!-- METRICS:END -->", txt, flags=re.S)
    readme.write_text(txt, encoding="utf-8")
    print("README metrics block refreshed")
else:
    print("README has no METRICS markers — table kept in reports/model_results.md only (no drift introduced)")
