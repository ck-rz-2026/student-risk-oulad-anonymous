"""Assemble the paper: facts -> content blocks -> two-column PDF."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paper_facts
import paper_content
import paper_layout as L

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main_table(F):
    models = ["DecisionTree", "LR", "Spline-GAM", "RandomForest", "XGBoost", "LightGBM"]
    label = {"RandomForest": "Random Forest", "DecisionTree": "Decision Tree"}
    head = ["Model", "AUC", "AUPRC", "Brier", "ECE", "Cov / Prec %"]
    rows = [head]
    for m in models:
        mm = F["main"][m]["metrics"]
        ex = F.get("extra", {}).get(m, {})
        op = ex.get("op80", {}) if ex else {}
        rows.append([
            label.get(m, m),
            f"{mm['auc'][0]:.3f}",
            f"{mm['auprc'][0]:.3f}",
            f"{mm['brier'][0]:.3f}",
            f"{mm['ece'][0]:.3f}",
            f"{op.get('cohort_flagged', 0)*100:.1f} / {op.get('precision', 0)*100:.1f}",
        ])
    return rows


if __name__ == "__main__":
    F = paper_facts.facts()
    blocks = paper_content.build(F)
    # replace placeholder table
    for i, b in enumerate(blocks):
        if b[0] == "tab" and b[1] == "PLACEHOLDER_MAIN":
            blocks[i] = ("tab", main_table(F), b[2], b[3], b[4])
    out = os.path.join(BASE, "paper", "Interpretable_Prediction_of_Student_Academic_Performance.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    L.build(blocks, out)

    # drop a trailing blank page if the flow ended exactly at the frame bottom
    try:
        from pypdf import PdfReader, PdfWriter
        r = PdfReader(out)
        while len(r.pages) > 1:
            txt = (r.pages[-1].extract_text() or "").strip()
            txt = "".join(ch for ch in txt if ch.isalnum())
            if len(txt) > 40:
                break
            w = PdfWriter()
            for p in r.pages[:-1]:
                w.add_page(p)
            with open(out, "wb") as fh:
                w.write(fh)
            print("dropped blank final page; pages ->", len(r.pages) - 1)
            break
    except Exception as e:
        print("blank-page cleanup skipped:", e)

    print("output:", out)
