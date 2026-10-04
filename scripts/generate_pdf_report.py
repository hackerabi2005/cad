"""
Generates the formal Track A Technical Report PDF from docs/report.md using ReportLab.
Ensures clean clinical styling, embedded evaluation tables, calibration plots,
and asserts page count <= 6 pages.
"""

import os
from pathlib import Path
from pypdf import PdfReader
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    KeepTogether,
    HRFlowable,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs"
REPORTS_DIR = BASE_DIR / "reports"
OUTPUT_PDF = DOCS_DIR / "report.pdf"


def build_pdf():
    print("Compiling docs/report.pdf with ReportLab...")
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#0284c7"),
        spaceAfter=10,
    )
    h1_style = ParagraphStyle(
        "H1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=10,
        spaceAfter=4,
    )
    h2_style = ParagraphStyle(
        "H2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=6,
        spaceAfter=3,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#334155"),
        spaceAfter=5,
    )
    bullet_style = ParagraphStyle(
        "Bullet",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#334155"),
        leftIndent=12,
        spaceAfter=3,
    )
    callout_style = ParagraphStyle(
        "Callout",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#b45309"),
        spaceAfter=4,
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("Cardio3D AI — Technical Report", title_style))
    story.append(
        Paragraph(
            "Multimodal AI Hackathon 2026 · Track A: Cardiovascular Risk Visualization & Prediction",
            subtitle_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceAfter=10))

    # 1. Executive Summary
    story.append(Paragraph("1. Executive Summary & Clinical Context", h1_style))
    story.append(
        Paragraph(
            "Coronary Artery Disease (CAD) remains the leading cause of death globally. Standard risk equations (Framingham, ASCVD) estimate abstract population percentages but cannot localize where arterial pathology is developing inside the human body. <b>Cardio3D AI</b> bridges predictive modeling and clinical anatomy by coupling machine learning (overall CAD and vessel-specific stenosis for LAD, LCX, RCA) with a client-side 3D interactive heart viewer. The system provides real-time risk color-mapping and exact additive SHAP feature attributions.",
            body_style,
        )
    )

    # 2. Dataset & Zero-Leakage Architecture
    story.append(Paragraph("2. Dataset Preprocessing & Zero-Leakage Architecture", h1_style))
    story.append(
        Paragraph(
            "The model is developed on the <b>UCI Extension of Z-Alizadeh Sani CAD Dataset</b> (303 patients, 59 columns):",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Accounting for 59 Columns vs. 54 Published Features:</b> 4 target columns (<code>Cath</code>, <code>LAD</code>, <code>LCX</code>, <code>RCA</code>), 1 constant column (<code>Exertional CP</code>, 100% 'N', offering zero predictive entropy and dropped), and 54 active physiological features (5 Demographics, 25 Symptoms/History, 7 ECG, 14 Lab biomarkers, 3 Echo metrics).",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Target Consistency Audit & Row 93 Alignment:</b> In the raw dataset, row 93 contains <code>LAD='Stenotic'</code>, <code>LCX='Normal'</code>, <code>RCA='Normal'</code>, but <code>Cath='Normal'</code>. By clinical definition, $\\ge 50\\%$ stenosis in any major vessel constitutes CAD. Row 93's target was aligned to CAD in the derived training pipeline, achieving 100% target consistency (<code>Cath == LAD | LCX | RCA</code> with 0 mismatches across 303 rows).",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Zero Leakage Guarantee:</b> All four targets and catheterization columns are strictly excluded from the input matrix <code>X</code>. All preprocessing (median/mode imputation, scaling, one-hot encoding) is contained inside scikit-learn <code>Pipeline</code> objects fitted strictly on training folds. Label-shuffled CV ROC-AUC confirmed random chance (0.50 ± 0.02).",
            bullet_style,
        )
    )

    # 3. Predictive Modeling & Evaluation Methodology
    story.append(Paragraph("3. Predictive Modeling & Cross-Validation Results", h1_style))
    story.append(
        Paragraph(
            "Evaluation used <b>Repeated Stratified 5-Fold Cross-Validation with 3 Repeats (15 total folds per candidate)</b> across Majority Baseline, Logistic Regression, Random Forest, and XGBoost:",
            body_style,
        )
    )

    # Table of Results
    table_data = [
        ["Target", "Model Selected", "ROC-AUC", "PR-AUC", "F1", "Recall", "Spec.", "Brier"],
        ["Cath (CAD)", "LogisticRegression", "0.929 ± 0.024", "0.965", "0.909", "0.934", "0.701", "0.098"],
        ["LAD (Anterior)", "RandomForest", "0.846 ± 0.049", "0.884", "0.819", "0.874", "0.638", "0.168"],
        ["LCX (Circumflex)", "XGBoost", "0.735 ± 0.060", "0.652", "0.541", "0.482", "0.803", "0.204"],
        ["RCA (Right Coronary)", "LogisticRegression", "0.733 ± 0.045", "0.655", "0.498", "0.451", "0.804", "0.202"],
    ]
    t = Table(table_data, colWidths=[75, 105, 75, 50, 45, 45, 45, 45])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 7.5),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
                ("TOPPADDING", (0, 0), (-1, 0), 4),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("ALIGN", (0, 1), (1, -1), "LEFT"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 7.5),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8fafc"), colors.HexColor("#f1f5f9")]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 6))

    # Thresholds
    story.append(
        Paragraph(
            "<b>Sensitivity-First Operating Thresholds:</b> In cardiovascular screening, missing an acute lesion is catastrophic. Operating points were determined on out-of-fold predictions to enforce $\\ge 90\\%$ sensitivity: Cath (thresh=0.38, sens=96.8%), LAD (thresh=0.42, sens=92.1%), LCX (thresh=0.25, sens=90.8%), RCA (thresh=0.26, sens=90.4%).",
            body_style,
        )
    )

    # 4. Risk Coherence Policy
    story.append(Paragraph("4. Logical Risk Coherence Policy", h1_style))
    story.append(
        Paragraph(
            "Because CAD is the union of individual vessels ($P(\\text{CAD}) \\ge \\max(P(\\text{LAD}), P(\\text{LCX}), P(\\text{RCA}))$), independent model outputs can violate logical consistency. Out-of-fold predictions showed a <b>21.5% violation rate</b>. Cardio3D AI automatically enforces the coherence display rule: $P(\\text{CAD})_{\\text{coherent}} = \\max(P(\\text{CAD}), \\max(P(\\text{vessels})))$, while displaying both raw and coherent scores transparently.",
            body_style,
        )
    )

    # 5. Explainable AI & SHAP
    story.append(Paragraph("5. Explainable AI & Exact Additive SHAP Pipeline", h1_style))
    story.append(
        Paragraph(
            "The system delivers real-time feature attributions using <b>LinearSHAP</b> (Logistic Regression) and <b>TreeSHAP</b> (Random Forest, XGBoost):",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Exact Additivity:</b> Base value plus sum of feature attributions identically matches the model's raw score (numerical error $\\le 10^{-5}$ across test cases).",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Parent Aggregation:</b> One-hot encoded dummy attributes are summed back to parent measurements ($\text{SHAP}(\text{Parent}) = \\sum \\text{SHAP}(\\text{dummies})$), so clinicians see concise, single-variable physiological attributions.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Top Population Risk Drivers:</b> Across the cohort, the most influential CAD risk drivers are <b>Typical Chest Pain</b>, <b>Age</b>, <b>Fasting Blood Sugar (FBS)</b>, <b>Regional Wall Motion Abnormality (RWMA)</b> count, and <b>Ejection Fraction (EF-TTE)</b>.",
            bullet_style,
        )
    )

    # Calibration Plot image if exists
    calib_img_path = REPORTS_DIR / "calibration_curves.png"
    if calib_img_path.exists():
        story.append(Spacer(1, 4))
        story.append(Paragraph("<b>Model Calibration Reliability Curves:</b>", h2_style))
        story.append(Image(str(calib_img_path), width=420, height=210))
        story.append(Spacer(1, 4))

    # 6. 3D Anatomical Visualization Pipeline
    story.append(Paragraph("6. 3D Anatomical Pipeline & WebGL Optimization", h1_style))
    story.append(
        Paragraph(
            "Built with <b>Three.js and React Three Fiber</b> utilizing BodyParts3D meshes (Branch A: Separate Artery Meshes):",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Segment Geometry:</b> Anatomically distinct meshes for the LAD (9 branches/trunks), LCX (4 branches/trunks), RCA (10 branches/trunks), and Ascending Aorta are centered at origin and scaled to viewport coordinates.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Triangle Budget Optimization:</b> Chamber and myocardium walls were decimated using quadric decimation by 70%, yielding <b>83,600 total triangles</b> (safely below the $\\le 100,000$ triangle limit). File size is 1.67 MB.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Software Rendering Benchmark:</b> Evaluated under software rendering (<code>--disable-gpu</code>), achieving a <b>median frame time of 16.5 ms (~60.6 FPS)</b>, exceeding the 20 FPS budget by 3x.",
            bullet_style,
        )
    )

    # 7. Methodological Decisions: TimesFM & TabPFN
    story.append(Paragraph("7. Architectural Decisions: TimesFM & TabPFN Rationale", h1_style))
    story.append(
        Paragraph(
            "• <b>TimesFM Considered & Rejected:</b> TimesFM 3.0 is a foundation model architected strictly for ordered, time-aligned sequential time series. The dataset consists of 303 independent static clinical snapshots with no longitudinal temporal dimension. Imposing a pseudo-time axis over tabular clinical rows creates spurious correlations and degrades predictive validity.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>TabPFN Evaluation:</b> TabPFN requires non-standard runtime dependencies and authentication tokens unsuitable for zero-friction deployment. Standard tuned tree and linear models achieved high discriminative power (Cath AUC 0.929, LAD AUC 0.846) with deterministic sub-millisecond inference and exact SHAP guarantees.",
            bullet_style,
        )
    )

    # 8. Limitations, Clinical Safety & Licenses
    story.append(Paragraph("8. Clinical Safety, Limitations & Attribution", h1_style))
    story.append(
        Paragraph(
            "• <b>Clinical Safety Disclaimer:</b> The application embeds prominent, non-dismissible disclaimer banners and footers: <i>'Decision support & educational use only — not a substitute for formal diagnostic imaging, catheterization, or physician judgment.'</i>",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Limitations:</b> Single-center retrospective dataset (n=303) requires multi-center validation prior to clinical adoption. <code>Region RWMA</code> is an echocardiographic wall motion abnormality count and does not represent spatial 3D lesion coordinates.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Attributions & Licenses:</b> Dataset: UCI Machine Learning Repository (CC BY 4.0). Anatomical 3D Mesh: BodyParts3D © The Database Center for Life Science (CC BY-SA 2.1 JP).",
            bullet_style,
        )
    )

    doc.build(story)

    # Verify page count <= 6
    reader = PdfReader(str(OUTPUT_PDF))
    page_count = len(reader.pages)
    print(f"Generated PDF: {OUTPUT_PDF} (Total Pages: {page_count})")
    assert page_count <= 6, f"PDF page count {page_count} exceeds 6-page limit!"
    print(f"Verification: PDF page count {page_count} <= 6 pages passed!")


if __name__ == "__main__":
    build_pdf()
