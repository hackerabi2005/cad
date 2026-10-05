"""
Generates the formal Track A Technical Report PDF from docs/report.md using ReportLab.
Ensures clean clinical styling, embedded evaluation tables, calibration plots,
ROC curves, architecture diagram, UI screenshot, and asserts page count <= 6 pages.
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
    PageBreak,
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
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=3,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#0284c7"),
        spaceAfter=6,
    )
    h1_style = ParagraphStyle(
        "H1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=14.5,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=8,
        spaceAfter=3,
    )
    h2_style = ParagraphStyle(
        "H2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=5,
        spaceAfter=2,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#334155"),
        spaceAfter=4,
    )
    bullet_style = ParagraphStyle(
        "Bullet",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#334155"),
        leftIndent=10,
        spaceAfter=2,
    )
    callout_style = ParagraphStyle(
        "Callout",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#b45309"),
        spaceAfter=3,
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
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceAfter=6))

    # 1. Executive Summary
    story.append(Paragraph("1. Executive Summary & Clinical Context", h1_style))
    story.append(
        Paragraph(
            "Coronary Artery Disease (CAD) is the leading cause of death globally. Standard risk equations (Framingham, ASCVD) estimate abstract population percentages but cannot localize where arterial pathology is developing inside the human body. <b>Cardio3D AI</b> bridges predictive modeling and clinical anatomy by coupling machine learning (overall CAD and vessel-specific significant stenosis for LAD, LCX, RCA) with a client-side 3D interactive heart viewer. The system provides real-time risk color-mapping and exact additive SHAP feature attributions.",
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
            "• <b>59 Columns vs. 54 Published Features:</b> 4 target columns (<code>Cath</code>, <code>LAD</code>, <code>LCX</code>, <code>RCA</code>), 1 constant column (<code>Exertional CP</code>, 100% 'N', offering zero predictive entropy and dropped), and 54 active physiological features (5 Demographics, 25 Symptoms/History, 7 ECG, 14 Lab biomarkers, 3 Echo metrics).",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Target Consistency Audit & Row 93 Alignment:</b> In the raw dataset, row 93 contains <code>LAD='Stenotic'</code>, <code>LCX='Normal'</code>, <code>RCA='Normal'</code>, but <code>Cath='Normal'</code>. By clinical definition, &ge; 50% stenosis in any major vessel constitutes CAD. To match the dataset's own definition (CAD = &ge; 1 stenotic vessel), row 93's target was aligned to CAD in derived training data (<code>Cath_aligned = Cath | LAD | LCX | RCA</code>), leaving the raw data file untouched and verified by SHA256. This aligns the released distribution from <b>216 CAD / 87 Normal &rarr; 217 CAD / 86 Normal</b> (0 mismatches across 303 rows).",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Zero Leakage Guarantee:</b> All four targets and catheterization columns are strictly excluded from the input matrix <code>X</code>. All preprocessing (median/mode imputation, scaling, one-hot encoding) is contained inside scikit-learn <code>Pipeline</code> objects fitted strictly on training folds. Label-shuffled CV ROC-AUC confirmed random chance (0.50 ± 0.02).",
            bullet_style,
        )
    )

    # Architecture Diagram
    arch_img_path = REPORTS_DIR / "architecture_diagram.png"
    if arch_img_path.exists():
        story.append(Spacer(1, 4))
        story.append(Paragraph("<b>Figure 1: End-to-End System Architecture & Data Flow</b>", h2_style))
        story.append(Image(str(arch_img_path), width=510, height=210))
        story.append(Spacer(1, 4))

    story.append(PageBreak())

    # 3. Predictive Modeling & Evaluation Methodology
    story.append(Paragraph("3. Predictive Modeling & Cross-Validation Results", h1_style))
    story.append(
        Paragraph(
            "Evaluation used <b>Repeated Stratified 5-Fold Cross-Validation with 3 Repeats (15 total folds per candidate)</b> across Baseline, Logistic Regression, Random Forest, and XGBoost:",
            body_style,
        )
    )

    # Table 1: Default Cutoff (0.50)
    story.append(Paragraph("<b>Table 1: Cross-Validation Benchmarks at Default Cutoff (0.50) (Mean of 15 Folds)</b>", h2_style))
    table_data = [
        ["Target", "Model Selected", "ROC-AUC", "PR-AUC", "F1", "Recall", "Spec.", "PPV", "NPV", "Brier"],
        ["Cath (CAD)", "LogisticRegression", "0.929 ± 0.024", "0.971", "0.909", "0.934", "0.698", "0.887", "0.812", "0.098"],
        ["LAD (Anterior)", "RandomForest", "0.846 ± 0.049", "0.883", "0.819", "0.874", "0.635", "0.773", "0.792", "0.168"],
        ["LCX (Circumflex)", "XGBoost", "0.735 ± 0.060", "0.615", "0.541", "0.482", "0.806", "0.628", "0.706", "0.204"],
        ["RCA (Right Coronary)", "LogisticRegression", "0.733 ± 0.045", "0.617", "0.498", "0.451", "0.799", "0.571", "0.710", "0.202"],
    ]
    t = Table(table_data, colWidths=[70, 95, 75, 45, 40, 42, 42, 42, 42, 42])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 7),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 2.5),
                ("TOPPADDING", (0, 0), (-1, 0), 2.5),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("ALIGN", (0, 1), (1, -1), "LEFT"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 7),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8fafc"), colors.HexColor("#f1f5f9")]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 4))

    # Table 2: Sensitivity-First Operating Thresholds
    story.append(Paragraph("<b>Table 2: Sensitivity-First Operating Thresholds (Tuned on Out-of-Fold Predictions)</b>", h2_style))
    op_table_data = [
        ["Target", "Model", "Default Cutoff", "Operating Cutoff", "Pooled Sens.", "Pooled Spec.", "Pooled PPV", "Pooled NPV"],
        ["Cath", "LogisticRegression", "0.50 (93.5% / 69.8%)", "0.611", "90.3%", "82.6%", "92.9%", "77.2%"],
        ["LAD", "RandomForest", "0.50 (87.6% / 63.5%)", "0.469", "90.4%", "60.3%", "76.2%", "81.7%"],
        ["LCX", "XGBoost", "0.50 (48.7% / 82.1%)", "0.217", "90.8%", "37.0%", "48.2%", "86.1%"],
        ["RCA", "LogisticRegression", "0.50 (43.9% / 80.4%)", "0.213", "90.4%", "39.2%", "47.3%", "87.1%"],
    ]
    t2 = Table(op_table_data, colWidths=[65, 95, 95, 75, 50, 50, 50, 50])
    t2.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 7),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 2.5),
                ("TOPPADDING", (0, 0), (-1, 0), 2.5),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("ALIGN", (0, 1), (1, -1), "LEFT"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 7),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8fafc"), colors.HexColor("#f1f5f9")]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ]
        )
    )
    story.append(t2)
    story.append(Spacer(1, 4))

    # Table 3: Nested-CV Operating Performance
    story.append(Paragraph("<b>Table 3: Nested Cross-Validation Operating Performance (Per-Fold Out-of-Sample)</b>", h2_style))
    nested_table_data = [
        ["Target", "Model Selected", "Operating Cutoff", "Nested-CV Sensitivity", "Nested-CV Specificity", "Nested-CV PPV", "Nested-CV NPV"],
        ["Cath", "LogisticRegression", "0.611", "90.2% ± 3.9%", "80.6% ± 6.7%", "92.2% ± 3.3%", "77.1% ± 8.6%"],
        ["LAD", "RandomForest", "0.469", "88.2% ± 7.2%", "62.2% ± 8.0%", "76.7% ± 6.1%", "79.9% ± 9.5%"],
        ["LCX", "XGBoost", "0.217", "91.6% ± 6.3%", "35.1% ± 7.9%", "47.9% ± 4.4%", "86.9% ± 8.8%"],
        ["RCA", "LogisticRegression", "0.213", "89.2% ± 6.9%", "38.7% ± 7.9%", "46.9% ± 4.8%", "86.9% ± 8.5%"],
    ]
    t3 = Table(nested_table_data, colWidths=[65, 95, 75, 80, 80, 65, 65])
    t3.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0369a1")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 7),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 2.5),
                ("TOPPADDING", (0, 0), (-1, 0), 2.5),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("ALIGN", (0, 1), (1, -1), "LEFT"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 7),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8fafc"), colors.HexColor("#f1f5f9")]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ]
        )
    )
    story.append(t3)
    story.append(Spacer(1, 3))

    story.append(
        Paragraph(
            "<b>Clinical Operating Tradeoff:</b> Because LCX and RCA targets exhibit moderate discrimination (ROC-AUC &asymp; 0.73), enforcing &ge; 90% sensitivity intentionally trades specificity to 35.1% for LCX and 38.7% for RCA. In cardiovascular screening, a false positive prompts confirmatory non-invasive imaging (CCTA), whereas a false negative risks untreated significant stenosis.",
            callout_style,
        )
    )

    # ROC and Calibration Curve Images side-by-side or stacked
    roc_img_path = REPORTS_DIR / "roc_curves.png"
    calib_img_path = REPORTS_DIR / "calibration_curves.png"
    if roc_img_path.exists() and calib_img_path.exists():
        story.append(Spacer(1, 4))
        story.append(Paragraph("<b>Figure 2: Receiver Operating Characteristic (ROC) & Calibration Reliability Curves</b>", h2_style))
        img_table = Table(
            [[Image(str(roc_img_path), width=250, height=200), Image(str(calib_img_path), width=250, height=200)]],
            colWidths=[255, 255],
        )
        img_table.setStyle(
            TableStyle(
                [
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ]
            )
        )
        story.append(img_table)
        story.append(Spacer(1, 4))

    story.append(PageBreak())

    # 4. Risk Coherence Policy
    story.append(Paragraph("4. Logical Risk Coherence Policy & Quantitative Evaluation", h1_style))
    story.append(
        Paragraph(
            "Because CAD is the clinical union of coronary stenosis ($P(\\text{CAD}) \\ge \\max(P(\\text{LAD}), P(\\text{LCX}), P(\\text{RCA}))$), independent model outputs can violate logical consistency. Out-of-fold cross-validation predictions were evaluated quantitatively: Raw CAD achieved ROC-AUC = 0.9302 and Brier = 0.0967; coherent CAD achieved ROC-AUC = 0.9291 and Brier = 0.1017 (AUC drop = 0.0012 &le; 0.010, Brier delta = +0.0050 &le; +0.010). The criteria were met and the coherence rule is <b>retained</b>. The CAD operating threshold is calibrated directly on the displayed score (0.611). The API transparently returns both <code>raw_prob</code> and <code>prob</code>.",
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
            "• <b>Exact Additivity:</b> Base value plus sum of feature attributions identically matches the model's raw score (numerical error &lt; 10<sup>-4</sup> across all presets and dataset test samples).",
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

    # 6. 3D Anatomical Visualization Pipeline
    story.append(Paragraph("6. 3D Anatomical Pipeline & Interactive Dashboard", h1_style))
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
            "• <b>Triangle Budget Optimization:</b> Chamber and myocardium walls were decimated using quadric decimation by 70%, yielding <b>83,600 total triangles</b> (safely below the &le; 100,000 triangle limit). File size is 1.67 MB.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Measured Frame-Rate Benchmark:</b> Evaluated under pure CPU software rendering (<code>--disable-gpu</code> at 1280&times;720 on 13th Gen Intel Core i5-13420H), achieving a <b>mean frame time of 49.3 ms (~20.3 FPS continuous orbit, p95 = 53.7 ms)</b>, recorded in <code>reports/fps_benchmark.json</code>. On a discrete GPU, rendering executes at the display refresh rate (not benchmarked).",
            bullet_style,
        )
    )

    # UI Screenshot
    ui_img_path = REPORTS_DIR / "screenshots" / "01_initial_dashboard.png"
    if ui_img_path.exists():
        story.append(Spacer(1, 4))
        story.append(Paragraph("<b>Figure 3: Interactive Clinical Dashboard & 3D Anatomical Heart Viewer</b>", h2_style))
        story.append(Image(str(ui_img_path), width=510, height=270))
        story.append(Spacer(1, 4))

    story.append(PageBreak())

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
            "• <b>TabPFN Not Evaluated:</b> TabPFN v2.5+ weights require non-commercial licensing agreements and interactive browser download authentication unsuitable for a self-contained, reproducible pipeline. In addition, TabPFN does not provide real-time exact additive TreeSHAP explainability. Tuned clinical models achieve high discriminative power (Cath AUC 0.929, LAD AUC 0.846) with sub-millisecond deterministic inference.",
            bullet_style,
        )
    )

    # 8. Limitations, Clinical Safety & Licenses
    story.append(Paragraph("8. Clinical Safety, Limitations & Attribution", h1_style))
    story.append(
        Paragraph(
            "• <b>Clinical Safety Disclaimer:</b> The application embeds prominent, non-dismissible disclaimer banners and footers: <i>'Decision support & educational use only &mdash; not a substitute for formal diagnostic imaging, catheterization, or physician judgment.'</i> Prototype is not validated for acute emergency triage.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Limitations:</b> Single-center retrospective dataset (n=303 from Tehran) requires multi-center external validation prior to clinical adoption. LCX labels have low precision under high-sensitivity tuning. <code>Region RWMA</code> is an echocardiographic wall motion abnormality count and does not represent spatial 3D lesion coordinates.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Attributions & Licenses:</b> Software: MIT License (see <code>LICENSE</code>). Third-Party Notices: <code>THIRD_PARTY_NOTICES.md</code>. Dataset: UCI Machine Learning Repository (CC BY 4.0). Anatomical 3D Mesh: BodyParts3D &copy; The Database Center for Life Science (CC BY-SA 2.1 JP).",
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
