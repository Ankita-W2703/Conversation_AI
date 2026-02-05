
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
# from evaluation import results

def generate_pdf_report(metrics, filename="report.pdf"):
    """
    metrics: dict
    filename: str or Path
    """
    filename = str(filename)

    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("Hybrid RAG Evaluation Report", styles["Title"]))

    for k, v in metrics.items():
        content.append(Paragraph(f"<b>{k}:</b> {v}", styles["Normal"]))

    doc.build(content)

def generate_html_report(metrics, filename="report.html"):
    filename = str(filename)

    html = "<html><body><h1>Hybrid RAG Evaluation Report</h1><ul>"
    for k, v in metrics.items():
        html += f"<li><b>{k}</b>: {v}</li>"
    html += "</ul></body></html>"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)

# generate_pdf_report(results)
# generate_html_report(results)