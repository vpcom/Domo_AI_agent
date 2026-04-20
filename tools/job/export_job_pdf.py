try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing dependency 'reportlab'. Install project dependencies with "
        "`pip install -r requirements.txt`."
    ) from exc

from tools.job.filesystem import ensure_new_file_path
from tools.job.models import JobState


def run(state: JobState) -> None:
    print(f"[pdf] reading cleaned_file={state.cleaned_file}")
    content = state.cleaned_file.read_text(encoding="utf-8")
    print(f"[pdf] content characters={len(content)}")

    ensure_new_file_path(state.pdf_file)
    print(f"[pdf] writing pdf_file={state.pdf_file}")
    pdf = canvas.Canvas(str(state.pdf_file), pagesize=A4)
    width, height = A4

    left_margin = 50
    top_margin = height - 50
    line_height = 16
    max_chars = 95

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(left_margin, top_margin, "Job Description")

    pdf.setFont("Helvetica", 11)
    y = top_margin - 30

    for paragraph in content.split("\n"):
        words = paragraph.split()
        line = ""

        if not words:
            y = _advance_or_new_page(pdf, y - line_height, top_margin)
            continue

        for word in words:
            test_line = f"{line} {word}".strip()

            if len(test_line) <= max_chars:
                line = test_line
                continue

            pdf.drawString(left_margin, y, line)
            y = _advance_or_new_page(pdf, y - line_height, top_margin)
            line = word

        if line:
            pdf.drawString(left_margin, y, line)
            y = _advance_or_new_page(pdf, y - line_height, top_margin)

        y = _advance_or_new_page(pdf, y - 4, top_margin)

    pdf.save()
    print(f"[pdf] wrote pdf_file={state.pdf_file}")


def _advance_or_new_page(pdf, y, top_margin):
    if y >= 50:
        return y

    pdf.showPage()
    pdf.setFont("Helvetica", 11)
    return top_margin
