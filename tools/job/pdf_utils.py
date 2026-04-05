from pathlib import Path

from assistant.config import is_debug_enabled


def extract_pdf_text(file_path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency: pypdf is required to extract text from PDF files. "
            "Install it with 'pip install pypdf' in your environment."
        ) from exc

    reader = PdfReader(str(file_path))
    text = ""
    if is_debug_enabled():
        print(f"DEBUG: Extracting text from PDF: {file_path}")
    for page in reader.pages:
        text += page.extract_text() or ""
        if is_debug_enabled():
            print(f"DEBUG: Extracted text from page {page}: {text[:100]}...")
    return text.strip()
