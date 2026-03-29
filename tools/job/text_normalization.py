import html
import re


TAG_BLOCK_BREAKS = (
    "address",
    "article",
    "aside",
    "blockquote",
    "br",
    "dd",
    "div",
    "dl",
    "dt",
    "fieldset",
    "figcaption",
    "figure",
    "footer",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "section",
    "table",
    "tr",
    "ul",
)


def decode_html_entities(text: str, max_passes: int = 3) -> str:
    decoded = text
    for _ in range(max_passes):
        next_value = html.unescape(decoded)
        if next_value == decoded:
            break
        decoded = next_value
    return decoded


def normalize_job_posting_text(text: str) -> str:
    normalized = decode_html_entities(text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u00a0", " ")

    # Remove clearly non-content blocks before stripping tags.
    normalized = re.sub(
        r"<(script|style)\b[^>]*>.*?</\1>",
        "",
        normalized,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Preserve some structure before removing the remaining tags.
    normalized = re.sub(r"(?i)<li\b[^>]*>", "\n- ", normalized)
    normalized = re.sub(r"(?i)</li\s*>", "\n", normalized)
    normalized = re.sub(r"(?i)<br\s*/?>", "\n", normalized)

    for tag in TAG_BLOCK_BREAKS:
        normalized = re.sub(rf"(?i)</?{tag}\b[^>]*>", "\n", normalized)

    normalized = re.sub(r"(?i)<[^>]+>", "", normalized)
    normalized = decode_html_entities(normalized)

    normalized = normalized.replace("•", "-")
    normalized = re.sub(r"[ \t]+\n", "\n", normalized)
    normalized = re.sub(r"\n[ \t]+", "\n", normalized)
    normalized = re.sub(r"[ \t]{2,}", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)

    lines = [line.strip() for line in normalized.splitlines()]
    cleaned = "\n".join(line for line in lines if line)

    return cleaned.strip()
