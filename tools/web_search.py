from base64 import b64decode
from binascii import Error as BinasciiError
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import requests

from tools.job.filesystem import save_text

SEARCH_URL = "https://html.duckduckgo.com/html/"
REQUEST_TIMEOUT = 30
DEFAULT_MAX_RESULTS = 5
MAX_ALLOWED_RESULTS = 10
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/135.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": SEARCH_URL,
    "Origin": "https://html.duckduckgo.com",
}
ANOMALY_MARKERS = (
    "anomaly-modal",
    "bots use duckduckgo too",
    "confirm this search was made by a human",
    "/anomaly.js",
)


class _DuckDuckGoResultParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._in_title = False
        self._current_href = ""
        self._title_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return

        attributes = {key: value or "" for key, value in attrs}
        css_class = attributes.get("class", "")
        href = attributes.get("href", "")
        if "result__a" in css_class and href:
            self._in_title = True
            self._current_href = href
            self._title_parts = []

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or not self._in_title:
            return

        title = unescape("".join(self._title_parts)).strip()
        href = _normalize_result_url(self._current_href)
        if title and href:
            self.results.append({"title": title, "url": href})
        self._in_title = False
        self._current_href = ""
        self._title_parts = []


def _normalize_result_url(raw_url: str) -> str:
    current = unescape(raw_url)
    seen: set[str] = set()

    while current and current not in seen:
        seen.add(current)
        parsed = urlparse(current)
        query = parse_qs(parsed.query)

        redirect_target = []
        for key in ("uddg", "u", "u3"):
            redirect_target = query.get(key, [])
            if redirect_target:
                current = _decode_redirect_value(redirect_target[0])
                break
        else:
            if current.startswith("//"):
                return f"https:{current}"
            return current

    if current.startswith("//"):
        return f"https:{current}"
    return current


def _decode_redirect_value(value: str) -> str:
    current = unescape(value)
    seen: set[str] = set()

    while current and current not in seen:
        seen.add(current)

        unquoted = unquote(current)
        if unquoted != current:
            current = unquoted
            continue

        try:
            padded = current + ("=" * (-len(current) % 4))
            decoded = b64decode(padded, validate=True).decode("utf-8")
        except (BinasciiError, UnicodeDecodeError, ValueError):
            break

        decoded_lower = decoded.lower()
        if decoded_lower.startswith(("http://", "https://", "http%3a", "https%3a")):
            current = decoded
            continue

        break

    return current


def _clamp_max_results(value: int | None) -> int:
    if value is None:
        return DEFAULT_MAX_RESULTS
    return max(1, min(int(value), MAX_ALLOWED_RESULTS))


def _looks_like_anomaly_page(html: str) -> bool:
    lowered = html.lower()
    return any(marker in lowered for marker in ANOMALY_MARKERS)


def search_web(
    query: str,
    max_results: int | None = None,
    output_path: str | None = None,
):
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("A search query is required.")

    result_limit = _clamp_max_results(max_results)
    response = requests.post(
        SEARCH_URL,
        data={"q": normalized_query},
        headers=REQUEST_HEADERS,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    if _looks_like_anomaly_page(response.text):
        raise RuntimeError(
            "DuckDuckGo blocked the automated search request with a human-verification challenge."
        )

    parser = _DuckDuckGoResultParser()
    parser.feed(response.text)
    results = parser.results[:result_limit]
    if not results:
        raise ValueError(f"No search results found for query: {normalized_query}")

    rendered_lines = [f"Found {len(results)} result(s) for: {normalized_query}\n"]
    for index, result in enumerate(results, start=1):
        rendered_lines.append(f"{index}. {result['title']}\n")
        rendered_lines.append(f"   URL: {result['url']}\n")

    rendered_output = "".join(rendered_lines)
    yield rendered_output

    if output_path:
        save_text(Path(output_path), rendered_output)
        yield f"Output written to: {output_path}\n"
