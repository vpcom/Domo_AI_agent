from assistant.config import is_debug_enabled
from tools.web_search import search_web


def run_web_search_workflow(
    query: str,
    max_results: int | None = None,
    output_path: str | None = None,
):
    yield "Starting web search workflow...\n"
    yield f"Query: {query}\n"
    if max_results is not None:
        yield f"Max results: {max_results}\n"
    if output_path:
        yield f"Resolved output: {output_path}\n"
    if is_debug_enabled():
        yield "DEBUG: debug mode is enabled.\n"

    yield from search_web(
        query=query,
        max_results=max_results,
        output_path=output_path,
    )
    yield "Workflow finished.\n"
