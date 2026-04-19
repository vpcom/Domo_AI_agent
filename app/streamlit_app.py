from base64 import b64encode
from html import escape
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from assistant.domo_agent import (
    apply_execution_result,
    apply_turn_result,
    create_conversation_state,
    execute_pending_tool_call,
    new_turn_id,
    plan_chat_turn,
    reset_conversation_state,
)
from assistant.runtime import WORKFLOW_PARAMETER_KEYS
from assistant.schemas import ActivityEvent, ContextValue, ConversationState

APP_FILE = Path(__file__).resolve()
PROJECT_ROOT = APP_FILE.parents[1]
IMG_DIR = PROJECT_ROOT / "img"
USER_AVATAR = str(IMG_DIR / "user_blue.svg")
ASSISTANT_AVATAR = str(IMG_DIR / "domo_icon.png")

st.set_page_config(page_title="Personal Assistant", layout="wide")

yellow_logo = b64encode(
    (IMG_DIR / "domo_yellow.webp").read_bytes()).decode("ascii")
blue_logo = b64encode(
    (IMG_DIR / "domo_blue.webp").read_bytes()).decode("ascii")

st.markdown(
    """
    <style>
    .domo-section {
      background: #050816;
      color: #f8fafc;
      border-radius: 12px;
      padding: 0.95rem 1rem;
      margin-bottom: 0.85rem;
    }
    .domo-section-title {
      font-size: 0.82rem;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: #94a3b8;
      margin-bottom: 0.55rem;
    }
    .domo-section-line {
      color: #f8fafc;
      font-size: 0.95rem;
      line-height: 1.45;
      word-break: break-word;
    }
    .domo-muted {
      color: #94a3b8;
    }
    .domo-log-line {
      color: #e2e8f0;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 0.9rem;
      line-height: 1.5;
      white-space: pre-wrap;
      margin: 0;
    }
    .domo-chat-preview {
      white-space: pre-wrap;
      overflow: hidden;
      display: -webkit-box;
      -webkit-box-orient: vertical;
      -webkit-line-clamp: 10;
      line-height: 1.5;
      position: relative;
    }
    .domo-sticky-context {
      position: sticky;
      top: 1rem;
      align-self: flex-start;
    }
    .domo-panel-heading {
      margin: 0 0 1rem 0;
    }
    div[data-testid="stChatInput"] > div {
      border: 1px solid rgba(245, 191, 36, 0.45);
      border-radius: 0.85rem;
      transition: border-color 0.18s ease, box-shadow 0.18s ease;
    }
    div[data-testid="stChatInput"] textarea,
    div[data-testid="stChatInput"] input {
      border: none !important;
      box-shadow: none !important;
      outline: none !important;
    }
    div[data-testid="stChatInput"]:focus-within > div {
      border-color: #f5bf24;
      box-shadow: 0 0 0 0.22rem rgba(245, 191, 36, 0.22);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div id="domo-header" style="display:flex; align-items:center; gap:0.75rem; margin-bottom:1rem;">
      <img
        id="domo-logo"
        src="data:image/webp;base64,{yellow_logo}"
        width="64"
        style="display:block"
      />
      <div>
        <h1 style="margin:0;">Domo</h1>
        <div style="color:#475569;">Chat, retained context, and workflow activity in one session.</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

components.html(
    f"""
    <script>
    const yellowLogo = "data:image/webp;base64,{yellow_logo}";
    const blueLogo = "data:image/webp;base64,{blue_logo}";

    function candidateDocuments() {{
      return [window.document, window.parent?.document, window.parent?.parent?.document]
        .filter(Boolean);
    }}

    function findLogo() {{
      for (const doc of candidateDocuments()) {{
        const logo = doc.getElementById("domo-logo");
        if (logo) return logo;
      }}
      return null;
    }}

    function syncLogoWithConnectionState() {{
      const logo = findLogo();
      if (!logo) return;

      const isConnecting = candidateDocuments().some((doc) =>
        doc.body && /Connecting/i.test(doc.body.innerText || "")
      );
      const nextLogo = isConnecting ? blueLogo : yellowLogo;
      if (logo.src !== nextLogo) {{
        logo.src = nextLogo;
      }}
    }}

    function installStickyContext() {{
      for (const doc of candidateDocuments()) {{
        const anchor = doc.getElementById("domo-context-anchor");
        if (!anchor) continue;

        const container = anchor.closest('[data-testid="stElementContainer"]');

        if (container && !container.classList.contains("domo-sticky-context")) {{
          container.classList.add("domo-sticky-context");
        }}
      }}
    }}

    function refreshUi() {{
      syncLogoWithConnectionState();
      installStickyContext();
    }}

    refreshUi();
    setInterval(refreshUi, 500);
    </script>
    """,
    height=0,
)

if "conversation_state" not in st.session_state:
    st.session_state.conversation_state = create_conversation_state()

state: ConversationState = st.session_state.conversation_state


def _format_value(value) -> str:
    if value is None or value == "":
        return "<span class='domo-muted'>Not set</span>"
    if isinstance(value, bool):
        return "True" if value else "False"
    return escape(str(value))


def _context_keys_for_display(active_workflow: str | None) -> dict[str, list[str]]:
    request_keys = [
        "request_summary",
        "selected_workflow",
        "confirmation_state",
        "run_status",
        "open_question",
    ]
    parameter_keys = list(WORKFLOW_PARAMETER_KEYS.get(active_workflow, ()))
    execution_keys = ["last_output_folder", "last_error"]
    return {
        "Request": request_keys,
        "Parameters": parameter_keys,
        "Execution": execution_keys,
    }


def _render_text_section(title: str, lines: list[str]) -> None:
    st.markdown(_text_section_html(title, lines), unsafe_allow_html=True)


def _text_section_html(title: str, lines: list[str]) -> str:
    if not lines:
        body = "<div class='domo-section-line'><span class='domo-muted'>No retained values yet.</span></div>"
    else:
        body = "".join(
            f"<div class='domo-section-line'>{line}</div>"
            for line in lines
        )
    return (
        "<div class='domo-section'>"
        f"<div class='domo-section-title'>{escape(title)}</div>"
        f"{body}"
        "</div>"
    )


def _context_line(value: ContextValue) -> str | None:
    if value.value in (None, ""):
        return None
    rendered_value = value.value
    if isinstance(rendered_value, str) and len(rendered_value) > 160:
        rendered_value = rendered_value[:157] + "..."
    return f"{escape(value.label)}: {_format_value(rendered_value)}"


def _render_context_panel(current_state: ConversationState) -> None:
    active_workflow = current_state.context.request.get("selected_workflow")
    workflow_name = None if active_workflow is None else active_workflow.value
    keys_by_section = _context_keys_for_display(workflow_name)

    sections_html: list[str] = []
    for section_name, keys in keys_by_section.items():
        section_map = {
            "Request": current_state.context.request,
            "Parameters": current_state.context.parameters,
            "Execution": current_state.context.execution,
        }[section_name]
        lines: list[str] = []
        for key in keys:
            value = section_map.get(key)
            if value is None:
                continue
            line = _context_line(value)
            if line is not None:
                lines.append(line)
        sections_html.append(_text_section_html(section_name, lines))

    st.markdown(
        (
            "<div id='domo-context-anchor'></div>"
            "<div>"
            "<h3 class='domo-panel-heading'>Context</h3>"
            f"{''.join(sections_html)}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_chat_panel(current_state: ConversationState) -> None:
    st.subheader("Chat")
    if not current_state.messages:
        st.info("Start with a request. Domo will clarify the task, retain parameters, and ask for confirmation before running anything.")
        return

    for message in current_state.messages:
        avatar = USER_AVATAR if message.role == "user" else ASSISTANT_AVATAR
        with st.chat_message(message.role, avatar=avatar):
            if message.role == "user" and _should_collapse_user_message(message.content):
                st.markdown(
                    (
                        "<div class='domo-chat-preview'>"
                        f"{escape(message.content)}"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )
                with st.expander("..."):
                    st.markdown(message.content)
            else:
                st.markdown(message.content)


def _should_collapse_user_message(content: str) -> bool:
    non_empty_lines = [line for line in content.splitlines() if line.strip()]
    return len(non_empty_lines) > 7 or len(content) > 500


def _activity_log_line(event: ActivityEvent) -> str:
    timestamp = event.timestamp.astimezone().strftime("%H:%M:%S")
    category = event.category.capitalize()
    summary = escape(event.summary.rstrip("."))
    detail = f": {escape(event.detail)}" if event.detail else ""
    return f"{escape(timestamp)}  {escape(category)}: {summary}{detail}"


def _render_activity_panel(current_state: ConversationState) -> None:
    st.subheader("Activity Logs")
    if not current_state.activity_events:
        st.caption("No activity recorded yet.")
        return

    lines = [_activity_log_line(event)
             for event in current_state.activity_events]
    _render_text_section(
        "Activity Logs",
        [f"<span class='domo-log-line'>{line}</span>" for line in lines],
    )

    raw_events = [
        event for event in current_state.activity_events if event.raw_lines]
    for index, event in enumerate(raw_events, start=1):
        title = event.summary or f"Workflow Run {index}"
        with st.expander(f"Details {index}: {title}"):
            st.code("".join(event.raw_lines), language="text")


header_left, header_right = st.columns([5, 1])
with header_right:
    if st.button("Reset Conversation", use_container_width=True):
        reset_conversation_state(state)
        st.rerun()

chat_col, context_col = st.columns([1.8, 1.1], gap="large")

with chat_col:
    _render_chat_panel(state)
    prompt = st.chat_input(
        "Describe what you want to do.",
        disabled=state.is_executing,
    )

with context_col:
    _render_context_panel(state)

st.divider()
_render_activity_panel(state)

if prompt:
    turn_id = new_turn_id()
    result = plan_chat_turn(state, prompt, turn_id=turn_id)
    apply_turn_result(state, prompt, result, turn_id=turn_id)

    if result.turn_intent == "execute":
        with st.spinner("Running workflow..."):
            execution_result = execute_pending_tool_call(
                state, turn_id=turn_id)
        apply_execution_result(state, execution_result, turn_id=turn_id)

    st.rerun()
