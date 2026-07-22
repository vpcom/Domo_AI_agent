"""Streamlit UI for the deterministic Domo agent."""

from base64 import b64encode
from html import escape
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from assistant.domo_agent import (
    build_state_view_model,
    create_agent_state,
    create_chat_history,
    create_ui_events,
    handle_user_message,
    reset_session,
)
from assistant.schemas import AgentState, ChatMessage, UiEvent


APP_FILE = Path(__file__).resolve()
PROJECT_ROOT = APP_FILE.parents[1]
IMG_DIR = PROJECT_ROOT / "img"
USER_AVATAR = str(IMG_DIR / "user_blue.svg")
ASSISTANT_AVATAR = str(IMG_DIR / "domo_icon.png")

st.set_page_config(page_title="Personal Assistant", layout="wide")

yellow_logo = b64encode((IMG_DIR / "domo_yellow.webp").read_bytes()).decode("ascii")
blue_logo = b64encode((IMG_DIR / "domo_blue.webp").read_bytes()).decode("ascii")

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
        <div style="color:#475569;">Your private digital assistant</div>
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

if "agent_state" not in st.session_state:
    st.session_state.agent_state = create_agent_state()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = create_chat_history()
if "ui_events" not in st.session_state:
    st.session_state.ui_events = create_ui_events()

state: AgentState = st.session_state.agent_state
chat_history: list[ChatMessage] = st.session_state.chat_history
ui_events: list[UiEvent] = st.session_state.ui_events


def _text_section_html(title: str, body: str) -> str:
    """Render a styled section used in the right-hand state panel."""

    return (
        "<div class='domo-section'>"
        f"<div class='domo-section-title'>{escape(title)}</div>"
        f"<div class='domo-section-line'>{body}</div>"
        "</div>"
    )


def _render_state_panel(current_state: AgentState) -> None:
    """Render a derived, read-only view of the deterministic agent state."""

    state_view = build_state_view_model(current_state)

    plan_lines = [
        (
            f"Step {item['step_id']}: "
            f"[{escape(str(item['type']))}] "
            f"{escape(str(item['tool_name']))} "
            f"({escape(str(item['status']))})"
        )
        for item in state_view["plan"]
    ]
    if not plan_lines:
        plan_lines = ["<span class='domo-muted'>No plan prepared yet.</span>"]

    artifact_lines = [
        f"{escape(artifact['kind'])}: {escape(artifact['path'])}"
        for artifact in state_view["memory"]["artifacts"]
    ]
    if not artifact_lines:
        artifact_lines = ["<span class='domo-muted'>No artifacts yet.</span>"]

    sections = [
        _text_section_html("Status", escape(str(state_view["status"]))),
        _text_section_html(
            "Goal",
            (
                f"User input: {escape(state_view['goal']['user_input'])}<br/>"
                f"Normalized: {escape(state_view['goal']['normalized_goal'])}"
            ),
        ),
        _text_section_html("Current Step", escape(str(state_view["current_step"]))),
        _text_section_html("Plan", "<br/>".join(plan_lines)),
        _text_section_html("Artifacts", "<br/>".join(artifact_lines)),
        _text_section_html(
            "Last Error",
            escape(str(state_view["last_error"]))
            if state_view["last_error"]
            else "<span class='domo-muted'>None</span>",
        ),
    ]

    st.markdown(
        (
            "<div id='domo-context-anchor'></div>"
            "<div>"
            "<h3 class='domo-panel-heading'>Agent State</h3>"
            f"{''.join(sections)}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_chat_panel(messages: list[ChatMessage]) -> None:
    """Render the chat column and collapse long user messages."""

    st.subheader("Chat")
    if not messages:
        st.info(
            "Describe a task. Domo will prepare a deterministic plan and execute it when policy allows."
        )
        return

    for message in messages:
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


def _render_activity_panel(events: list[UiEvent]) -> None:
    """Render UI-owned planner, execution, and state events."""

    st.subheader("Activity Logs")
    if not events:
        st.caption("No activity recorded yet.")
        return

    for event in events:
        timestamp = event.timestamp.astimezone().strftime("%H:%M:%S")
        detail = f": {escape(event.detail)}" if event.detail else ""
        line = (
            f"{escape(timestamp)}  "
            f"{escape(event.category.capitalize())}: "
            f"{escape(event.message)}{detail}"
        )
        st.markdown(
            f"<div class='domo-section-line domo-log-line'>{line}</div>",
            unsafe_allow_html=True,
        )
        if event.expanded_text:
            with st.expander(_activity_expander_label(event)):
                language = "json" if _looks_like_json(event.expanded_text) else "text"
                st.code(event.expanded_text, language=language)


def _should_collapse_user_message(content: str) -> bool:
    """Collapse long user messages to keep the chat column readable."""

    non_empty_lines = [line for line in content.splitlines() if line.strip()]
    return len(non_empty_lines) > 7 or len(content) > 500


def _activity_expander_label(event: UiEvent) -> str:
    """Return a stable expander label for detailed activity entries."""

    if event.category == "state":
        return "State"
    if "prompt" in event.message.lower():
        return "Prompt"
    if "response" in event.message.lower():
        return "Response"
    return "Details"


def _looks_like_json(value: str) -> bool:
    """Use JSON syntax highlighting when the payload appears to be structured."""

    stripped = value.strip()
    return stripped.startswith("{") or stripped.startswith("[")


header_left, header_right = st.columns([5, 1])
with header_right:
    if st.button("Reset Conversation", use_container_width=True):
        reset_session(state, chat_history, ui_events)
        st.rerun()

chat_col, context_col = st.columns([1.8, 1.1], gap="large")

with chat_col:
    _render_chat_panel(chat_history)
    prompt = st.chat_input(
        "Describe what you want to do.",
        disabled=state.status == "executing",
    )

with context_col:
    _render_state_panel(state)

st.divider()
_render_activity_panel(ui_events)

if prompt:
    with chat_col:
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            with st.spinner("Working..."):
                handle_user_message(state, chat_history, ui_events, prompt)
    st.rerun()
