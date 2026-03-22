from base64 import b64encode
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from assistant.domo_agent import plan_domo, run_domo_tool
from assistant.schemas import PlannedToolCall

APP_FILE = Path(__file__).resolve()
PROJECT_ROOT = APP_FILE.parents[1]
IMG_DIR = PROJECT_ROOT / "img"


st.set_page_config(page_title="Personal Assistant")

yellow_logo = b64encode(
    (IMG_DIR / "domo_yellow.webp").read_bytes()).decode("ascii")
blue_logo = b64encode(
    (IMG_DIR / "domo_blue.webp").read_bytes()).decode("ascii")

st.markdown(
    f"""
    <div id="domo-header" style="display:flex; align-items:center; gap:0.75rem; margin-bottom:1rem;">
      <img
        id="domo-logo"
        src="data:image/webp;base64,{yellow_logo}"
        width="64"
        style="display:block"
      />
      <h1 style="margin:0;">Domo</h1>
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
      logo.src = isConnecting ? blueLogo : yellowLogo;
    }}

    syncLogoWithConnectionState();
    setInterval(syncLogoWithConnectionState, 500);
    </script>
    """,
    height=0,
)

components.html(
    """
    <script>
    function installShiftEnterSubmit() {
      const doc = window.parent.document;
      const textarea = doc.querySelector('textarea[aria-label="What can I do for you today?"]');
      const runButton = Array.from(doc.querySelectorAll("button")).find(
        (button) => button.innerText.trim() === "Run"
      );

      if (!textarea || !runButton || textarea.dataset.shiftEnterBound === "true") {
        return;
      }

      textarea.dataset.shiftEnterBound = "true";
      textarea.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && event.shiftKey) {
          event.preventDefault();
          runButton.click();
        }
      });
    }

    installShiftEnterSubmit();
    setInterval(installShiftEnterSubmit, 500);
    </script>
    """,
    height=0,
)

with st.form("domo-input-form", clear_on_submit=False):
    user_input = st.text_area(
        "What can I do for you today?")
    submitted = st.form_submit_button("Run")

if "domo_output" not in st.session_state:
    st.session_state.domo_output = ""

if "pending_tool_call" not in st.session_state:
    st.session_state.pending_tool_call = None

if submitted:
    outcome = plan_domo(user_input)

    if outcome.kind in {"respond", "error"}:
        st.session_state.pending_tool_call = None
        st.session_state.domo_output = outcome.message
    elif outcome.kind == "approval_required" and outcome.tool_call is not None:
        st.session_state.pending_tool_call = outcome.tool_call.model_dump()
        st.session_state.domo_output = outcome.message
    elif outcome.kind == "tool" and outcome.tool_call is not None:
        st.session_state.pending_tool_call = None
        placeholder = st.empty()
        output = ""

        for chunk in run_domo_tool(outcome.tool_call):
            output += chunk
            placeholder.text(output)

        st.session_state.domo_output = output

if st.session_state.pending_tool_call:
    st.info("Pending approval required before executing the proposed tool action.")
    pending = PlannedToolCall.model_validate(
        st.session_state.pending_tool_call)
    st.code(
        f"Tool: {pending.tool_name}\n"
        f"Parameters: {pending.parameters.model_dump()}",
        language="text",
    )

    approve_col, cancel_col = st.columns(2)
    with approve_col:
        if st.button("Approve and Run"):
            placeholder = st.empty()
            output = ""

            for chunk in run_domo_tool(pending):
                output += chunk
                placeholder.text(output)

            st.session_state.domo_output = output
            st.session_state.pending_tool_call = None
            st.rerun()
    with cancel_col:
        if st.button("Cancel"):
            st.session_state.pending_tool_call = None
            st.session_state.domo_output = "Tool execution cancelled."
            st.rerun()

if st.session_state.domo_output:
    st.text(st.session_state.domo_output)
