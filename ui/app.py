"""Streamlit UI for the AI Text Detector backend.

Pick a model, paste text, and get a human/AI verdict. The backend keeps a
single active model at a time, so selecting a different one here triggers a
`POST /v1/models/switch` before detection.

Run with:
    streamlit run ui/app.py
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

DEFAULT_BACKEND_URL = os.environ.get("BACKEND_BASE_URL", "http://localhost:8000")
MAX_INPUT_CHARS = 20_000

VERDICT_DISPLAY: dict[str, tuple[str, str]] = {
    "ai": ("🤖 Likely AI-generated", "#d9534f"),
    "human": ("👤 Likely human-written", "#5cb85c"),
    "unknown": ("🤷 Uncertain", "#f0ad4e"),
}


def _client(base_url: str) -> httpx.Client:
    return httpx.Client(base_url=base_url.rstrip("/"), timeout=httpx.Timeout(connect=5, read=120, write=10, pool=5))


def fetch_models(base_url: str) -> tuple[str, list[dict]]:
    """Return (active_name, available_models) from the backend."""
    with _client(base_url) as client:
        resp = client.get("/v1/models")
        resp.raise_for_status()
        data = resp.json()
    return data["active"], data["available"]


def switch_model(base_url: str, name: str) -> dict:
    with _client(base_url) as client:
        resp = client.post("/v1/models/switch", json={"name": name})
        resp.raise_for_status()
        return resp.json()["active"]


def detect(base_url: str, text: str) -> dict:
    with _client(base_url) as client:
        resp = client.post("/v1/detect", json={"text": text})
        resp.raise_for_status()
        return resp.json()


def _error_detail(exc: httpx.HTTPStatusError) -> str:
    try:
        body = exc.response.json()
        if isinstance(body, dict):
            return str(body.get("detail") or body.get("message") or body)
    except Exception:
        pass
    return exc.response.text or str(exc)


# ----------------------------------------------------------------------- layout

st.set_page_config(page_title="AI Text Detector", page_icon="🤖", layout="centered")
st.title("🤖 AI Text Detector")
st.caption("Detect whether a piece of text was written by a human or an AI.")

with st.sidebar:
    st.header("Settings")
    base_url = st.text_input("Backend URL", value=DEFAULT_BACKEND_URL)

    if st.button("🔄 Refresh models", use_container_width=True):
        st.session_state.pop("models", None)

    if "models" not in st.session_state:
        try:
            active, available = fetch_models(base_url)
            st.session_state["models"] = available
            st.session_state["active_model"] = active
        except Exception as exc:  # noqa: BLE001 — surface any connection issue to the user
            st.error(f"Could not reach backend: {exc}")
            st.stop()

    available = st.session_state["models"]
    active = st.session_state["active_model"]
    names = [m["name"] for m in available]

    selected = st.selectbox(
        "Model",
        options=names,
        index=names.index(active) if active in names else 0,
        help="Selecting a different model switches the backend's active model before detection.",
    )

    info = next((m for m in available if m["name"] == selected), None)
    if info is not None:
        loaded = "✅ loaded" if info.get("loaded") else "⬇️ loads on first use"
        st.markdown(f"**Active:** `{active}`")
        bits = [loaded]
        if info.get("version"):
            bits.append(f"v{info['version']}")
        if info.get("device"):
            bits.append(info["device"])
        st.caption(" · ".join(bits))

text = st.text_area(
    "Text to analyze",
    height=260,
    max_chars=MAX_INPUT_CHARS,
    placeholder="Paste the text you want to check here…",
)
st.caption(f"{len(text)} / {MAX_INPUT_CHARS} characters")

if st.button("Analyze", type="primary", disabled=not text.strip()):
    try:
        if selected != st.session_state.get("active_model"):
            with st.spinner(f"Switching to model '{selected}'…"):
                new_active = switch_model(base_url, selected)
                st.session_state["active_model"] = new_active["name"]

        with st.spinner("Analyzing…"):
            result = detect(base_url, text)
    except httpx.HTTPStatusError as exc:
        st.error(f"Backend error ({exc.response.status_code}): {_error_detail(exc)}")
        st.stop()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Request failed: {exc}")
        st.stop()

    verdict = result["verdict"]
    label, color = VERDICT_DISPLAY.get(verdict, (verdict, "#999"))
    st.markdown(
        f"<h2 style='color:{color};margin-bottom:0'>{label}</h2>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("AI probability", f"{result['ai_probability'] * 100:.1f}%")
    c2.metric("Human probability", f"{result['human_probability'] * 100:.1f}%")
    c3.metric("Confidence", f"{result['confidence'] * 100:.1f}%")

    st.progress(result["ai_probability"], text="AI probability")

    diag = result.get("diagnostics", {})
    model = result.get("model", {})
    meta = [f"Model: **{model.get('name', '?')}**"]
    if model.get("version"):
        meta.append(f"version `{model['version']}`")
    meta.append(f"{result.get('processing_time_ms', 0)} ms")
    if diag.get("tokens") is not None:
        meta.append(f"{diag['tokens']} tokens")
    if diag.get("chunks", 1) > 1:
        meta.append(f"{diag['chunks']} windows")
    st.caption(" · ".join(meta))
    if diag.get("truncated"):
        st.warning("⚠️ Text was truncated to the model's input limit.")

    with st.expander("Raw response"):
        st.json(result)
