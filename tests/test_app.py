"""Dashboard tests.

Static checks run everywhere. The ``AppTest`` tests run the real script with Streamlit's
test harness and are skipped when Streamlit is not installed.
"""

import ast
import importlib.util

import pytest

from src import config as cfg

APP = cfg.PROJECT_ROOT / "app.py"
HAS_STREAMLIT = importlib.util.find_spec("streamlit") is not None
needs_streamlit = pytest.mark.skipif(not HAS_STREAMLIT, reason="streamlit is not installed (pip install -r requirements.txt)")

# Long-stable Streamlit API used by the app (documented, not experimental).
STABLE_API = {
    "set_page_config", "title", "header", "subheader", "markdown", "caption", "info", "warning", "error",
    "success", "radio", "selectbox", "multiselect", "number_input", "slider", "file_uploader",
    "download_button", "metric", "dataframe", "image", "columns", "tabs", "expander", "stop", "divider",
    "sidebar", "cache_data", "cache_resource", "write",
}


def streamlit_attributes():
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    return {
        node.attr for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "st"
    }


def test_app_compiles():
    compile(APP.read_text(encoding="utf-8"), str(APP), "exec")


def test_app_uses_only_stable_streamlit_api():
    assert streamlit_attributes() <= STABLE_API, streamlit_attributes() - STABLE_API


def test_app_avoids_removed_or_unsafe_options():
    source = APP.read_text(encoding="utf-8")
    for banned in ("use_column_width", "use_container_width", "st.cache(", "unsafe_allow_html", "experimental_", "st.pyplot"):
        assert banned not in source, banned


@pytest.fixture()
def app():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    assert not at.exception, at.exception
    return at


@needs_streamlit
@pytest.mark.parametrize("page", ["Overview", "Data exploration", "Model evaluation", "Machine assessment"])
def test_every_page_renders_without_exceptions(app, page):
    app.sidebar.radio(key="page").set_value(page).run()
    assert not app.exception, app.exception


@needs_streamlit
def test_overview_states_scope(app):
    text = " ".join(m.value for m in app.info) + " ".join(m.value for m in app.markdown)
    assert "synthetic" in text.lower()
    assert len(app.metric) >= 4


@needs_streamlit
def test_assessment_typical_machine_is_low(app):
    app.sidebar.radio(key="page").set_value("Machine assessment").run()
    assert any("Low risk" in s.value for s in app.success)


@needs_streamlit
def test_assessment_real_overstrain_record_is_high(app):
    app.sidebar.radio(key="page").set_value("Machine assessment").run()
    app.selectbox(key="example_choice").set_value("overstrain").run()
    assert not app.exception
    assert any("High risk" in e.value for e in app.error)


@needs_streamlit
def test_assessment_celsius_input_is_rejected_with_a_message(app):
    app.sidebar.radio(key="page").set_value("Machine assessment").run()
    app.number_input(key="typical_air_temp_k").set_value(27.0).run()
    assert not app.exception
    assert any("cannot be assessed" in e.value for e in app.error)


@needs_streamlit
def test_threshold_explorer(app):
    app.sidebar.radio(key="page").set_value("Model evaluation").run()
    app.slider(key="threshold_slider").set_value(0.6).run()
    assert not app.exception
