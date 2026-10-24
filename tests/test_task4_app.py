"""Offline smoke tests for the Task 4 Streamlit app.

These use Streamlit's AppTest to run app.py in a test harness (no
browser, no model calls). They check that the app loads cleanly and
the main controls exist; live chat behaviour was verified manually
in milestones 4.2-4.5.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parent.parent / "task4_streamlit_app" / "app.py"


def _run_app() -> AppTest:
    """Run the app once in the test harness and return the result."""
    app = AppTest.from_file(str(APP_PATH), default_timeout=60)
    app.run()
    return app


def test_app_loads_without_exception() -> None:
    """The app script runs top to bottom with no exception."""
    app = _run_app()
    assert not app.exception


def test_title_and_mode_radio() -> None:
    """The title is shown and the sidebar radio offers both modes."""
    app = _run_app()
    assert app.title[0].value == "Chatbot"
    radio = app.sidebar.radio[0]
    assert list(radio.options) == ["PDF questions", "Agent"]


def test_empty_chat_hint_is_shown() -> None:
    """A fresh session (empty history) shows the starter hint."""
    app = _run_app()
    assert app.info
    assert "Ask a question to start" in app.info[0].value
