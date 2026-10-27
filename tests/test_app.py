from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent

PAGES = ["views/data_page.py", "views/explore.py", "views/prepare.py", "views/train.py", "views/evaluate.py",
         "views/explain_page.py", "views/decide.py", "views/act.py"]


def test_autopilot_then_every_page():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    at.run()
    assert not at.exception
    autopilot = next(b for b in at.button if b.label == "Run autopilot")
    autopilot.click().run()
    assert not at.exception, at.exception
    assert at.session_state.exp is not None and at.session_state.decided
    for page in PAGES:
        at.switch_page(page).run()
        assert not at.exception, f"{page}: {at.exception}"
