"""Phase 6: end-to-end Streamlit smoke test via Streamlit's AppTest harness.

Marked slow (it trains models + runs the whole script) so it stays out of the fast CI
lane, but it proves the demo renders without raising.
"""

from __future__ import annotations

import pytest

from readmitrisk.paths import get_paths


@pytest.mark.slow
def test_streamlit_app_runs_without_exception() -> None:
    from streamlit.testing.v1 import AppTest

    app_path = get_paths().root / "src" / "readmitrisk" / "ui" / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=240)
    at.run()
    assert not at.exception, f"Streamlit app raised: {at.exception}"
    # The three tabs and the title should be present.
    assert any("ReadmitRisk" in (m.value if hasattr(m, "value") else "") for m in at.title)
    assert len(at.tabs) == 3
