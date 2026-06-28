"""Entry point for Streamlit Community Cloud.

Streamlit Cloud installs requirements.txt and runs this file. We add the src layout to
the path (the package is not pip installed there) and execute the real app module fresh on
every rerun so Streamlit's interaction model works.
"""

import pathlib
import runpy
import sys

ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

runpy.run_path(str(ROOT / "src" / "readmitrisk" / "ui" / "app.py"), run_name="__main__")
