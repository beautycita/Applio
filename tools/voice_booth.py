"""Moved to tools/studio/ (server.py, booth.html, editor.html). Kept so old commands still work."""

import os
import runpy
import sys

if __name__ == "__main__":
    sys.argv[0] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "studio", "server.py"
    )
    runpy.run_path(sys.argv[0], run_name="__main__")
