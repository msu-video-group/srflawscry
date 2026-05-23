import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from srflawscryui.app import LAUNCH_KWARGS, build_app

app = build_app()

if __name__ == "__main__":
    app.launch(**LAUNCH_KWARGS)
