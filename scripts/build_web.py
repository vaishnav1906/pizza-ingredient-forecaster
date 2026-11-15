"""Copy the app's source into docs/app/src so the in-browser (stlite) version can load it.

    python scripts/build_web.py

Run this after changing the app, then redeploy. The copies are committed so GitHub Pages and
Vercel can serve them as static files.
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "docs" / "app" / "src"
# source -> published name (static hosts often skip dot-folders, so the theme is published as a plain file)
FILES = {"app.py": "app.py", "forecast_lib.py": "forecast_lib.py", "ui.py": "ui.py",
         "generate_data.py": "generate_data.py", "recipes.csv": "recipes.csv", "sales.csv": "sales.csv",
         ".streamlit/config.toml": "streamlit-config.toml"}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for src, name in FILES.items():
        shutil.copy2(ROOT / src, DEST / name)
        print("copied", src, "->", name)


if __name__ == "__main__":
    main()
