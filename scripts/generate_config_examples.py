#!/usr/bin/env python3
"""
Generate examples/config.json from the default values in config.py.

The committed file must equal this script's output: tests/test_example_config.py
fails when it does not. After changing a default, adding a setting or removing
one, run this script and commit the result.
"""

import json
import sys
from pathlib import Path

# Add src to path so we can import jarvis modules
script_dir = Path(__file__).parent
project_root = script_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

from jarvis.config import export_example_config

EXAMPLE_PATH = project_root / "examples" / "config.json"


def build_example_config() -> dict:
    """The example configuration: every default that is the same on every machine."""
    return export_example_config(include_db_path=False)


def render_example_config(config: dict) -> str:
    """The exact text written to examples/config.json."""
    return json.dumps(config, indent=2) + "\n"


def generate_config_example() -> Path:
    """Write examples/config.json from defaults and return its path."""
    # newline="\n" so the file is identical whichever platform writes it.
    with EXAMPLE_PATH.open("w", encoding="utf-8", newline="\n") as f:
        f.write(render_example_config(build_example_config()))
    return EXAMPLE_PATH


def main() -> None:
    """Generate all example configuration files."""
    # The console may not be UTF-8 (cp1252 on Windows); never fail on an emoji.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    print("⚙️ Generating configuration examples from defaults...")

    path = generate_config_example()

    print(f"  📄 Wrote {path}")
    print("  ✅ Example files are now in sync with config.py defaults.")


if __name__ == "__main__":
    main()
