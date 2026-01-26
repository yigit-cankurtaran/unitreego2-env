from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export local pip packages to a requirements file.")
    parser.add_argument("--output", type=Path, default=Path("requirements_local.txt"))
    args = parser.parse_args()

    try:
        result = subprocess.run(
            ["python", "-m", "pip", "freeze"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"pip freeze failed. Make sure your venv is active.\n{exc.stderr}"
        ) from exc

    args.output.write_text(result.stdout)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
