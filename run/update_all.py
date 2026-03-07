from __future__ import annotations

import runpy
import sys


def run(script: str) -> None:
    print(f"\n--- Running {script} ---")
    runpy.run_path(script, run_name="__main__")


def main() -> None:
    sys.path.insert(0, "src")

    run("run/build_country_panel.py")
    run("run/build_country_scores.py")
    run("run/build_portfolio.py")
    run("run/weekly_actions.py")


if __name__ == "__main__":
    main()