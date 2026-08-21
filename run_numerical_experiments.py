#!/usr/bin/env python3
"""Run every published numerical experiment and write one JSON report."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np

import verify_closed_forms
import verify_quadrature
import verify_turning_point
import verify_uniaxial_branch


PACKAGE_DIR = Path(__file__).resolve().parent
RESULT_FILE = PACKAGE_DIR / "results" / "numerical-experiments.json"


def main() -> None:
    experiments = {
        "quadrature": verify_quadrature.run_checks(),
        "turning_point": verify_turning_point.run_checks(),
        "uniaxial_branch": verify_uniaxial_branch.run_checks(),
        "closed_forms": verify_closed_forms.run_checks(),
    }
    report = {
        "status": "passed",
        "runtime": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "experiments": experiments,
    }
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = RESULT_FILE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(RESULT_FILE)
    print(f"All numerical experiments passed. Results: {RESULT_FILE}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Numerical experiments failed: {error}", file=sys.stderr)
        raise
