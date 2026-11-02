#!/usr/bin/env python3
"""Clear v2 entry point. All CLI flags are supported by the integrated runner."""
import sys
from run_baseline_evidence_integration import main


if __name__ == "__main__":
    if "--ranker" in sys.argv:
        raise SystemExit("This v2 entry point always uses --ranker v2; remove --ranker")
    sys.argv.extend(["--ranker", "v2"])
    main()
