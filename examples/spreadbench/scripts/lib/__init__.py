"""Support library for the spreadbench eval/optimize CLI scripts.

Not part of the public ``spreadsheet_rlm`` package — these modules are
benchmark-harness glue (dataset loading, scoring, LM config, the eval
loop) shared between ``scripts/eval.py`` and ``scripts/optimize.py``.

Importing from here makes sure ``examples/spreadbench/`` is on
``sys.path`` so modules in this package can ``import spreadsheet_rlm.*``
without each CLI having to repeat the path dance.
"""

from __future__ import annotations

import sys
from pathlib import Path

_EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))
