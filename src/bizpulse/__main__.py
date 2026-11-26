"""Allow ``python -m bizpulse``."""

from __future__ import annotations

import sys

from bizpulse.cli import main

sys.exit(main())
