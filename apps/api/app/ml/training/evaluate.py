from __future__ import annotations

import json
from pathlib import Path


def read_evaluation_artifact(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
