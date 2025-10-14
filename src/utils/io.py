import json
from pathlib import Path
from typing import Dict, Any


def write_json(obj: Dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)