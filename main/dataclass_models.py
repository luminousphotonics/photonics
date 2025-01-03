from dataclasses import dataclass
from typing import Dict

@dataclass
class Fixture:
    file_path: str  # Now stores a URL to the static file
    type: str
    layer: int
    position: Dict[str, float]
    rotation: Dict[str, float]
    index: int = -1