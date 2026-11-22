from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(ROOT / "config.yaml", "r") as f:
        return yaml.safe_load(f)


def resolve(path: str) -> Path:
    """Resolve a config-relative path against the repo root."""
    return ROOT / path
