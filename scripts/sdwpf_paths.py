"""Bootstrap chemin d’import : racine du dépôt + ``src`` pour le package ``sdwpf``."""

from __future__ import annotations

import sys
from pathlib import Path


def repo_root_from_here(script_file: str) -> Path:
    """Racine du dépôt depuis ``__file__`` d’un script dans ``scripts/``."""
    return Path(script_file).resolve().parent.parent


def prepend_src(root: Path) -> None:
    """Insère ``root/src`` en tête de ``sys.path`` si besoin."""
    s = str(root / "src")
    if s not in sys.path:
        sys.path.insert(0, s)
