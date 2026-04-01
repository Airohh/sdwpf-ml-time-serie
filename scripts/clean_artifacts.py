"""
Supprime des artefacts locaux (non versionnés ou jetables).

Depuis la racine du dépôt :
  python scripts/clean_artifacts.py              # figures obsolètes h72 + mlruns
  python scripts/clean_artifacts.py --dry-run
  python scripts/clean_artifacts.py --reports    # efface aussi reports/*.csv et *.md
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser(description="Nettoyage local : figures h72, mlruns, rapports optionnels")
    p.add_argument(
        "--reports",
        action="store_true",
        help="Supprimer les CSV/MD directs sous reports/ (pas figures/)",
    )
    p.add_argument("--dry-run", action="store_true", help="Afficher seulement ce qui serait supprimé")
    args = p.parse_args()

    removed: list[str] = []

    fig_dir = _ROOT / "reports" / "figures"
    if fig_dir.is_dir():
        for png in fig_dir.glob("*_h72_*.png"):
            removed.append(str(png))
            if not args.dry_run:
                png.unlink(missing_ok=True)

    mlruns = _ROOT / "mlruns"
    if mlruns.is_dir():
        removed.append(str(mlruns))
        if not args.dry_run:
            shutil.rmtree(mlruns, ignore_errors=True)

    if args.reports:
        rdir = _ROOT / "reports"
        for pat in ("*.csv", "*.md"):
            for f in rdir.glob(pat):
                removed.append(str(f))
                if not args.dry_run:
                    f.unlink(missing_ok=True)

    for path in removed:
        print(("would remove " if args.dry_run else "removed ") + path)

    if not removed:
        print("nothing to clean")


if __name__ == "__main__":
    main()
    sys.exit(0)
