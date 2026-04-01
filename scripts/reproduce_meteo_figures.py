"""
Rejoue les générations de figures PNG SDWPF que tu as utilisées en mode météo.

Chaque preset lance ``scripts/sdwpf_visualize.py`` avec ``--meteo-mode`` et
``--horizon-days 1`` (cible Patv à J+1, 144 pas de 10 min), comme pour les
fichiers ``01_*`` … ``05_kpi_*`` dans ``reports/figures/``.

Exemples (depuis la racine du projet) :

  python scripts/reproduce_meteo_figures.py --preset single
  python scripts/reproduce_meteo_figures.py --preset multi5
  python scripts/reproduce_meteo_figures.py --preset multi20
  python scripts/reproduce_meteo_figures.py --preset multi100
  python scripts/reproduce_meteo_figures.py --preset all --open-folder
  python scripts/reproduce_meteo_figures.py --dry-run --preset all
  python scripts/reproduce_meteo_figures.py --preset multi20 -- --xgb-device cuda
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parent.parent
_VIS = _ROOT / "scripts" / "sdwpf_visualize.py"


def _turb_ids_1_to_n(n: int) -> str:
    return ",".join(str(i) for i in range(1, n + 1))


PRESETS: dict[str, tuple[str, str]] = {
    "single": ("1", "Turbine n°1 → tag ``t1``"),
    "multi5": ("1,2,3,5,8", "5 turbines → tag ``multi_1-2-3-5-8``"),
    "multi20": (_turb_ids_1_to_n(20), "20 turbines (1–20) → tag ``multi_n20_1-20``"),
    "multi100": (_turb_ids_1_to_n(100), "100 turbines (1–100) → tag ``multi_n100_1-100``"),
}


def build_cmd(
    turb_ids: str,
    *,
    open_folder: bool,
    extra: list[str],
) -> list[str]:
    cmd: list[str] = [
        sys.executable,
        str(_VIS),
        "--meteo-mode",
        "--horizon-days",
        "1",
        "--turb-ids",
        turb_ids,
    ]
    if open_folder:
        cmd.append("--open-folder")
    cmd.extend(extra)
    return cmd


def run_one(name: str, turb_ids: str, *, open_folder: bool, dry_run: bool, extra: list[str]) -> int:
    cmd = build_cmd(turb_ids, open_folder=open_folder, extra=extra)
    print(f"\n=== Preset « {name} » ===")
    print(" ", subprocess.list2cmdline(cmd))
    if dry_run:
        return 0
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    r = subprocess.run(cmd, cwd=_ROOT, env=env)
    return int(r.returncode)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Rejoue les runs figures SDWPF (météo + horizon 1 jour)."
    )
    p.add_argument(
        "--preset",
        choices=[*sorted(PRESETS.keys()), "all"],
        default="single",
        help="single | multi5 | multi20 | multi100 | all (enchaîne single→multi5→multi20→multi100)",
    )
    p.add_argument(
        "--open-folder",
        action="store_true",
        help="Ouvre le dossier des figures après chaque run (Windows : explorateur).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche les commandes sans exécuter.",
    )
    p.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Arguments supplémentaires pour sdwpf_visualize.py (après -- si besoin).",
    )
    args = p.parse_args()
    extra = args.extra
    if extra and extra[0] == "--":
        extra = extra[1:]

    if not _VIS.is_file():
        raise SystemExit(f"Script introuvable : {_VIS}")

    order = (
        ["single", "multi5", "multi20", "multi100"] if args.preset == "all" else [args.preset]
    )
    code = 0
    for key in order:
        spec, blurb = PRESETS[key]
        print(f"({blurb})")
        code = run_one(key, spec, open_folder=args.open_folder, dry_run=args.dry_run, extra=extra)
        if code != 0:
            raise SystemExit(code)
    if args.preset == "all":
        print("\nTous les presets ont été exécutés (fin : multi100 peut être très long).")


if __name__ == "__main__":
    main()
