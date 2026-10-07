"""XGBoost : choix du device (CPU / CUDA), entraînement et prédiction sur ce device."""

from __future__ import annotations

import contextlib
import os
import re
import warnings
from collections.abc import Iterator
from typing import Any

import numpy as np

try:
    import xgboost as xgb
except ImportError:
    xgb = None

XGBOOST_AVAILABLE = xgb is not None

# Variable d'environnement : voir ``resolve_xgb_device_spec``.
XGB_DEVICE_ENV = "SDWPF_XGB_DEVICE"


def resolve_xgb_device_spec(explicit: str | None) -> str:
    """Résout le device XGBoost (doc GPU : ``device="cuda"`` ou ``cuda:<ordinal>``).

    Valeurs acceptées : ``auto``, ``cpu``, ``cuda``, ``cuda:0``, ``cuda:1``, …
    Ordre : argument explicite > ``SDWPF_XGB_DEVICE`` > ``cpu``. Le GPU est opt-in :
    CPU et GPU ne donnent pas exactement les mêmes arbres, les chiffres publiés sont en CPU.
    """
    raw = (explicit or "").strip().lower()
    if not raw:
        raw = os.environ.get(XGB_DEVICE_ENV, "cpu").strip().lower()
    if raw in ("auto", "cpu", "cuda"):
        return raw
    if re.fullmatch(r"cuda:\d+", raw):
        return raw
    raise ValueError(
        f"xgb_device invalide {raw!r}: attendu auto, cpu, cuda, ou cuda:<n> "
        f"(ex. cuda:0). Variable {XGB_DEVICE_ENV}."
    )


def _xgboost_train_device(dev: str) -> str:
    """Le booster signale souvent ``cuda:0`` ; un device ``cuda`` nu peut provoquer des replis."""
    if dev == "cuda":
        return "cuda:0"
    return dev


def xgboost_built_with_cuda() -> bool:
    if xgb is None:
        return False
    try:
        bi = xgb.build_info()
    except Exception:  # noqa: BLE001 - build_info varie selon les versions / builds
        return False
    if isinstance(bi, dict) and (bi.get("USE_CUDA") or bi.get("CUDA_VERSION")):
        return True
    if isinstance(bi, (list, tuple)):
        for e in bi:
            if isinstance(e, dict):
                name = str(e.get("name", "")).upper()
                val = str(e.get("value", "")).upper()
                if name == "USE_CUDA" and val in ("ON", "1", "TRUE", "YES"):
                    return True
    return "CUDA" in str(bi).upper() and "OFF" not in str(bi).upper()


@contextlib.contextmanager
def device_mismatch_warnings_silenced(spec: str) -> Iterator[None]:
    """L'API sklearn + NumPy CPU peut déclencher ce UserWarning pendant fit/predict GPU."""
    if spec == "cpu":
        yield
        return
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*(mismatched devices|Falling back to prediction using DMatrix).*",
            category=UserWarning,
        )
        yield


def fit_xgb_regressor(
    params: dict[str, Any],
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    *,
    fit_kw: dict[str, Any],
    spec: str,
) -> tuple[Any, str]:
    """Entraîne un ``XGBRegressor`` ; renvoie ``(modèle, device d'entraînement)``.

    ``auto`` tente CUDA puis se replie sur CPU avec un avertissement.
    """
    if xgb is None:
        raise RuntimeError("xgboost not installed; pip install xgboost")

    def _fit(p: dict[str, Any]) -> Any:
        reg = xgb.XGBRegressor(**p)
        reg.fit(X_tr, y_tr, **fit_kw)
        return reg

    p_cpu = {**params, "tree_method": "hist", "device": "cpu", "n_jobs": -1}
    if spec == "cpu":
        return _fit(p_cpu), "cpu"
    if spec == "auto":
        dev = _xgboost_train_device("cuda")
        p_gpu = {**params, "tree_method": "hist", "device": dev}
        p_gpu.pop("n_jobs", None)
        try:
            return _fit(p_gpu), dev
        except Exception as e:  # noqa: BLE001 - toute erreur GPU déclenche le repli CPU
            warnings.warn(
                f"XGBoost GPU indisponible ({type(e).__name__}: {e}). Repli CPU.",
                UserWarning,
                stacklevel=2,
            )
            return _fit(p_cpu), "cpu"
    if not xgboost_built_with_cuda():
        raise RuntimeError(
            f"xgb_device={spec!r} requiert XGBoost compilé avec CUDA "
            f"(voir https://xgboost.readthedocs.io/en/stable/gpu/index.html ). "
            "Installe un build GPU ou utilise auto/cpu."
        )
    dev = _xgboost_train_device(spec)
    p = {**params, "tree_method": "hist", "device": dev}
    p.pop("n_jobs", None)
    return _fit(p), dev


def predict_on_device(reg: Any, X: np.ndarray, device: str) -> np.ndarray:
    """Prédiction alignée sur le device d'entraînement (évite « mismatched devices »)."""
    if device.startswith("cuda"):
        dev = _xgboost_train_device(device)
        for factory in (xgb.QuantileDMatrix, xgb.DMatrix):
            try:
                dm = factory(X, device=dev)
                return reg.get_booster().predict(dm)
            except Exception:  # noqa: BLE001, S112 - on essaie le constructeur suivant
                continue
    return reg.predict(X)
