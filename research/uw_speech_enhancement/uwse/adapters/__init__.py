"""SE model adapters (RESEARCH_PLAN.md §7.1-7.4).

Importing this package registers every adapter whose module imports cleanly.
Each adapter module is imported inside try/except, so a missing optional
dependency (e.g. deepfilternet, speechbrain) only disables that adapter and
logs a warning. Heavy model code is imported lazily inside ``load_pretrained``,
so registration itself is cheap.

>>> from uwse.adapters import build, available
>>> available()            # e.g. ['dfn3', 'identity', 'metricganplus', ...]
>>> se = build("wpe4", taps=30)
"""

from __future__ import annotations

import importlib
import logging

from .base import SEAdapter, available, build, register

log = logging.getLogger(__name__)

_MODULES = (
    "identity",
    "wpe",
    "speechbrain_models",
    "dfn3",
    "tfgridnet",
    "kiel_unet",  # C10, written separately; optional
)

#: module name -> error message for adapter modules that failed to import
import_errors: dict[str, str] = {}

for _m in _MODULES:
    try:
        importlib.import_module(f"{__name__}.{_m}")
    except ModuleNotFoundError as _e:
        if _e.name == f"{__name__}.{_m}":
            continue  # module not written (yet); silently skip
        import_errors[_m] = repr(_e)
        log.warning("adapter module %s disabled: %r", _m, _e)
    except Exception as _e:  # noqa: BLE001 - optional deps can fail in many ways
        import_errors[_m] = repr(_e)
        log.warning("adapter module %s disabled: %r", _m, _e)

__all__ = ["SEAdapter", "available", "build", "register", "import_errors"]
