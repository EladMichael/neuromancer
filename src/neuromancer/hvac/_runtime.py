"""
Single source of truth for runtime type-checking in neuromancer.hvac.

Runtime type checking (via beartype) is ON by default so that shape/type
contract violations fail fast with a clear error. It can be disabled for
heavy workloads by setting the environment variable RUNTIME_TYPING to one of
{"0", "false", "no", ""} (case-insensitive) before importing this package.

Import `beartype` from here everywhere instead of re-implementing the toggle:

    from .._runtime import beartype   # building_components/*, actuators/*
    from ._runtime import beartype    # top-level hvac modules
"""
import os


def _typecheck_enabled() -> bool:
    return str(os.environ.get("RUNTIME_TYPING", "1")).strip().lower() not in (
        "0", "false", "no", "",
    )


def _noop_beartype(fn):
    return fn


if _typecheck_enabled():
    try:
        from beartype import beartype
    except ImportError:
        import warnings
        warnings.warn(
            "RUNTIME_TYPING is enabled but 'beartype' is not installed; "
            "runtime type checking is disabled. `pip install beartype` to enable it.",
            RuntimeWarning,
        )
        beartype = _noop_beartype
else:
    beartype = _noop_beartype
