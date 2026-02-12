"""Optuna hyperparameter search API routes.

Reads Optuna's SQLite storage directly — no separate data pipeline needed.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime as _dt
from typing import Any

from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/optuna", tags=["optuna"])


def _get_storage_url() -> str:
    return os.environ.get("OPTUNA_STORAGE", "sqlite:///optuna_results.db")


def _load_study(study_name: str):
    """Load an Optuna study from storage. Returns None on failure."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        return optuna.load_study(study_name=study_name, storage=_get_storage_url())
    except Exception as e:
        logger.debug("Failed to load study %s: %s", study_name, e)
        return None


def _trial_to_dict(trial) -> dict[str, Any]:
    duration = None
    if trial.datetime_start and trial.datetime_complete:
        duration = (trial.datetime_complete - trial.datetime_start).total_seconds()
    return {
        "number": trial.number,
        "value": trial.value,
        "state": trial.state.name,
        "params": trial.params,
        "user_attrs": trial.user_attrs,
        "duration_seconds": duration,
        "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else None,
        "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else None,
    }


def _db_exists() -> bool:
    """Check if the SQLite DB file exists (for sqlite:/// URLs)."""
    url = _get_storage_url()
    if url.startswith("sqlite:///"):
        path = url.replace("sqlite:///", "")
        return os.path.isfile(path)
    # For non-sqlite URLs, assume reachable
    return True


# --- Endpoints ---


@router.get("/status")
async def optuna_status():
    """Whether Optuna DB exists and list active studies."""
    if not _db_exists():
        return {"available": False, "studies": [], "storage": _get_storage_url()}

    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        summaries = optuna.get_all_study_summaries(storage=_get_storage_url())
        # Return both names (for backward compat) and detailed list with timestamps
        studies = [s.study_name for s in summaries]
        studies_detail = [
            {
                "name": s.study_name,
                "datetime_start": s.datetime_start.isoformat() if s.datetime_start else None,
                "n_trials": s.n_trials,
            }
            for s in sorted(
                summaries,
                key=lambda x: x.datetime_start or _dt.min,
                reverse=True,
            )
        ]
        return {
            "available": True,
            "studies": studies,
            "studies_detail": studies_detail,
            "storage": _get_storage_url(),
        }
    except Exception as e:
        logger.debug("Optuna status error: %s", e)
        return {"available": False, "studies": [], "error": str(e)}


@router.get("/studies")
async def list_studies():
    """List all studies with summary info."""
    if not _db_exists():
        return {"studies": []}

    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        summaries = optuna.get_all_study_summaries(storage=_get_storage_url())
        result = []
        for s in summaries:
            best_value = None
            best_trial_number = None
            if s.best_trial:
                best_value = s.best_trial.value
                best_trial_number = s.best_trial.number
            # StudySummary doesn't expose per-state counts; load study for accuracy
            n_complete = 0
            try:
                study = optuna.load_study(study_name=s.study_name, storage=_get_storage_url())
                from optuna.trial import TrialState
                n_complete = len([t for t in study.trials if t.state == TrialState.COMPLETE])
            except Exception:
                # Fallback: at least 1 if best_trial exists
                n_complete = 1 if s.best_trial else 0

            result.append({
                "name": s.study_name,
                "direction": s.direction.name if hasattr(s.direction, "name") else str(s.direction),
                "n_trials": s.n_trials,
                "n_complete": n_complete,
                "best_value": best_value,
                "best_trial_number": best_trial_number,
                "datetime_start": s.datetime_start.isoformat() if s.datetime_start else None,
            })
        return {"studies": result}
    except Exception as e:
        logger.debug("List studies error: %s", e)
        return {"studies": [], "error": str(e)}


@router.get("/studies/{study_name}")
async def get_study(study_name: str):
    """Study summary with best trial and param importances."""
    study = _load_study(study_name)
    if study is None:
        return {"error": f"Study '{study_name}' not found"}

    # Trial state counts
    from optuna.trial import TrialState

    trials = study.trials
    n_complete = len([t for t in trials if t.state == TrialState.COMPLETE])
    n_pruned = len([t for t in trials if t.state == TrialState.PRUNED])
    n_fail = len([t for t in trials if t.state == TrialState.FAIL])
    n_running = len([t for t in trials if t.state == TrialState.RUNNING])

    best = None
    if n_complete > 0:
        best = _trial_to_dict(study.best_trial)

    # Parameter importances (need at least 2 completed trials)
    importances = {}
    if n_complete >= 2:
        try:
            from optuna.importance import get_param_importances

            importances = get_param_importances(study)
        except Exception:
            pass

    # Get study start time from first trial
    datetime_start = None
    if trials:
        starts = [t.datetime_start for t in trials if t.datetime_start]
        if starts:
            datetime_start = min(starts).isoformat()

    return {
        "name": study.study_name,
        "direction": study.direction.name,
        "n_trials": len(trials),
        "n_complete": n_complete,
        "n_pruned": n_pruned,
        "n_fail": n_fail,
        "n_running": n_running,
        "best_trial": best,
        "param_importances": importances,
        "datetime_start": datetime_start,
    }


@router.get("/studies/{study_name}/trials")
async def get_trials(study_name: str):
    """All trials with params, values, state, user_attrs."""
    study = _load_study(study_name)
    if study is None:
        return {"error": f"Study '{study_name}' not found", "trials": []}

    return {"trials": [_trial_to_dict(t) for t in study.trials]}


@router.get("/studies/{study_name}/best")
async def get_best_trial(study_name: str):
    """Best trial details."""
    study = _load_study(study_name)
    if study is None:
        return {"error": f"Study '{study_name}' not found"}

    from optuna.trial import TrialState

    complete = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not complete:
        return {"error": "No completed trials yet", "best_trial": None}

    return {"best_trial": _trial_to_dict(study.best_trial)}
