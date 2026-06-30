from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any


def _has_mlflow() -> bool:
    try:
        import mlflow  # noqa: F401
        return True
    except Exception:
        return False


@contextmanager
def mlflow_run(project_name: str, run_name: str, params: dict[str, Any] | None = None):
    if not _has_mlflow():
        yield None
        return

    import mlflow

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(project_name)
    with mlflow.start_run(run_name=run_name):
        if params:
            flat: dict[str, Any] = {}
            for k, v in params.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        flat[f"{k}.{kk}"] = vv
                else:
                    flat[k] = v
            mlflow.log_params({k: str(v) for k, v in flat.items()})
        yield mlflow


def mlflow_log_metrics(mlflow_ctx: Any, metrics: dict[str, float]) -> None:
    if mlflow_ctx is None:
        return
    for k, v in metrics.items():
        try:
            mlflow_ctx.log_metric(k, float(v))
        except Exception:
            pass


def mlflow_log_artifact(mlflow_ctx: Any, path: str) -> None:
    if mlflow_ctx is None:
        return
    try:
        mlflow_ctx.log_artifact(path)
    except Exception:
        pass
