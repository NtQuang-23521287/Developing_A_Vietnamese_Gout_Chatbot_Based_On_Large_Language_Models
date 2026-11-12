from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def mlflow_run(config: dict, run_name: str, params: dict):
    if not config.get("mlflow_enabled", False):
        yield None
        return
    try:
        import mlflow
    except ImportError:
        yield None
        return
    mlflow.set_tracking_uri(config.get("tracking_uri", "file:./mlruns"))
    mlflow.set_experiment(config.get("experiment_name", "gout_llmops"))
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({k: v for k, v in params.items() if v is not None})
        yield mlflow
