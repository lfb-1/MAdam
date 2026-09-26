import json
import logging
import os
from pathlib import Path
from typing import Any

from hydra.utils import get_original_cwd

from src.callbacks.callback import Callback
from src.utils.results_io import normalize_fim_method


TRAINING_HISTORY_PREFIXES = ("train/", "insta/train/")
TRAINING_HISTORY_STEP_KEYS = ("mystep",)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


class LocalTrainingDynamicsCallback(Callback):
    def __init__(self, dataset_name: str, config):
        super().__init__()
        self.dataset_name = dataset_name
        self.config = config
        self.records: list[dict[str, Any]] = []
        self.metric_keys: set[str] = set()
        self.output_path = self._resolve_output_path()

    def _resolve_project_root(self) -> Path:
        try:
            return Path(get_original_cwd())
        except ValueError:
            logging.warning("Hydra not initialized; using cwd for training dynamics output.")
            return Path.cwd()

    def _resolve_output_path(self) -> Path:
        explicit_path = os.environ.get("MADAM_TRAINING_DYNAMICS_PATH")
        if explicit_path:
            return Path(explicit_path)

        project_root = self._resolve_project_root()
        results_version = os.environ.get("MADAM_RESULTS_VERSION", "v5")
        method_name = str(getattr(self.config.method, "name", "unknown"))
        fim_method = normalize_fim_method(getattr(self.config.method, "fim_method", "none"))
        seed = int(getattr(self.config, "seed", 0))
        filename = f"{method_name}_{fim_method}_s{seed}.json"
        return project_root / "reports" / "training_dynamics" / self.dataset_name / results_version / filename

    def _payload(self) -> dict[str, Any]:
        return {
            "source": "local",
            "dataset": self.dataset_name,
            "results_version": os.environ.get("MADAM_RESULTS_VERSION", "v5"),
            "run": {
                "method": str(getattr(self.config.method, "name", "unknown")),
                "fim_method": normalize_fim_method(getattr(self.config.method, "fim_method", "none")),
                "seed": int(getattr(self.config, "seed", 0)),
                "output_path": str(self.output_path),
            },
            "filter": {
                "prefixes": list(TRAINING_HISTORY_PREFIXES),
                "step_keys": list(TRAINING_HISTORY_STEP_KEYS),
            },
            "record_count": len(self.records),
            "metric_keys": sorted(self.metric_keys),
            "records": self.records,
        }

    def _write_payload(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(self._payload(), indent=2) + "\n")

    def connect(self, trainer, *args, **kwargs):
        super().connect(trainer, *args, **kwargs)
        trainer.training_dynamics_recorder = self

    def record(self, payload: dict[str, Any]) -> None:
        record = {str(key): _json_safe(value) for key, value in payload.items() if value is not None}
        self.metric_keys.update(
            key for key in record if any(key.startswith(prefix) for prefix in TRAINING_HISTORY_PREFIXES)
        )
        self.records.append(record)
        self._write_payload()

    def on_before_fit(self, trainer, *args, **kwargs):
        self._write_payload()

    def on_after_fit(self, trainer, *args, **kwargs):
        self._write_payload()
