import mlflow
import re
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pickle
from contextlib import contextmanager


@contextmanager
def mlflow_tracker(enabled: bool, run_name: str | None = None):
        if not enabled:
            yield None
            return

        with mlflow.start_run(run_name=run_name) as run:
            yield run

def _log_numeric_metrics(trend: str, values: dict, model_name: str, suffix: str = "") -> None:
        if not isinstance(values, dict):
            return
        for metric_name, metric_value in values.items():
            if isinstance(metric_value, (int, float, np.number)) and np.isfinite(metric_value):
                mlflow.log_metric(f"{model_name}_{trend}_{metric_name}_{suffix}", float(metric_value))

def _map_feature_name(raw_feat: Any, feature_names: list[str]) -> str:
        feat = str(raw_feat)
        m = re.fullmatch(r"f(\d+)", feat)
        if not m:
            return feat
        idx = int(m.group(1))
        return feature_names[idx] if 0 <= idx < len(feature_names) else feat


def _compute_pct_importance(
        booster: Any,
        importance_type: str,
        feature_names: list[str],
) -> dict[str, float]:
        raw = booster.get_score(importance_type=importance_type) or {}
        total = float(sum(raw.values()))
        if total <= 0:
            return {}

        pct: dict[str, float] = {}
        for feat, score in raw.items():
            mapped = _map_feature_name(feat, feature_names)
            pct[mapped] = pct.get(mapped, 0.0) + (100.0 * float(score) / total)

        return dict(sorted(pct.items(), key=lambda x: x[1], reverse=True))

def _collect_model_importance_and_params(
        boosting_model: Any,
        feature_names: list[str],
        importance_types: list[str],
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
        booster = boosting_model.get_booster()

        model_importance = {
            imp_type: _compute_pct_importance(booster, imp_type, feature_names)
            for imp_type in importance_types
        }

        wrapper_params = boosting_model.get_params(deep=True)
        xgb_params = (
            boosting_model.get_xgb_params()
            if hasattr(boosting_model, "get_xgb_params")
            else {}
        )
        booster_attrs = booster.attributes() or {}

        model_params = {
            "wrapper_params": wrapper_params,
            "xgb_params": xgb_params,
            "booster_attrs": booster_attrs,
        }
        return model_importance, model_params

def _plot_importance_types(
        model_key: str,
        trend: str,
        model_importance: dict[str, dict[str, float]],
        importance_types: list[str],
        out_dir: Path,
        top_k: int = 10,
) -> Path:
        n_types = len(importance_types)
        fig, axes = plt.subplots(1, n_types, figsize=(6 * n_types, 8))
        if n_types == 1:
            axes = [axes]

        for ax, imp_type in zip(axes, importance_types):
            top_items = list(model_importance.get(imp_type, {}).items())[:top_k]
            labels = [k for k, _ in top_items]
            values = [v for _, v in top_items]
            ax.barh(labels, values, color="steelblue")
            ax.set_xlabel(f"{imp_type} (%)")
            ax.set_title(imp_type)
            ax.invert_yaxis()

        fig.suptitle(
            f"{model_key} - Top {top_k} Features per Importance Type ({trend})", fontsize=14
        )
        plt.tight_layout()

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_path = out_dir / f"{trend}_{model_key}_all_importance_types.png"
        fig.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return plot_path

def _log_scalar_params_to_mlflow(prefix: str, params: dict[str, Any]) -> None:
        for k, v in (params or {}).items():
            if isinstance(v, (int, float, str, bool)):
                mlflow.log_param(f"{prefix}_{k}", v)
