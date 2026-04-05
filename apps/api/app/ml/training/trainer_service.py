from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import numpy as np

from app.ml.sequence_model_architectures import build_model
from app.ml.training.callbacks import LIGHTNING_AVAILABLE, build_callbacks
from app.ml.training.config import TrainingConfig
from app.ml.training.datamodule import LIGHTNING_AVAILABLE as DATAMODULE_LIGHTNING_AVAILABLE
from app.ml.training.datamodule import TORCH_AVAILABLE, OceanRouteDataModule
from app.ml.training.dataset_reader import load_dataset_bundle
from app.ml.training.export_artifact import export_model_artifact
from app.ml.training.losses import multitask_loss
from app.ml.training.metrics import compute_eval_metrics
from app.services.data_lake_service import training_runs_root

try:  # pragma: no cover - optional dependency
    import lightning.pytorch as pl
    from lightning.pytorch.loggers import TensorBoardLogger
except Exception:  # pragma: no cover
    try:
        import pytorch_lightning as pl  # type: ignore[no-redef]
        from pytorch_lightning.loggers import TensorBoardLogger  # type: ignore[assignment]
    except Exception:  # pragma: no cover
        pl = None
        TensorBoardLogger = None

try:  # pragma: no cover - optional dependency
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except Exception:  # pragma: no cover
    plt = None
    MATPLOTLIB_AVAILABLE = False

if TORCH_AVAILABLE:  # pragma: no cover - deep path
    import torch
else:  # pragma: no cover
    torch = None


def _tiny_png(path: Path) -> None:
    payload = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9sF3W9kAAAAASUVORK5CYII="
    )
    path.write_bytes(payload)


def _write_curves(run_dir: Path, train_losses: list[float], val_losses: list[float], metrics: dict[str, Any]) -> None:
    plots_dir = run_dir / "plots"
    sample_dir = plots_dir / "sample_predictions"
    plots_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    if MATPLOTLIB_AVAILABLE and train_losses:
        figure = plt.figure()
        plt.plot(train_losses, label="train_loss")
        if val_losses:
            plt.plot(val_losses, label="val_loss")
        plt.legend()
        plt.title("Loss")
        figure.savefig(plots_dir / "loss.png")
        plt.close(figure)

        figure = plt.figure()
        keys = ["precision_at_10", "kg_mae", "route_uplift_pct"]
        values = [float(metrics.get(key, 0.0)) for key in keys]
        plt.bar(keys, values)
        plt.xticks(rotation=20)
        plt.tight_layout()
        figure.savefig(plots_dir / "val_metrics.png")
        plt.close(figure)

        figure = plt.figure()
        plt.text(0.1, 0.5, json.dumps(metrics, indent=2), family="monospace")
        plt.axis("off")
        figure.savefig(sample_dir / "sample_prediction.png")
        plt.close(figure)
        return
    _tiny_png(plots_dir / "loss.png")
    _tiny_png(plots_dir / "val_metrics.png")
    _tiny_png(sample_dir / "sample_prediction.png")


if TORCH_AVAILABLE and DATAMODULE_LIGHTNING_AVAILABLE and LIGHTNING_AVAILABLE and pl is not None:  # pragma: no cover - deep path only
    class SequenceForecastLightningModule(pl.LightningModule):
        def __init__(
            self,
            *,
            architecture: str,
            input_channels: int,
            num_horizons: int,
            learning_rate: float,
            hotspot_loss: str,
            focal_gamma: float,
        ) -> None:
            super().__init__()
            self.save_hyperparameters()
            self.model = build_model(architecture, input_channels=input_channels, num_horizons=num_horizons)
            self.learning_rate = learning_rate
            self.hotspot_loss = hotspot_loss
            self.focal_gamma = focal_gamma

        def forward(self, inputs):  # type: ignore[override]
            return self.model(inputs)

        def _probability_diagnostics(self, prediction_probability, target_probability) -> dict[str, float]:
            positive_mask = target_probability > 0.5
            negative_mask = target_probability <= 0.5
            positive_prob_mean = (
                float(prediction_probability[positive_mask].mean().detach().cpu().item())
                if bool(positive_mask.any())
                else 0.0
            )
            negative_prob_mean = (
                float(prediction_probability[negative_mask].mean().detach().cpu().item())
                if bool(negative_mask.any())
                else 0.0
            )
            flattened_prediction = prediction_probability.reshape(prediction_probability.shape[0], -1)
            flattened_target = target_probability.reshape(target_probability.shape[0], -1)
            top_k = min(10, flattened_prediction.shape[1])
            if top_k <= 0:
                return {
                    "positive_prob_mean": positive_prob_mean,
                    "negative_prob_mean": negative_prob_mean,
                    "top10_score_mean": 0.0,
                    "top10_true_hits_mean": 0.0,
                }
            top_scores, top_indexes = torch.topk(flattened_prediction, k=top_k, dim=1)
            top_hits = torch.gather(flattened_target, 1, top_indexes)
            return {
                "positive_prob_mean": positive_prob_mean,
                "negative_prob_mean": negative_prob_mean,
                "top10_score_mean": float(top_scores.mean().detach().cpu().item()),
                "top10_true_hits_mean": float(top_hits.sum(dim=1).float().mean().detach().cpu().item()),
            }

        def _shared_step(self, batch, stage: str):
            prediction_probability, prediction_expected_kg, prediction_uncertainty = self(batch["inputs"])
            total_loss, loss_parts = multitask_loss(
                prediction_probability=prediction_probability,
                prediction_expected_kg=prediction_expected_kg,
                prediction_uncertainty=prediction_uncertainty,
                target_probability=batch["target_probability"],
                target_expected_kg=batch["target_expected_kg"],
                target_uncertainty=batch["target_uncertainty"],
                hotspot_loss=self.hotspot_loss,
                focal_gamma=self.focal_gamma,
            )
            self.log(f"{stage}_loss", total_loss, prog_bar=True, on_epoch=True, on_step=(stage == "train"))
            for name, value in loss_parts.items():
                self.log(f"{stage}_{name}", value, prog_bar=False, on_epoch=True, on_step=False)
            for name, value in self._probability_diagnostics(prediction_probability, batch["target_probability"]).items():
                self.log(f"{stage}_{name}", value, prog_bar=False, on_epoch=True, on_step=False)
            return total_loss

        def training_step(self, batch, batch_idx):  # type: ignore[override]
            return self._shared_step(batch, "train")

        def validation_step(self, batch, batch_idx):  # type: ignore[override]
            self._shared_step(batch, "val")

        def test_step(self, batch, batch_idx):  # type: ignore[override]
            self._shared_step(batch, "test")

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def _resolve_accelerator(device: str) -> tuple[str, int]:
    lowered = device.lower()
    if lowered == "cuda":
        return "gpu", 1
    if lowered == "cpu":
        return "cpu", 1
    return "auto", 1


def _evaluate_model(module, datamodule: OceanRouteDataModule, horizons: list[int]) -> dict[str, Any]:  # pragma: no cover - deep path only
    test_loader = datamodule.test_dataloader()
    probabilities: list[np.ndarray] = []
    kilograms: list[np.ndarray] = []
    uncertainties: list[np.ndarray] = []
    targets_probability: list[np.ndarray] = []
    targets_kg: list[np.ndarray] = []
    targets_uncertainty: list[np.ndarray] = []
    baseline_density: list[np.ndarray] = []
    module.eval()
    with torch.no_grad():
        for batch in test_loader:
            pred_probability, pred_expected_kg, pred_uncertainty = module(batch["inputs"])
            probabilities.append(pred_probability.detach().cpu().numpy())
            kilograms.append(pred_expected_kg.detach().cpu().numpy())
            uncertainties.append(pred_uncertainty.detach().cpu().numpy())
            targets_probability.append(batch["target_probability"].detach().cpu().numpy())
            targets_kg.append(batch["target_expected_kg"].detach().cpu().numpy())
            targets_uncertainty.append(batch["target_uncertainty"].detach().cpu().numpy())
            baseline_density.append(batch["inputs"][:, -1, 4:5].detach().cpu().numpy())
    return compute_eval_metrics(
        probabilities=np.concatenate(probabilities, axis=0),
        predicted_kg=np.concatenate(kilograms, axis=0),
        predicted_uncertainty=np.concatenate(uncertainties, axis=0),
        baseline_density=np.concatenate(baseline_density, axis=0),
        target_probability=np.concatenate(targets_probability, axis=0),
        target_expected_kg=np.concatenate(targets_kg, axis=0),
        target_uncertainty=np.concatenate(targets_uncertainty, axis=0),
        horizons=horizons,
    )


def train_sequence_model(config: TrainingConfig, dataset_root: Path) -> dict[str, Any]:
    if not TORCH_AVAILABLE or pl is None or TensorBoardLogger is None:
        raise ValueError(
            "Deep sequence training requires torch, pytorch-lightning/lightning, and tensorboard. "
            "Install the ML environment from apps/api/requirements-ml.txt and a matching torch wheel."
        )

    bundle = load_dataset_bundle(dataset_root)
    run_dir = training_runs_root() / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "stdout.log"

    datamodule = OceanRouteDataModule(
        bundle,
        channel_names=list(bundle.metadata["input_channels"]),
        region_ids=list(config.region_ids) or None,
        horizons=list(config.horizons) or None,
        batch_size=config.batch_size,
    )
    datamodule.setup()
    callbacks, checkpoint_callback, history_callback = build_callbacks(run_dir)
    logger = TensorBoardLogger(save_dir=str(run_dir), name="tensorboard")
    accelerator, devices = _resolve_accelerator(config.device)
    lightning_module = SequenceForecastLightningModule(
        architecture=config.architecture,
        input_channels=int(bundle.x_tensor.shape[2]),
        num_horizons=len(config.horizons),
        learning_rate=config.learning_rate,
        hotspot_loss=config.hotspot_loss,
        focal_gamma=config.focal_gamma,
    )
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        accelerator=accelerator,
        devices=devices,
        logger=logger,
        callbacks=callbacks,
        enable_progress_bar=True,
        log_every_n_steps=1,
    )
    trainer.fit(lightning_module, datamodule=datamodule)
    trainer.test(lightning_module, datamodule=datamodule)

    best_checkpoint_path = checkpoint_callback.best_model_path if checkpoint_callback is not None else ""
    best_val_loss = float(checkpoint_callback.best_model_score.detach().cpu().item()) if checkpoint_callback and checkpoint_callback.best_model_score is not None else None
    trained_module = lightning_module
    if best_checkpoint_path:
        trained_module = SequenceForecastLightningModule.load_from_checkpoint(  # type: ignore[attr-defined]
            best_checkpoint_path,
            architecture=config.architecture,
            input_channels=int(bundle.x_tensor.shape[2]),
            num_horizons=len(config.horizons),
            learning_rate=config.learning_rate,
            hotspot_loss=config.hotspot_loss,
            focal_gamma=config.focal_gamma,
        )
    metrics = _evaluate_model(trained_module, datamodule, list(config.horizons))
    _write_curves(
        run_dir,
        history_callback.train_losses if history_callback is not None else [],
        history_callback.val_losses if history_callback is not None else [],
        metrics,
    )
    feature_stats_path = str(bundle.root_path / "feature_stats.json")
    feature_schema = {
        "input_channels": bundle.metadata["input_channels"],
        "target_channels": bundle.metadata["target_channels"],
        "horizons": bundle.metadata["horizons"],
        "trained_horizons": list(config.horizons),
        "lookback_hours": int(bundle.metadata.get("lookback_hours", bundle.x_tensor.shape[1])),
        "compatible_regions": list(config.region_ids),
        "tensor_layout": {
            "feature_snapshot": "T,C,Y,X",
            "model_input": "B,T,C,Y,X",
            "target": "B,H,1,Y,X",
        },
        "tensor_shapes": bundle.metadata["tensor_shapes"],
    }
    artifact_paths = export_model_artifact(
        model=trained_module.model,
        model_id=config.model_id,
        architecture=config.architecture,
        dataset_id=config.dataset_id,
        dataset_version=str(bundle.metadata["dataset_version"]),
        training_scope=config.training_scope,
        trained_regions=list(config.region_ids),
        compatible_regions=list(config.region_ids),
        horizons=list(config.horizons),
        input_channels=list(bundle.metadata["input_channels"]),
        output_heads=list(config.output_heads),
        training_command=config.training_command,
        created_at=config.created_at.isoformat(),
        eval_metrics=metrics,
        normalization_stats_path=feature_stats_path,
        feature_schema=feature_schema,
        best_checkpoint_path=best_checkpoint_path,
        framework=config.framework,
        lookback_hours=int(bundle.metadata.get("lookback_hours", bundle.x_tensor.shape[1])),
        tensor_layout=feature_schema["tensor_layout"],
    )
    stdout_path.write_text(
        json.dumps(
            {
                "config": config.as_dict(),
                "metrics": metrics,
                "best_checkpoint_path": best_checkpoint_path,
                "best_val_loss": best_val_loss,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return {
        "metrics": metrics,
        "run_dir": str(run_dir),
        "tensorboard_dir": str(run_dir / "tensorboard"),
        "artifact_paths": artifact_paths,
        "feature_schema": feature_schema,
        "feature_stats_path": feature_stats_path,
        "best_checkpoint_path": best_checkpoint_path or None,
        "best_val_loss": best_val_loss,
        "current_epoch": config.epochs,
    }
