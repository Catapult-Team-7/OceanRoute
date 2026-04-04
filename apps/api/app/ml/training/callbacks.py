from __future__ import annotations

from pathlib import Path

try:  # pragma: no cover - optional dependency
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
except Exception:  # pragma: no cover
    try:
        import pytorch_lightning as pl  # type: ignore[no-redef]
        from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint  # type: ignore[assignment]
    except Exception:  # pragma: no cover
        pl = None
        EarlyStopping = None
        LearningRateMonitor = None
        ModelCheckpoint = None


LIGHTNING_AVAILABLE = pl is not None and ModelCheckpoint is not None


if LIGHTNING_AVAILABLE:  # pragma: no cover - deep path only
    class HistoryCallback(pl.Callback):
        def __init__(self) -> None:
            super().__init__()
            self.train_losses: list[float] = []
            self.val_losses: list[float] = []

        def on_train_epoch_end(self, trainer, pl_module) -> None:
            loss = trainer.callback_metrics.get("train_loss")
            if loss is not None:
                self.train_losses.append(float(loss.detach().cpu().item()))

        def on_validation_epoch_end(self, trainer, pl_module) -> None:
            loss = trainer.callback_metrics.get("val_loss")
            if loss is not None:
                self.val_losses.append(float(loss.detach().cpu().item()))


def build_callbacks(run_dir: Path):
    if not LIGHTNING_AVAILABLE:
        return [], None, None
    checkpoints_dir = run_dir / "checkpoints"
    checkpoint = ModelCheckpoint(
        dirpath=str(checkpoints_dir),
        filename="best",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    )
    history = HistoryCallback()
    callbacks = [
        checkpoint,
        history,
        EarlyStopping(monitor="val_loss", patience=5, mode="min"),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    return callbacks, checkpoint, history
