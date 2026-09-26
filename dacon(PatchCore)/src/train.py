"""GPU 학습, TensorBoard 기록, 최고 성능 체크포인트 저장을 수행한다."""

from pathlib import Path

from anomalib.engine import Engine
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger


def train_model(model, datamodule, run_name: str, max_epochs: int, output_dir: str) -> Engine:
    """모델을 학습하고 validation image AUROC 최고 체크포인트를 저장한다."""
    logger = TensorBoardLogger(save_dir=Path(output_dir) / run_name, name="tensorboard")
    checkpoint_callback = ModelCheckpoint(
        monitor=None if run_name == "patchcore" else "image_AUROC",
        mode="max",
        save_top_k=1,
        filename="patchcore-{epoch}" if run_name == "patchcore" else "best-{epoch}-{image_AUROC:.4f}",
        save_on_train_epoch_end=False,  # validation에서 계산된 image AUROC를 기준으로 저장한다.
    )
    engine = Engine(
        default_root_dir=f"{output_dir}/{run_name}",
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=1,
        logger=logger,
        log_every_n_steps=1,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
    )
    engine.fit(model=model, datamodule=datamodule)
    return engine
