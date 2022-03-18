# ------------------------------------------------------------------------------
# Base Segmentation Datamodule for PyTorch and PyTorch Lighning
# ------------------------------------------------------------------------------
from typing import Any, Dict, cast

import torch
from torch.nn import functional as F

import segmentation_models_pytorch as smp
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pl_bolts.models.vision.unet import UNet
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, DeviceStatsMonitor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from torchmetrics import MetricCollection, Accuracy, IoU

from ..datasets.utils import unbind_samples
from ..datamodules.segmentation_datamodule import SegmentationDataModule

# -------------------------------------------------------------------------------
# class UNet
# This class performs training and classification of satellite imagery using a
# UNet CNN.
# -------------------------------------------------------------------------------
@MODEL_REGISTRY
class UNetSegmentation(LightningModule):

    # ---------------------------------------------------------------------------
    # __init__
    # ---------------------------------------------------------------------------
    def __init__(self, **kwargs: Any) -> None:
        
        super().__init__()
        self.save_hyperparameters()  # creates `self.hparams` from kwargs

        print(self.hparams)

        self.ignore_zeros = None if kwargs["ignore_zeros"] else 0

        self.model = smp.Unet(
            encoder_name=self.hparams["encoder_name"],
            encoder_weights=self.hparams["encoder_weights"],
            in_channels=self.hparams["input_channels"],
            classes=self.hparams["num_classes"],
        )

        self.train_metrics = MetricCollection(
            [
                Accuracy(
                    num_classes=self.hparams["num_classes"],
                    ignore_index=self.ignore_zeros,
                ),
                IoU(
                    num_classes=self.hparams["num_classes"],
                    ignore_index=self.ignore_zeros,
                ),
            ],
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")

    # ---------------------------------------------------------------------------
    # model methods
    # ---------------------------------------------------------------------------
    def forward(self, x):
        return self.model(x)

    def compute_loss(self, out, mask):
        return F.cross_entropy(out, mask)

    def training_step(
            self, batch: Dict[str, Any], batch_idx: int
        ) -> torch.Tensor:
        x, y = batch["image"].float(), batch["label"].long()
        y_hat = self.forward(x)
        y_hat_bin = y_hat.argmax(dim=1)
        loss = self.compute_loss(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.train_metrics(y_hat_bin, y)
        return cast(torch.Tensor, loss)

    def training_epoch_end(self, outputs: Any) -> None:
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(
            self, batch: Dict[str, Any], batch_idx: int
        ) -> None:
        x, y = batch["image"].float(), batch["label"].long()
        y_hat = self.forward(x)
        y_hat_bin = y_hat.argmax(dim=1)
        loss = self.compute_loss(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.val_metrics(y_hat_bin, y)

        if batch_idx < 10:
            try:
                datamodule = self.trainer.datamodule  # type: ignore[attr-defined]
                batch["prediction"] = y_hat_bin
                for key in ["image", "label", "prediction"]:
                    batch[key] = batch[key].cpu()
                sample = unbind_samples(batch)[0]
                fig = datamodule.plot(sample)
                summary_writer = self.logger.experiment
                summary_writer.add_figure(
                    f"image/{batch_idx}", fig, global_step=self.global_step
                )
            except AttributeError:
                pass

    def validation_epoch_end(self, outputs: Any) -> None:
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def test_step(
            self, batch: Dict[str, Any], batch_idx: int
        ) -> None:
        x, y = batch["image"].float(), batch["label"].long()
        y_hat = self.forward(x)
        y_hat_bin = y_hat.argmax(dim=1)
        loss = self.compute_loss(y_hat, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.test_metrics(y_hat_bin, y)

    def test_epoch_end(self, outputs: Any) -> None:
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()
    
    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.hparams["learning_rate"]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer, patience=self.hparams["learning_rate_schedule_patience"]
                ),
                "monitor": "val_loss",
            },
        }

def cli_main():

    seed_everything(1234)
    
    # model args
    model = UNetSegmentation(
        input_channels=3,
        num_classes=2,
        num_layers=5,
        features_start=64,
        bilinear=False,
        lr=0.0001
    )

    datamodule =  SegmentationDataModule(

        # Dataset parameters
        input_bands=['CB', 'B', 'G', 'Y', 'R', 'RE', 'N1', 'N2'],
        output_bands=['B', 'G', 'R'],
        tile_size=256,
        seed=42,
        max_patches=500,
        dataset_dir='/lscratch/jacaraba/terragpu/clouds/senegal',
        generate_dataset=True,
        images_regex='/adapt/nobackup/projects/ilab/projects/Senegal/LCLUC_Senegal_Cloud/training/data/*.tif',
        labels_regex='/adapt/nobackup/projects/ilab/projects/Senegal/LCLUC_Senegal_Cloud/training/labels/*.tif',
        transform=False,
        test_size=0.20,
        normalize=False,
        downscale=False,
        standardize=False,

        # Datamodule parameters
        val_split=0.2,
        test_split=0.1,
        
        # Performance parameters
        batch_size=32,
        shuffle=True,
    )

    # default logger used by trainer
    logger = TensorBoardLogger(
        save_dir="/lscratch/jacaraba/terragpu/clouds/senegal/tensor",
        version=1, name="lightning_logs")

    # train
    trainer = Trainer(
        gpus=4,
        num_processes=40,
        strategy='ddp',
        precision=16,
        logger=logger,
        default_root_dir="/lscratch/jacaraba/terragpu/clouds/senegal/saving_model",
        callbacks=[
            EarlyStopping(monitor="val_loss"),
            ModelCheckpoint(dirpath="/lscratch/jacaraba/terragpu/clouds/senegal/saving_model", save_top_k=2, monitor="val_loss"),
            DeviceStatsMonitor()
        ],
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    cli_main()