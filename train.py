import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from dataset import PansharpeningDataModule, RSDataset
from network import CSSL
from src.plot_log import SimplePlotCallback


def infer_ms_channels(sensor):
    return 8 if sensor.lower() in {"worldview-2", "worldview-3", "wv2", "wv3"} else 4


def infer_num_classes(sensor):
    return 8 if sensor.lower() in {"worldview-2", "wv2"} else 7


def main(args):
    seed_everything(args.seed, workers=True)

    data_root = Path(args.data_dir) / args.sensor
    train_dataset = RSDataset(data_root / "Train", satellite=args.sensor, resolution="Hybrid")
    val_dataset = RSDataset(data_root / "val", satellite=args.sensor, resolution="Hybrid")
    datamodule = PansharpeningDataModule(
        train_dataset,
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    ms_channels = args.ms_channels or infer_ms_channels(args.sensor)
    num_classes = args.num_classes or infer_num_classes(args.sensor)
    model = CSSL(
        ms_channels=ms_channels,
        num_classes=num_classes,
        task=args.task,
        sensor=args.sensor,
        spectral_weight=args.spectral_weight,
        spatial_weight=args.spatial_weight,
        learning_rate=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
    )

    logger = CSVLogger(save_dir=args.log_dir, name=args.experiment_name, version=args.version)
    plot_callback = SimplePlotCallback(
        plot_interval=args.plot_interval,
        input_dir=str(Path(args.log_dir) / args.experiment_name / str(args.version)),
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        monitor=args.monitor,
        mode="min",
        save_top_k=1,
        auto_insert_metric_name=False,
        filename=f"{args.experiment_name}-{args.version}",
        every_n_epochs=args.val_freq,
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        logger=logger,
        check_val_every_n_epoch=args.val_freq,
        callbacks=[checkpoint_callback, plot_callback],
        deterministic=args.deterministic,
    )
    trainer.fit(model, datamodule)


def build_parser():
    parser = argparse.ArgumentParser(description="Train CSSL for pansharpening.")
    parser.add_argument("--data-dir", default="data/psc")
    parser.add_argument("--sensor", default="worldview-2")
    parser.add_argument("--experiment-name", default="CSSL")
    parser.add_argument("--version", default="worldview-2")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--epochs", default=600, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--ms-channels", default=None, type=int)
    parser.add_argument("--num-classes", default=None, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--lr-step-size", default=50, type=int)
    parser.add_argument("--lr-gamma", default=0.8, type=float)
    parser.add_argument("--spectral-weight", default=1.0, type=float)
    parser.add_argument("--spatial-weight", default=1.0, type=float)
    parser.add_argument("--val-freq", default=1, type=int)
    parser.add_argument("--plot-interval", default=1, type=int)
    parser.add_argument("--monitor", default="val_loss")
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", default=1, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--task", action=argparse.BooleanOptionalAction, default=True)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
