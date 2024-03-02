from kws.model import ViT_Lightning, ConfMatrixLogging
from kws.data import AudioDataModule
from kws.preprocessing_data.preproccesing import KNOWN_COMMANDS

import lightning as L
from lightning.pytorch.loggers import WandbLogger
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

import hydra
from hydra.core.config_store import ConfigStore
from config import Params

cs = ConfigStore.instance()
cs.store(name="params", node=Params)

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: Params) -> None:
    working_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    dm = AudioDataModule(
        data_destination=cfg.data.path_to_data,
        batch_size=cfg.training.batch,
        n_mels=cfg.data.n_mels,
        hop_length=cfg.data.hop_length
    )
    
    model = ViT_Lightning(
        time_window=cfg.data.time_window,
        frequency=cfg.data.frequency,
        patch_size_t=cfg.data.patch_size_time,
        patch_size_f=cfg.data.patch_size_freq,
        # -----
        embed_dim=cfg.model.embedding_dim,
        num_classes=cfg.model.num_output_classes,
        depth=cfg.model.layers,
        num_heads=cfg.model.heads,
        mlp_dim=cfg.model.mlp_dim,
        drop_rate=cfg.model.dropout,
        norm_type=cfg.model.norm_type,
        # -----
        lr=cfg.training.lr,
        # -----
        type_of_scheduler = cfg.scheduler_train.type_of_scheduler, 
        patience_reduce = cfg.scheduler_train.patience_reduce, 
        factor_reduce = cfg.scheduler_train.factor_reduce,
        lr_coef_cycle = cfg.scheduler_train.lr_coef_cycle, 
        total_num_of_epochs = cfg.training.epochs,
        # -----
        # sample_rate = cfg.mfcc_settings.sample_rate, 
        # n_mffc = cfg.mfcc_settings.n_mfcc, 
        # n_mels = cfg.mfcc_settings.n_mels, 
        # n_fft = cfg.mfcc_settings.n_fft, 
        # hop_length=cfg.mfcc_settings.hop_length,
        # previous_model = None
    )

    # model = ViT_Lightning.load_from_checkpoint("outputs/2023-12-19/13-26-05/weights/epoch=48-step=8134.ckpt")

    wandb.login(key="dec2ee769ce2e455dd463be9b11767cf8190d658")
    wandb_log = WandbLogger(project=cfg.training.project_name, name=cfg.training.train_name, save_dir=working_dir + cfg.training.wandb_path)

    checkpoint = ModelCheckpoint(
        dirpath=working_dir + cfg.training.model_path,
        filename="epoch_{epoch}-val_loss_{val_loss:.2f}-val_acc_{val_acc:.2f}",
        save_top_k=cfg.training.save_best_of,
        monitor=cfg.training.checkpoint_monitor
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    conf_matrix_logger = ConfMatrixLogging(KNOWN_COMMANDS)
    early_stop = EarlyStopping(monitor=cfg.training.checkpoint_monitor, patience=cfg.training.early_stopping_patience)

    trainer = L.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator="auto",
        devices=1,
        logger=wandb_log,
        callbacks=[checkpoint, lr_monitor, conf_matrix_logger, early_stop],
        fast_dev_run=5
    )
    trainer.fit(model=model, datamodule=dm)

    wandb.finish()


if __name__ == "__main__":
    L.seed_everything(1702)
    main()