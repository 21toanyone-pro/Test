import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import torchtext
torchtext.disable_torchtext_deprecation_warning()

from modelmodule import RobertaClassifier
from datamodule import AGNewsDataModule

torch.set_float32_matmul_precision('high')

def main():
    os.makedirs('ckpt', exist_ok=True)
    data_module = AGNewsDataModule(batch_size=8)
    model = RobertaClassifier()
    logger = CSVLogger("csv_logs", name="roberta_agnews")

    # Early Stopping, val_loss 기준
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=3,
        verbose=True,
        mode='min'
    )

    #모니터링, val_loss 기준
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        dirpath='ckpt',
        filename='best_resnet',
        save_top_k=1,
        save_last=False
    )

    #학습
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1 if torch.cuda.is_available() else None,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger, 
        enable_progress_bar=True
    )

    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    main()