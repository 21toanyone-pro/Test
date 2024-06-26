import os
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from modelmodule import ResNetClassifier
from datamodule import CIFAR10DataModule


torch.set_float32_matmul_precision('high')

def main():
    os.makedirs('ckpt', exist_ok=True)
    data_module = CIFAR10DataModule(data_dir='./data', batch_size=64)
    model = ResNetClassifier()
    logger = TensorBoardLogger("tb_logs", name="resnet_cifar10")

    # Early Stopping, val_acc 기준
    early_stop_callback = EarlyStopping(
        monitor='val_acc',
        min_delta=0.00,
        patience=3,
        verbose=True,
        mode='max'
    )
    
    #모니터링, val_acc 기준
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        mode='max',
        dirpath='ckpt',
        filename='best_resnet',
        save_top_k=1,
        save_last=False
    )
    
    #학습
    trainer = Trainer(
        max_epochs=3,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1 if torch.cuda.is_available() else None,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        enable_progress_bar=True
    )

    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    main()