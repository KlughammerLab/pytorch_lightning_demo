#!/usr/bin/env python3
"""
Training script for the linear model

author: jy
"""
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import GradientAccumulationScheduler
from linear import LinearModel



if __name__ == "__main__":
    seed_everything(42)
    torch.set_float32_matmul_precision("medium")
    ndevices = torch.cuda.device_count()
    
    ###########################################################################
    # Set fake data and dataloaders

    # 10 features, 2 modalities -> 10 targets
    train_data = torch.randn(100, 10, 2)
    train_targets = torch.randn(100, 10)

    val_data = torch.randn(10, 10, 2)
    val_targets = torch.randn(10, 10)

    test_data = torch.randn(10, 10, 2)
    test_targets = torch.randn(10, 10)

    train_loader = DataLoader(
        list(zip(train_data, train_targets)),
        batch_size=10,
        shuffle=True
    )

    val_loader = DataLoader(
        list(zip(train_data, train_targets)),
        batch_size=10,
        shuffle=False
    )

    test_loader = DataLoader(
        list(zip(train_data, train_targets)),
        batch_size=10,
        shuffle=False
    )

    ###########################################################################
    # Init model
    demo_model = LinearModel(
        input_dim=2,
        output_dim=1,
    )

    ###########################################################################
    # Init trainer
    callbacks = [
        ModelCheckpoint(
            every_n_epochs=1,
            save_last=True,
            monitor="vali.loss",
            save_top_k=2,
        ),
        GradientAccumulationScheduler(
            scheduling={0: 1, 10: 2}
        )
    ]
    trainer_params = dict(
        max_epochs = 1,
        devices = ndevices,
        accelerator = "cpu" if ndevices == 0 else "cuda",
        precision = 32,
        num_nodes = 1,
        default_root_dir = "../logs",
        log_every_n_steps = 1,
        accumulate_grad_batches = 1,
        gradient_clip_val = 0.5,
        gradient_clip_algorithm = "value",
    )
    trainer = Trainer(
        **trainer_params,
        # enable_checkpointing = False, # for deactivating checkpointing
        callbacks = callbacks,
        enable_progress_bar=False, # deactivate progress bar for slurm job
    )

    ###########################################################################
    # Train the model
    torch.cuda.empty_cache()
    trainer.fit(
        model = demo_model,
        train_dataloaders = train_loader,
        val_dataloaders = val_loader,
        ckpt_path=None,
    )

    # Test the model
    torch.cuda.empty_cache()
    trainer.test(
        model = demo_model,
        dataloaders=test_loader,
    )
    
    ###########################################################################
    # loading checkpoint
    ckpt_path = trainer.checkpoint_callback.best_model_path
    demo_model = LinearModel.load_from_checkpoint(
        ckpt_path,
        **demo_model.hparams,
    )
    print(f"Loading checkpoint from {ckpt_path}")

    trainer_params["max_epochs"] *= 2
    trainer = Trainer(
        **trainer_params,
        callbacks = callbacks,
        enable_progress_bar=False,
    )

    ###########################################################################
    # Train the model
    torch.cuda.empty_cache()
    trainer.fit(
        model = demo_model,
        train_dataloaders = train_loader,
        val_dataloaders = val_loader,
        ckpt_path=None,
    )

    # Test the model
    torch.cuda.empty_cache()
    trainer.test(
        model = demo_model,
        dataloaders=test_loader,
    )