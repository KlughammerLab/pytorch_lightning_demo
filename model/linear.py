#!/usr/bin/env python3
"""
Simple Linear Model

author: jy
"""
import torch
from torch import nn
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts



class LinearModel(pl.LightningModule):
    """
    A simple linear model implemented with PyTorch Lightning

    Args:
        input_dim (int): The number of input features
        output_dim (int): The number of output features
        criterion (nn.Module): The loss function to use

        learning_rate (float): The learning rate for the optimizer
        weight_decay (float): The weight decay for the optimizer
        betas (tuple): The betas for the optimizer

        sch_T_0 (int): The initial number of iterations for the learning rate scheduler
        sch_T_mult (int): The multiplier for the number of iterations for the learning rate scheduler
        sch_eta_min (float): The minimum learning rate for the learning rate scheduler
    """
    def __init__(
            self,
            input_dim:int,
            output_dim:int,
            criterion:nn.Module = nn.MSELoss(reduction='mean'),
            learning_rate:float = 1e-4,
            weight_decay:float = 0.001,
            betas:tuple = (0.9, 0.999),
            sch_T_0:int = 20,
            sch_T_mult:int = 2,
            sch_eta_min:float = 1e-6
        ) -> None:
        super(LinearModel, self).__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Flatten(start_dim=1, end_dim=-1),
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )
            elif isinstance(
                m, (nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Loss function
        self.criterion = criterion

        # Optimizer hyper
        self.optimizer_hyper = dict(
            lr = learning_rate,
            weight_decay = weight_decay,
            betas = betas
        )

        # Learning rate scheduler hyper
        self.scheduler_hyper = dict(
            T_0 = sch_T_0,
            T_mult = sch_T_mult,
            eta_min = sch_eta_min
        )   

    def forward(self, x):
        return self.model(x)
    
    @torch.no_grad()
    def cal_grad_norm(self,) -> float:
        """
        Calculate the gradient norm
        """
        grad_norm = 0.0
        for p in self.parameters():
            if p.grad is None or p.requires_grad == False:
                continue
            grad_norm += torch.abs(p.grad.data.norm(2)).item()
        return grad_norm

    def training_step(self, batch, batch_idx:int = None) -> torch.Tensor:
        x, y = batch
        out = self(x)
        loss = self.criterion(out, y)

        # Update log
        self.log_dict(
            {
                "train.loss": loss,
                "train.grad_norm": self.cal_grad_norm()
            }, 
            sync_dist=True,
            prog_bar=True
        )
        return loss
    
    def validation_step(self, batch, batch_idx:int = None) -> torch.Tensor:
        x, y = batch
        out = self(x)
        loss = self.criterion(out, y)

        # Update log
        self.log_dict(
            {"vali.loss": loss,}, 
            sync_dist=True,
            prog_bar=True
        )
        return loss

    def test_step(self, batch, batch_idx:int = None) -> torch.Tensor:
        x, y = batch
        out = self(x)
        loss = self.criterion(out, y)

        # Update log
        self.log_dict(
            {"test.loss": loss,}, 
            sync_dist=True,
            prog_bar=True
        )
        return loss

    def configure_optimizers(self) -> list:
        """
        Create the optimizer and the training schedule:
        """
        optimizer = AdamW(
            self.parameters(),
            **self.optimizer_hyper
        )

        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            **self.scheduler_hyper
        )

        return [optimizer], [scheduler]