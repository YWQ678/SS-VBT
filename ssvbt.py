"""
Self-Supervised Visible Blind Trace (SS-VBT) Denoising Implementation
Paper: "Self-Supervised Visible Blind Trace Framework for Low-SNR Seismic Data Denoising"
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional

class GlobalSeismicTraceMasker:
    def __init__(self, unit_size: int = 4):
        """
        Args:
            unit_size: Number of traces per masking unit (default=4 as per paper)
        """
        self.unit_size = unit_size

    def __call__(self, seismic: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate mask volume and masked input
        Args:
            seismic: Input seismic (H x W)
        Returns:
            mask_volume: Mask volume (H x W x W//s)
            masked_input: Masked seismic (H x W x W//s)
        """
        H, W = seismic.shape
        num_units = W // self.unit_size
        mask_volume = torch.zeros(H, W, num_units)
        masked_input = seismic.unsqueeze(2).repeat(1, 1, num_units)

        for i in range(num_units):
            # Create trace-wise mask
            mask = torch.ones_like(seismic)
            start = i * self.unit_size
            mask[:, start:start+self.unit_size] = 0
            mask_volume[:, :, i] = mask
            masked_input[:, start:start+self.unit_size, i] = 0

        return mask_volume, masked_input

class BlindTraceVisibilityLoss(nn.Module):
    def __init__(self, lambda_bt: float = 20.0, eta: float = 1.0):
        """
        Args:
            lambda_bt: Blind-to-visible weight (default=20 per paper Table 4)
            eta: Regularization weight (default=1 per paper Table 5)
        """
        super().__init__()
        self.lambda_bt = lambda_bt
        self.eta = eta
        self.mse = nn.MSELoss()

    def forward(self, 
               denoised: torch.Tensor, 
               noisy: torch.Tensor, 
               mask: torch.Tensor) -> torch.Tensor:
        """
        Compute hybrid loss
        Args:
            denoised: Network output (H x W)
            noisy: Original noisy input (H x W)
            mask: Binary mask (H x W)
        """
        # Blind trace term
        blind_loss = self.mse(denoised * mask, noisy * mask)
        
        # Visible term with gradient stop
        visible_term = (denoised - noisy).detach() + noisy
        visible_loss = self.mse(denoised, visible_term)
        
        # Regularization
        reg_loss = torch.mean(torch.abs(denoised))
        
        return blind_loss + self.lambda_bt * visible_loss + self.eta * reg_loss

class SSVBT(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels*2, 3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, in_channels, 3, padding=1),
            nn.Tanh()  # Constrain output to [-1,1]
        )
        self.masker = GlobalSeismicTraceMasker()
        self.loss_fn = BlindTraceVisibilityLoss()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask_volume, masked = self.masker(x)
        denoised_volume = torch.stack([self.decoder(self.encoder(m.unsqueeze(0).unsqueeze(0)).squeeze() 
                                     for m in masked.unbind(2)], dim=2)
        return denoised_volume.mean(dim=2), self.loss_fn(denoised_volume.mean(dim=2), x, mask_volume)
