import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalSeismicTraceMasker:
    """
    A module that masks seismic data by systematically masking columns of traces.
    Creates s different masked versions of the input where each version has 1/s of its traces masked.
    """

    def __init__(self, s=4):
        """
        Args:
            s (int): Number of masking patterns to create (also determines mask ratio as 1/s)
        """
        self.s = s

    def __call__(self, y):
        """
        Args:
            y (torch.Tensor): Input seismic data of shape (batch_size, channels, height, width)

        Returns:
            torch.Tensor: Masked data volume of shape (batch_size, s, channels, height, width)
            torch.Tensor: Mask positions (for reconstruction) of shape (batch_size, s, 1, height, width)
        """
        batch_size, channels, H, W = y.shape
        device = y.device

        # Create s different masking patterns
        masked_data = []
        mask_positions = []

        for i in range(self.s):
            # Create mask for the i-th column in each s-sized block
            mask = torch.ones_like(y)
            for w in range(W):
                if w % self.s == i:
                    mask[:, :, :, w] = 0  # Mask this column

            # Apply mask
            masked = y * mask
            masked_data.append(masked)
            mask_positions.append(1 - mask)  # 1 where masked, 0 otherwise

        # Stack all masked versions along a new dimension
        masked_volume = torch.stack(masked_data, dim=1)  # (batch_size, s, C, H, W)
        mask_positions = torch.stack(mask_positions, dim=1)  # (batch_size, s, C, H, W)

        return masked_volume, mask_positions


class GlobalFusionModule(nn.Module):
    """
    A module that fuses the predictions from s different masked versions of the input
    by combining the predicted values for the masked regions.
    """

    def __init__(self, s=4):
        """
        Args:
            s (int): Number of masking patterns that were used (should match masker)
        """
        super().__init__()
        self.s = s

    def forward(self, denoised_volume, mask_positions):
        """
        Args:
            denoised_volume (torch.Tensor): Denoised predictions for each masked version
                Shape: (batch_size, s, channels, height, width)
            mask_positions (torch.Tensor): Positions that were masked in each version
                Shape: (batch_size, s, channels, height, width)

        Returns:
            torch.Tensor: Fused denoised output of shape (batch_size, channels, height, width)
        """
        batch_size, s, channels, H, W = denoised_volume.shape

        # Initialize output with zeros
        output = torch.zeros(batch_size, channels, H, W, device=denoised_volume.device)
        count = torch.zeros(batch_size, channels, H, W, device=denoised_volume.device)

        # Sum all predictions for masked positions
        for i in range(self.s):
            # Add the denoised values where this version had masked traces
            output += denoised_volume[:, i] * mask_positions[:, i]
            count += mask_positions[:, i]

        # For positions that were never masked (count=0), use average of all predictions
        never_masked = (count == 0)
        if never_masked.any():
            avg_prediction = denoised_volume.mean(dim=1)  # Average across s versions
            output[never_masked] = avg_prediction[never_masked]
            count[never_masked] = 1  # To avoid division by zero

        # Normalize by the count (average the predictions)
        output = output / count

        return output


# Example usage:
if __name__ == "__main__":
    # Create dummy seismic data (batch_size=2, channels=1, height=64, width=64)
    dummy_data = torch.randn(2, 1, 64, 64)

    # Create masker and apply it
    masker = GlobalSeismicTraceMasker(s=4)
    masked_volume, mask_positions = masker(dummy_data)

    print(f"Original shape: {dummy_data.shape}")
    print(f"Masked volume shape: {masked_volume.shape}")  # Should be (2, 4, 1, 64, 64)

    # Simulate denoising process (in practice, this would be a neural network)
    # Here we just add some noise to simulate denoising
    denoised_volume = masked_volume + 0.1 * torch.randn_like(masked_volume)

    # Create fusion module and apply it
    fusion = GlobalFusionModule(s=4)
    final_output = fusion(denoised_volume, mask_positions)

    print(f"Final output shape: {final_output.shape}")  # Should match original (2, 1, 64, 64)
