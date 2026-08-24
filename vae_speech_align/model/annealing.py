import logging

import torch

from vae_speech_align.forwardsum.matrix_util import lengths_to_matrix_mask

_logger = logging.getLogger(__name__)

EPS = 1e-6


class AnnealingConv(torch.nn.Module):
    """Gaussian convolution for smoothing the alignment gamma matrix.

    The sigma parameter controls the smoothing width and is decayed by
    update_rate at each call to update_phase().
    """

    # registered buffer; declared so mypy sees a Tensor, not
    # the Tensor | Module union of nn.Module attribute lookup
    sigma: torch.Tensor

    def __init__(
        self,
        initial_sigma: float,
        update_rate: float,
        kernel_size: int,
    ) -> None:
        super().__init__()

        if initial_sigma <= 0:
            raise ValueError(
                f"initial_sigma must be positive, got {initial_sigma}"
            )
        if not 0 < update_rate <= 1:
            raise ValueError(
                f"update_rate must be in (0, 1], got {update_rate}"
            )

        self.kernel_size = kernel_size

        padding = (kernel_size - 1) // 2

        self.conv = torch.nn.Conv1d(
            1,
            1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.conv.requires_grad_(False)

        self.update_rate = update_rate
        self.initial_sigma = initial_sigma
        self.register_buffer("sigma", torch.tensor([initial_sigma]))

        self.set_weight()

    def set_weight(self) -> None:
        """Set conv kernel weights from a Gaussian with current sigma."""
        padding = (self.kernel_size - 1) // 2

        idx = (
            (torch.arange(0, self.kernel_size) - padding)
            .float()
            .to(self.conv.weight.device)
        )
        self.conv.weight[:] = torch.exp(-((idx / self.sigma) ** 2.0))

    def update_phase(self) -> None:
        """Decay sigma by update_rate and refresh kernel weights."""
        self.sigma[:] *= self.update_rate

        self.set_weight()

    def reset_weight(self) -> None:
        """Reset sigma to the initial value and refresh kernel weights."""
        self.sigma[:] = self.initial_sigma

        self.set_weight()

    def forward(
        self,
        x: torch.Tensor,
        output_lengths: torch.Tensor,
        state_lengths: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:

        B = x.size(0)
        T = x.size(1)
        K = x.size(2)

        mask = lengths_to_matrix_mask(output_lengths, state_lengths, x.device)

        padded_mask = torch.nn.functional.pad(
            mask, (1, 1, 1, 1), "constant", False
        )

        # x: [B, T, K]

        x = x * padded_mask
        x = x.unsqueeze(-2)  # [B, T, 1, K]
        x = x.view(B * T, 1, K)  # [BxT, 1, K]
        y: torch.Tensor = self.conv(x)  # [BxT, 1, K]
        y = y.squeeze(-2)  # [BxT, K]
        y = y.view(B, T, K)  # [B, T, K]

        y = y * padded_mask

        if normalize:
            y /= y.sum(dim=2, keepdim=True) + EPS

        return y
