import torch


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average values over the True positions of mask.

    Raises:
        ValueError: If the mask has no valid positions, which would
            otherwise turn the loss into a silent 0/0 NaN.
    """
    mask_total = mask.sum()
    if mask_total <= 0:
        raise ValueError("mask has no valid positions")
    return (values * mask).sum() / mask_total


class MaskedVAEKLDLoss(torch.nn.Module):
    """VAE KL divergence loss with sequence-length masking."""

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, mean: torch.Tensor, lvar: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        loss_all = (-0.5 * (1 + lvar - (mean**2.0) - torch.exp(lvar))).sum(
            dim=-1
        )
        return _masked_mean(loss_all, mask)


class MaskedMSELoss(torch.nn.Module):
    """MSE loss with sequence-length masking."""

    def __init__(self) -> None:
        super().__init__()
        self.loss_func = torch.nn.MSELoss(reduction="none")

    def forward(
        self, output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        loss_all: torch.Tensor = self.loss_func(output, target).sum(dim=-1)
        return _masked_mean(loss_all, mask)


class MaskedCrossEntropyLoss(torch.nn.Module):
    """Cross-entropy loss with sequence-length masking."""

    def __init__(self) -> None:
        super().__init__()
        self.loss_func = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(
        self, output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        B, N, D = output.size()

        loss_all: torch.Tensor = self.loss_func(
            output.view(B * N, D), target.view(B * N)
        ).view(B, N)

        return _masked_mean(loss_all, mask)
