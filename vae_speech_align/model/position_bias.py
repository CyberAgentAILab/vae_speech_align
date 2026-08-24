import torch


def log_binomial_coef(n: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Compute log(C(n, k)) using the log-gamma function."""
    return torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)


def log_beta_func(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Compute log(B(z1, z2)) using the log-gamma function."""
    return torch.lgamma(z1) + torch.lgamma(z2) - torch.lgamma(z1 + z2)


def calc_beta_binomial_matrix(
    output_lengths: torch.Tensor,
    state_lengths: torch.Tensor,
    scaling_factor: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Compute log beta-binomial matrix [B, T, K] for position bias."""

    T_max = int(output_lengths.max().item())
    K_max = int(state_lengths.max().item())
    B = len(output_lengths)

    device = output_lengths.device

    nt = torch.arange(0, T_max, device=device)[None, :, None]
    nk = torch.arange(0, K_max, device=device)[None, None, :]

    nt = torch.tile(nt, (B, 1, 1))  # [B, T, 1]
    nk = torch.tile(nk, (B, 1, 1))  # [B, 1, K]

    # clamp out-of-range state indices down to the last valid state so the
    # binomial/beta terms below stay defined for padded positions
    nk = torch.where(
        nk >= state_lengths[:, None, None],
        state_lengths[:, None, None] - 1,
        nk,
    )

    alpha = scaling_factor * nt + 1
    beta = torch.clamp(
        scaling_factor * (output_lengths[:, None, None] - 1 - nt) + 1, min=1.0
    )

    K_minus_1 = state_lengths[:, None, None] - 1  # [B, 1, 1]

    bb1 = log_binomial_coef(K_minus_1, nk)
    bb2 = log_beta_func(nk + alpha, K_minus_1 - nk + beta)
    bb3 = log_beta_func(alpha, beta)

    return bb1 + bb2 - bb3


class PositionBiasModule(torch.nn.Module):
    """Position bias based on beta-binomial distribution.

    The scaling factor (scale) is multiplied by update_rate at each call to
    update_phase(), allowing the bias strength to be gradually reduced over
    training steps.
    """

    # registered buffer; declared so mypy sees a Tensor, not
    # the Tensor | Module union of nn.Module attribute lookup
    scale: torch.Tensor

    def __init__(
        self,
        initial_scale: float,
        update_rate: float,
    ) -> None:
        super().__init__()

        if initial_scale <= 0:
            raise ValueError(
                f"initial_scale must be positive, got {initial_scale}"
            )
        if not 0 < update_rate <= 1:
            raise ValueError(
                f"update_rate must be in (0, 1], got {update_rate}"
            )

        self.update_rate = update_rate
        self.initial_scale = initial_scale
        self.register_buffer("scale", torch.tensor([initial_scale]))

    def update_phase(self) -> None:
        """Decay the scale by update_rate."""
        self.scale[:] *= self.update_rate

    def reset_scale(self) -> None:
        """Reset scale to the initial value."""
        self.scale[:] = self.initial_scale

    def forward(
        self,
        output_lengths: torch.Tensor,
        state_lengths: torch.Tensor,
    ) -> torch.Tensor:

        return calc_beta_binomial_matrix(
            output_lengths=output_lengths,
            state_lengths=state_lengths,
            scaling_factor=self.scale,
        )
