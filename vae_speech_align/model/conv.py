import torch


class ConvModuleLayer(torch.nn.Module):
    """Single conv layer with layer norm, ReLU, and dropout."""

    def __init__(
        self,
        hidden_size: int,
        dropout_rate: float,
        kernel_size: int,
    ) -> None:
        super().__init__()

        self.layer_norm = torch.nn.LayerNorm(hidden_size)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.activation = torch.nn.ReLU()
        self.conv = torch.nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]

        h: torch.Tensor = x
        h = self.layer_norm(h)
        h = h.transpose(-1, -2)
        h = self.conv(h)
        h = h.transpose(-1, -2)
        h = self.activation(h)
        h = self.dropout(h)

        return h


class UpsampleConvModule(torch.nn.Module):
    """Upsamples a sequence by a fixed scale using transposed convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample_scale: int,
    ) -> None:
        super().__init__()

        self.conv = torch.nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=upsample_scale,
            stride=upsample_scale,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]

        h: torch.Tensor = x
        h = h.transpose(-1, -2)
        h = self.conv(h)
        h = h.transpose(-1, -2)
        h = h.contiguous()

        return h


class DownsampleConvModule(torch.nn.Module):
    """Downsamples a sequence by a fixed scale using strided convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample_scale: int,
    ) -> None:
        super().__init__()

        self.conv = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=downsample_scale,
            stride=downsample_scale,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]

        h: torch.Tensor = x
        h = h.transpose(-1, -2)
        h = self.conv(h)
        h = h.transpose(-1, -2)
        h = h.contiguous()

        return h


class ConvModule(torch.nn.Module):
    """Stack of convolutional layers with optional residual connections."""

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        dropout_rate: float,
        kernel_size: int,
        residual: bool,
    ) -> None:
        super().__init__()

        self.residual = residual
        layers = []
        for _ in range(num_layers):
            layers.append(
                ConvModuleLayer(
                    hidden_size=hidden_size,
                    kernel_size=kernel_size,
                    dropout_rate=dropout_rate,
                )
            )

        self.layers = torch.nn.ModuleList(layers)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # x: [B, T, D]
        # mask: [B, T]

        if mask is not None:
            x = x * mask[:, :, None]

        h: torch.Tensor = x
        for layer in self.layers:
            if self.residual:
                h = h + layer(h)
            else:
                h = layer(h)

        return h
