import numpy as np
import torch

OUTPUT_LEN = 5


def make_forwardsum_data_numberable_pathes() -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Return a toy example whose alignment paths can be enumerated.

    Builds a single arc-shaped 2-D observation sequence of T=5 frames
    together with K=3 state Gaussians placed along the arc. With only
    five frames and three left-to-right states there are exactly six
    monotonic alignment paths, so tests can brute-force all of them
    and compare against the DP results.

    Returns:
        A tuple of the observations [1, 5, 2], the state means
        [1, 3, 2], and the state variances [1, 3, 2].
    """
    ny_n = np.linspace(0, 1, OUTPUT_LEN)  # [T]
    y1 = ((1 + 0.1 * ny_n) * np.cos(0.5 * np.pi * ny_n))[:, None]  # [T, 1]
    y2 = ((1 + 0.1 * ny_n) * np.sin(0.5 * np.pi * ny_n))[:, None]  # [T, 1]
    y = torch.from_numpy(np.hstack([y1, y2]).astype(np.float32))  # [T, 2]
    y = y.unsqueeze(0)  # [1, T, 2]

    output_mean: torch.Tensor = torch.FloatTensor(
        [
            [1.2, 0.0],
            [0.6, 0.6],
            [0, 1.2],
        ]
    )  # [K, 2]
    output_mean = output_mean.unsqueeze(0)

    output_var = (0.5**2.0) * torch.ones_like(output_mean)

    return y, output_mean, output_var


def make_data_simple_2d(
    mel_b1_len: int = 40,
    text_b1_len: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a rough two-element batch of 2-D mel/text-like features.

    The second batch element is a half-length, noise-perturbed copy of
    the first, so the batch exercises the length masking of the DP
    backends. Both sequences are zero-padded to a common length.

    Args:
        mel_b1_len: Frame count of the first element; the second gets
            half of it.
        text_b1_len: State count of the first element; the second gets
            half of it.

    Returns:
        A tuple of the padded mel features [2, T, 2], the padded text
        features [2, K, 2], the mel lengths, and the text lengths.
    """
    mel_b2_len = mel_b1_len // 2
    text_b2_len = text_b1_len // 2

    ny = np.arange(1, mel_b1_len + 1)
    ny_n = ny / mel_b1_len
    mel_y1 = ((1 + ny_n) * np.cos(4 * np.pi * ny_n))[:, None]
    mel_y2 = ((1 + ny_n) * np.sin(4 * np.pi * ny_n))[:, None]
    mel_b1 = torch.from_numpy(np.hstack([mel_y1, mel_y2]).astype(np.float32))

    if text_b1_len == 8:
        text_b1 = torch.from_numpy(
            np.array(
                [
                    [1.0, 1.0],
                    [-0.5, 1.0],
                    [-1.5, 0.0],
                    [0, -1.5],
                    [1.0, 1.0],
                    [-0.5, 1.0],
                    [-1.5, 0.0],
                    [0, -1.5],
                ]
            ).astype(np.float32)
        )
    else:
        torch.manual_seed(553)
        text_b1 = torch.randn((text_b1_len, 2))

    torch.manual_seed(543)
    text_b2 = text_b1[:text_b2_len, :].clone()
    mel_b2 = mel_b1[:mel_b2_len, :].clone()
    mel_b2 += 0.2 * torch.randn_like(mel_b2)

    text = torch.nn.utils.rnn.pad_sequence(
        [text_b1, text_b2], batch_first=True
    )
    mel = torch.nn.utils.rnn.pad_sequence([mel_b1, mel_b2], batch_first=True)

    text_lengths = torch.LongTensor([text_b1_len, text_b2_len])
    mel_lengths = torch.LongTensor([mel_b1_len, mel_b2_len])

    return mel, text, mel_lengths, text_lengths
