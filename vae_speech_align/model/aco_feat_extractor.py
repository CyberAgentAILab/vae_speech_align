import logging
from dataclasses import dataclass
from typing import Any

import torch
import torchaudio
from transformers import HubertModel, Wav2Vec2Model, WavLMModel

from vae_speech_align import config
from vae_speech_align.utils import length_to_input_mask

SSL_SAMPLING_RATE = 16000
SSL_FRAME_RATE = 50

_logger = logging.getLogger(__name__)

_SSL_MODEL_CLASSES = {
    config.AcousticFeatureType.WAV2VEC2: Wav2Vec2Model,
    config.AcousticFeatureType.HUBERT: HubertModel,
    config.AcousticFeatureType.WAVLM: WavLMModel,
}


class AcousticFeatureExtractorBase(torch.nn.Module):
    """Base class of the acoustic feature extractors.

    Subclasses set frame_rate, the number of feature frames per second
    of audio, so downstream code can convert frame counts to seconds.
    """

    frame_rate: float


def create_aco_feat_extractor(
    aco_feat_config: config.AcousticFeatureExtractor,
) -> AcousticFeatureExtractorBase:
    """Factory function to create an acoustic feature extractor from config."""
    if aco_feat_config.model_type in _SSL_MODEL_CLASSES:
        return SSLFeatureExtractor(
            model_class=_SSL_MODEL_CLASSES[aco_feat_config.model_type],
            model_name=aco_feat_config.model_name,
            output_layer=aco_feat_config.layer_idx,
            num_dims=aco_feat_config.num_dims,
        )
    elif aco_feat_config.model_type == config.AcousticFeatureType.MELSPEC:
        return MelspecExtractor(num_dims=aco_feat_config.num_dims)
    else:
        raise NotImplementedError(
            f"unsupported acoustic feature type: {aco_feat_config.model_type}"
        )


class SSLFeatureExtractor(AcousticFeatureExtractorBase):
    """Extract features from a pretrained SSL model (Wav2Vec2/HuBERT/WavLM)."""

    def __init__(
        self,
        model_class: type[Any],
        model_name: str,
        output_layer: int,
        num_dims: int,
    ) -> None:
        super().__init__()
        _logger.info(f"Load model: {model_name}")
        self.ssl_model = model_class.from_pretrained(model_name)
        self.ssl_model.eval()

        self.output_layer = output_layer
        self.frame_rate = float(SSL_FRAME_RATE)

        self.layer_norm = torch.nn.LayerNorm(num_dims)

    def forward(
        self, wav: torch.Tensor, wav_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # wav: [B, num_samples]

        with torch.no_grad():

            # Without the attention mask the transformer attends to the
            # padded samples, so the features of a short utterance would
            # depend on the batch it is grouped into.  Models with a
            # group-norm conv feature extractor were pretrained without
            # attention masks, so per the Hugging Face guidance on
            # batched inference they must not receive one.
            if self.ssl_model.config.feat_extract_norm == "layer":
                attention_mask = length_to_input_mask(
                    wav_lengths, max_length=wav.size(1)
                ).to(torch.long)
            else:
                attention_mask = None

            res = self.ssl_model(
                wav, attention_mask=attention_mask, output_hidden_states=True
            )
            feat = res["hidden_states"][self.output_layer]  # [B, T, H]

            feat_lengths = torch.ceil(
                wav_lengths / SSL_SAMPLING_RATE * SSL_FRAME_RATE
            )
            feat_lengths = feat_lengths.to(torch.int64)
            # Cap by each utterance's true conv output length; clamping by
            # feat.size(1) only works when the batch size is 1.
            conv_lengths = self.ssl_model._get_feat_extract_output_lengths(
                wav_lengths
            ).to(torch.int64)
            feat_lengths = torch.minimum(feat_lengths, conv_lengths)

            mask = length_to_input_mask(feat_lengths)

            feat = feat * mask.unsqueeze(-1)
            feat = self.layer_norm(feat)

            return feat.detach(), feat_lengths


@dataclass
class MelspecConfig:
    """Arguments of torchaudio.transforms.MelSpectrogram."""

    sample_rate: int = 16000
    n_fft: int = 2048
    hop_length: int = 160
    win_length: int = 1200
    f_min: int = 80
    f_max: int = 7600


class MelspecExtractor(AcousticFeatureExtractorBase):
    """Extract mel-spectrogram features from waveforms."""

    def __init__(
        self, num_dims: int, melspec_config: MelspecConfig | None = None
    ) -> None:
        super().__init__()

        self.melspec_config = (
            melspec_config if melspec_config is not None else MelspecConfig()
        )

        self.melspec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.melspec_config.sample_rate,
            n_fft=self.melspec_config.n_fft,
            hop_length=self.melspec_config.hop_length,
            win_length=self.melspec_config.win_length,
            f_min=self.melspec_config.f_min,
            f_max=self.melspec_config.f_max,
            n_mels=num_dims,
        )

        self.layer_norm = torch.nn.LayerNorm(num_dims)
        self.frame_rate = (
            self.melspec_config.sample_rate / self.melspec_config.hop_length
        )

    def forward(
        self, wav: torch.Tensor, wav_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.melspec_transform(wav).transpose(-2, -1)

        feat_lengths = (
            torch.ceil(wav_lengths / self.melspec_config.hop_length) + 1
        )
        feat_lengths = torch.clamp(feat_lengths, max=feat.size(1))
        feat_lengths = feat_lengths.to(torch.int64)

        mask = length_to_input_mask(feat_lengths)

        feat = feat * mask.unsqueeze(-1)
        feat = self.layer_norm(feat)

        return feat.detach(), feat_lengths
