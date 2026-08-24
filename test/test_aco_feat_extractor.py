"""Tests for vae_speech_align.model.aco_feat_extractor."""

from types import SimpleNamespace

import pytest
import torch

from vae_speech_align import config
from vae_speech_align.model.aco_feat_extractor import (
    MelspecConfig,
    MelspecExtractor,
    SSLFeatureExtractor,
    create_aco_feat_extractor,
)

SR = 16000


# ---------------------------------------------------------------------------
# create_aco_feat_extractor
# ---------------------------------------------------------------------------


class TestCreateAcoFeatExtractor:
    def test_melspec(self):
        """Test that the MELSPEC feature type creates a MelspecExtractor."""
        conf = config.AcousticFeatureExtractor(
            model_type=config.AcousticFeatureType.MELSPEC,
            model_name="",
            layer_idx=0,
            num_dims=80,
        )
        extractor = create_aco_feat_extractor(conf)
        assert isinstance(extractor, MelspecExtractor)

    def test_unsupported_type_raises(self):
        """Test that an unknown feature type raises NotImplementedError."""
        conf = config.AcousticFeatureExtractor(
            model_type=None,
            model_name="",
            layer_idx=0,
            num_dims=39,
        )
        with pytest.raises(NotImplementedError):
            create_aco_feat_extractor(conf)


# ---------------------------------------------------------------------------
# SSLFeatureExtractor
# ---------------------------------------------------------------------------


class _StubSSLModel(torch.nn.Module):
    """Stand-in for a HF wav2vec2-style model (50 Hz frame rate)."""

    DOWNSAMPLE = SR // 50
    HIDDEN_DIM = 8
    FEAT_EXTRACT_NORM = "layer"

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(feat_extract_norm=self.FEAT_EXTRACT_NORM)
        self.received_attention_mask: torch.Tensor | None = None

    @classmethod
    def from_pretrained(cls, model_name: str) -> "_StubSSLModel":
        return cls()

    def _get_feat_extract_output_lengths(
        self, wav_lengths: torch.Tensor
    ) -> torch.Tensor:
        return torch.div(wav_lengths, self.DOWNSAMPLE, rounding_mode="floor")

    def forward(
        self,
        wav: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> dict[str, list[torch.Tensor]]:
        self.received_attention_mask = attention_mask
        num_frames = wav.size(1) // self.DOWNSAMPLE
        frames = wav[:, : num_frames * self.DOWNSAMPLE]
        frames = frames.reshape(wav.size(0), num_frames, self.DOWNSAMPLE)
        hidden = frames.mean(dim=-1, keepdim=True).tile(1, 1, self.HIDDEN_DIM)
        return {"hidden_states": [hidden]}


class _StubGroupNormSSLModel(_StubSSLModel):
    """Stub whose conv feature extractor uses group norm."""

    FEAT_EXTRACT_NORM = "group"


class TestSSLFeatureExtractor:
    def _create_extractor(
        self, model_class: type[_StubSSLModel] = _StubSSLModel
    ) -> SSLFeatureExtractor:
        return SSLFeatureExtractor(
            model_class=model_class,
            model_name="stub",
            output_layer=0,
            num_dims=_StubSSLModel.HIDDEN_DIM,
        )

    def test_attention_mask_covers_valid_samples(self):
        """Test that the sample-level attention mask reaching a
        layer-norm SSL model marks exactly the first wav_lengths
        samples of each item.

        Without it the transformer attends to the padded samples and
        the features of a short utterance depend on its batch."""
        extractor = self._create_extractor()
        wav = torch.randn(2, 4 * SR // 50)
        wav_lengths = torch.tensor([4 * SR // 50, 2 * SR // 50])
        extractor(wav, wav_lengths)
        mask = extractor.ssl_model.received_attention_mask
        assert mask is not None
        assert mask.shape == wav.shape
        assert mask.sum(dim=1).tolist() == wav_lengths.tolist()

    def test_group_norm_model_gets_no_attention_mask(self):
        """Test that a group-norm SSL model is called without an
        attention mask, since such models were pretrained without one
        (Hugging Face guidance on batched inference)."""
        extractor = self._create_extractor(_StubGroupNormSSLModel)
        wav = torch.randn(2, 4 * SR // 50)
        wav_lengths = torch.tensor([4 * SR // 50, 2 * SR // 50])
        extractor(wav, wav_lengths)
        assert extractor.ssl_model.received_attention_mask is None

    def test_feat_lengths_capped_per_item(self):
        """Test that each item's frame count is capped by its own conv
        output length, not by the padded batch length.

        wav_lengths [1000, 700]: ceil(50 * len / 16000) gives [4, 3]
        but the conv stack yields floor(len / 320) = [3, 2] frames."""
        extractor = self._create_extractor()
        wav = torch.randn(2, 1000)
        wav_lengths = torch.tensor([1000, 700])
        _, feat_lengths = extractor(wav, wav_lengths)
        assert feat_lengths.tolist() == [3, 2]
        assert feat_lengths.dtype == torch.int64

    def test_frames_beyond_length_are_masked(self):
        """Test that frames past each item's length are zero before the
        layer norm (the layer norm maps zero vectors to its bias)."""
        extractor = self._create_extractor()
        wav = torch.randn(2, 1000)
        wav_lengths = torch.tensor([1000, 700])
        feat, feat_lengths = extractor(wav, wav_lengths)
        short_item_length = feat_lengths[1].item()
        layer_norm_bias = extractor.layer_norm.bias
        assert torch.allclose(
            feat[1, short_item_length:],
            layer_norm_bias.tile(feat.size(1) - short_item_length, 1),
        )


# ---------------------------------------------------------------------------
# MelspecExtractor
# ---------------------------------------------------------------------------


class TestMelspecExtractor:
    def test_output_shape(self):
        """Test that output is [B, T, num_dims] with int64 frame lengths."""
        num_dims = 80
        extractor = MelspecExtractor(num_dims=num_dims)
        wav = torch.randn(2, SR)
        wav_lengths = torch.tensor([SR, SR // 2])
        feat, feat_lengths = extractor(wav, wav_lengths)
        assert feat.ndim == 3
        assert feat.shape[0] == 2
        assert feat.shape[2] == num_dims
        assert feat_lengths.shape == (2,)
        assert feat_lengths.dtype == torch.int64

    def test_lengths_proportional_to_wav_lengths(self):
        """Test that frame lengths scale with wav lengths at the frame
        rate implied by hop_length."""
        extractor = MelspecExtractor(
            num_dims=80, melspec_config=MelspecConfig(hop_length=160)
        )
        wav = torch.randn(2, SR)
        wav_lengths = torch.tensor([SR, SR // 2])
        _, feat_lengths = extractor(wav, wav_lengths)
        # 16000 / hop_length = 100 frames/sec (+1);
        # a half-length wav gives ~half the frames
        assert feat_lengths[0].item() == pytest.approx(101, abs=2)
        assert feat_lengths[1].item() == pytest.approx(51, abs=2)

    def test_frame_rate_follows_hop_length(self):
        """Test that frame_rate is sample_rate / hop_length."""
        assert MelspecExtractor(num_dims=80).frame_rate == pytest.approx(100.0)
        halved = MelspecExtractor(
            num_dims=80, melspec_config=MelspecConfig(hop_length=320)
        )
        assert halved.frame_rate == pytest.approx(50.0)

    def test_hop_length_controls_frame_rate(self):
        """Test that doubling hop_length halves the frame lengths."""
        extractor = MelspecExtractor(
            num_dims=80, melspec_config=MelspecConfig(hop_length=320)
        )
        wav = torch.randn(1, SR)
        _, feat_lengths = extractor(wav, torch.tensor([SR]))
        assert feat_lengths[0].item() == pytest.approx(51, abs=2)

    def test_masking(self):
        """Test that frames beyond the valid length are zero-masked."""
        extractor = MelspecExtractor(num_dims=80)
        wav = torch.randn(2, SR)
        wav_lengths = torch.tensor([SR, SR // 2])
        feat, feat_lengths = extractor(wav, wav_lengths)
        L = feat_lengths[1].item()
        if feat.shape[1] > L:
            assert (feat[1, L:] == 0).all()

    def test_no_grad(self):
        """Test that extracted features do not require gradients."""
        extractor = MelspecExtractor(num_dims=80)
        wav = torch.randn(1, SR)
        feat, _ = extractor(wav, torch.tensor([SR]))
        assert not feat.requires_grad
