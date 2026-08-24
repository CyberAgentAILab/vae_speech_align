"""
End-to-end smoke tests for train.py and align.py.

Uses synthetic WAV files and phoneme labels so that no external model
downloads or real corpora are required.  The acoustic feature extractor
is set to MELSPEC (num_dims=80) to avoid Wav2Vec2/HuBERT downloads.
All computation runs on CPU with a tiny model.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
import torchaudio
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file

from vae_speech_align.cli.align import run_alignment_steps
from vae_speech_align.cli.train import run_training_steps
from vae_speech_align.config import AlignmentExpConfig, AlignmentImplementation
from vae_speech_align.dataset import create_dataloader
from vae_speech_align.model.aco_feat_extractor import create_aco_feat_extractor
from vae_speech_align.model.model import Model

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PHONEMES = ["a", "i", "u", "e", "o"]
NUM_UTTS = 4
SR = 16000
WAV_DURATION_SEC = 1.0  # 1-second utterances


def _make_conf(token_list_file: Path) -> AlignmentExpConfig:
    """Build a minimal AlignmentExpConfig using MELSPEC (no model
    downloads)."""
    cfg_dict = {
        "features": {
            "acoustic_feature_extractor": {
                "model_type": "MELSPEC",
                "model_name": "",
                "layer_idx": 0,
                "num_dims": 80,
            },
            "linguistic": {
                "tag": "phoneme",
                "num_tokens": len(PHONEMES),
                "token_list_file": str(token_list_file),
            },
        },
        "model": {
            "acoustic_encoder": {
                "residual": False,
                "num_layers": 2,
                "hidden_dim": 32,
                "conv_kernel_size": 3,
                "dropout_rate": 0.0,
            },
            "linguistic_encoder": {
                "residual": False,
                "num_layers": 2,
                "hidden_dim": 32,
                "conv_kernel_size": 3,
                "dropout_rate": 0.0,
            },
            "acoustic_decoder": {
                "residual": False,
                "num_layers": 2,
                "hidden_dim": 32,
                "conv_kernel_size": 3,
                "dropout_rate": 0.0,
            },
            "linguistic_decoder": {
                "residual": False,
                "num_layers": 2,
                "hidden_dim": 32,
                "conv_kernel_size": 3,
                "dropout_rate": 0.0,
            },
            "acoustic_upsample_scale": 1,
            "linguistic_upsample_scale": 1,
            "matching_dim": 16,
            "loss": {
                "alignment": 1.0,
                "recon_x": 0.0,
                "recon_y": 0.0,
                "vae_beta_x": 0.0,
                "vae_beta_y": 0.0,
            },
            "position_bias": {
                "bias_type": "NONE",
                "initial_scale": 1.0,
                "update_rate": 1.0,
            },
            "annealing_weight": {
                "initial_sigma": 2.0,
                "update_rate": 0.9,
                "kernel_size": 3,
            },
        },
        "initial_checkpoint": "",
        "training": {
            "batch_size": 2,
            "num_workers": 0,
            "num_steps": 3,
            "optimizer": {
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
            },
            "scheduler": {
                "warmup_steps": 0,
            },
            "intervals": {
                "training_loss_interval": 1,
                "plot_interval": 1000,
                "checkpoint_interval": 1000,
                "annealing_interval": 1000,
                "position_bias_scale_interval": 1000,
            },
        },
        "rand_seed": 0,
    }
    yaml_conf = OmegaConf.create(cfg_dict)
    conf: AlignmentExpConfig = OmegaConf.merge(
        OmegaConf.structured(AlignmentExpConfig), yaml_conf
    )
    return conf


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    """Create synthetic WAV files and phoneme label files."""
    base = tmp_path_factory.mktemp("data")
    wav_dir = base / "wav"
    wav_dir.mkdir()
    phoneme_dir = base / "phoneme"
    phoneme_dir.mkdir()
    token_list_file = base / "phonemes.txt"
    token_list_file.write_text("\n".join(PHONEMES))

    rng = np.random.default_rng(42)
    num_samples = int(SR * WAV_DURATION_SEC)
    for i in range(NUM_UTTS):
        wav = (
            torch.from_numpy(
                rng.standard_normal(num_samples).astype(np.float32)
            )
            * 0.01
        )
        torchaudio.save(str(wav_dir / f"utt{i:03d}.wav"), wav.unsqueeze(0), SR)

        num_phones = rng.integers(3, 6)
        phones = [PHONEMES[j % len(PHONEMES)] for j in range(num_phones)]
        (phoneme_dir / f"utt{i:03d}.txt").write_text(" ".join(phones))

    return {
        "base": base,
        "wav_dir": wav_dir,
        "phoneme_dir": phoneme_dir,
        "token_list_file": token_list_file,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTrainAlign:
    """End-to-end smoke tests over the synthetic corpus."""

    def test_train(self, data_dir, tmp_path):
        """train.py: run a few training steps without error."""
        conf = _make_conf(data_dir["token_list_file"])

        torch.manual_seed(0)
        np.random.seed(0)

        device = torch.device("cpu")

        train_dataloader = create_dataloader(
            conf,
            wav_dir=data_dir["wav_dir"],
            token_dir=data_dir["phoneme_dir"],
            shuffle=False,
        )

        model = Model(
            conf.model,
            conf.features,
            alignment_impl=AlignmentImplementation.NUMBA,
        )
        model.alignment.annealing_conv.reset_weight()
        model.to(device)

        aco_feat_extractor = create_aco_feat_extractor(
            conf.features.acoustic_feature_extractor
        )
        aco_feat_extractor.to(device)

        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir()
        plot_dir = tmp_path / "plot"
        plot_dir.mkdir()

        run_training_steps(
            model=model,
            aco_feat_extractor=aco_feat_extractor,
            training_config=conf.training,
            train_dataloader=train_dataloader,
            device=device,
            summary_writer=None,
            checkpoint_dir=checkpoint_dir,
            plot_dir=plot_dir,
        )

        # Verify the model state can be saved
        model_path = tmp_path / "model.safetensors"
        save_file(model.state_dict(), model_path)
        assert model_path.exists()

    def test_align(self, data_dir, tmp_path):
        """align.py: run alignment and verify output files are created."""
        conf = _make_conf(data_dir["token_list_file"])

        torch.manual_seed(0)
        np.random.seed(0)

        device = torch.device("cpu")

        # Build and save a randomly initialised model
        model = Model(
            conf.model,
            conf.features,
            alignment_impl=AlignmentImplementation.NUMBA,
        )
        model.alignment.annealing_conv.reset_weight()
        model_path = tmp_path / "model.safetensors"
        save_file(model.state_dict(), model_path)

        # Reload (same as align.py does)
        model.load_state_dict(load_file(model_path))
        model.to(device)

        aco_feat_extractor = create_aco_feat_extractor(
            conf.features.acoustic_feature_extractor
        )
        aco_feat_extractor.to(device)

        wav_dir = data_dir["wav_dir"]
        base_list = sorted([fn.stem for fn in wav_dir.glob("*.wav")])

        dataloader = create_dataloader(
            conf,
            wav_dir=wav_dir,
            token_dir=data_dir["phoneme_dir"],
            batch_size=1,
            shuffle=False,
        )

        out_dir = tmp_path / "align_out"
        out_dir.mkdir()

        run_alignment_steps(
            model=model,
            aco_feat_extractor=aco_feat_extractor,
            base_list=base_list,
            dataloader=dataloader,
            device=device,
            out_base_dir=out_dir,
        )

        # Verify output files exist for every utterance
        for base in base_list:
            assert (
                out_dir / "state" / f"{base}.npy"
            ).exists(), f"Missing state/{base}.npy"
            assert (
                out_dir / "token" / f"{base}.npy"
            ).exists(), f"Missing token/{base}.npy"
            assert (
                out_dir / "textgrid" / f"{base}.textgrid"
            ).exists(), f"Missing textgrid/{base}.textgrid"

        assert (out_dir / "likelihood.txt").exists()
