"""Tests for vae_speech_align.config (OmegaConf structured config)."""

from omegaconf import OmegaConf

from vae_speech_align.config import (
    AcousticFeatureType,
    AlignmentExpConfig,
    PositionBiasType,
)


def _merge(cfg_dict):
    return OmegaConf.merge(
        OmegaConf.structured(AlignmentExpConfig), OmegaConf.create(cfg_dict)
    )


class TestEnumConversion:
    def test_acoustic_feature_type_from_string(self):
        """Test that a model_type string is converted to the
        AcousticFeatureType enum on merge."""
        conf = _merge(
            {
                "features": {
                    "acoustic_feature_extractor": {
                        "model_type": "MELSPEC",
                        "model_name": "",
                        "layer_idx": 0,
                        "num_dims": 80,
                    },
                },
            }
        )
        assert (
            conf.features.acoustic_feature_extractor.model_type
            == AcousticFeatureType.MELSPEC
        )

    def test_all_acoustic_feature_types_parse(self):
        """Test that every AcousticFeatureType member name parses from
        its string form."""
        for name in ["WAV2VEC2", "HUBERT", "WAVLM", "MELSPEC"]:
            conf = _merge(
                {
                    "features": {
                        "acoustic_feature_extractor": {
                            "model_type": name,
                            "model_name": "",
                            "layer_idx": 0,
                            "num_dims": 39,
                        },
                    },
                }
            )
            assert (
                conf.features.acoustic_feature_extractor.model_type
                == AcousticFeatureType[name]
            )

    def test_position_bias_type_from_string(self):
        """Test that a bias_type string is converted to the
        PositionBiasType enum on merge."""
        conf = _merge(
            {
                "model": {
                    "position_bias": {
                        "bias_type": "LOGPROB",
                        "initial_scale": 1.0,
                        "update_rate": 0.9,
                    },
                },
            }
        )
        assert conf.model.position_bias.bias_type == PositionBiasType.LOGPROB


class TestDefaults:
    def test_rand_seed_accepts_none(self):
        """Test that the optional rand_seed field accepts None."""
        conf = _merge({"rand_seed": None})
        assert conf.rand_seed is None

    def test_rand_seed_accepts_int(self):
        """Test that the rand_seed field accepts an integer value."""
        conf = _merge({"rand_seed": 42})
        assert conf.rand_seed == 42


class TestLayerIdx:
    def test_layer_idx_accepts_int(self):
        """Test that layer_idx accepts an int layer index."""
        conf = _merge(
            {
                "features": {
                    "acoustic_feature_extractor": {
                        "model_type": "HUBERT",
                        "model_name": "m",
                        "layer_idx": 6,
                        "num_dims": 768,
                    },
                },
            }
        )
        assert conf.features.acoustic_feature_extractor.layer_idx == 6
