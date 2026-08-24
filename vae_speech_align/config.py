from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path


class PositionBiasType(Enum):
    """Type of position bias applied to the alignment matrix."""

    NONE = auto()
    LOGPROB = auto()


class AcousticFeatureType(Enum):
    """Supported acoustic feature extractor types."""

    WAV2VEC2 = auto()
    HUBERT = auto()
    WAVLM = auto()
    MELSPEC = auto()


class AlignmentImplementation(str, Enum):
    """Supported forward-sum computation backends."""

    NUMBA = "numba"
    TRITON = "triton"

    def __str__(self) -> str:
        return self.value


@dataclass
class AcousticFeatureExtractor:
    """Configuration for the acoustic feature extractor (SSL model or
    mel-spectrogram)."""

    model_type: AcousticFeatureType
    model_name: str
    layer_idx: int
    num_dims: int


@dataclass
class LinguisticFeatures:
    """Configuration for linguistic (phoneme) features."""

    tag: str
    num_tokens: int
    token_list_file: Path


@dataclass
class FeaturesWithExtractor:
    """Combined acoustic and linguistic feature configuration."""

    acoustic_feature_extractor: AcousticFeatureExtractor
    linguistic: LinguisticFeatures


@dataclass
class Optimizer:
    learning_rate: float
    weight_decay: float


@dataclass
class Scheduler:
    warmup_steps: int


@dataclass
class TrainingIntervals:
    """Step intervals for plotting, logging, checkpoints, annealing."""

    plot_interval: int
    training_loss_interval: int
    checkpoint_interval: int
    annealing_interval: int
    position_bias_scale_interval: int


@dataclass
class TrainingLoss:
    """Loss coefficients for alignment, reconstruction, and VAE KLD terms."""

    alignment: float
    recon_x: float
    recon_y: float
    vae_beta_x: float
    vae_beta_y: float


@dataclass
class TrainingConfig:
    """Training hyperparameters: batch size, steps, optimizer, intervals."""

    batch_size: int
    num_workers: int
    num_steps: int
    optimizer: Optimizer
    scheduler: Scheduler
    intervals: TrainingIntervals
    pin_memory: bool = False


@dataclass
class ConvStack:
    """Conv-stack config (conv layers with optional residual
    connections), shared by the encoders and decoders."""

    residual: bool
    num_layers: int
    hidden_dim: int
    conv_kernel_size: int
    dropout_rate: float


@dataclass
class PositionBias:
    """Position bias configuration for the alignment matrix."""

    bias_type: PositionBiasType
    initial_scale: float
    update_rate: float


@dataclass
class AnnealingWeight:
    """Annealing conv config for smoothing the alignment gamma."""

    initial_sigma: float
    update_rate: float
    kernel_size: int


@dataclass
class AlignmentModel:
    """Model architecture configuration for the VAE-based alignment model."""

    acoustic_encoder: ConvStack
    linguistic_encoder: ConvStack
    acoustic_decoder: ConvStack
    linguistic_decoder: ConvStack
    acoustic_upsample_scale: int  # e.g., 2 upsamples 20ms frames to 10ms
    linguistic_upsample_scale: (
        int  # e.g., 3 expands each phoneme into 3 states
    )
    matching_dim: int
    loss: TrainingLoss
    position_bias: PositionBias
    annealing_weight: AnnealingWeight


@dataclass
class AlignmentExpConfig:
    """Top-level experiment config: features, model, training settings."""

    features: FeaturesWithExtractor
    model: AlignmentModel
    initial_checkpoint: str
    training: TrainingConfig
    rand_seed: int | None
