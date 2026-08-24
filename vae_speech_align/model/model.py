import logging
from dataclasses import dataclass

import torch

import vae_speech_align.forwardsum.matrix_util
from vae_speech_align import config
from vae_speech_align.forwardsum.backend import create_forwardsum_backend
from vae_speech_align.model.annealing import AnnealingConv
from vae_speech_align.model.conv import (
    ConvModule,
    DownsampleConvModule,
    UpsampleConvModule,
)
from vae_speech_align.model.loss import (
    MaskedCrossEntropyLoss,
    MaskedMSELoss,
    MaskedVAEKLDLoss,
)
from vae_speech_align.model.position_bias import PositionBiasModule
from vae_speech_align.utils import length_to_input_mask

_logger = logging.getLogger(__name__)


@dataclass
class LinguisticEncoderOutput:
    """Linguistic encoder output: Gaussian params for alignment matching."""

    mean: torch.Tensor
    var: torch.Tensor
    vae_lvar: torch.Tensor
    lengths: torch.Tensor


@dataclass
class AcousticEncoderOutput:
    """Acoustic encoder output: Gaussian params for alignment matching."""

    mean: torch.Tensor
    vae_lvar: torch.Tensor
    lengths: torch.Tensor


@dataclass
class LikelihoodResult:
    """Forward-sum alignment log-likelihood and the derived loss."""

    log_likelihood: torch.Tensor
    alignment_loss: torch.Tensor


@dataclass
class AnnealingAlignmentResult:
    """Alignment losses computed with annealing-smoothed gamma."""

    log_likelihood: torch.Tensor
    annealed_alignment_loss: torch.Tensor
    alignment_loss: torch.Tensor


@dataclass
class ViterbiResult:
    """Viterbi decoding output: per-item log-likelihoods and the path."""

    log_likelihoods: torch.Tensor
    path: torch.Tensor


@dataclass
class ViterbiAlignmentResult:
    """Full Viterbi alignment pipeline output.

    state_durations and token_durations are padded to the max length in
    the batch; use state_lengths / token_lengths to extract valid entries.
    """

    viterbi_likelihoods: torch.Tensor
    path: torch.Tensor
    state_durations: torch.Tensor
    token_durations: torch.Tensor
    state_lengths: torch.Tensor
    token_lengths: torch.Tensor


@dataclass
class LinguisticReconstructionLoss:
    """Linguistic decoder reconstruction and VAE KLD losses."""

    x_recon_loss: torch.Tensor
    x_vae_kld_loss: torch.Tensor


@dataclass
class AcousticReconstructionLoss:
    """Acoustic decoder reconstruction and VAE KLD losses."""

    y_recon_loss: torch.Tensor
    y_vae_kld_loss: torch.Tensor


class LinguisticEncoder(torch.nn.Module):
    """Encode token indices into Gaussian params for alignment matching."""

    def __init__(
        self,
        encoder_config: config.ConvStack,
        num_classes: int,
        matching_dim: int,
        upsample_scale: int,
    ) -> None:
        super().__init__()

        if upsample_scale < 1:
            raise ValueError(
                f"upsample_scale must be >= 1, got {upsample_scale}"
            )

        output_dim = matching_dim * 2

        self.embedding = torch.nn.Embedding(
            num_embeddings=num_classes, embedding_dim=encoder_config.hidden_dim
        )

        self.conv_module = ConvModule(
            num_layers=encoder_config.num_layers,
            hidden_size=encoder_config.hidden_dim,
            dropout_rate=encoder_config.dropout_rate,
            kernel_size=encoder_config.conv_kernel_size,
            residual=encoder_config.residual,
        )

        self.final_layer = torch.nn.Linear(
            encoder_config.hidden_dim, output_dim
        )

        self.upsample_scale = upsample_scale

        if upsample_scale == 1:
            self.upsample_module = None
        else:
            self.upsample_module = UpsampleConvModule(
                encoder_config.hidden_dim,
                encoder_config.hidden_dim,
                upsample_scale,
            )

    def forward(
        self, x: torch.Tensor, x_lengths: torch.Tensor
    ) -> LinguisticEncoderOutput:
        # x: [B, K, D]
        # mask: [B, K]

        h = self.embedding(x)

        x_out_lengths = x_lengths * self.upsample_scale
        out_mask = length_to_input_mask(x_out_lengths)

        if self.upsample_module is not None:
            h = self.upsample_module(h)

        h = self.conv_module(h, out_mask)

        out = self.final_layer(h)

        mean, vae_lvar = torch.chunk(out, chunks=2, dim=-1)
        # var is fixed to 1 for a uniform variance across all states, which
        # stabilizes training
        return LinguisticEncoderOutput(
            mean, torch.ones_like(mean), vae_lvar, x_out_lengths
        )


class AcousticEncoder(torch.nn.Module):
    """Encode acoustic features into Gaussian params for alignment."""

    def __init__(
        self,
        encoder_config: config.ConvStack,
        input_dim: int,
        matching_dim: int,
        upsample_scale: int,
    ) -> None:
        super().__init__()

        if upsample_scale < 1:
            raise ValueError(
                f"upsample_scale must be >= 1, got {upsample_scale}"
            )

        hidden_dim = encoder_config.hidden_dim
        output_dim = matching_dim * 2

        self.upsample_scale = upsample_scale
        self.input_linear: torch.nn.Module
        if upsample_scale == 1:
            self.input_linear = torch.nn.Linear(input_dim, hidden_dim)
        else:
            self.input_linear = UpsampleConvModule(
                input_dim, hidden_dim, upsample_scale
            )

        self.conv_module = ConvModule(
            num_layers=encoder_config.num_layers,
            hidden_size=encoder_config.hidden_dim,
            dropout_rate=encoder_config.dropout_rate,
            kernel_size=encoder_config.conv_kernel_size,
            residual=encoder_config.residual,
        )

        self.final_layer = torch.nn.Linear(hidden_dim, output_dim)

    def forward(
        self, x: torch.Tensor, x_lengths: torch.Tensor
    ) -> AcousticEncoderOutput:
        # x: [B, K, D]
        # mask: [B, K]

        x_out_lengths = x_lengths * self.upsample_scale
        out_mask = length_to_input_mask(x_out_lengths)

        h = self.input_linear(x)

        h = self.conv_module(h, out_mask)

        h = self.final_layer(h)

        out = h * out_mask[:, :, None]

        mean, vae_lvar = torch.chunk(out, chunks=2, dim=-1)

        return AcousticEncoderOutput(mean, vae_lvar, x_out_lengths)


class DecoderModule(torch.nn.Module):
    """Conv decoder with optional downsampling to reconstruct features."""

    def __init__(
        self,
        decoder_config: config.ConvStack,
        input_dim: int,
        output_dim: int,
        downsample_scale: int,
    ) -> None:
        super().__init__()

        if downsample_scale < 1:
            raise ValueError(
                f"downsample_scale must be >= 1, got {downsample_scale}"
            )

        self.input_layer = torch.nn.Linear(
            input_dim, decoder_config.hidden_dim
        )

        self.conv_module = ConvModule(
            num_layers=decoder_config.num_layers,
            hidden_size=decoder_config.hidden_dim,
            dropout_rate=decoder_config.dropout_rate,
            kernel_size=decoder_config.conv_kernel_size,
            residual=decoder_config.residual,
        )

        self.downsample_scale = downsample_scale
        self.final_layer: torch.nn.Module
        if downsample_scale == 1:
            self.final_layer = torch.nn.Linear(
                decoder_config.hidden_dim, output_dim
            )
        else:
            self.final_layer = DownsampleConvModule(
                decoder_config.hidden_dim, output_dim, downsample_scale
            )

    def forward(
        self, x: torch.Tensor, x_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, K, D]
        # mask: [B, K]

        mask = length_to_input_mask(x_lengths)

        h = self.input_layer(x)

        downsampled_lengths = x_lengths // self.downsample_scale
        downsampled_mask = length_to_input_mask(downsampled_lengths)

        h = self.conv_module(h, mask)

        out = self.final_layer(h)

        return out * downsampled_mask[:, :, None], downsampled_lengths


class AlignmentModule(torch.nn.Module):
    """Forward-sum alignment computation with annealing and position bias."""

    def __init__(
        self,
        position_bias_config: config.PositionBias,
        annealing_config: config.AnnealingWeight,
        alignment_loss_coef: float,
        alignment_impl: config.AlignmentImplementation = (
            config.AlignmentImplementation.TRITON
        ),
    ) -> None:
        super().__init__()

        self.set_alignment_impl(alignment_impl)
        self.alignment_loss_coef = alignment_loss_coef

        self.position_bias_type = position_bias_config.bias_type
        self.position_bias = PositionBiasModule(
            initial_scale=position_bias_config.initial_scale,
            update_rate=position_bias_config.update_rate,
        )

        self.annealing_conv = AnnealingConv(
            initial_sigma=annealing_config.initial_sigma,
            update_rate=annealing_config.update_rate,
            kernel_size=annealing_config.kernel_size,
        )

    def set_alignment_impl(
        self, alignment_impl: config.AlignmentImplementation
    ) -> None:
        """Set the forward-sum implementation ('triton' or 'numba')."""
        self.alignment_impl = config.AlignmentImplementation(alignment_impl)
        self.forwardsum_backend = create_forwardsum_backend(
            self.alignment_impl
        )

    def annealing_sigma(self) -> float:
        """Return the current annealing sigma value."""
        return self.annealing_conv.sigma.item()

    def update_annealing(self) -> float:
        """Decay the annealing sigma by update_rate and return the value."""
        self.annealing_conv.update_phase()
        return self.annealing_sigma()

    def update_position_bias_scale(self) -> float:
        """Decay the position bias scale by update_rate; return the value."""
        self.position_bias.update_phase()
        return self.position_bias.scale.item()

    def calc_log_prob_output_matrix(
        self,
        y: torch.Tensor,
        output_lengths: torch.Tensor,
        state_lengths: torch.Tensor,
        output_mean: torch.Tensor,
        output_var: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the padded log-prob output matrix with position bias."""
        lout = vae_speech_align.forwardsum.matrix_util.make_padded_log_prob_output_matrix(  # noqa: E501
            y=y,
            output_lengths=output_lengths,
            state_lengths=state_lengths,
            output_mean=output_mean,
            output_var=output_var,
            time_wise_normalize=True,
            bias=None,
        )

        if self.position_bias_type == config.PositionBiasType.LOGPROB:
            lposition = self.position_bias.forward(
                output_lengths, state_lengths
            )
            lposition = (
                vae_speech_align.forwardsum.matrix_util.make_padded_BTK_matrix(
                    lposition,
                    output_lengths,
                    state_lengths,
                )
            )
            lout = lout + lposition

        return lout

    def _get_log_prob_matrix(
        self, x_out: LinguisticEncoderOutput, y_out: AcousticEncoderOutput
    ) -> torch.Tensor:
        return self.calc_log_prob_output_matrix(
            y=y_out.mean,
            output_lengths=y_out.lengths,
            state_lengths=x_out.lengths,
            output_mean=x_out.mean,
            output_var=x_out.var,
        )

    def calc_likelihood(
        self, x_out: LinguisticEncoderOutput, y_out: AcousticEncoderOutput
    ) -> LikelihoodResult:
        """Compute forward-sum log-likelihood for the alignment."""
        y_lengths = y_out.lengths
        x_lengths = x_out.lengths

        if y_lengths.sum() <= 0:
            raise ValueError("y_lengths must sum to a positive value")

        log_prob_output_matrix = self._get_log_prob_matrix(x_out, y_out)

        log_likelihoods = self.forwardsum_backend.likelihood(
            log_prob_output_matrix,
            y_lengths,
            x_lengths,
        )

        log_likelihood = (
            self.alignment_loss_coef * log_likelihoods.sum() / y_lengths.sum()
        )
        return LikelihoodResult(
            log_likelihood=log_likelihood,
            alignment_loss=-log_likelihood,
        )

    def calc_annealing_alignment(
        self, x_out: LinguisticEncoderOutput, y_out: AcousticEncoderOutput
    ) -> AnnealingAlignmentResult:
        """Compute alignment loss with annealing-smoothed gamma."""
        y_lengths = y_out.lengths
        x_lengths = x_out.lengths

        if y_lengths.sum() <= 0:
            raise ValueError("y_lengths must sum to a positive value")

        log_prob_output_matrix = self._get_log_prob_matrix(x_out, y_out)

        with torch.no_grad():
            gamma, log_likelihoods = self.forwardsum_backend.gamma(
                log_prob_output_matrix,
                y_lengths,
                x_lengths,
                return_likelihood=True,
            )

        annealed_gamma = self.annealing_conv(gamma, y_lengths, x_lengths)
        # Surrogate likelihood following Koriyama (Interspeech 2024,
        # isca-archive.org/interspeech_2024/koriyama24_interspeech.pdf):
        # the gradient of the forward-sum log-likelihood w.r.t.
        # log_prob_output_matrix is gamma, so multiplying the detached
        # (annealed) gamma by log_prob_output_matrix makes autograd
        # reproduce that gradient — smoothed by the annealing kernel —
        # without differentiating through the DP itself.
        annealing_align_likelihood = (
            annealed_gamma.detach() * log_prob_output_matrix
        ).sum()

        log_likelihood = (
            self.alignment_loss_coef * log_likelihoods.sum() / y_lengths.sum()
        )
        annealing_align_likelihood = (
            self.alignment_loss_coef
            * annealing_align_likelihood
            / y_lengths.sum()
        )
        return AnnealingAlignmentResult(
            log_likelihood=log_likelihood,
            annealed_alignment_loss=-annealing_align_likelihood,
            alignment_loss=-log_likelihood,
        )

    def calc_gamma(
        self, x_out: LinguisticEncoderOutput, y_out: AcousticEncoderOutput
    ) -> torch.Tensor:
        """Compute the posterior alignment matrix (gamma)."""
        log_prob_output_matrix = self._get_log_prob_matrix(x_out, y_out)

        gamma = self.forwardsum_backend.gamma(
            log_prob_output_matrix,
            y_out.lengths,
            x_out.lengths,
        )
        assert isinstance(gamma, torch.Tensor)
        return gamma

    def calc_viterbi(
        self, x_out: LinguisticEncoderOutput, y_out: AcousticEncoderOutput
    ) -> ViterbiResult:
        """Find the best alignment path via Viterbi decoding."""
        log_prob_output_matrix = self._get_log_prob_matrix(x_out, y_out)

        log_likelihoods, path = self.forwardsum_backend.viterbi(
            log_prob_output_matrix,
            y_out.lengths,
            x_out.lengths,
        )
        return ViterbiResult(log_likelihoods=log_likelihoods, path=path)


class Model(torch.nn.Module):
    """VAE-based speech alignment model.

    Combines linguistic and acoustic encoders with forward-sum alignment
    and reconstruction decoders.
    """

    def __init__(
        self,
        model_config: config.AlignmentModel,
        features: config.FeaturesWithExtractor,
        alignment_impl: config.AlignmentImplementation = (
            config.AlignmentImplementation.TRITON
        ),
    ) -> None:
        super().__init__()

        num_classes = 1 + features.linguistic.num_tokens

        self.matching_dim = model_config.matching_dim
        self.linguistic_encoder = LinguisticEncoder(
            encoder_config=model_config.linguistic_encoder,
            num_classes=num_classes,
            matching_dim=model_config.matching_dim,
            upsample_scale=model_config.linguistic_upsample_scale,
        )
        self.acoustic_encoder = AcousticEncoder(
            encoder_config=model_config.acoustic_encoder,
            input_dim=features.acoustic_feature_extractor.num_dims,
            matching_dim=model_config.matching_dim,
            upsample_scale=model_config.acoustic_upsample_scale,
        )

        self.linguistic_decoder = DecoderModule(
            decoder_config=model_config.linguistic_decoder,
            input_dim=model_config.matching_dim,
            output_dim=num_classes,
            downsample_scale=model_config.linguistic_upsample_scale,
        )
        self.acoustic_decoder = DecoderModule(
            decoder_config=model_config.acoustic_decoder,
            input_dim=model_config.matching_dim,
            output_dim=features.acoustic_feature_extractor.num_dims,
            downsample_scale=model_config.acoustic_upsample_scale,
        )

        self.alignment = AlignmentModule(
            position_bias_config=model_config.position_bias,
            annealing_config=model_config.annealing_weight,
            alignment_loss_coef=model_config.loss.alignment,
            alignment_impl=alignment_impl,
        )

        self.mse_loss = MaskedMSELoss()
        self.cross_entropy_loss = MaskedCrossEntropyLoss()
        self.vae_kld_loss = MaskedVAEKLDLoss()

        self.loss_coef = model_config.loss

    # --- Alignment delegation ---

    def set_alignment_impl(
        self, alignment_impl: config.AlignmentImplementation
    ) -> None:
        self.alignment.set_alignment_impl(alignment_impl)

    def annealing_sigma(self) -> float:
        return self.alignment.annealing_sigma()

    def update_annealing(self) -> float:
        return self.alignment.update_annealing()

    def update_position_bias_scale(self) -> float:
        return self.alignment.update_position_bias_scale()

    def calc_likelihood(
        self, x_out: LinguisticEncoderOutput, y_out: AcousticEncoderOutput
    ) -> LikelihoodResult:
        return self.alignment.calc_likelihood(x_out, y_out)

    def calc_annealing_alignment(
        self, x_out: LinguisticEncoderOutput, y_out: AcousticEncoderOutput
    ) -> AnnealingAlignmentResult:
        return self.alignment.calc_annealing_alignment(x_out, y_out)

    def calc_gamma(
        self, x_out: LinguisticEncoderOutput, y_out: AcousticEncoderOutput
    ) -> torch.Tensor:
        return self.alignment.calc_gamma(x_out, y_out)

    def calc_viterbi(
        self, x_out: LinguisticEncoderOutput, y_out: AcousticEncoderOutput
    ) -> ViterbiResult:
        return self.alignment.calc_viterbi(x_out, y_out)

    @torch.no_grad()
    def run_viterbi(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        wav: torch.Tensor,
        wav_lengths: torch.Tensor,
        aco_feat_extractor: torch.nn.Module,
    ) -> ViterbiAlignmentResult:
        """Run the full Viterbi alignment pipeline and return durations."""
        y, y_lengths = aco_feat_extractor(wav, wav_lengths)

        x_out = self.forward_x(x, x_lengths)
        y_out = self.forward_y(y, y_lengths)

        viterbi = self.calc_viterbi(x_out, y_out)
        path = viterbi.path

        scale = self.linguistic_encoder.upsample_scale
        state_durations = path[:, 1:-1, 1:-1].sum(dim=1)  # [B, K]
        token_durations = state_durations.reshape(
            state_durations.size(0), -1, scale
        ).sum(
            dim=-1
        )  # [B, num_tokens]

        state_lengths = x_lengths * scale
        return ViterbiAlignmentResult(
            viterbi_likelihoods=viterbi.log_likelihoods,
            path=path,
            state_durations=state_durations,
            token_durations=token_durations,
            state_lengths=state_lengths,
            token_lengths=x_lengths,
        )

    # --- VAE ---

    def reparameterize(
        self, mean: torch.Tensor, lvar: torch.Tensor
    ) -> torch.Tensor:
        """Apply the VAE reparameterization trick (mean during eval)."""
        if self.training:
            std = torch.exp(0.5 * lvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        else:
            return mean

    def forward_x(
        self, x: torch.Tensor, x_lengths: torch.Tensor
    ) -> LinguisticEncoderOutput:
        """Encode linguistic (token) input."""
        out: LinguisticEncoderOutput = self.linguistic_encoder(x, x_lengths)
        return out

    def adjust_y_lengths(
        self,
        y: torch.Tensor,
        y_lengths: torch.Tensor,
        x_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure y is at least as long as x after upsampling, because the
        # forward-sum algorithm requires len(y) >= len(x) to find a valid
        # alignment path via dynamic programming.
        y_upsample_scale = self.acoustic_encoder.upsample_scale
        x_upsample_scale = self.linguistic_encoder.upsample_scale
        y_out_lengths = y_lengths * y_upsample_scale
        x_out_lengths = x_lengths * x_upsample_scale

        if (x_out_lengths > y_out_lengths).any():
            y_required_lengths = torch.ceil(
                x_out_lengths / y_upsample_scale
            ).long()
            y_new_lengths = torch.where(
                y_required_lengths > y_lengths, y_required_lengths, y_lengths
            )
            _logger.info(f"y_lengths modified: {y_lengths} -> {y_new_lengths}")
            y_lengths = y_new_lengths
            if y_lengths.max() > y.size(-2):
                pad_size = int(y_lengths.max().item()) - y.size(-2)
                # pad the time axis (dim -2) at the end; the last dim
                # keeps its size
                y = torch.nn.functional.pad(
                    y, (0, 0, 0, pad_size), "constant", 0
                )

        return y, y_lengths

    def forward_y(
        self, y: torch.Tensor, y_lengths: torch.Tensor
    ) -> AcousticEncoderOutput:
        """Encode acoustic features."""
        out: AcousticEncoderOutput = self.acoustic_encoder(y, y_lengths)
        return out

    def reconstruct_x(
        self, x_out: LinguisticEncoderOutput
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct linguistic input via the linguistic decoder."""
        x = self.reparameterize(x_out.mean, x_out.vae_lvar)
        recon_x, recon_x_lengths = self.linguistic_decoder(x, x_out.lengths)
        return recon_x, recon_x_lengths

    def reconstruct_y(
        self, y_out: AcousticEncoderOutput
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct acoustic features via the acoustic decoder."""
        y = self.reparameterize(y_out.mean, y_out.vae_lvar)
        recon_y, recon_y_lengths = self.acoustic_decoder(y, y_out.lengths)
        return recon_y, recon_y_lengths

    # --- Reconstruction loss ---

    def calc_linguistic_reconstruction_loss(
        self,
        recon_x: torch.Tensor,
        x_out: LinguisticEncoderOutput,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
    ) -> LinguisticReconstructionLoss:
        """Compute linguistic reconstruction and KLD losses."""
        x_mask = length_to_input_mask(x_lengths)
        x_out_mask = length_to_input_mask(x_out.lengths)
        return LinguisticReconstructionLoss(
            x_recon_loss=self.loss_coef.recon_x
            * self.cross_entropy_loss(recon_x, x, x_mask),
            x_vae_kld_loss=self.loss_coef.vae_beta_x
            * self.vae_kld_loss(x_out.mean, x_out.vae_lvar, x_out_mask),
        )

    def calc_acoustic_reconstruction_loss(
        self,
        recon_y: torch.Tensor,
        y_out: AcousticEncoderOutput,
        y: torch.Tensor,
        y_lengths: torch.Tensor,
    ) -> AcousticReconstructionLoss:
        """Compute acoustic reconstruction and KLD losses."""
        y_mask = length_to_input_mask(y_lengths)
        y_out_mask = length_to_input_mask(y_out.lengths)
        return AcousticReconstructionLoss(
            y_recon_loss=self.loss_coef.recon_y
            * self.mse_loss(recon_y, y, y_mask),
            y_vae_kld_loss=self.loss_coef.vae_beta_y
            * self.vae_kld_loss(y_out.mean, y_out.vae_lvar, y_out_mask),
        )
