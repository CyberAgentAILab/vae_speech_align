"""Tests for vae_speech_align.model submodules."""

import pytest
import torch

from vae_speech_align import config
from vae_speech_align.model.aco_feat_extractor import MelspecExtractor
from vae_speech_align.model.annealing import AnnealingConv
from vae_speech_align.model.conv import (
    ConvModule,
    ConvModuleLayer,
    DownsampleConvModule,
    UpsampleConvModule,
)
from vae_speech_align.model.loss import (
    MaskedCrossEntropyLoss,
    MaskedMSELoss,
    MaskedVAEKLDLoss,
)
from vae_speech_align.model.model import (
    AcousticEncoder,
    AlignmentModule,
    DecoderModule,
    LinguisticEncoder,
    Model,
)
from vae_speech_align.model.position_bias import (
    PositionBiasModule,
    calc_beta_binomial_matrix,
    log_beta_func,
    log_binomial_coef,
)

# ---------------------------------------------------------------------------
# ConvModule / ConvModuleLayer
# ---------------------------------------------------------------------------


class TestConvModuleLayer:
    def test_output_shape(self):
        """Test that ConvModuleLayer preserves the [B, T, H] input shape."""
        layer = ConvModuleLayer(
            hidden_size=16, dropout_rate=0.0, kernel_size=3
        )
        x = torch.randn(2, 10, 16)
        y = layer(x)
        assert y.shape == (2, 10, 16)


class TestConvModule:
    def test_output_shape_no_residual(self):
        """Test that a non-residual ConvModule preserves the input shape."""
        m = ConvModule(
            num_layers=2,
            hidden_size=16,
            dropout_rate=0.0,
            kernel_size=3,
            residual=False,
        )
        x = torch.randn(2, 10, 16)
        y = m(x)
        assert y.shape == x.shape

    def test_output_shape_with_residual(self):
        """Test that a residual ConvModule preserves the input shape."""
        m = ConvModule(
            num_layers=2,
            hidden_size=16,
            dropout_rate=0.0,
            kernel_size=3,
            residual=True,
        )
        x = torch.randn(2, 10, 16)
        y = m(x)
        assert y.shape == x.shape

    def test_mask_applied(self):
        """Test that a forward pass with a padding mask keeps the shape."""
        m = ConvModule(
            num_layers=1,
            hidden_size=8,
            dropout_rate=0.0,
            kernel_size=3,
            residual=False,
        )
        x = torch.randn(2, 5, 8)
        mask = torch.tensor(
            [
                [True, True, True, False, False],
                [True, True, True, True, True],
            ]
        )
        y = m(x, mask=mask)
        assert y.shape == x.shape


class TestUpsampleConvModule:
    def test_upsample_2x(self):
        """Test that upsample_scale=2 doubles the time axis length."""
        m = UpsampleConvModule(in_channels=8, out_channels=8, upsample_scale=2)
        x = torch.randn(2, 5, 8)
        y = m(x)
        assert y.shape == (2, 10, 8)

    def test_upsample_3x(self):
        """Test that upsample_scale=3 triples the time axis and maps the
        channel dimension from 8 to 16."""
        m = UpsampleConvModule(
            in_channels=8, out_channels=16, upsample_scale=3
        )
        x = torch.randn(1, 4, 8)
        y = m(x)
        assert y.shape == (1, 12, 16)


class TestDownsampleConvModule:
    def test_downsample_2x(self):
        """Test that downsample_scale=2 halves the time axis length."""
        m = DownsampleConvModule(
            in_channels=8, out_channels=8, downsample_scale=2
        )
        x = torch.randn(2, 10, 8)
        y = m(x)
        assert y.shape == (2, 5, 8)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


class TestMaskedMSELoss:
    def test_zero_loss_on_identical(self):
        """Test that the masked MSE loss is zero for identical tensors."""
        loss_fn = MaskedMSELoss()
        x = torch.randn(2, 5, 8)
        mask = torch.ones(2, 5, dtype=torch.bool)
        loss = loss_fn(x, x, mask)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_all_masked_raises(self):
        """Test that an all-False mask raises ValueError instead of
        returning a silent 0/0 NaN."""
        loss_fn = MaskedMSELoss()
        x = torch.randn(1, 3, 4)
        mask = torch.zeros(1, 3, dtype=torch.bool)
        with pytest.raises(ValueError, match="no valid positions"):
            loss_fn(x, x, mask)

    def test_masked_positions_ignored(self):
        """Test that only mask-valid positions contribute to the MSE loss."""
        loss_fn = MaskedMSELoss()
        output = torch.zeros(1, 3, 4)
        target = torch.ones(1, 3, 4)
        # Only first position is valid
        mask = torch.tensor([[True, False, False]])
        loss = loss_fn(output, target, mask)
        # MSE at position 0: mean over dim=-1 of (0-1)^2 = 1 per dim, sum 4
        assert loss.item() == pytest.approx(4.0)


class TestMaskedCrossEntropyLoss:
    def test_output_is_scalar(self):
        """Test that the masked cross-entropy loss reduces to a scalar."""
        loss_fn = MaskedCrossEntropyLoss()
        output = torch.randn(2, 5, 10)
        target = torch.randint(0, 10, (2, 5))
        mask = torch.ones(2, 5, dtype=torch.bool)
        loss = loss_fn(output, target, mask)
        assert loss.ndim == 0

    def test_all_masked_raises(self):
        """Test that an all-False mask raises ValueError instead of
        returning a silent 0/0 NaN."""
        loss_fn = MaskedCrossEntropyLoss()
        output = torch.randn(1, 3, 5)
        target = torch.tensor([[2, 0, 0]])
        mask = torch.zeros(1, 3, dtype=torch.bool)
        with pytest.raises(ValueError, match="no valid positions"):
            loss_fn(output, target, mask)

    def test_masked_positions_ignored(self):
        """Test that changing targets at masked positions leaves the
        cross-entropy loss unchanged."""
        loss_fn = MaskedCrossEntropyLoss()
        output = torch.randn(1, 3, 5)
        target = torch.tensor([[2, 0, 0]])
        mask = torch.tensor([[True, False, False]])
        loss1 = loss_fn(output, target, mask)
        target2 = torch.tensor([[2, 3, 4]])
        loss2 = loss_fn(output, target2, mask)
        assert loss1.item() == pytest.approx(loss2.item())


class TestMaskedVAEKLDLoss:
    def test_zero_at_prior(self):
        """Test that the KLD loss is zero when mean=0 and lvar=0, i.e.
        the posterior equals the N(0, 1) prior."""
        loss_fn = MaskedVAEKLDLoss()
        mean = torch.zeros(2, 5, 8)
        lvar = torch.zeros(2, 5, 8)
        mask = torch.ones(2, 5, dtype=torch.bool)
        loss = loss_fn(mean, lvar, mask)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_all_masked_raises(self):
        """Test that an all-False mask raises ValueError instead of
        returning a silent 0/0 NaN."""
        loss_fn = MaskedVAEKLDLoss()
        mean = torch.zeros(1, 3, 4)
        lvar = torch.zeros(1, 3, 4)
        mask = torch.zeros(1, 3, dtype=torch.bool)
        with pytest.raises(ValueError, match="no valid positions"):
            loss_fn(mean, lvar, mask)

    def test_positive_for_non_prior(self):
        """Test that a nonzero posterior mean yields a positive KLD loss."""
        loss_fn = MaskedVAEKLDLoss()
        mean = torch.ones(2, 5, 8)
        lvar = torch.zeros(2, 5, 8)
        mask = torch.ones(2, 5, dtype=torch.bool)
        loss = loss_fn(mean, lvar, mask)
        assert loss.item() > 0


# ---------------------------------------------------------------------------
# AnnealingConv
# ---------------------------------------------------------------------------


class TestAnnealingConv:
    def test_update_phase_decreases_sigma(self):
        """Test that update_phase multiplies sigma by the update rate."""
        ac = AnnealingConv(initial_sigma=2.0, update_rate=0.5, kernel_size=3)
        assert ac.sigma.item() == pytest.approx(2.0)
        ac.update_phase()
        assert ac.sigma.item() == pytest.approx(1.0)

    def test_reset_weight(self):
        """Test that reset_weight restores sigma to its initial value."""
        ac = AnnealingConv(initial_sigma=2.0, update_rate=0.5, kernel_size=3)
        ac.update_phase()
        ac.reset_weight()
        assert ac.sigma.item() == pytest.approx(2.0)

    def test_non_positive_initial_sigma_raises(self):
        """Test that a zero or negative initial_sigma raises ValueError
        at construction."""
        with pytest.raises(ValueError, match="must be positive"):
            AnnealingConv(initial_sigma=0.0, update_rate=0.5, kernel_size=3)
        with pytest.raises(ValueError, match="must be positive"):
            AnnealingConv(initial_sigma=-2.0, update_rate=0.5, kernel_size=3)

    def test_out_of_range_update_rate_raises(self):
        """Test that an update_rate outside (0, 1] raises ValueError at
        construction."""
        with pytest.raises(ValueError, match="must be in"):
            AnnealingConv(initial_sigma=2.0, update_rate=0.0, kernel_size=3)
        with pytest.raises(ValueError, match="must be in"):
            AnnealingConv(initial_sigma=2.0, update_rate=1.5, kernel_size=3)

    def test_output_shape(self):
        """Test that the forward pass preserves the padded input shape."""
        ac = AnnealingConv(initial_sigma=2.0, update_rate=0.9, kernel_size=3)
        B, T, K = 2, 6, 4
        # AnnealingConv expects padded input [B, T+2, K+2]
        x = torch.rand(B, T + 2, K + 2)
        output_lengths = torch.tensor([T, T - 1])
        state_lengths = torch.tensor([K, K - 1])
        y = ac(x, output_lengths, state_lengths)
        assert y.shape == x.shape

    def test_kernel_weights_gaussian(self):
        """Test that the kernel is a symmetric Gaussian whose center
        weight is exp(0) = 1."""
        ac = AnnealingConv(initial_sigma=2.0, update_rate=0.9, kernel_size=5)
        w = ac.conv.weight.squeeze()
        assert w[2].item() == pytest.approx(1.0)
        assert w[0].item() == pytest.approx(w[4].item())
        assert w[1].item() == pytest.approx(w[3].item())
        assert w[1] > w[0]

    def test_forward_normalizes_valid_rows(self):
        """Test that normalize=True makes each valid row sum to one."""
        ac = AnnealingConv(initial_sigma=2.0, update_rate=0.9, kernel_size=3)
        B, T, K = 2, 6, 4
        x = torch.rand(B, T + 2, K + 2)
        output_lengths = torch.tensor([T, T - 1])
        state_lengths = torch.tensor([K, K - 1])
        y = ac(x, output_lengths, state_lengths, normalize=True)
        for b in range(B):
            row_sums = y[b, 1 : output_lengths[b] + 1].sum(dim=-1)
            assert torch.allclose(
                row_sums, torch.ones_like(row_sums), atol=1e-4
            )

    def test_forward_zeroes_masked_positions(self):
        """Test that padded borders and positions beyond the sequence
        lengths are zeroed in the output."""
        ac = AnnealingConv(initial_sigma=2.0, update_rate=0.9, kernel_size=3)
        B, T, K = 2, 6, 4
        x = torch.rand(B, T + 2, K + 2)
        output_lengths = torch.tensor([T, T - 1])
        state_lengths = torch.tensor([K, K - 1])
        y = ac(x, output_lengths, state_lengths)
        # padded borders are always zero
        assert (y[:, 0, :] == 0).all()
        assert (y[:, -1, :] == 0).all()
        assert (y[:, :, 0] == 0).all()
        assert (y[:, :, -1] == 0).all()
        # positions beyond the second element's lengths are zero
        assert (y[1, output_lengths[1] + 1 :, :] == 0).all()
        assert (y[1, :, state_lengths[1] + 1 :] == 0).all()

    def test_forward_without_normalize(self):
        """Test that normalize=False leaves the row sums unnormalized."""
        ac = AnnealingConv(initial_sigma=2.0, update_rate=0.9, kernel_size=3)
        B, T, K = 1, 4, 3
        x = torch.rand(B, T + 2, K + 2)
        output_lengths = torch.tensor([T])
        state_lengths = torch.tensor([K])
        y = ac(x, output_lengths, state_lengths, normalize=False)
        row_sums = y[0, 1 : T + 1].sum(dim=-1)
        assert not torch.allclose(row_sums, torch.ones_like(row_sums))


# ---------------------------------------------------------------------------
# Position bias
# ---------------------------------------------------------------------------


class TestPositionBias:
    def test_log_binomial_coef(self):
        """Test that log_binomial_coef reproduces C(5, 2) = 10."""
        result = log_binomial_coef(torch.tensor(5.0), torch.tensor(2.0))
        assert torch.exp(result).item() == pytest.approx(10.0, rel=1e-4)

    def test_log_beta_func(self):
        """Test that log_beta_func reproduces B(1, 1) = 1."""
        result = log_beta_func(torch.tensor(1.0), torch.tensor(1.0))
        assert torch.exp(result).item() == pytest.approx(1.0, rel=1e-4)

    def test_calc_beta_binomial_matrix_shape(self):
        """Test that the bias matrix shape is [B, max_T, max_K]."""
        output_lengths = torch.tensor([10, 8])
        state_lengths = torch.tensor([5, 4])
        mat = calc_beta_binomial_matrix(output_lengths, state_lengths)
        assert mat.shape == (2, 10, 5)

    def test_position_bias_module_update(self):
        """Test that update_phase multiplies the scale by the update rate."""
        m = PositionBiasModule(initial_scale=10.0, update_rate=0.5)
        m.update_phase()
        assert m.scale.item() == pytest.approx(5.0)

    def test_position_bias_module_reset(self):
        """Test that reset_scale restores the scale's initial value."""
        m = PositionBiasModule(initial_scale=10.0, update_rate=0.5)
        m.update_phase()
        m.reset_scale()
        assert m.scale.item() == pytest.approx(10.0)

    def test_non_positive_initial_scale_raises(self):
        """Test that a zero or negative initial_scale raises ValueError
        at construction."""
        with pytest.raises(ValueError, match="must be positive"):
            PositionBiasModule(initial_scale=0.0, update_rate=0.5)
        with pytest.raises(ValueError, match="must be positive"):
            PositionBiasModule(initial_scale=-1.0, update_rate=0.5)

    def test_out_of_range_update_rate_raises(self):
        """Test that an update_rate outside (0, 1] raises ValueError at
        construction."""
        with pytest.raises(ValueError, match="must be in"):
            PositionBiasModule(initial_scale=1.0, update_rate=0.0)
        with pytest.raises(ValueError, match="must be in"):
            PositionBiasModule(initial_scale=1.0, update_rate=1.5)

    def test_beta_binomial_rows_sum_to_one(self):
        """Test that each valid row is a beta-binomial distribution
        over states whose probabilities sum to one."""
        output_lengths = torch.tensor([10, 8])
        state_lengths = torch.tensor([5, 4])
        mat = calc_beta_binomial_matrix(
            output_lengths, state_lengths, scaling_factor=1.0
        )
        probs = torch.exp(mat)
        for b in range(2):
            for t in range(output_lengths[b]):
                row_sum = probs[b, t, : state_lengths[b]].sum()
                assert row_sum.item() == pytest.approx(
                    1.0, abs=1e-4
                ), f"b={b}, t={t}"

    def test_beta_binomial_diagonal_bias(self):
        """Test that early frames favor early states and late frames
        favor late states."""
        output_lengths = torch.tensor([20])
        state_lengths = torch.tensor([5])
        mat = calc_beta_binomial_matrix(
            output_lengths, state_lengths, scaling_factor=1.0
        )
        # frame t is scored by BetaBinomial(K - 1, t + 1, T - t), whose
        # mode sits near the diagonal (K - 1) * t / (T - 1); with T=20,
        # K=5 the first frame (t=0) peaks at state 0 and the last frame
        # (t=19) at the last state 4
        assert mat[0, 0].argmax().item() == 0
        assert mat[0, 19].argmax().item() == 4

    def test_position_bias_module_forward(self):
        """Test that forward matches calc_beta_binomial_matrix computed
        with the module's current scale."""
        m = PositionBiasModule(initial_scale=1.0, update_rate=0.9)
        output_lengths = torch.tensor([10, 8])
        state_lengths = torch.tensor([5, 4])
        result = m(output_lengths, state_lengths)
        expected = calc_beta_binomial_matrix(
            output_lengths, state_lengths, scaling_factor=m.scale
        )
        assert result.shape == (2, 10, 5)
        assert torch.allclose(result, expected)


# ---------------------------------------------------------------------------
# LinguisticEncoder / AcousticEncoder / DecoderModule
# ---------------------------------------------------------------------------


def _make_conv_stack_config():
    return config.ConvStack(
        residual=False,
        num_layers=2,
        hidden_dim=16,
        conv_kernel_size=3,
        dropout_rate=0.0,
    )


class TestLinguisticEncoder:
    def test_output_shape(self):
        """Test that mean/var outputs are [B, K, matching_dim] and the
        input lengths pass through unchanged."""
        enc = LinguisticEncoder(
            encoder_config=_make_conv_stack_config(),
            num_classes=10,
            matching_dim=8,
            upsample_scale=1,
        )
        x = torch.randint(0, 10, (2, 5))
        x_lengths = torch.tensor([5, 3])
        out = enc(x, x_lengths)
        assert out.mean.shape == (2, 5, 8)
        assert out.var.shape == (2, 5, 8)
        assert torch.equal(out.lengths, x_lengths)

    def test_upsample_scale(self):
        """Test that upsample_scale=2 doubles the output time axis and
        the output lengths."""
        enc = LinguisticEncoder(
            encoder_config=_make_conv_stack_config(),
            num_classes=10,
            matching_dim=8,
            upsample_scale=2,
        )
        x = torch.randint(0, 10, (2, 5))
        x_lengths = torch.tensor([5, 3])
        out = enc(x, x_lengths)
        assert out.mean.shape[1] == 10  # 5 * 2
        assert out.lengths[0].item() == 10
        assert out.lengths[1].item() == 6


class TestAcousticEncoder:
    def test_output_shape(self):
        """Test that the mean output is [B, T, matching_dim] with the
        input lengths unchanged."""
        enc = AcousticEncoder(
            encoder_config=_make_conv_stack_config(),
            input_dim=80,
            matching_dim=8,
            upsample_scale=1,
        )
        y = torch.randn(2, 20, 80)
        y_lengths = torch.tensor([20, 15])
        out = enc(y, y_lengths)
        assert out.mean.shape == (2, 20, 8)
        assert torch.equal(out.lengths, y_lengths)


class TestDecoderModule:
    def test_output_shape(self):
        """Test that the decoder outputs [B, T, output_dim] features."""
        dec = DecoderModule(
            decoder_config=_make_conv_stack_config(),
            input_dim=8,
            output_dim=10,
            downsample_scale=1,
        )
        x = torch.randn(2, 10, 8)
        x_lengths = torch.tensor([10, 7])
        out, out_lengths = dec(x, x_lengths)
        assert out.shape == (2, 10, 10)

    def test_downsample(self):
        """Test that downsample_scale=2 halves the time axis and the
        output lengths."""
        dec = DecoderModule(
            decoder_config=_make_conv_stack_config(),
            input_dim=8,
            output_dim=10,
            downsample_scale=2,
        )
        x = torch.randn(2, 10, 8)
        x_lengths = torch.tensor([10, 8])
        out, out_lengths = dec(x, x_lengths)
        assert out.shape == (2, 5, 10)
        assert out_lengths[0].item() == 5
        assert out_lengths[1].item() == 4


# ---------------------------------------------------------------------------
# Model (smoke test)
# ---------------------------------------------------------------------------


class TestModel:
    @staticmethod
    def _make_model():
        model_config = config.AlignmentModel(
            acoustic_encoder=_make_conv_stack_config(),
            linguistic_encoder=_make_conv_stack_config(),
            acoustic_decoder=_make_conv_stack_config(),
            linguistic_decoder=_make_conv_stack_config(),
            acoustic_upsample_scale=1,
            linguistic_upsample_scale=1,
            matching_dim=8,
            loss=config.TrainingLoss(
                alignment=1.0,
                recon_x=0.0,
                recon_y=0.0,
                vae_beta_x=0.0,
                vae_beta_y=0.0,
            ),
            position_bias=config.PositionBias(
                bias_type="NONE", initial_scale=1.0, update_rate=1.0
            ),
            annealing_weight=config.AnnealingWeight(
                initial_sigma=2.0, update_rate=0.9, kernel_size=3
            ),
        )
        features = config.FeaturesWithExtractor(
            acoustic_feature_extractor=config.AcousticFeatureExtractor(
                model_type="MELSPEC",
                model_name="",
                layer_idx=0,
                num_dims=80,
            ),
            linguistic=config.LinguisticFeatures(
                tag="phoneme", num_tokens=5, token_list_file=""
            ),
        )
        return Model(
            model_config,
            features,
            alignment_impl=config.AlignmentImplementation.NUMBA,
        )

    def test_forward_x(self):
        """Test that forward_x runs the linguistic encoder over a batch."""
        model = self._make_model()
        x = torch.randint(0, 6, (2, 5))
        x_lengths = torch.tensor([5, 3])
        out = model.forward_x(x, x_lengths)
        assert out.mean.shape[0] == 2

    def test_forward_y(self):
        """Test that forward_y runs the acoustic encoder over a batch."""
        model = self._make_model()
        y = torch.randn(2, 20, 80)
        y_lengths = torch.tensor([20, 15])
        out = model.forward_y(y, y_lengths)
        assert out.mean.shape[0] == 2

    def test_reparameterize_eval(self):
        """Test that reparameterize returns the mean in eval mode."""
        model = self._make_model()
        model.eval()
        mean = torch.randn(2, 5, 8)
        lvar = torch.zeros(2, 5, 8)
        result = model.reparameterize(mean, lvar)
        assert torch.equal(result, mean)

    def test_reparameterize_train(self):
        """Test that training-mode reparameterization draws a noisy
        sample of the same shape as the mean."""
        model = self._make_model()
        model.train()
        mean = torch.randn(2, 5, 8)
        lvar = torch.zeros(2, 5, 8)
        result = model.reparameterize(mean, lvar)
        assert result.shape == mean.shape

    def test_set_alignment_impl(self):
        """Test that set_alignment_impl switches the alignment backend."""
        pytest.importorskip("triton")
        model = self._make_model()
        model.set_alignment_impl(config.AlignmentImplementation.TRITON)
        assert (
            model.alignment.alignment_impl
            is config.AlignmentImplementation.TRITON
        )

    def test_set_alignment_impl_rejects_unknown_backend(self):
        """Test that an unknown alignment backend raises ValueError."""
        model = self._make_model()
        with pytest.raises(ValueError, match="invalid"):
            model.set_alignment_impl("invalid")  # type: ignore[arg-type]

    def test_annealing_delegation(self):
        """Test that the model-level accessors report sigma and decay
        it by the annealing update rate."""
        model = self._make_model()
        sigma0 = model.annealing_sigma()
        assert sigma0 == pytest.approx(2.0)
        sigma1 = model.update_annealing()
        assert sigma1 == pytest.approx(2.0 * 0.9)
        assert model.annealing_sigma() == pytest.approx(sigma1)

    def test_position_bias_scale_delegation(self):
        """Test that update_position_bias_scale returns the updated
        position-bias scale."""
        model = self._make_model()
        scale = model.update_position_bias_scale()
        assert scale == pytest.approx(
            1.0
        )  # initial_scale 1.0 * update_rate 1.0


class TestScaleValidation:
    def test_non_positive_upsample_scale_raises(self):
        """Test that an upsample_scale below 1 raises ValueError at
        encoder construction."""
        with pytest.raises(ValueError, match="must be >= 1"):
            LinguisticEncoder(
                encoder_config=_make_conv_stack_config(),
                num_classes=5,
                matching_dim=8,
                upsample_scale=0,
            )
        with pytest.raises(ValueError, match="must be >= 1"):
            AcousticEncoder(
                encoder_config=_make_conv_stack_config(),
                input_dim=80,
                matching_dim=8,
                upsample_scale=0,
            )

    def test_non_positive_downsample_scale_raises(self):
        """Test that a downsample_scale below 1 raises ValueError at
        decoder construction."""
        with pytest.raises(ValueError, match="must be >= 1"):
            DecoderModule(
                decoder_config=_make_conv_stack_config(),
                output_dim=80,
                input_dim=8,
                downsample_scale=0,
            )


# ---------------------------------------------------------------------------
# AlignmentModule (numba implementation on CPU)
# ---------------------------------------------------------------------------


def _make_alignment_module(
    bias_type=config.PositionBiasType.NONE, initial_scale=1.0
):
    return AlignmentModule(
        position_bias_config=config.PositionBias(
            bias_type=bias_type,
            initial_scale=initial_scale,
            update_rate=0.9,
        ),
        annealing_config=config.AnnealingWeight(
            initial_sigma=2.0, update_rate=0.9, kernel_size=3
        ),
        alignment_loss_coef=1.0,
        alignment_impl=config.AlignmentImplementation.NUMBA,
    )


def _make_encoder_outputs(B=2, T=20, K=5, D=8, seed=0):
    """Build random LinguisticEncoderOutput / AcousticEncoderOutput pairs."""
    from vae_speech_align.model.model import (
        AcousticEncoderOutput,
        LinguisticEncoderOutput,
    )

    torch.manual_seed(seed)
    x_out = LinguisticEncoderOutput(
        mean=torch.randn(B, K, D),
        var=torch.ones(B, K, D),
        vae_lvar=torch.zeros(B, K, D),
        lengths=torch.tensor([K, K - 1]),
    )
    y_out = AcousticEncoderOutput(
        mean=torch.randn(B, T, D),
        vae_lvar=torch.zeros(B, T, D),
        lengths=torch.tensor([T, T - 4]),
    )
    return x_out, y_out


class TestAlignmentModule:
    def test_calc_likelihood(self):
        """Test that the log-likelihood is finite and the alignment
        loss is its negation."""
        module = _make_alignment_module()
        x_out, y_out = _make_encoder_outputs()
        result = module.calc_likelihood(x_out, y_out)
        assert torch.isfinite(result.log_likelihood)
        assert result.alignment_loss.item() == pytest.approx(
            -result.log_likelihood.item()
        )

    def test_zero_total_y_length_raises(self):
        """Test that all-zero acoustic lengths raise ValueError before
        the likelihood is divided by their sum."""
        module = _make_alignment_module()
        x_out, y_out = _make_encoder_outputs()
        y_out.lengths = torch.zeros_like(y_out.lengths)
        with pytest.raises(ValueError, match="positive"):
            module.calc_likelihood(x_out, y_out)
        with pytest.raises(ValueError, match="positive"):
            module.calc_annealing_alignment(x_out, y_out)

    def test_calc_gamma_rows_sum_to_one(self):
        """Test that each valid row of the padded [B, T+2, K+2] gamma
        matrix is a posterior summing to one."""
        module = _make_alignment_module()
        x_out, y_out = _make_encoder_outputs()
        gamma = module.calc_gamma(x_out, y_out)
        B = gamma.size(0)
        for b in range(B):
            T_b = y_out.lengths[b].item()
            row_sums = gamma[b, 1 : T_b + 1].sum(dim=-1)
            assert torch.allclose(
                row_sums, torch.ones_like(row_sums), atol=1e-3
            ), f"b={b}"

    def test_calc_annealing_alignment(self):
        """Test that annealed alignment losses are finite and the
        log-likelihood matches calc_likelihood."""
        module = _make_alignment_module()
        x_out, y_out = _make_encoder_outputs()
        result = module.calc_annealing_alignment(x_out, y_out)
        assert torch.isfinite(result.log_likelihood)
        assert torch.isfinite(result.annealed_alignment_loss)
        assert torch.isfinite(result.alignment_loss)
        # the forward-sum likelihood must match calc_likelihood
        likelihood = module.calc_likelihood(x_out, y_out)
        assert result.log_likelihood.item() == pytest.approx(
            likelihood.log_likelihood.item()
        )

    def test_calc_viterbi_path_is_valid(self):
        """Test that the Viterbi path is one-hot per valid frame and
        moves monotonically from the first to the last state."""
        module = _make_alignment_module()
        x_out, y_out = _make_encoder_outputs()
        result = module.calc_viterbi(x_out, y_out)
        log_likelihoods = result.log_likelihoods
        path = result.path
        B = path.size(0)
        assert torch.isfinite(log_likelihoods).all()
        # each valid frame occupies exactly one state
        assert path.sum().item() == y_out.lengths.sum().item()
        for b in range(B):
            T_b = y_out.lengths[b].item()
            K_b = x_out.lengths[b].item()
            frame_states = path[b, 1 : T_b + 1].argmax(dim=-1)
            assert (path[b, 1 : T_b + 1].sum(dim=-1) == 1).all()
            # monotonic non-decreasing path from the first to the last state
            assert (frame_states[1:] >= frame_states[:-1]).all()
            assert frame_states[0].item() == 1
            assert frame_states[-1].item() == K_b

    def test_position_bias_logprob_changes_likelihood(self):
        """Test that enabling the LOGPROB position bias changes the
        forward-sum log-likelihood."""
        x_out, y_out = _make_encoder_outputs()
        module_none = _make_alignment_module(
            bias_type=config.PositionBiasType.NONE
        )
        module_bias = _make_alignment_module(
            bias_type=config.PositionBiasType.LOGPROB,
            initial_scale=1.0,
        )
        ll_none = module_none.calc_likelihood(x_out, y_out).log_likelihood
        ll_bias = module_bias.calc_likelihood(x_out, y_out).log_likelihood
        assert torch.isfinite(ll_bias)
        assert ll_none.item() != pytest.approx(ll_bias.item())


# ---------------------------------------------------------------------------
# Model: adjust_y_lengths / run_viterbi / reconstruction
# ---------------------------------------------------------------------------


class TestAdjustYLengths:
    def test_noop_when_y_is_long_enough(self):
        """Test that adjust_y_lengths returns the inputs unchanged when
        y is already long enough."""
        model = TestModel._make_model()
        y = torch.randn(1, 20, 80)
        y_lengths = torch.tensor([20])
        x_lengths = torch.tensor([5])
        y2, y2_lengths = model.adjust_y_lengths(y, y_lengths, x_lengths)
        assert y2 is y
        assert torch.equal(y2_lengths, y_lengths)

    def test_pads_when_y_is_too_short(self):
        """Test that y is zero-padded up to the required length while
        the original frames are kept."""
        model = TestModel._make_model()
        y = torch.randn(1, 5, 80)
        y_lengths = torch.tensor([5])
        x_lengths = torch.tensor([10])
        y2, y2_lengths = model.adjust_y_lengths(y, y_lengths, x_lengths)
        assert y2_lengths[0].item() == 10
        assert y2.size(-2) == 10
        # newly padded frames are zero, original frames are unchanged
        assert (y2[:, 5:] == 0).all()
        assert torch.equal(y2[:, :5], y)

    def test_extends_only_short_elements_in_batch(self):
        """Test that only batch items shorter than their x length get
        extended lengths, without padding the tensor."""
        model = TestModel._make_model()
        y = torch.randn(2, 12, 80)
        y_lengths = torch.tensor([12, 5])
        x_lengths = torch.tensor([5, 8])
        y2, y2_lengths = model.adjust_y_lengths(y, y_lengths, x_lengths)
        assert y2_lengths[0].item() == 12  # long enough already
        assert y2_lengths[1].item() == 8  # extended to x length
        assert y2.size(-2) == 12  # no tensor padding needed


class TestRunViterbi:
    def test_durations_sum_to_frame_counts(self):
        """Test that per-state and per-token durations each sum to the
        number of feature frames per item."""
        model = TestModel._make_model()
        model.eval()
        extractor = MelspecExtractor(num_dims=80)

        sr = 16000
        torch.manual_seed(0)
        wav = torch.randn(2, sr) * 0.01
        wav_lengths = torch.tensor([sr, sr // 2])
        x = torch.randint(1, 6, (2, 5))
        x_lengths = torch.tensor([5, 3])

        result = model.run_viterbi(
            x,
            x_lengths,
            wav,
            wav_lengths,
            extractor,
        )

        _, feat_lengths = extractor(wav, wav_lengths)

        assert result.viterbi_likelihoods.shape == (2,)
        # upsample_scale == 1
        assert torch.equal(result.state_lengths, x_lengths)
        assert torch.equal(result.token_lengths, x_lengths)
        for b in range(2):
            n_states = result.state_lengths[b].item()
            assert (
                result.state_durations[b, :n_states].sum().item()
                == feat_lengths[b].item()
            )
            assert (
                result.token_durations[b, :n_states].sum().item()
                == feat_lengths[b].item()
            )
            # padded entries hold no duration
            assert result.state_durations[b, n_states:].sum().item() == 0

    def test_token_durations_with_upsample_scale(self):
        """Test that with linguistic upsample scale 3 each token
        duration is the sum of its three state durations."""
        model_config = config.AlignmentModel(
            acoustic_encoder=_make_conv_stack_config(),
            linguistic_encoder=_make_conv_stack_config(),
            acoustic_decoder=_make_conv_stack_config(),
            linguistic_decoder=_make_conv_stack_config(),
            acoustic_upsample_scale=1,
            linguistic_upsample_scale=3,
            matching_dim=8,
            loss=config.TrainingLoss(
                alignment=1.0,
                recon_x=0.0,
                recon_y=0.0,
                vae_beta_x=0.0,
                vae_beta_y=0.0,
            ),
            position_bias=config.PositionBias(
                bias_type="NONE", initial_scale=1.0, update_rate=1.0
            ),
            annealing_weight=config.AnnealingWeight(
                initial_sigma=2.0, update_rate=0.9, kernel_size=3
            ),
        )
        features = config.FeaturesWithExtractor(
            acoustic_feature_extractor=config.AcousticFeatureExtractor(
                model_type="MELSPEC",
                model_name="",
                layer_idx=0,
                num_dims=80,
            ),
            linguistic=config.LinguisticFeatures(
                tag="phoneme", num_tokens=5, token_list_file=""
            ),
        )
        model = Model(
            model_config,
            features,
            alignment_impl=config.AlignmentImplementation.NUMBA,
        )
        model.eval()
        extractor = MelspecExtractor(num_dims=80)

        sr = 16000
        torch.manual_seed(0)
        wav = torch.randn(1, sr) * 0.01
        wav_lengths = torch.tensor([sr])
        x = torch.randint(1, 6, (1, 4))
        x_lengths = torch.tensor([4])

        result = model.run_viterbi(
            x,
            x_lengths,
            wav,
            wav_lengths,
            extractor,
        )

        assert result.state_lengths[0].item() == 12  # 4 tokens x 3 states
        assert result.token_lengths[0].item() == 4
        # each token duration is the sum of its 3 state durations
        expected = result.state_durations[0, :12].reshape(4, 3).sum(dim=-1)
        assert torch.equal(result.token_durations[0, :4], expected)


class TestReconstruction:
    def test_reconstruct_x_shape(self):
        """Test that reconstruct_x outputs [B, K, num_tokens + 1]
        logits with the input lengths preserved."""
        model = TestModel._make_model()
        model.eval()
        x = torch.randint(1, 6, (2, 5))
        x_lengths = torch.tensor([5, 3])
        x_out = model.forward_x(x, x_lengths)
        recon_x, recon_lengths = model.reconstruct_x(x_out)
        assert recon_x.shape == (2, 5, 6)  # num_tokens + 1 classes
        assert torch.equal(recon_lengths, x_lengths)

    def test_reconstruct_y_shape(self):
        """Test that reconstruct_y restores the [B, T, input_dim] shape
        with the input lengths preserved."""
        model = TestModel._make_model()
        model.eval()
        y = torch.randn(2, 20, 80)
        y_lengths = torch.tensor([20, 15])
        y_out = model.forward_y(y, y_lengths)
        recon_y, recon_lengths = model.reconstruct_y(y_out)
        assert recon_y.shape == (2, 20, 80)
        assert torch.equal(recon_lengths, y_lengths)

    def test_calc_reconstruction_loss(self):
        """Test that reconstruction and VAE KLD loss terms are finite."""
        model = TestModel._make_model()
        model.eval()
        x = torch.randint(1, 6, (2, 5))
        x_lengths = torch.tensor([5, 3])
        y = torch.randn(2, 20, 80)
        y_lengths = torch.tensor([20, 15])

        x_out = model.forward_x(x, x_lengths)
        y_out = model.forward_y(y, y_lengths)
        recon_x, _ = model.reconstruct_x(x_out)
        recon_y, _ = model.reconstruct_y(y_out)

        linguistic_loss = model.calc_linguistic_reconstruction_loss(
            recon_x,
            x_out,
            x,
            x_lengths,
        )
        acoustic_loss = model.calc_acoustic_reconstruction_loss(
            recon_y,
            y_out,
            y,
            y_lengths,
        )
        for v in (
            linguistic_loss.x_recon_loss,
            linguistic_loss.x_vae_kld_loss,
            acoustic_loss.y_recon_loss,
            acoustic_loss.y_vae_kld_loss,
        ):
            assert torch.isfinite(v)


class TestAcousticEncoderUpsample:
    def test_upsample_scale_2(self):
        """Test that upsample_scale=2 doubles the acoustic time axis
        and the output lengths."""
        enc = AcousticEncoder(
            encoder_config=_make_conv_stack_config(),
            input_dim=80,
            matching_dim=8,
            upsample_scale=2,
        )
        y = torch.randn(2, 10, 80)
        y_lengths = torch.tensor([10, 7])
        out = enc(y, y_lengths)
        assert out.mean.shape == (2, 20, 8)
        assert out.lengths[0].item() == 20
        assert out.lengths[1].item() == 14
