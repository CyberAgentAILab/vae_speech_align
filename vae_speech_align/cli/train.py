import argparse
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import transformers
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file
from torch.utils.tensorboard import SummaryWriter

from vae_speech_align.config import (
    AlignmentExpConfig,
    AlignmentImplementation,
    TrainingConfig,
)
from vae_speech_align.dataset import create_dataloader
from vae_speech_align.model.aco_feat_extractor import create_aco_feat_extractor
from vae_speech_align.model.model import Model
from vae_speech_align.utils import RunningAverage

_logger = logging.getLogger(__name__)


def plot_matrix(
    *,
    model: Model,
    aco_feat_extractor: torch.nn.Module,
    dataloader: torch.utils.data.dataloader.DataLoader,
    device: torch.device,
    training_step: int,
    plot_dir: Path,
) -> None:
    """Plot gamma and Viterbi alignment heatmaps for the first batch."""
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            break
        x, wav, _, x_lengths, wav_lengths = batch

        x = x.to(device)
        wav = wav.to(device)
        x_lengths = x_lengths.to(device)
        wav_lengths = wav_lengths.to(device)

        y, y_lengths = aco_feat_extractor(wav, wav_lengths)
        # Match train_one_batch so the plotted alignment reflects the same
        # y shapes the model is actually trained on.
        y, y_lengths = model.adjust_y_lengths(y, y_lengths, x_lengths)

        x_out = model.forward_x(x, x_lengths)
        y_out = model.forward_y(y, y_lengths)

        gamma = model.calc_gamma(x_out, y_out)
        gamma_map = (
            gamma[0, 1 : y_out.lengths[0] + 1, 1 : x_out.lengths[0] + 1]
            .detach()
            .cpu()
            .numpy()
        )

        out_file = plot_dir / f"{training_step}-gamma.png"
        fig, ax = plt.subplots(1, 1)
        sns.heatmap(gamma_map, ax=ax)
        fig.savefig(out_file)
        plt.close(fig)

        path = model.calc_viterbi(x_out, y_out).path
        path_map = (
            path[0, 1 : y_out.lengths[0] + 1, 1 : x_out.lengths[0] + 1]
            .detach()
            .cpu()
            .numpy()
        )

        out_file = plot_dir / f"{training_step}-viterbi.png"
        fig, ax = plt.subplots(1, 1)
        sns.heatmap(path_map, ax=ax)
        fig.savefig(out_file)
        plt.close(fig)


def build_optimizer_and_scheduler(
    model: Model,
    training_config: TrainingConfig,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """Create the Adam optimizer and warmup scheduler from config."""
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=training_config.optimizer.learning_rate,
        weight_decay=training_config.optimizer.weight_decay,
    )
    scheduler = transformers.get_constant_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=training_config.scheduler.warmup_steps,
    )
    return optimizer, scheduler


def train_one_batch(
    *,
    model: Model,
    aco_feat_extractor: torch.nn.Module,
    batch: tuple,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> dict[str, torch.Tensor]:
    """Run one forward/backward/optimizer step and return the loss values."""
    model.train()

    x, wav, _, x_lengths, wav_lengths = batch

    optimizer.zero_grad()

    x = x.to(device)
    wav = wav.to(device)
    x_lengths = x_lengths.to(device)
    wav_lengths = wav_lengths.to(device)

    with torch.no_grad():
        y, y_lengths = aco_feat_extractor(wav, wav_lengths)

    y, y_lengths = model.adjust_y_lengths(y, y_lengths, x_lengths)

    x_out = model.forward_x(x, x_lengths)
    y_out = model.forward_y(y, y_lengths)

    alignment_result = model.calc_annealing_alignment(x_out, y_out)

    recon_x, _ = model.reconstruct_x(x_out)
    recon_y, _ = model.reconstruct_y(y_out)

    linguistic_loss = model.calc_linguistic_reconstruction_loss(
        recon_x, x_out, x, x_lengths
    )
    acoustic_loss = model.calc_acoustic_reconstruction_loss(
        recon_y, y_out, y, y_lengths
    )
    recon_loss = (
        linguistic_loss.x_recon_loss
        + linguistic_loss.x_vae_kld_loss
        + acoustic_loss.y_recon_loss
        + acoustic_loss.y_vae_kld_loss
    )

    loss = alignment_result.annealed_alignment_loss + recon_loss

    loss.backward()

    optimizer.step()
    scheduler.step()

    return {
        "log_likelihood": alignment_result.log_likelihood,
        "annealed_alignment_loss": alignment_result.annealed_alignment_loss,
        "alignment_loss": alignment_result.alignment_loss,
        "x_recon_loss": linguistic_loss.x_recon_loss,
        "x_vae_kld_loss": linguistic_loss.x_vae_kld_loss,
        "y_recon_loss": acoustic_loss.y_recon_loss,
        "y_vae_kld_loss": acoustic_loss.y_vae_kld_loss,
        "recon_loss": recon_loss,
        "loss": loss,
    }


def accumulate_loss_stats(
    loss_stats: dict[str, RunningAverage],
    loss_dict: dict[str, torch.Tensor],
) -> None:
    """Add each scalar loss value to its running-average tracker."""
    for key, val in loss_dict.items():
        loss_stats[key].add(val.detach().cpu().item())


def log_training_stats(
    summary_writer: torch.utils.tensorboard.writer.SummaryWriter | None,
    loss_stats: dict[str, RunningAverage],
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    model: Model,
    step: int,
) -> None:
    """Log running-average losses and learning stats to TensorBoard."""
    _logger.info(f"Running average loss: {loss_stats['loss'].value:.6e}")

    if summary_writer is None:
        return

    for key, ave in loss_stats.items():
        summary_writer.add_scalar(f"training/{key}", ave.value, step)

    summary_writer.add_scalar("lr", scheduler.get_last_lr()[0], step)
    summary_writer.add_scalar("annealing/sigma", model.annealing_sigma(), step)


def run_training_steps(  # noqa: CFQ002
    *,
    model: Model,
    aco_feat_extractor: torch.nn.Module,
    training_config: TrainingConfig,
    train_dataloader: torch.utils.data.dataloader.DataLoader,
    device: torch.device,
    summary_writer: torch.utils.tensorboard.writer.SummaryWriter | None,
    checkpoint_dir: Path,
    plot_dir: Path,
) -> None:
    """Run the training loop with annealing, plotting, and checkpointing.

    This is the training entry point; its several dependencies (model,
    feature extractor, dataloader, device, and the summary/checkpoint/plot
    output sinks) are all distinct and are intentionally passed explicitly,
    so CFQ002 (argument count) is waived here.
    """
    if len(train_dataloader) == 0:
        raise ValueError("train_dataloader is empty")

    num_steps = training_config.num_steps
    intervals = training_config.intervals
    num_epochs = int(np.ceil(num_steps / len(train_dataloader)))

    optimizer, scheduler = build_optimizer_and_scheduler(
        model, training_config
    )

    loss_stats: dict[str, RunningAverage] = defaultdict(RunningAverage)
    step = 0

    plot_matrix(
        model=model,
        aco_feat_extractor=aco_feat_extractor,
        dataloader=train_dataloader,
        device=device,
        training_step=step,
        plot_dir=plot_dir,
    )

    for epoch in range(num_epochs):
        _logger.info(f"Epoch: {epoch}/{num_epochs}")

        for batch in train_dataloader:
            loss_dict = train_one_batch(
                model=model,
                aco_feat_extractor=aco_feat_extractor,
                batch=batch,
                device=device,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            accumulate_loss_stats(loss_stats, loss_dict)
            step += 1

            if step % intervals.training_loss_interval == 0:
                _logger.info(f"Iteration: {step}/{num_steps}")
                log_training_stats(
                    summary_writer, loss_stats, scheduler, model, step
                )

            if step % intervals.plot_interval == 0:
                plot_matrix(
                    model=model,
                    aco_feat_extractor=aco_feat_extractor,
                    dataloader=train_dataloader,
                    device=device,
                    training_step=step,
                    plot_dir=plot_dir,
                )

            if step % intervals.annealing_interval == 0:
                sigma = model.update_annealing()
                _logger.info("Update annealing sigma: {:f}".format(sigma))

            if step % intervals.position_bias_scale_interval == 0:
                scale = model.update_position_bias_scale()
                _logger.info("Update position bias scale: {:f}".format(scale))

            if step % intervals.checkpoint_interval == 0:
                save_file(
                    model.state_dict(),
                    checkpoint_dir / f"{step}.safetensors",
                )

            if step >= num_steps:
                _logger.info(f"Training reached to {num_steps} iters")
                break


def main(args: argparse.Namespace) -> None:

    exp_config_file = Path(args.exp_config)

    yaml_conf = OmegaConf.load(exp_config_file)
    conf = cast(
        AlignmentExpConfig,
        OmegaConf.merge(OmegaConf.structured(AlignmentExpConfig), yaml_conf),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = out_dir / "run.log"
    print(f"Log to {log_file}")

    fmt = "%(asctime)s [%(name)s] %(message)s"
    logging.basicConfig(filename=log_file, level=logging.INFO, format=fmt)

    _logger.info(OmegaConf.to_yaml(conf))

    # rand_seed is optional: a null seed means "do not fix the RNG state".
    # np.random.seed tolerates None but torch.manual_seed raises on it, so
    # guard both together rather than seeding only one source.
    if conf.rand_seed is not None:
        np.random.seed(conf.rand_seed)
        torch.manual_seed(conf.rand_seed)

    train_dataloader = create_dataloader(
        conf,
        wav_dir=Path(args.wav_dir),
        token_dir=Path(args.phoneme_dir),
        shuffle=True,
    )

    _logger.info(len(train_dataloader))

    model = Model(conf.model, conf.features, alignment_impl=args.impl)

    if conf.initial_checkpoint != "":
        _logger.info(f"Load pretrained model: {conf.initial_checkpoint}")
        model.load_state_dict(load_file(conf.initial_checkpoint))
    else:
        _logger.info("No pretrained model loaded.")

    model.alignment.annealing_conv.reset_weight()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    _logger.info(model)

    aco_feat_extractor = create_aco_feat_extractor(
        conf.features.acoustic_feature_extractor
    )
    aco_feat_extractor.to(device)

    checkpoint_dir = out_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    plot_dir = out_dir / "plot"
    plot_dir.mkdir(parents=True, exist_ok=True)

    summary_dir = out_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer = SummaryWriter(log_dir=(summary_dir / now_str))

    run_training_steps(
        model=model,
        aco_feat_extractor=aco_feat_extractor,
        training_config=conf.training,
        train_dataloader=train_dataloader,
        device=device,
        summary_writer=summary_writer,
        checkpoint_dir=checkpoint_dir,
        plot_dir=plot_dir,
    )

    save_file(model.state_dict(), out_dir / "model.safetensors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_config", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--wav_dir", type=str, required=True)
    parser.add_argument("--phoneme_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--impl",
        type=AlignmentImplementation,
        choices=tuple(AlignmentImplementation),
        default=AlignmentImplementation.TRITON,
    )

    args = parser.parse_args()

    main(args)
