import argparse
import logging
from pathlib import Path
from typing import cast

import numpy as np
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file
from tqdm import tqdm

from vae_speech_align.config import AlignmentExpConfig, AlignmentImplementation
from vae_speech_align.dataset import Dataset, create_dataloader
from vae_speech_align.model.aco_feat_extractor import (
    AcousticFeatureExtractorBase,
    create_aco_feat_extractor,
)
from vae_speech_align.model.model import Model
from vae_speech_align.utils import write_textgrid

_logger = logging.getLogger(__name__)


def run_alignment_steps(
    *,
    model: Model,
    aco_feat_extractor: AcousticFeatureExtractorBase,
    base_list: list[str],
    dataloader: torch.utils.data.dataloader.DataLoader,
    device: torch.device,
    out_base_dir: Path,
) -> None:
    """Run Viterbi decoding and save durations and TextGrids."""
    out_state_dur_dir = out_base_dir / "state"
    out_token_dur_dir = out_base_dir / "token"
    out_textgrid_dir = out_base_dir / "textgrid"

    out_state_dur_dir.mkdir(parents=True, exist_ok=True)
    out_token_dur_dir.mkdir(parents=True, exist_ok=True)
    out_textgrid_dir.mkdir(parents=True, exist_ok=True)

    log_likelihood_sum = 0.0

    # seconds per aligned frame: the extractor's frame rate times the
    # acoustic upsampling of the model
    frame_shift = 1.0 / (
        aco_feat_extractor.frame_rate * model.acoustic_encoder.upsample_scale
    )

    model.eval()

    batch_idx = 0
    for batch in tqdm(dataloader, total=len(dataloader)):
        x, wav, phonemes, x_lengths, wav_lengths = batch

        x = x.to(device)
        wav = wav.to(device)
        x_lengths = x_lengths.to(device)
        wav_lengths = wav_lengths.to(device)

        result = model.run_viterbi(
            x, x_lengths, wav, wav_lengths, aco_feat_extractor
        )

        log_likelihood_sum += result.viterbi_likelihoods.sum().detach().item()

        for i in range(x.size(0)):
            base = base_list[batch_idx]
            batch_idx += 1

            state_dur_i = result.state_durations[i, : result.state_lengths[i]]
            token_dur_i = result.token_durations[i, : result.token_lengths[i]]

            state_dur_npy = state_dur_i.detach().cpu().numpy()
            token_dur_npy = token_dur_i.detach().cpu().numpy()

            np.save(
                out_state_dur_dir / f"{base}.npy",
                state_dur_npy.ravel().astype(np.float32),
            )
            np.save(
                out_token_dur_dir / f"{base}.npy",
                token_dur_npy.astype(np.float32),
            )

            write_textgrid(
                out_textgrid_dir / f"{base}.textgrid",
                phonemes[i],
                token_dur_i,
                state_durations=state_dur_i,
                frame_shift=frame_shift,
            )

    _logger.info("Viterbi likelihood: {:e}".format(log_likelihood_sum))

    with open(out_base_dir / "likelihood.txt", "w") as f:
        print("{:e}".format(log_likelihood_sum), file=f)


def main(args: argparse.Namespace) -> None:

    fmt = "%(asctime)s [%(name)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)

    exp_config_file = Path(args.exp_config)

    yaml_conf = OmegaConf.load(exp_config_file)
    conf = cast(
        AlignmentExpConfig,
        OmegaConf.merge(OmegaConf.structured(AlignmentExpConfig), yaml_conf),
    )

    _logger.info(OmegaConf.to_yaml(conf))

    if conf.rand_seed is not None:
        np.random.seed(conf.rand_seed)
        torch.manual_seed(conf.rand_seed)

    model = Model(conf.model, conf.features, alignment_impl=args.impl)

    model_ckpt = Path(args.model_ckpt)
    _logger.info(f"Load model: {model_ckpt}")

    model.load_state_dict(load_file(model_ckpt))

    out_base_dir = Path(args.out_dir)
    _logger.info(f"Out base dir: {out_base_dir}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    _logger.info(model)

    aco_feat_extractor = create_aco_feat_extractor(
        conf.features.acoustic_feature_extractor
    )
    aco_feat_extractor.to(device)

    dataloader = create_dataloader(
        conf,
        wav_dir=Path(args.wav_dir),
        token_dir=Path(args.phoneme_dir),
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Reuse the dataset's own ordering instead of re-globbing, so the output
    # filenames cannot drift out of sync with the samples the loader yields.
    dataset = cast(Dataset, dataloader.dataset)
    base_list = dataset.base_list

    run_alignment_steps(
        model=model,
        aco_feat_extractor=aco_feat_extractor,
        base_list=base_list,
        dataloader=dataloader,
        device=device,
        out_base_dir=out_base_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_config", type=str, required=True)
    parser.add_argument("--model_ckpt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--wav_dir", type=str, required=True)
    parser.add_argument("--phoneme_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for Viterbi decoding",
    )
    parser.add_argument(
        "--impl",
        type=AlignmentImplementation,
        choices=tuple(AlignmentImplementation),
        default=AlignmentImplementation.NUMBA,
    )

    args = parser.parse_args()

    main(args)
