"""Command-line interface for the Spine codec: train, encode, decode, recon."""

from __future__ import annotations

import argparse
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spine.model import Spine


def _default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_model(checkpoint: str | None, device: str) -> Spine:
    if checkpoint:
        from spine.tools import load_model_from_checkpoint

        return load_model_from_checkpoint(checkpoint, device)
    from spine.model import Spine

    return Spine.from_pretrained(device=device)


def cmd_train(args: argparse.Namespace) -> None:
    from spine.config import from_yaml
    from spine.train import train

    train(from_yaml(args.config), resume_from=args.resume)


def cmd_encode(args: argparse.Namespace) -> None:
    import torch

    from spine.tools import encode_file

    device = _default_device()
    model = _load_model(args.checkpoint, device)
    codes = encode_file(model, args.input, device)
    torch.save([c.cpu() for c in codes], args.output)
    print(f"encoded {args.input} -> {args.output} ({len(codes)} scales)")


def cmd_decode(args: argparse.Namespace) -> None:
    import torch
    import torchaudio

    device = _default_device()
    model = _load_model(args.checkpoint, device)
    codes = [c.to(device) for c in torch.load(args.input, map_location=device)]
    with torch.no_grad():
        audio_hat = model.decode(codes)
    torchaudio.save(args.output, audio_hat.squeeze(0).cpu(), model.sample_rate)
    print(f"decoded {args.input} -> {args.output}")


def cmd_recon(args: argparse.Namespace) -> None:
    from spine.tools import reconstruct_file

    device = _default_device()
    model = _load_model(args.checkpoint, device)
    reconstruct_file(model, args.input, args.output, device)
    print(f"reconstructed {args.input} -> {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spine", description="Spine neural audio codec")
    sub = parser.add_subparsers(dest="command")

    p_train = sub.add_parser("train", help="Train the codec")
    p_train.add_argument("--config", required=True)
    p_train.add_argument("--resume", default=None, help="Checkpoint to resume from")
    p_train.set_defaults(func=cmd_train)

    p_encode = sub.add_parser("encode", help="Encode an audio file to tokens")
    p_encode.add_argument(
        "--checkpoint", default=None, help="Local checkpoint (default: Hub model)"
    )
    p_encode.add_argument("--input", required=True)
    p_encode.add_argument("--output", required=True)
    p_encode.set_defaults(func=cmd_encode)

    p_decode = sub.add_parser("decode", help="Decode a token file to audio")
    p_decode.add_argument(
        "--checkpoint", default=None, help="Local checkpoint (default: Hub model)"
    )
    p_decode.add_argument("--input", required=True)
    p_decode.add_argument("--output", required=True)
    p_decode.set_defaults(func=cmd_decode)

    p_recon = sub.add_parser("recon", help="Roundtrip an audio file through the codec")
    p_recon.add_argument("--checkpoint", default=None, help="Local checkpoint (default: Hub model)")
    p_recon.add_argument("--input", required=True)
    p_recon.add_argument("--output", required=True)
    p_recon.set_defaults(func=cmd_recon)

    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
