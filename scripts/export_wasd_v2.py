from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from srflawscry import WASD


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the clean-selected WASD v2 checkpoint for Hugging Face."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    detector = WASD(
        embedding_dim=128,
        pretrained=False,
        use_norm=True,
        margin=1.0,
        prominence_head_hidden=256,
        encoder="mobnet",
        refiner="nafnet",
        crop_size=100,
        nafnet_variant="small",
        use_dists_map=True,
        dists_patch_size=64,
        dists_stride=32,
        dists_batch_size=32,
        inference_tile_size=512,
        inference_tile_overlap=64,
        artifact_threshold=0.21,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    detector.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    detector.eval()
    detector.save_pretrained(args.output_dir)

    restored = WASD.from_pretrained(args.output_dir).eval()
    expected = detector.state_dict()
    actual = restored.state_dict()
    if expected.keys() != actual.keys():
        raise RuntimeError("Round-trip changed state-dict keys")
    if any(not torch.equal(expected[key], actual[key]) for key in expected):
        raise RuntimeError("Round-trip changed model tensors")

    manifest = {
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_sha256": sha256(args.checkpoint),
        "exported_model_sha256": sha256(args.output_dir / "model.safetensors"),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "strict_state_dict_load": True,
        "round_trip_equal": True,
        "operating_threshold": 0.21,
    }
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
