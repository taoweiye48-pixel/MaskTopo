from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


EXPECTED = {"train": 600, "test": 200}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare FIVES for topology-connectivity validation."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("real_data/FIVES/dataset/extracted"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("real_data/FIVES/dataset/preprocessed512"),
    )
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def normalized_name(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def find_pair_directories(root: Path, split: str) -> tuple[Path, Path]:
    candidates: list[tuple[Path, Path]] = []
    for directory in root.rglob("*"):
        if not directory.is_dir() or directory.name.lower() != split:
            continue
        children = {
            normalized_name(child.name): child
            for child in directory.iterdir()
            if child.is_dir()
        }
        original = children.get("original")
        ground_truth = children.get("groundtruth")
        if original is not None and ground_truth is not None:
            candidates.append((original, ground_truth))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one {split}/Original + Ground Truth pair, "
            f"found {len(candidates)} under {root}."
        )
    return candidates[0]


def image_map(directory: Path) -> dict[str, Path]:
    supported = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    return {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in supported
    }


def max_pool_mask(mask: np.ndarray, output_size: int) -> np.ndarray:
    height, width = mask.shape
    if height % output_size == 0 and width % output_size == 0:
        row_factor = height // output_size
        col_factor = width // output_size
        return mask.reshape(
            output_size,
            row_factor,
            output_size,
            col_factor,
        ).max(axis=(1, 3))
    resized = Image.fromarray(mask.astype(np.uint8) * 255).resize(
        (output_size, output_size),
        Image.Resampling.BOX,
    )
    return np.asarray(resized) > 0


def prepare_pair(
    image_path: Path,
    mask_path: Path,
    image_output: Path,
    mask_output: Path,
    size: int,
) -> dict[str, object]:
    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    mask = np.asarray(Image.open(mask_path).convert("L")) > 0
    if rgb.shape[:2] != mask.shape:
        raise ValueError(f"Image/mask mismatch for {image_path.name}.")
    green = rgb[..., 1]
    green_small = np.asarray(
        Image.fromarray(green).resize(
            (size, size), Image.Resampling.LANCZOS
        )
    )
    vessel_small = max_pool_mask(mask, size)
    Image.fromarray(green_small.astype(np.uint8)).save(image_output)
    Image.fromarray(vessel_small.astype(np.uint8) * 255).save(mask_output)
    return {
        "source_shape": list(mask.shape),
        "working_shape": [size, size],
        "source_vessel_fraction": float(mask.mean()),
        "working_vessel_fraction": float(vessel_small.mean()),
    }


def fingerprint(output_root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(output_root.rglob("*.png")):
        digest.update(str(path.relative_to(output_root)).encode())
        digest.update(str(path.stat().st_size).encode())
    return digest.hexdigest()[:16]


def run_self_test() -> None:
    mask = np.zeros((8, 8), dtype=bool)
    mask[1, 1] = True
    mask[6:8, 6:8] = True
    pooled = max_pool_mask(mask, 2)
    assert pooled.shape == (2, 2)
    assert pooled[0, 0] and pooled[1, 1]
    print("PREPARE_FIVES_SELF_TEST_PASS", flush=True)


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, object]] = {}
    for split, expected_count in EXPECTED.items():
        originals, ground_truths = find_pair_directories(source_root, split)
        images = image_map(originals)
        masks = image_map(ground_truths)
        if set(images) != set(masks) or len(images) != expected_count:
            raise ValueError(
                f"{split} pairs incomplete: images={len(images)}, "
                f"masks={len(masks)}, overlap={len(set(images) & set(masks))}."
            )
        image_output_dir = output_root / f"{split}_img"
        mask_output_dir = output_root / f"{split}_lab"
        image_output_dir.mkdir(parents=True, exist_ok=True)
        mask_output_dir.mkdir(parents=True, exist_ok=True)
        for index, stem in enumerate(sorted(images), start=1):
            image_output = image_output_dir / f"{stem}.png"
            mask_output = mask_output_dir / f"{stem}.png"
            records[f"{split}:{stem}"] = prepare_pair(
                images[stem],
                masks[stem],
                image_output,
                mask_output,
                args.size,
            )
            if index % 50 == 0 or index == expected_count:
                print(
                    f"FIVES_PREPARE split={split} "
                    f"progress={index}/{expected_count}",
                    flush=True,
                )
    manifest = {
        "dataset": "FIVES",
        "source_record": "10.6084/m9.figshare.19688169.v1",
        "license": "CC BY 4.0",
        "preprocessing": {
            "image": (
                "green channel, Lanczos to 512x512; the downstream shared "
                "sample generator performs the single inversion"
            ),
            "mask": "binary annotation, 4x4 max pooling to 512x512",
        },
        "counts": EXPECTED,
        "fingerprint": fingerprint(output_root),
        "records": records,
    }
    (output_root / "preparation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"FIVES_PREPARATION_COMPLETE fingerprint={manifest['fingerprint']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
