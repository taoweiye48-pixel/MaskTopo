from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import math
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable
from xml.etree import ElementTree

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation


EXPERIMENT_ID = "TB-B-260803-033"
AMBIGUOUS_IMAGES = (
    "val/RN2,2,41.tif",
    "val/RN2,2,42.tif",
    "val/RN2,3,32.tif",
    "val/RN2,4,39.tif",
    "val/RN2,7,29.tif",
)
ORPHAN_RSML = (
    "RSML/RN2,0,40 .rsml",
    "RSML/RN2,0,41.rsml",
    "RSML/RN2,0,42.rsml",
    "RSML/RN2,3,33.rsml",
    "RSML/RN2,4,29.rsml",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/dev-only RootNav RSML pairing audit.")
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("data/RootNav2A/archives/RootNav2_Arabidopsis.zip"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("实验记录/TB-B-260803-033_pairing_audit.json"),
    )
    parser.add_argument("--calibration-pairs", type=int, default=40)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_gray(handle: zipfile.ZipFile, member: str) -> np.ndarray:
    if member.startswith("test/") or ",testset." in member:
        raise ValueError(f"Sealed test member requested: {member}")
    raw = handle.read(member)
    with Image.open(io.BytesIO(raw)) as image:
        gray = np.asarray(image.convert("L"), dtype=np.float32)
    if gray.ndim != 2:
        raise ValueError(f"Unexpected image rank for {member}: {gray.shape}")
    return gray


def load_polylines(handle: zipfile.ZipFile, member: str) -> tuple[list[np.ndarray], str]:
    if ",testset." in member:
        raise ValueError(f"Sealed test RSML requested: {member}")
    raw = handle.read(member)
    tree = ElementTree.fromstring(raw)
    polylines: list[np.ndarray] = []
    for polyline in tree.findall(".//geometry/polyline"):
        points: list[tuple[float, float]] = []
        for point in polyline.findall("point"):
            points.append((float(point.attrib["x"]), float(point.attrib["y"])))
        if len(points) >= 2:
            polylines.append(np.asarray(points, dtype=np.float32))
    if not polylines:
        raise ValueError(f"No RSML polylines in {member}")
    return polylines, sha256_bytes(raw)


def coordinate_transforms(height: int, width: int) -> dict[str, Callable[[float, float], tuple[float, float]]]:
    return {
        "yx": lambda x, y: (y, x),
        "yx_flip_row": lambda x, y: (height - 1 - y, x),
        "yx_flip_col": lambda x, y: (y, width - 1 - x),
        "yx_flip_both": lambda x, y: (height - 1 - y, width - 1 - x),
        "xy": lambda x, y: (x, y),
        "xy_flip_row": lambda x, y: (height - 1 - x, y),
        "xy_flip_col": lambda x, y: (x, width - 1 - y),
        "xy_flip_both": lambda x, y: (height - 1 - x, width - 1 - y),
    }


def render_mask(
    shape: tuple[int, int],
    polylines: list[np.ndarray],
    transform_name: str,
    width: int,
) -> np.ndarray:
    height, image_width = shape
    transforms = coordinate_transforms(height, image_width)
    transform = transforms[transform_name]
    canvas = Image.new("1", (image_width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for polyline in polylines:
        coordinates: list[tuple[float, float]] = []
        for x, y in polyline:
            row, col = transform(float(x), float(y))
            coordinates.append((col, row))
        draw.line(coordinates, fill=1, width=width, joint="curve")
    return np.asarray(canvas, dtype=bool)


def contrast_score(image: np.ndarray, centerline: np.ndarray) -> float:
    inner = binary_dilation(centerline, iterations=2)
    ring = binary_dilation(centerline, iterations=8) & ~binary_dilation(centerline, iterations=4)
    if centerline.sum() < 20 or ring.sum() < 100:
        return float("nan")
    foreground = image[inner].mean()
    local_background = image[ring].mean()
    return float((foreground - local_background) / (image.std() + 1e-6))


def exact_train_pairs(names: list[str]) -> list[tuple[str, str]]:
    rsml_names = set(names)
    pairs: list[tuple[str, str]] = []
    for image_member in sorted(
        name for name in names if name.startswith("train/") and name.lower().endswith(".tif")
    ):
        source_id = PurePosixPath(image_member).stem
        rsml_member = f"RSML/{source_id}.rsml"
        if rsml_member in rsml_names:
            pairs.append((image_member, rsml_member))
    return pairs


def pairing_audit(archive: Path, calibration_pairs: int) -> dict[str, Any]:
    with zipfile.ZipFile(archive) as handle:
        names = [info.filename for info in handle.infolist() if not info.is_dir()]
        for member in (*AMBIGUOUS_IMAGES, *ORPHAN_RSML):
            if member not in names:
                raise ValueError(f"Expected audit member missing: {member}")
        exact = exact_train_pairs(names)
        if len(exact) < calibration_pairs:
            raise ValueError(f"Only {len(exact)} exact train pairs available.")
        chosen = exact[:calibration_pairs]
        calibration: list[tuple[np.ndarray, list[np.ndarray]]] = []
        for image_member, rsml_member in chosen:
            image = load_gray(handle, image_member)
            polylines, _ = load_polylines(handle, rsml_member)
            calibration.append((image, polylines))

        transform_scores: dict[str, list[float]] = {}
        for transform_name in coordinate_transforms(*calibration[0][0].shape):
            scores = [
                contrast_score(image, render_mask(image.shape, polylines, transform_name, 1))
                for image, polylines in calibration
            ]
            transform_scores[transform_name] = scores
        transform_median_abs = {
            name: abs(float(np.nanmedian(scores))) for name, scores in transform_scores.items()
        }
        selected_transform = max(transform_median_abs, key=transform_median_abs.get)
        raw_polarity = float(np.nanmedian(transform_scores[selected_transform]))
        polarity = 1.0 if raw_polarity >= 0 else -1.0

        width_scores: dict[str, list[float]] = {}
        for width in (1, 3, 5):
            scores = [
                polarity
                * contrast_score(image, render_mask(image.shape, polylines, selected_transform, width))
                for image, polylines in calibration
            ]
            width_scores[str(width)] = scores
        selected_width = max(
            (1, 3, 5), key=lambda width: float(np.nanmedian(width_scores[str(width)]))
        )
        reference_scores = np.asarray(width_scores[str(selected_width)], dtype=np.float64)

        images = {member: load_gray(handle, member) for member in AMBIGUOUS_IMAGES}
        rsml_payload: dict[str, tuple[list[np.ndarray], str]] = {
            member: load_polylines(handle, member) for member in ORPHAN_RSML
        }
        matrix = np.empty((len(AMBIGUOUS_IMAGES), len(ORPHAN_RSML)), dtype=np.float64)
        for row, image_member in enumerate(AMBIGUOUS_IMAGES):
            for column, rsml_member in enumerate(ORPHAN_RSML):
                polylines, _ = rsml_payload[rsml_member]
                mask = render_mask(
                    images[image_member].shape, polylines, selected_transform, selected_width
                )
                matrix[row, column] = polarity * contrast_score(images[image_member], mask)

        assignments: list[dict[str, Any]] = []
        for permutation in itertools.permutations(range(len(ORPHAN_RSML))):
            per_pair = [float(matrix[row, column]) for row, column in enumerate(permutation)]
            assignments.append(
                {
                    "total_score": float(np.sum(per_pair)),
                    "minimum_pair_score": float(np.min(per_pair)),
                    "permutation": list(permutation),
                    "pairs": [
                        {
                            "image_member": AMBIGUOUS_IMAGES[row],
                            "rsml_member": ORPHAN_RSML[column],
                            "score": per_pair[row],
                        }
                        for row, column in enumerate(permutation)
                    ],
                }
            )
        assignments.sort(key=lambda item: (item["total_score"], item["minimum_pair_score"]), reverse=True)
        best = assignments[0]
        second = assignments[1]
        row_margins = []
        for row, chosen_column in enumerate(best["permutation"]):
            alternatives = [matrix[row, column] for column in range(matrix.shape[1]) if column != chosen_column]
            row_margins.append(float(matrix[row, chosen_column] - max(alternatives)))

    normal_low = float(np.nanquantile(reference_scores, 0.05))
    best_scores = np.asarray([pair["score"] for pair in best["pairs"]], dtype=np.float64)
    unique_assignment = bool(
        best["total_score"] > second["total_score"] + 0.05
        and min(row_margins) > 0.02
    )
    quality_consistent = bool(np.all(best_scores >= normal_low - 0.10))
    return {
        "experiment_id": EXPERIMENT_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "access_scope": "train/val TIFF and non-testset RSML only",
        "test_tiff_opened": 0,
        "test_rsml_opened": 0,
        "calibration": {
            "pair_count": len(chosen),
            "pairs": [list(pair) for pair in chosen],
            "transform_median_absolute_contrast": transform_median_abs,
            "selected_transform": selected_transform,
            "raw_selected_transform_median_contrast": raw_polarity,
            "selected_polarity": "root_lighter" if polarity > 0 else "root_darker",
            "width_median_signed_contrast": {
                width: float(np.nanmedian(scores)) for width, scores in width_scores.items()
            },
            "selected_width": selected_width,
            "reference_signed_contrast_median": float(np.nanmedian(reference_scores)),
            "reference_signed_contrast_q05": normal_low,
        },
        "ambiguous_images": list(AMBIGUOUS_IMAGES),
        "orphan_rsml": list(ORPHAN_RSML),
        "orphan_rsml_content_sha256": {
            member: payload[1] for member, payload in rsml_payload.items()
        },
        "score_matrix_rows_images_columns_rsml": matrix.tolist(),
        "top_assignments": assignments[:10],
        "best_minus_second_total_score": float(best["total_score"] - second["total_score"]),
        "best_row_margins": row_margins,
        "unique_assignment_gate": unique_assignment,
        "quality_consistent_with_exact_train_gate": quality_consistent,
        "pairing_gate_pass": bool(unique_assignment and quality_consistent),
    }


def self_test() -> None:
    image = np.zeros((32, 32), dtype=np.float32)
    image[4:29, 16] = 255
    polylines = [np.asarray([[16, 4], [16, 28]], dtype=np.float32)]
    mask = render_mask(image.shape, polylines, "yx", 1)
    assert mask[4, 16] and mask[28, 16]
    assert contrast_score(image, mask) > 0
    print(json.dumps({"stage": "self-test", "status": "PASS"}))


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    result = pairing_audit(args.archive.resolve(), args.calibration_pairs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS" if result["pairing_gate_pass"] else "FAIL",
        "selected_transform": result["calibration"]["selected_transform"],
        "selected_polarity": result["calibration"]["selected_polarity"],
        "selected_width": result["calibration"]["selected_width"],
        "best_assignment": result["top_assignments"][0],
        "best_minus_second_total_score": result["best_minus_second_total_score"],
        "best_row_margins": result["best_row_margins"],
        "quality_gate": result["quality_consistent_with_exact_train_gate"],
        "pairing_gate": result["pairing_gate_pass"],
        "test_tiff_opened": 0,
        "test_rsml_opened": 0,
        "output": str(args.output.resolve()),
    }, indent=2))
    if not result["pairing_gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
