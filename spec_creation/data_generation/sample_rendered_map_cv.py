#!/usr/bin/env python3
"""Compare OpenCV edge detectors on rendered map samples.

The script randomly selects rendered trajectory-map PNGs and writes one labelled
four-panel diagnostic image per sample plus a contact sheet. It reads only local
renders and does not make any network requests.

Run from ``mobilitynet-analysis-scripts``::

    python spec_creation/data_generation/sample_rendered_map_cv.py
"""

import argparse
from email.mime import image
import json
import random
from pathlib import Path

import cv2
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = REPOSITORY_ROOT / "datasets" / "standard_map_renders"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "datasets" / "standard_map_cv_diagnostics"
DEFAULT_SAMPLE_COUNT = 10
DEFAULT_SEED = 20260722


def labelled(image, label):
    """Add a stable title strip above a diagnostic image."""
    title_height = 24
    panel = np.full(
        (image.shape[0] + title_height, image.shape[1], 3), 255, dtype=np.uint8)
    panel[title_height:] = image
    cv2.putText(panel, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (20, 20, 20), 1, cv2.LINE_AA)
    return panel


def detect_edges(image, soft_threshold=0.2, soft_temperature=15,
                 color_threshold=3):
    """Return Canny, color-filter, and soft Sobel edge images."""
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscale, (3, 3), 0)
    canny_edges = cv2.Canny(blurred, 50, 150)

    red = image[:, :, 2].astype(np.uint16)
    blue = image[:, :, 0].astype(np.uint16)
    green = image[:, :, 1]
    # 1. Create a 2D boolean mask (True/False)
    mask = (red + blue >= 255 - color_threshold) & (green <= color_threshold)

    # 2. Apply the mask to the 3D image by adding a channel dimension to the mask
    color_filter = np.where(mask[..., np.newaxis], image, 0).astype(np.uint8)

    normalized = grayscale.astype(np.float32) / 255.0
    sobel_x = cv2.Sobel(normalized, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(normalized, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    magnitude /= np.max(magnitude) + 1e-8
    soft_edges = 1.0 / (1.0 + np.exp(
        -soft_temperature * (magnitude - soft_threshold)))

    return canny_edges, color_filter, np.uint8(soft_edges * 255)


def diagnostic_panel(image):
    """Build a 2x2 original and edge-method comparison panel."""
    canny_edges, color_filter, soft_edges = detect_edges(image)
    canny_panel = cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2BGR)
    color_filter_panel = color_filter #cv2.cvtColor(color_filter, cv2.COLOR_GRAY2BGR)
    soft_panel = cv2.cvtColor(soft_edges, cv2.COLOR_GRAY2BGR)
    top_row = np.hstack((labelled(image, "Original"), labelled(canny_panel, "Canny edges")))
    bottom_row = np.hstack((
        labelled(color_filter_panel, "R+B color filter"),
        labelled(soft_panel, "Soft Canny edges"),
    ))
    return np.vstack((top_row, bottom_row))


def contact_sheet(panels, columns=2):
    """Arrange equal-size sample panels into a compact review sheet."""
    rows = (len(panels) + columns - 1) // columns
    height, width = panels[0].shape[:2]
    sheet = np.full((rows * height, columns * width, 3), 255, dtype=np.uint8)
    for index, panel in enumerate(panels):
        row, column = divmod(index, columns)
        top = row * height
        left = column * width
        sheet[top:top + height, left:left + width] = panel
    return sheet


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
                        help="Directory containing rendered trajectory PNGs")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory for diagnostic PNGs and sample metadata")
    parser.add_argument("--count", type=int, default=DEFAULT_SAMPLE_COUNT,
                        help="Number of random renders to inspect (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random selection seed (default: %(default)s)")
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.count <= 0:
        raise ValueError("--count must be positive")
    image_paths = sorted(args.input_dir.glob("*.png"))
    if len(image_paths) < args.count:
        raise ValueError("Requested %d samples but found only %d PNGs in %s" %
                         (args.count, len(image_paths), args.input_dir))

    selected_paths = random.Random(args.seed).sample(image_paths, args.count)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panels = []
    samples = []
    for index, image_path in enumerate(selected_paths, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not read image: %s" % image_path)
        panel = diagnostic_panel(image)
        cv2.putText(panel, "Sample %02d" % index, (6, panel.shape[0] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
        output_path = args.output_dir / ("sample_%02d.png" % index)
        if not cv2.imwrite(str(output_path), panel):
            raise OSError("Could not write diagnostic: %s" % output_path)
        panels.append(panel)
        samples.append({
            "sample": index,
            "source_image": image_path.name,
            "diagnostic_image": output_path.name,
        })

    sheet_path = args.output_dir / "contact_sheet.png"
    if not cv2.imwrite(str(sheet_path), contact_sheet(panels)):
        raise OSError("Could not write contact sheet: %s" % sheet_path)
    metadata_path = args.output_dir / "samples.json"
    with metadata_path.open("w") as output_file:
        json.dump({"seed": args.seed, "sample_count": args.count, "samples": samples},
                  output_file, indent=2)
        output_file.write("\n")

    print("Wrote %d diagnostics and contact sheet: %s" % (args.count, sheet_path))


if __name__ == "__main__":
    main()