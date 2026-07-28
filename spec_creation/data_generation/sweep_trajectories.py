#!/usr/bin/env python3
"""Sweep all available phone trajectories and save representative samples.

The sweep keeps each phone, operating-system setting, evaluation, trip, and
section separate. Every saved sample includes longitude, latitude, and the
source location timestamp (``ts``).

Run from ``mobilitynet-analysis-scripts``::

    python spec_creation/data_generation/sweep_trajectories.py
    python spec_creation/data_generation/sweep_trajectories.py --samples-per-trajectory 10
"""

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from spec_creation.data_generation.trajectory_tile_sampler import (
    DEFAULT_AUTHOR_EMAIL,
    DEFAULT_DATA_ROOT,
    TrajectoryTileSampler,
)


DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "datasets" / "trajectories"
DEFAULT_LOCATION_KEY = "location_df"
DEFAULT_SAMPLES_PER_TRAJECTORY = 5
DEFAULT_SEED = 20260722


def json_value(value):
    """Convert common pandas/numpy scalar values to JSON-compatible values."""
    return value.item() if hasattr(value, "item") else value


def records_from_tiles(tiles):
    """Convert sampler tiles into JSON-compatible timestamped point records."""
    records = []
    for tile in tiles:
        records.append({
            "source_id": tile["source_id"],
            "spec_id": tile["spec_id"],
            "phone_os": tile["phone_os"],
            "phone_label": tile["phone_label"],
            "evaluation_range_index": tile["evaluation_range_index"],
            "evaluation_trip_range_index": tile["evaluation_trip_range_index"],
            "evaluation_section_range_index": tile["evaluation_section_range_index"],
            "tile_id": tile["tile_id"],
            "bounds_utm_m": tile["bounds_utm_m"],
            "bounds_wgs84": tile["bounds_wgs84"],
            "epsg": tile["epsg"],
            "tile_point_count": len(tile["tile_trajectory_points"]),
            "tile_points": [
                {
                    "longitude": json_value(longitude),
                    "latitude": json_value(latitude),
                    "ts": json_value(timestamp),
                }
                for longitude, latitude, timestamp in tile["tile_trajectory_points"]
            ],
        })
    return records


def write_outputs(output_dir, records, args, spec_ids):
    """Write the sampled records and a reproducibility summary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "trajectory_samples.jsonl"
    with records_path.open("w") as output_file:
        for record in records:
            output_file.write(json.dumps(record, sort_keys=True) + "\n")

    summary = {
        "data_root": str(args.data_root),
        "author_email": args.author_email,
        "spec_ids": spec_ids,
        "location_key": args.location_key,
        "samples_per_trajectory": args.samples_per_trajectory,
        "seed": args.seed,
        "trajectory_count": len(records),
        "sample_count": sum(record["tile_point_count"] for record in records),
        "records": records_path.name,
    }
    with (output_dir / "summary.json").open("w") as output_file:
        json.dump(summary, output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                        help="Root containing <author-email>/<spec-id> data")
    parser.add_argument("--author-email", default=DEFAULT_AUTHOR_EMAIL,
                        help="Persisted datastore author directory")
    parser.add_argument("--spec-id", action="append", dest="spec_ids",
                        help="Evaluation spec to include; repeat to select multiple")
    parser.add_argument("--location-key", default=DEFAULT_LOCATION_KEY,
                        help="PhoneView range dataframe to use")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory (default: %(default)s)")
    parser.add_argument("--samples-per-trajectory", type=int,
                        default=DEFAULT_SAMPLES_PER_TRAJECTORY,
                        help="Number of timestamped points to save per trajectory")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Reproducible sampling seed")
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.samples_per_trajectory <= 0:
        raise ValueError("--samples-per-trajectory must be positive")

    sampler = TrajectoryTileSampler(
        tiles_per_trajectory=args.samples_per_trajectory,
        tile_side_meters=640,
        seed=args.seed,
        data_root=args.data_root,
        author_email=args.author_email,
    )
    spec_ids = args.spec_ids or sampler.discover_spec_ids(args.data_root, args.author_email)
    if not spec_ids:
        raise ValueError("No evaluation specs discovered; verify --data-root and --author-email")

    trajectories = sampler.load_phone_trajectories(spec_ids, args.location_key)
    tiles = sampler.find_tiles(trajectories)
    records = records_from_tiles(tiles)
    write_outputs(args.output_dir, records, args, spec_ids)
    print("Saved %d samples from %d trajectories to %s" % (
        sum(record["tile_point_count"] for record in records),
        len(records),
        args.output_dir,
    ))


if __name__ == "__main__":
    main()
