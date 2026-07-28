#!/usr/bin/env python3
"""Verify cached zoom-15 raster coverage for every sampled trajectory tile.

Each trajectory tile is displayed by ``raster_tile_viewer.py`` in a fixed 3x3
slippy-tile viewport. This script checks that the nine zoom-15 tiles centered
on each trajectory tile's WGS84 bounds are present in the local raster cache.
It never requests map data.

Run from ``mobilitynet-analysis-scripts``::

    python spec_creation/data_generation/verify_raster_tile_coverage.py
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from spec_creation.data_generation.raster_tile_viewer import (
    DEFAULT_CACHE_DIR,
    DEFAULT_TRAJECTORY_DIR,
    GRID_DIMENSION,
    VIEWER_ZOOM,
)
from spec_creation.data_generation.standard_map_tile_source import StandardMapTileSource


def tile_coordinate(longitude, latitude, zoom):
    """Return the slippy-map tile containing a WGS84 coordinate."""
    latitude = max(min(latitude, 85.05112878), -85.05112878)
    tile_count = 2 ** zoom
    x = math.floor((longitude + 180.0) / 360.0 * tile_count)
    latitude_radians = math.radians(latitude)
    y = math.floor((1.0 - math.asinh(math.tan(latitude_radians)) / math.pi) / 2.0 * tile_count)
    return x % tile_count, min(max(y, 0), tile_count - 1)


def required_tiles(record, zoom=VIEWER_ZOOM, grid_dimension=GRID_DIMENSION):
    """Yield the fixed viewer grid centered on a trajectory tile's bounds."""
    bounds = record["bounds_wgs84"]
    longitude = (bounds["west"] + bounds["east"]) / 2
    latitude = (bounds["south"] + bounds["north"]) / 2
    center_x, center_y = tile_coordinate(longitude, latitude, zoom)
    radius = grid_dimension // 2
    tile_count = 2 ** zoom
    for y in range(center_y - radius, center_y + radius + 1):
        if 0 <= y < tile_count:
            for x in range(center_x - radius, center_x + radius + 1):
                yield zoom, x % tile_count, y


def load_records(trajectory_dir):
    """Read trajectory samples emitted by ``sweep_trajectories.py``."""
    records_path = Path(trajectory_dir) / "trajectory_samples.jsonl"
    if not records_path.is_file():
        raise ValueError("Trajectory samples were not found: %s" % records_path)
    with records_path.open() as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-dir", type=Path, default=DEFAULT_TRAJECTORY_DIR,
                        help="Directory containing trajectory_samples.jsonl")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                        help="Raster tile cache directory (default: %(default)s)")
    parser.add_argument("--report", type=Path,
                        help="Optional JSON report path for missing trajectory tiles")
    return parser.parse_args()


def main():
    args = parse_arguments()
    records = load_records(args.trajectory_dir)
    source = StandardMapTileSource(args.cache_dir, url_template="", offline=True)
    missing_by_trajectory = []
    required = set()
    for record in records:
        tile_ids = set(required_tiles(record))
        required.update(tile_ids)
        missing = [
            {"zoom": zoom, "x": x, "y": y}
            for zoom, x, y in sorted(tile_ids)
            if not source.cache_path(zoom, x, y).is_file()
        ]
        if missing:
            missing_by_trajectory.append({"tile_id": record["tile_id"], "missing_tiles": missing})

    cached = len(required) - len({
        (item["zoom"], item["x"], item["y"])
        for trajectory in missing_by_trajectory
        for item in trajectory["missing_tiles"]
    })
    report = {
        "zoom": VIEWER_ZOOM,
        "grid_dimension": GRID_DIMENSION,
        "trajectory_tile_count": len(records),
        "unique_required_raster_tiles": len(required),
        "cached_unique_raster_tiles": cached,
        "missing_trajectory_tile_count": len(missing_by_trajectory),
        "missing_trajectories": missing_by_trajectory,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w") as output_file:
            json.dump(report, output_file, indent=2, sort_keys=True)
            output_file.write("\n")

    print("Checked %d trajectory tiles requiring %d unique zoom-%d raster tiles." %
          (len(records), len(required), VIEWER_ZOOM))
    if missing_by_trajectory:
        print("Coverage incomplete: %d trajectory tiles have missing raster tiles." %
              len(missing_by_trajectory))
        for trajectory in missing_by_trajectory[:20]:
            print("  %s: %d missing" % (trajectory["tile_id"], len(trajectory["missing_tiles"])))
        if len(missing_by_trajectory) > 20:
            print("  ... plus %d more" % (len(missing_by_trajectory) - 20))
        raise SystemExit(1)
    print("Coverage complete: all required viewer tiles are cached in %s." % args.cache_dir)


if __name__ == "__main__":
    main()
