#!/usr/bin/env python3
"""Interactively cache OpenStreetMap-standard tiles around trajectory samples.

The local viewer displays a fixed 3x3 grid of 256-pixel tiles at zoom level 15
and overlays the sampled GPS trajectories from ``sweep_trajectories.py``.
Raster tiles are requested only as the operator pans or selects a trajectory;
responses are stored through ``StandardMapTileSource`` for later dataset use.

Run from ``mobilitynet-analysis-scripts``::

    python spec_creation/data_generation/raster_tile_viewer.py

Then open http://127.0.0.1:5056. Use the trajectory list or pan the map to
cache the map context you want to inspect. This tool does not prefetch tiles.
"""

import argparse
import io
import json
import math
import sys
from pathlib import Path

from flask import Flask, abort, jsonify, send_file


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from spec_creation.data_generation.standard_map_tile_source import (
    DEFAULT_STANDARD_TILE_URL,
    STANDARD_RENDER_ATTRIBUTION,
    StandardMapTileSource,
)


VIEWER_ZOOM = 15
GRID_DIMENSION = 3
DEFAULT_TRAJECTORY_DIR = REPOSITORY_ROOT / "datasets" / "trajectories"
DEFAULT_CACHE_DIR = REPOSITORY_ROOT / "datasets" / "raster_tiles"


class RasterTileViewer:
    """Own the local trajectory dataset and manually populated raster cache."""

    def __init__(self, trajectory_dir, cache_dir, tile_url, offline, retries):
        self.trajectory_dir = Path(trajectory_dir).resolve()
        self.records_path = self.trajectory_dir / "trajectory_samples.jsonl"
        self.tile_source = StandardMapTileSource(cache_dir, tile_url, offline, retries)
        self.records = self.load_records()

    def load_records(self):
        """Load the sweep output without requesting any map data."""
        if not self.records_path.is_file():
            raise ValueError("Trajectory samples were not found: %s" % self.records_path)
        records = []
        with self.records_path.open() as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except ValueError as error:
                    raise ValueError("Invalid JSON at %s:%d" % (self.records_path, line_number)) from error
                if not record.get("tile_points"):
                    continue
                records.append(record)
        if not records:
            raise ValueError("No trajectory samples with tile points were found: %s" % self.records_path)
        return records

    def serialized_records(self):
        """Return local GPS overlays and metadata for the browser map."""
        serialized = []
        for index, record in enumerate(self.records):
            required_tiles = self.required_tiles(record)
            missing_tile_count = sum(
                not self.tile_source.cache_path(zoom, x, y).is_file()
                for zoom, x, y in required_tiles
            )
            serialized.append({
                "index": index,
                "tile_id": record["tile_id"],
                "source_id": record["source_id"],
                "spec_id": record["spec_id"],
                "phone_os": record["phone_os"],
                "phone_label": record["phone_label"],
                "tile_point_count": record["tile_point_count"],
                "bounds_wgs84": record["bounds_wgs84"],
                "tile_points": record["tile_points"],
                "missing_tile_count": missing_tile_count,
                "required_tile_count": len(required_tiles),
            })
        return serialized

    @staticmethod
    def slippy_tile_coordinate(longitude, latitude, zoom):
        """Return the slippy-map tile containing a WGS84 coordinate."""
        latitude = max(min(latitude, 85.05112878), -85.05112878)
        tile_count = 2 ** zoom
        x = math.floor((longitude + 180.0) / 360.0 * tile_count)
        latitude_radians = math.radians(latitude)
        y = math.floor((1.0 - math.asinh(math.tan(latitude_radians)) / math.pi) /
                       2.0 * tile_count)
        return x % tile_count, min(max(y, 0), tile_count - 1)

    def required_tiles(self, record):
        """Return the fixed 3x3 viewer grid centered on a trajectory tile."""
        bounds = record["bounds_wgs84"]
        longitude = (bounds["west"] + bounds["east"]) / 2
        latitude = (bounds["south"] + bounds["north"]) / 2
        center_x, center_y = self.slippy_tile_coordinate(longitude, latitude, VIEWER_ZOOM)
        radius = GRID_DIMENSION // 2
        tile_count = 2 ** VIEWER_ZOOM
        return [
            (VIEWER_ZOOM, x % tile_count, y)
            for y in range(center_y - radius, center_y + radius + 1)
            if 0 <= y < tile_count
            for x in range(center_x - radius, center_x + radius + 1)
        ]


def create_app(viewer):
    """Create the local-only raster tile viewer application."""
    app = Flask(__name__)

    @app.get("/")
    def index():
        return send_file(Path(__file__).with_name("raster_tile_viewer.html"))

    @app.get("/api/trajectories")
    def trajectories():
        return jsonify({
            "zoom": VIEWER_ZOOM,
            "grid_dimension": GRID_DIMENSION,
            "attribution": STANDARD_RENDER_ATTRIBUTION,
            "trajectories": viewer.serialized_records(),
        })

    @app.get("/tiles/<int:zoom>/<int:x>/<int:y>.png")
    def tile(zoom, x, y):
        if zoom != VIEWER_ZOOM:
            abort(404, "Only zoom level %d is available." % VIEWER_ZOOM)
        try:
            image = viewer.tile_source.tile(zoom, x, y)
        except FileNotFoundError as error:
            abort(404, str(error))
        except RuntimeError as error:
            abort(502, str(error))
        payload = io.BytesIO()
        image.save(payload, format="PNG")
        payload.seek(0)
        response = send_file(payload, mimetype="image/png", max_age=31536000)
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response

    return app


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-dir", type=Path, default=DEFAULT_TRAJECTORY_DIR,
                        help="Directory containing trajectory_samples.jsonl")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                        help="Raster tile cache directory (default: %(default)s)")
    parser.add_argument("--standard-tile-url", default=DEFAULT_STANDARD_TILE_URL,
                        help="Slippy-tile URL with {z}, {x}, and {y} placeholders")
    parser.add_argument("--offline", action="store_true",
                        help="Show only tiles already present in the raster cache")
    parser.add_argument("--retries", type=int, default=3,
                        help="Retries for a manually requested uncached tile")
    parser.add_argument("--host", default="127.0.0.1", help="Local server host")
    parser.add_argument("--port", type=int, default=5056, help="Local server port")
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.retries < 0 or not 1 <= args.port <= 65535:
        raise ValueError("--retries must be non-negative and --port must be valid")
    viewer = RasterTileViewer(
        args.trajectory_dir, args.cache_dir, args.standard_tile_url, args.offline, args.retries)
    print("Loaded %d trajectory tile samples from %s" % (len(viewer.records), viewer.records_path))
    print("Raster cache: %s" % viewer.tile_source.cache_dir)
    create_app(viewer).run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
