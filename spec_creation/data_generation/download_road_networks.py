#!/usr/bin/env python3
"""Download highway and rail network graphs for every swept trajectory tile.

The script calls ``osmnx.graph.graph_from_bbox`` with the same highway and
railway conditions used by ``overpass_tile_source.py``. It saves each graph as a
GraphML file under ``datasets/road_networks/graphs`` and records outcomes in a
JSONL manifest, allowing interrupted batches to resume.

Run from ``mobilitynet-analysis-scripts``::

    python spec_creation/data_generation/download_road_networks.py
"""

import argparse
import json
import time
from pathlib import Path

import osmnx as ox
from pyproj import CRS, Transformer
import requests


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAJECTORIES = REPOSITORY_ROOT / "datasets" / "trajectories" / "trajectory_samples.jsonl"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "datasets" / "road_networks"
DEFAULT_OVERPASS_URL = "https://overpass-api.de/api"
MINIMUM_PARENT_SIDE_METERS = 15_000
MAXIMUM_PARENT_SIDE_METERS = 20_000
CUSTOM_FILTER = [
    "[highway]",
    '[railway~"^(subway|rail|tram|light_rail)$"][service!~"^(yard|siding)$"]',
]


def trajectory_records(path):
    """Yield trajectory-tile records from the sweep JSONL file."""
    with path.open() as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError("Invalid JSON on line %d of %s" % (line_number, path)) from error


def bbox_for_record(record):
    """Return an OSMnx v2 bbox in (left, bottom, right, top) order."""
    bounds = record["bounds_wgs84"]
    return (bounds["west"], bounds["south"], bounds["east"], bounds["north"])


def graph_path(output_dir, tile_id):
    """Return the stable GraphML location for a trajectory tile."""
    return output_dir / (tile_id + ".graphml")


def append_manifest(path, entry):
    """Append one durable download result to the batch manifest."""
    with path.open("a") as output_file:
        output_file.write(json.dumps(entry, sort_keys=True) + "\n")


def graph_for_bbox(bbox):
    """Download one simplified, largest-component road and rail network graph."""
    return ox.graph.graph_from_bbox(
        bbox=bbox,
        network_type="all",
        simplify=True,
        retain_all=False,
        truncate_by_edge=False,
        custom_filter=CUSTOM_FILTER,
    )


def utm_bounds_for_record(record):
    """Return an unprojected tile's stored UTM bounds as west, south, east, north."""
    west, south, east, north = record["bounds_utm_m"]
    return west, south, east, north


def union_bounds(first_bounds, second_bounds):
    """Return the minimal rectangle containing two UTM rectangles."""
    return (
        min(first_bounds[0], second_bounds[0]),
        min(first_bounds[1], second_bounds[1]),
        max(first_bounds[2], second_bounds[2]),
        max(first_bounds[3], second_bounds[3]),
    )


def dimensions_meters(bounds):
    """Return a UTM rectangle's width and height in metres."""
    return bounds[2] - bounds[0], bounds[3] - bounds[1]


def fits_maximum_parent(bounds):
    """Return whether a UTM rectangle fits within the 20 km parent limit."""
    width, height = dimensions_meters(bounds)
    return width <= MAXIMUM_PARENT_SIDE_METERS and height <= MAXIMUM_PARENT_SIDE_METERS


def padded_parent_bounds(bounds):
    """Pad a group rectangle to at least 15 km per side without exceeding 20 km."""
    west, south, east, north = bounds
    width, height = dimensions_meters(bounds)
    if width < MINIMUM_PARENT_SIDE_METERS:
        padding = (MINIMUM_PARENT_SIDE_METERS - width) / 2
        west -= padding
        east += padding
    if height < MINIMUM_PARENT_SIDE_METERS:
        padding = (MINIMUM_PARENT_SIDE_METERS - height) / 2
        south -= padding
        north += padding
    parent_bounds = (west, south, east, north)
    if not fits_maximum_parent(parent_bounds):
        raise ValueError("Could not pad parent bounds within the 20 km limit: %s" % (parent_bounds,))
    return parent_bounds


def candidate_score(group_bounds, candidate_bounds):
    """Prioritize candidates that satisfy 15 km coverage with the smallest expansion."""
    expanded = union_bounds(group_bounds, candidate_bounds)
    width, height = dimensions_meters(expanded)
    minimum_deficit = max(0, MINIMUM_PARENT_SIDE_METERS - width) + max(
        0, MINIMUM_PARENT_SIDE_METERS - height)
    area = width * height
    return minimum_deficit, area, expanded


def greedy_parent_groups(records):
    """Greedily group UTM tile rectangles into 15-20 km parent query areas."""
    records_by_epsg = {}
    for record in records:
        records_by_epsg.setdefault(record["epsg"], []).append(record)

    groups = []
    for epsg, projected_records in sorted(records_by_epsg.items()):
        remaining = sorted(projected_records, key=lambda record: (
            utm_bounds_for_record(record)[0], utm_bounds_for_record(record)[1], record["tile_id"]))
        while remaining:
            group = [remaining.pop(0)]
            group_bounds = utm_bounds_for_record(group[0])
            while True:
                eligible = []
                for candidate in remaining:
                    expanded = union_bounds(group_bounds, utm_bounds_for_record(candidate))
                    if fits_maximum_parent(expanded):
                        eligible.append((candidate_score(group_bounds, utm_bounds_for_record(candidate)), candidate))
                if not eligible:
                    break
                eligible.sort(key=lambda item: (item[0][0], item[0][1], item[1]["tile_id"]))
                _, selected = eligible[0]
                group.append(selected)
                remaining.remove(selected)
                group_bounds = union_bounds(group_bounds, utm_bounds_for_record(selected))
                width, height = dimensions_meters(group_bounds)
                if width >= MINIMUM_PARENT_SIDE_METERS and height >= MINIMUM_PARENT_SIDE_METERS:
                    break
            groups.append({
                "epsg": epsg,
                "records": group,
                "bounds_utm_m": padded_parent_bounds(group_bounds),
            })
    return groups


def wgs84_bbox_from_utm(bounds, epsg):
    """Project a UTM parent rectangle to an OSMnx bbox in left/bottom/right/top order."""
    transformer = Transformer.from_crs(CRS.from_epsg(epsg), CRS.from_epsg(4326), always_xy=True)
    west, south, east, north = bounds
    corners = [transformer.transform(x, y) for x, y in (
        (west, south), (west, north), (east, south), (east, north))]
    longitudes, latitudes = zip(*corners)
    return min(longitudes), min(latitudes), max(longitudes), max(latitudes)


def parent_graph_path(output_dir, parent_index, epsg):
    """Return the stable GraphML location for a grouped parent query area."""
    return output_dir / "parent_graphs" / ("parent_%04d_epsg%d.graphml" % (parent_index, epsg))


def is_connection_failure(error):
    """Return whether an error means the Overpass endpoint cannot be reached."""
    return isinstance(error, requests.exceptions.ConnectionError)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, default=DEFAULT_TRAJECTORIES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--delay-seconds", type=float, default=1.0,
                        help="Pause after each Overpass request (default: %(default)s)")
    parser.add_argument("--overpass-url", default=DEFAULT_OVERPASS_URL,
                        help="OSMnx Overpass API base URL (default: %(default)s)")
    parser.add_argument("--request-timeout", type=int, default=180,
                        help="OSMnx request timeout in seconds (default: %(default)s)")
    parser.add_argument("--max-consecutive-connection-failures", type=int, default=2,
                        help="Stop during an endpoint outage after this many failures")
    parser.add_argument("--max-parent-areas", type=int,
                        help="Limit parent graph queries for a controlled partial run")
    parser.add_argument("--overwrite", action="store_true",
                        help="Redownload graphs whose GraphML files already exist")
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.delay_seconds < 0:
        raise ValueError("--delay-seconds must not be negative")
    if args.max_parent_areas is not None and args.max_parent_areas <= 0:
        raise ValueError("--max-parent-areas must be positive")
    if args.request_timeout <= 0:
        raise ValueError("--request-timeout must be positive")
    if args.max_consecutive_connection_failures <= 0:
        raise ValueError("--max-consecutive-connection-failures must be positive")
    if not args.trajectories.is_file():
        raise FileNotFoundError("Trajectory dataset not found: %s" % args.trajectories)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tile_manifest_path = args.output_dir / "download_manifest.jsonl"
    parent_manifest_path = args.output_dir / "parent_download_manifest.jsonl"
    parent_areas = greedy_parent_groups(list(trajectory_records(args.trajectories)))
    downloaded = skipped = failed = parent_downloaded = parent_skipped = 0
    consecutive_connection_failures = 0
    ox.settings.overpass_url = args.overpass_url
    ox.settings.requests_timeout = args.request_timeout
    ox.settings.user_agent = "OpenPathTrajectoryNetworks/1.0"
    # ox.settings.use_cache = True
    print("Using Overpass endpoint: %s" % ox.settings.overpass_url)

    print("Planned %d grouped parent areas for %d trajectory tiles." % (
        len(parent_areas), sum(len(parent_area["records"]) for parent_area in parent_areas)))

    for parent_index, parent_area in enumerate(parent_areas, start=1):
        if args.max_parent_areas is not None and parent_index > args.max_parent_areas:
            break
        pending_records = [
            record for record in parent_area["records"]
            if args.overwrite or not graph_path(args.output_dir, record["tile_id"]).is_file()
        ]
        if not pending_records:
            skipped += len(parent_area["records"])
            parent_skipped += 1
            continue
        parent_bbox = wgs84_bbox_from_utm(parent_area["bounds_utm_m"], parent_area["epsg"])
        parent_path = parent_graph_path(args.output_dir, parent_index, parent_area["epsg"])
        try:
            if parent_path.is_file() and not args.overwrite:
                parent_graph = ox.io.load_graphml(parent_path)
                parent_skipped += 1
            else:
                parent_path.parent.mkdir(parents=True, exist_ok=True)
                parent_graph = graph_for_bbox(parent_bbox)
                ox.io.save_graphml(parent_graph, parent_path)
                parent_downloaded += 1
            append_manifest(parent_manifest_path, {
                "parent_index": parent_index,
                "bbox_utm_m": parent_area["bounds_utm_m"],
                "bbox_wgs84": parent_bbox,
                "edge_count": len(parent_graph.edges),
                "node_count": len(parent_graph.nodes),
                "overpass_url": args.overpass_url,
                "path": str(parent_path.relative_to(args.output_dir)),
                "tile_count": len(parent_area["records"]),
                "status": "downloaded",
            })
            consecutive_connection_failures = 0
            print("parent %d: %d tiles (%d nodes, %d edges)" % (
                parent_index, len(parent_area["records"]), len(parent_graph.nodes),
                len(parent_graph.edges)))
            for record in pending_records:
                tile_id = record["tile_id"]
                output_path = graph_path(args.output_dir, tile_id)
                try:
                    subgraph = ox.truncate.truncate_graph_bbox(
                        parent_graph, bbox=bbox_for_record(record), truncate_by_edge=False)
                    ox.io.save_graphml(subgraph, output_path)
                    append_manifest(tile_manifest_path, {
                        "tile_id": tile_id,
                        "bbox": bbox_for_record(record),
                        "edge_count": len(subgraph.edges),
                        "node_count": len(subgraph.nodes),
                        "parent_index": parent_index,
                        "parent_path": str(parent_path.relative_to(args.output_dir)),
                        "path": str(output_path.relative_to(args.output_dir)),
                        "status": "extracted",
                    })
                    downloaded += 1
                except Exception as error:
                    append_manifest(tile_manifest_path, {
                        "tile_id": tile_id,
                        "bbox": bbox_for_record(record),
                        "error": str(error),
                        "parent_index": parent_index,
                        "status": "failed",
                    })
                    failed += 1
                    print("failed extraction: %s: %s" % (tile_id, error))
        except Exception as error:
            append_manifest(parent_manifest_path, {
                "parent_index": parent_index,
                "bbox_utm_m": parent_area["bounds_utm_m"],
                "bbox_wgs84": parent_bbox,
                "error": str(error),
                "overpass_url": args.overpass_url,
                "status": "failed",
            })
            failed += 1
            print("failed parent %d: %s" % (parent_index, error))
            if is_connection_failure(error):
                consecutive_connection_failures += 1
                if consecutive_connection_failures >= args.max_consecutive_connection_failures:
                    print("Stopping after %d consecutive endpoint connection failures." % (
                        consecutive_connection_failures))
                    break
            else:
                consecutive_connection_failures = 0
        if args.delay_seconds:
            time.sleep(args.delay_seconds)

    print("parents_downloaded=%d parents_reused=%d tiles_extracted=%d tiles_skipped=%d failed=%d" % (
        parent_downloaded, parent_skipped, downloaded, skipped, failed))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()