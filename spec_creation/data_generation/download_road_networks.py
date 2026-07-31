#!/usr/bin/env python3
"""Create per-tile road-network line graphs from the local Bay Area OSM XML.

The parent process loads ``bay_area.osm`` once, keeps all highway ways plus the
requested railway ways, and removes every other OSM way. Workers inherit the
filtered graph, truncate tiles by edge/bounding-box intersection (not by node
containment), enrich edges, convert each tile to a line graph, and save it with
the established ``<tile_id>.graphml`` convention. No API requests occur.

Run from ``mobilitynet-analysis-scripts``::

    python spec_creation/data_generation/download_road_networks.py
"""

import argparse
import json
import multiprocessing
from collections import defaultdict
from pathlib import Path

import networkx as nx
import osmnx as ox
from shapely.geometry import LineString, box
from shapely.strtree import STRtree


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OSM_XML = REPOSITORY_ROOT / "bay_area.osm"
DEFAULT_TRAJECTORIES = REPOSITORY_ROOT / "datasets" / "trajectories" / "trajectory_samples.jsonl"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "datasets" / "road_networks"
RAILWAY_VALUES = {"subway", "rail", "tram", "light_rail", "lightrail"}
EXCLUDED_RAIL_SERVICES = {"yard", "siding"}
EXCLUDE_KEYS = {"d12", "d17", "d18", "d19", "d20", "name", "bridge", "ref", "service", "access"}

_PARENT_GRAPH = None
_EDGE_RECORDS = None
_EDGE_INDEX = None


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
    """Return a tile bbox in (left, bottom, right, top) order."""
    bounds = record["bounds_wgs84"]
    return bounds["west"], bounds["south"], bounds["east"], bounds["north"]


def graph_path(output_dir, tile_id):
    """Return the established GraphML location for a trajectory tile."""
    return output_dir / (tile_id + ".graphml")


def include_edge(data):
    """Return whether an OSM edge satisfies the requested highway/rail filter."""
    if data.get("highway") is not None:
        return True
    return (
        data.get("railway") in RAILWAY_VALUES
        and data.get("service") not in EXCLUDED_RAIL_SERVICES
    )


def filter_parent_graph(graph):
    """Return a copy containing only the requested highway and railway edges."""
    included_edges = [
        (source, destination, key)
        for source, destination, key, data in graph.edges(keys=True, data=True)
        if include_edge(data)
    ]
    filtered = graph.edge_subgraph(included_edges).copy()
    filtered.remove_nodes_from(list(nx.isolates(filtered)))
    return filtered


def edge_geometry(graph, source, destination, data):
    """Return an edge's geometry, creating a straight line if absent."""
    geometry = data.get("geometry")
    if geometry is not None:
        return geometry
    source_node = graph.nodes[source]
    destination_node = graph.nodes[destination]
    return LineString([
        (float(source_node["x"]), float(source_node["y"])),
        (float(destination_node["x"]), float(destination_node["y"])),
    ])


def truncate_graph_by_edge_bbox(graph, bbox, edge_records, edge_index):
    """Include every edge intersecting a tile bbox and all its connected nodes."""
    left, bottom, right, top = bbox
    tile_polygon = box(left, bottom, right, top)
    edge_indices = edge_index.query(tile_polygon, predicate="intersects")
    intersecting_edges = [edge_records[int(index)][0] for index in edge_indices]
    if not intersecting_edges:
        raise ValueError("No graph edges intersect the requested bounding box.")
    subgraph = graph.edge_subgraph(intersecting_edges).copy()
    return subgraph


def enrich_and_line_graph(graph):
    """Fill edge geometry/coordinates and return its filtered-attribute line graph."""
    for source, destination, key, data in graph.edges(keys=True, data=True):
        start_x, start_y = float(graph.nodes[source]["x"]), float(graph.nodes[source]["y"])
        end_x, end_y = float(graph.nodes[destination]["x"]), float(graph.nodes[destination]["y"])
        data["start_x"] = start_x
        data["start_y"] = start_y
        data["end_x"] = end_x
        data["end_y"] = end_y
        if data.get("geometry") is None:
            data["geometry"] = LineString([(start_x, start_y), (end_x, end_y)])

    line_graph = nx.line_graph(graph)
    for source, destination, key, data in graph.edges(keys=True, data=True):
        edge_node = source, destination, key
        if edge_node in line_graph.nodes:
            line_graph.nodes[edge_node].update({
                attribute: value for attribute, value in data.items()
                if attribute not in EXCLUDE_KEYS
            })
    return line_graph


def build_edge_index(graph):
    """Build the parent-edge STRtree once for worker truncation queries."""
    records = [
        ((source, destination, key), edge_geometry(graph, source, destination, data))
        for source, destination, key, data in graph.edges(keys=True, data=True)
    ]
    geometries = [geometry for _, geometry in records]
    return records, geometries, STRtree(geometries)


def initialize_worker(parent_graph):
    """Initialize one worker with the already-filtered parent graph."""
    global _PARENT_GRAPH, _EDGE_RECORDS, _EDGE_INDEX
    _PARENT_GRAPH = parent_graph
    _EDGE_RECORDS, _, _EDGE_INDEX = build_edge_index(parent_graph)


def process_record(record, output_dir):
    """Extract, enrich, line-graph, and save one trajectory tile graph."""
    tile_id = record["tile_id"]
    output_path = graph_path(output_dir, tile_id)
    tile_graph = truncate_graph_by_edge_bbox(
        _PARENT_GRAPH,
        bbox_for_record(record),
        _EDGE_RECORDS,
        _EDGE_INDEX,
    )
    line_graph = enrich_and_line_graph(tile_graph)
    ox.io.save_graphml(line_graph, output_path)
    return {
        "tile_id": tile_id,
        "bbox": bbox_for_record(record),
        "edge_count": len(tile_graph.edges),
        "line_graph_edge_count": len(line_graph.edges),
        "line_graph_node_count": len(line_graph.nodes),
        "path": str(output_path.relative_to(output_dir)),
        "status": "written",
    }


def worker_main(task):
    """Run one worker task and return a serializable success/failure record."""
    record, output_dir = task
    try:
        return process_record(record, Path(output_dir))
    except Exception as error:
        return {
            "tile_id": record["tile_id"],
            "bbox": bbox_for_record(record),
            "error": str(error),
            "status": "failed",
        }


def append_manifest(path, entry):
    """Append one durable per-tile result."""
    with path.open("a") as output_file:
        output_file.write(json.dumps(entry, sort_keys=True) + "\n")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--osm-xml", type=Path, default=DEFAULT_OSM_XML)
    parser.add_argument("--trajectories", type=Path, default=DEFAULT_TRAJECTORIES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() // 2),
                        help="Number of worker processes (default: half the CPUs)")
    parser.add_argument("--max-tiles", type=int,
                        help="Limit tile outputs for a controlled partial run")
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.max_tiles is not None and args.max_tiles <= 0:
        raise ValueError("--max-tiles must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if not args.osm_xml.is_file():
        raise FileNotFoundError("OSM XML not found: %s" % args.osm_xml)
    if not args.trajectories.is_file():
        raise FileNotFoundError("Trajectory dataset not found: %s" % args.trajectories)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "download_manifest.jsonl"
    manifest_path.unlink(missing_ok=True)
    print("Loading offline parent graph: %s" % args.osm_xml, flush=True)
    parent_graph = ox.graph_from_xml(args.osm_xml)
    print("Loaded %d nodes and %d edges." % (len(parent_graph.nodes), len(parent_graph.edges)), flush=True)
    filtered_graph = filter_parent_graph(parent_graph)
    del parent_graph
    print("Filtered to %d nodes and %d highway/rail edges." % (
        len(filtered_graph.nodes), len(filtered_graph.edges)), flush=True)

    records = list(trajectory_records(args.trajectories))
    if args.max_tiles is not None:
        records = records[:args.max_tiles]
    tasks = ((record, str(args.output_dir)) for record in records)
    written = failed = 0

    context = multiprocessing.get_context("fork")
    with context.Pool(
        processes=args.workers,
        initializer=initialize_worker,
        initargs=(filtered_graph,),
        maxtasksperchild=25,
    ) as pool:
        for index, result in enumerate(pool.imap_unordered(worker_main, tasks), start=1):
            append_manifest(manifest_path, result)
            if result["status"] == "written":
                written += 1
                print("written %d/%d: %s (%d line nodes, %d line edges)" % (
                    index, len(records), result["tile_id"],
                    result["line_graph_node_count"], result["line_graph_edge_count"]), flush=True)
            else:
                failed += 1
                print("failed %d/%d: %s: %s" % (
                    index, len(records), result["tile_id"], result["error"]), flush=True)

    print("written=%d failed=%d" % (written, failed), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
