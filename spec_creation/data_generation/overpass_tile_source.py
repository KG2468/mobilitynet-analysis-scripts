"""Resumable Overpass tile source."""

import hashlib
import json
import os
import time
from pathlib import Path

import requests


DEFAULT_OVERPASS_URL = "https://overpass-api.de/api/interpreter"


RAILWAY_VALUES = {"subway", "rail", "tram", "light_rail"}
EXCLUDED_RAIL_SERVICES = {"yard", "siding"}


def load_json(path):
    """Read JSON, returning ``None`` for an absent or invalid cache file."""
    try:
        with Path(path).open() as input_file:
            return json.load(input_file)
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return None


def atomic_json(path, payload):
    """Write JSON atomically so interrupted requests remain resumable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.%d.tmp" % (path.name, os.getpid()))
    with temporary.open("w") as output_file:
        json.dump(payload, output_file, separators=(",", ":"))
        output_file.write("\n")
    os.replace(temporary, path)


def classify_way(tags):
    """Return the highest-priority requested OSM class, or ``None``."""
    if "highway" in tags:
        return "highway"
    if (tags.get("railway") in RAILWAY_VALUES and
            tags.get("service") not in EXCLUDED_RAIL_SERVICES):
        return "railway"
    for feature_name in ("building", "natural", "waterway", "barrier", "surface", "amenity", "landuse"):
        if feature_name in tags:
            return feature_name
    return None


def parse_overpass_ways(payload):
    """Return requested tagged OSM ways with complete node coordinates."""
    nodes = {
        element["id"]: element
        for element in payload.get("elements", [])
        if element.get("type") == "node" and "lat" in element and "lon" in element
    }
    ways = []
    for element in payload.get("elements", []):
        if element.get("type") != "way":
            continue
        tags = element.get("tags", {})
        class_name = classify_way(tags)
        node_ids = element.get("nodes", [])
        if class_name is None or not node_ids or any(node_id not in nodes for node_id in node_ids):
            continue
        ways.append({
            "osm_way_id": element["id"],
            "tags": tags,
            "class_name": class_name,
            "nodes": [
                {"longitude": nodes[node_id]["lon"], "latitude": nodes[node_id]["lat"]}
                for node_id in node_ids
            ],
        })
    return ways


def overpass_query(south, west, north, east):
    """Build the way-only Overpass query for all requested classes."""
    bbox = "%.7f,%.7f,%.7f,%.7f" % (south, west, north, east)
    return """[out:json][timeout:180];
(
  way[\"highway\"](%(bbox)s);
  way[\"railway\"~\"^(subway|rail|tram|light_rail)$\"][\"service\"!~\"^(yard|siding)$\"](%(bbox)s);
  way[\"building\"](%(bbox)s);
  way[\"natural\"](%(bbox)s);
  way[\"waterway\"](%(bbox)s);
  way[\"barrier\"](%(bbox)s);
  way[\"surface\"](%(bbox)s);
  way[\"amenity\"](%(bbox)s);
  way[\"landuse\"](%(bbox)s);
);
out body;
>;
out body;""" % {"bbox": bbox}


class OverpassTileSource:
    """Resumable Overpass tile cache for the exact OSM vector query."""

    def __init__(self, cache_dir, overpass_url, offline=False, retries=3):
        self.cache_dir = Path(cache_dir)
        self.overpass_url = overpass_url
        self.offline = offline
        self.retries = retries
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "OpenPATH rendered-map segmentation builder/1.0"})

    def cache_path(self, tile_id):
        digest = hashlib.sha256(tile_id.encode("utf-8")).hexdigest()
        return self.cache_dir / (digest + ".json")

    def ways_for_tile(self, tile_id, south, west, north, east):
        """Load cached vectors or query the Overpass API for this tile."""
        cache_path = self.cache_path(tile_id)
        payload = load_json(cache_path)
        if payload is None:
            if self.offline:
                raise FileNotFoundError(
                    "No cached OSM response for %s while --offline is set" % tile_id)
            query = overpass_query(south, west, north, east)
            last_error = None
            for attempt in range(self.retries + 1):
                try:
                    response = self.session.post(
                        self.overpass_url, data={"data": query}, timeout=240)
                    response.raise_for_status()
                    payload = response.json()
                    atomic_json(cache_path, payload)
                    break
                except (requests.RequestException, ValueError) as error:
                    last_error = error
                    if attempt < self.retries:
                        time.sleep(2 ** attempt)
            if payload is None:
                raise RuntimeError("Overpass request failed for %s: %s" % (tile_id, last_error))
        return parse_overpass_ways(payload)
