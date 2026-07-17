#!/usr/bin/env python3
"""Build north-up OSM/GPS map tiles and pixel-aligned segmentation masks.

The builder samples a reproducible, equal number of 640 m grid tiles from
every available phone trajectory.  Each tile has one trajectory rendered on
top of the OSM ways that intersect its boundary.  Its input render and target
mask share the same projected bounds and pixel transform, so they can be used
directly for semantic-segmentation training.

The default image engine is the MapLibre GL JS renderer in
``maplibre_renderer/``. It receives one exported WGS84 OSM/GPS payload per
tile and renders the input through a configurable MapLibre style in headless
Chromium. Use ``--render-engine python`` only for the legacy PIL comparison
renders. Use ``--preview-count N`` to render N randomly selected examples plus
a contact sheet, or ``--execute`` to write the full dataset.

Examples (run from mobilitynet-analysis-scripts)::

    python spec_creation/rendered_map_segmentation_dataset.py --preview-count 6
    python spec_creation/rendered_map_segmentation_dataset.py --execute \
        --tiles-per-trajectory 12 --output-dir datasets/osm_gps_segmentation

Use ``--offline`` to create previews only from previously cached Overpass tile
responses.  Each cache file is keyed by the tile's projected grid identity.
"""

import argparse
import hashlib
import io
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw
from pyproj import CRS, Transformer
from shapely.geometry import LineString, Polygon, box


# Direct execution puts ``spec_creation`` rather than the repository root on
# ``sys.path``.  Keep the documented ``python spec_creation/...py`` invocation
# able to import the sibling ``emeval`` package.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


TILE_SIDE_METERS = 640
RESOLUTION = 128
METERS_PER_PIXEL = TILE_SIDE_METERS / RESOLUTION
DEFAULT_DATA_ROOT = Path("bin/data")
DEFAULT_AUTHOR_EMAIL = "shankari@eecs.berkeley.edu"
DEFAULT_OUTPUT_DIR = Path("datasets/osm_gps_segmentation")
DEFAULT_OVERPASS_URL = "https://overpass-api.de/api/interpreter"
DEFAULT_STANDARD_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
STANDARD_RENDER_ATTRIBUTION = "© OpenStreetMap contributors"

# Values are intentionally stable: a trained model and saved masks therefore
# remain compatible when the builder is rerun.  OSM priority is low value first
# in this table.  The rasterizer draws it in reverse, allowing highways to
# overwrite lower-priority features.  GPS is explicitly top-most.
CLASS_DEFINITIONS = (
    (0, "background"),
    (1, "highway"),
    (2, "railway"),
    (3, "building"),
    (4, "natural"),
    (5, "waterway"),
    (6, "barrier"),
    (7, "surface"),
    (8, "amenity"),
    (9, "landuse"),
    (10, "gps_trajectory"),
)
CLASS_IDS = {name: class_id for class_id, name in CLASS_DEFINITIONS}
OSM_CLASS_NAMES = tuple(name for _, name in CLASS_DEFINITIONS[1:-1])
RAILWAY_VALUES = {"subway", "rail", "tram", "light_rail"}
EXCLUDED_RAIL_SERVICES = {"yard", "siding"}

# OSM feature line widths expressed in rendered pixels (5 m/pixel at default
# settings).  Closed ways are filled regardless of class; open ways use these
# widths so they remain observable after downsampling.
LINE_WIDTH_PIXELS = {
    "highway": 3,
    "railway": 2,
    "building": 1,
    "natural": 2,
    "waterway": 2,
    "barrier": 1,
    "surface": 3,
    "amenity": 2,
    "landuse": 2,
    "gps_trajectory": 3,
}

VIVID_PALETTE = {
    0: (12, 18, 33), 1: (255, 211, 105), 2: (191, 135, 255),
    3: (255, 144, 102), 4: (111, 219, 145), 5: (88, 181, 232),
    6: (221, 221, 221), 7: (255, 176, 104), 8: (78, 210, 196),
    9: (145, 190, 106), 10: (255, 71, 142),
}
INK_PALETTE = {
    0: (0, 0, 0), 1: (255, 255, 255), 2: (210, 210, 210),
    3: (135, 135, 135), 4: (95, 95, 95), 5: (175, 175, 175),
    6: (70, 70, 70), 7: (150, 150, 150), 8: (200, 200, 200),
    9: (110, 110, 110), 10: (0, 255, 255),
}
MONO_PALETTE = {
    0: 0, 1: 255, 2: 220, 3: 115, 4: 150, 5: 190, 6: 80,
    7: 170, 8: 205, 9: 135, 10: 245,
}
MASK_PALETTE = {
    0: (28, 28, 28), **VIVID_PALETTE,
}


def utm_crs_for(longitude, latitude):
    """Return the local WGS84 UTM CRS for a longitude/latitude coordinate."""
    zone = int((longitude + 180) / 6) + 1
    epsg = (32600 if latitude >= 0 else 32700) + zone
    return CRS.from_epsg(epsg)


def transformers_for(longitude, latitude):
    """Return local UTM CRS plus WGS84-to-UTM and inverse transformers."""
    crs = utm_crs_for(longitude, latitude)
    return (
        crs,
        Transformer.from_crs("EPSG:4326", crs, always_xy=True),
        Transformer.from_crs(crs, "EPSG:4326", always_xy=True),
    )


def classify_way(tags):
    """Return the highest-priority requested OSM class, or ``None``.

    A way with, for example, both ``highway`` and ``surface`` is a highway.
    This classification is also the overwrite order used during rasterization.
    """
    if "highway" in tags:
        return "highway"
    if (tags.get("railway") in RAILWAY_VALUES and
            tags.get("service") not in EXCLUDED_RAIL_SERVICES):
        return "railway"
    for feature_name in ("building", "natural", "waterway", "barrier", "surface", "amenity", "landuse"):
        if feature_name in tags:
            return feature_name
    return None


def feature_geometry(way, transformer):
    """Project an Overpass way record to a Shapely geometry, if valid."""
    coordinates = [
        transformer.transform(node["longitude"], node["latitude"])
        for node in way.get("nodes", [])
        if "longitude" in node and "latitude" in node
    ]
    if len(coordinates) < 2:
        return None
    if len(coordinates) >= 4 and coordinates[0] == coordinates[-1]:
        polygon = Polygon(coordinates)
        if not polygon.is_empty and polygon.is_valid and polygon.area > 0:
            return polygon
    line = LineString(coordinates)
    return line if not line.is_empty else None


def pixel_coordinates(coordinates, bounds, resolution):
    """Map projected coordinates to north-up image coordinates."""
    minimum_x, minimum_y, maximum_x, maximum_y = bounds
    scale_x = resolution / (maximum_x - minimum_x)
    scale_y = resolution / (maximum_y - minimum_y)
    return [
        ((x - minimum_x) * scale_x, (maximum_y - y) * scale_y)
        for x, y in coordinates
    ]


def draw_geometry(draw, geometry, bounds, resolution, value, line_width):
    """Draw a Shapely geometry into an image using a shared tile transform."""
    if geometry.is_empty:
        return
    geometry_type = geometry.geom_type
    if geometry_type == "Polygon":
        exterior = pixel_coordinates(geometry.exterior.coords, bounds, resolution)
        draw.polygon(exterior, fill=value)
        for interior in geometry.interiors:
            draw.polygon(pixel_coordinates(interior.coords, bounds, resolution), fill=0)
    elif geometry_type == "LineString":
        coordinates = pixel_coordinates(geometry.coords, bounds, resolution)
        if len(coordinates) >= 2:
            draw.line(coordinates, fill=value, width=line_width, joint="curve")
    elif geometry_type == "Point":
        x, y = pixel_coordinates([(geometry.x, geometry.y)], bounds, resolution)[0]
        radius = max(1, line_width // 2)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=value)
    elif geometry_type.startswith("Multi") or geometry_type == "GeometryCollection":
        for member in geometry.geoms:
            draw_geometry(draw, member, bounds, resolution, value, line_width)


def rasterize_tile(ways, trajectory_xy, bounds, resolution=RESOLUTION):
    """Rasterize OSM ways low-to-high priority, then the GPS trajectory.

    ``bounds`` are UTM metres in ``(min_x, min_y, max_x, max_y)`` order.  The
    vector clipping and all drawing use the same bounds, ensuring an exact
    affine alignment between inputs and labels.
    """
    mask = Image.new("L", (resolution, resolution), 0)
    draw = ImageDraw.Draw(mask)
    tile_shape = box(*bounds)

    # Inverse OSM priority: low priority is written first, then a higher
    # priority way overwrites it.  GPS is rendered after all OSM features.
    for class_name in reversed(OSM_CLASS_NAMES):
        class_id = CLASS_IDS[class_name]
        for way in ways:
            if way.get("class_name") != class_name:
                continue
            clipped = way["geometry"].intersection(tile_shape)
            draw_geometry(
                draw, clipped, bounds, resolution, class_id,
                LINE_WIDTH_PIXELS[class_name],
            )

    if trajectory_xy:
        trajectory = LineString(trajectory_xy)
        clipped_trajectory = trajectory.intersection(tile_shape)
        draw_geometry(
            draw, clipped_trajectory, bounds, resolution,
            CLASS_IDS["gps_trajectory"], LINE_WIDTH_PIXELS["gps_trajectory"],
        )
    return np.asarray(mask, dtype=np.uint8)


def colorize_mask(mask, palette, mode="RGB"):
    """Turn class-ID pixels into an RGB image or one-channel intensity image."""
    if mode == "L":
        lookup = np.array([palette.get(index, 0) for index in range(256)], dtype=np.uint8)
        return Image.fromarray(lookup[mask], mode="L")
    lookup = np.array([palette.get(index, (0, 0, 0)) for index in range(256)], dtype=np.uint8)
    return Image.fromarray(lookup[mask], mode="RGB")


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


def load_json(path):
    """Read JSON, returning ``None`` for an absent or invalid cache file."""
    try:
        with path.open() as input_file:
            return json.load(input_file)
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return None


def atomic_json(path, payload):
    """Write JSON atomically so interrupted dataset builds remain resumable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.%d.tmp" % (path.name, os.getpid()))
    with temporary.open("w") as output_file:
        json.dump(payload, output_file, separators=(",", ":"))
        output_file.write("\n")
    os.replace(temporary, path)


def atomic_image(path, image):
    """Write an image atomically, preserving its PNG mode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.%d.tmp.png" % (path.stem, os.getpid()))
    image.save(temporary, format="PNG")
    os.replace(temporary, path)


def web_mercator_world_pixel(longitude, latitude, zoom, tile_size=256):
    """Return north-up global slippy-map pixels for a WGS84 coordinate."""
    latitude = max(min(latitude, 85.05112878), -85.05112878)
    world_size = tile_size * (2 ** zoom)
    x = (longitude + 180.0) / 360.0 * world_size
    latitude_radians = math.radians(latitude)
    y = (1.0 - math.asinh(math.tan(latitude_radians)) / math.pi) / 2.0 * world_size
    return x, y


class StandardMapTileSource:
    """Small cached OpenStreetMap-standard tile client for visual comparisons.

    It is deliberately opt-in, intended for small review contact sheets, and
    sends a descriptive User-Agent.  A full build should use a separately
    hosted/provider-approved URL via ``--standard-tile-url``.
    """

    def __init__(self, cache_dir, url_template, offline=False, retries=3):
        self.cache_dir = Path(cache_dir)
        self.url_template = url_template
        self.offline = offline
        self.retries = retries
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "OpenPATH rendered-map comparison preview/1.0"})

    def cache_path(self, zoom, x, y):
        return self.cache_dir / str(zoom) / str(x) / (str(y) + ".png")

    def tile(self, zoom, x, y):
        """Load one standard tile from cache or the configured tile endpoint."""
        limit = 2 ** zoom
        if y < 0 or y >= limit:
            return Image.new("RGB", (256, 256), "white")
        x %= limit
        cache_path = self.cache_path(zoom, x, y)
        try:
            with Image.open(cache_path) as cached:
                return cached.convert("RGB").copy()
        except (FileNotFoundError, OSError):
            pass
        if self.offline:
            raise FileNotFoundError("No cached standard map tile %d/%d/%d while --offline is set" % (zoom, x, y))
        url = self.url_template.format(z=zoom, x=x, y=y)
        last_error = None
        for attempt in range(self.retries + 1):
            try:
                response = self.session.get(url, timeout=60)
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content)).convert("RGB")
                atomic_image(cache_path, image)
                return image
            except (requests.RequestException, OSError) as error:
                last_error = error
                if attempt < self.retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError("Standard map tile request failed for %d/%d/%d: %s" % (zoom, x, y, last_error))


def standard_map_render(tile, trajectory_xy, resolution, source, zoom):
    """Render a north-up standard map baseline and overlay the GPS in cyan."""
    bounds = tile["bounds_wgs84"]
    left, top = web_mercator_world_pixel(bounds["west"], bounds["north"], zoom)
    right, bottom = web_mercator_world_pixel(bounds["east"], bounds["south"], zoom)
    minimum_tile_x, maximum_tile_x = math.floor(left / 256), math.floor((right - 1) / 256)
    minimum_tile_y, maximum_tile_y = math.floor(top / 256), math.floor((bottom - 1) / 256)
    mosaic = Image.new("RGB", ((maximum_tile_x - minimum_tile_x + 1) * 256,
                               (maximum_tile_y - minimum_tile_y + 1) * 256), "white")
    for tile_y in range(minimum_tile_y, maximum_tile_y + 1):
        for tile_x in range(minimum_tile_x, maximum_tile_x + 1):
            mosaic.paste(source.tile(zoom, tile_x, tile_y), ((tile_x - minimum_tile_x) * 256,
                                                              (tile_y - minimum_tile_y) * 256))
    crop = mosaic.crop((round(left - minimum_tile_x * 256), round(top - minimum_tile_y * 256),
                        round(right - minimum_tile_x * 256), round(bottom - minimum_tile_y * 256)))
    render = crop.resize((resolution, resolution), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(render)
    if trajectory_xy:
        coordinates = pixel_coordinates(trajectory_xy, tile["bounds_utm_m"], resolution)
        if len(coordinates) >= 2:
            # Black underlay keeps the high-contrast GPS trace visible over
            # the conventional map's roads, labels, parks, and water.
            draw.line(coordinates, fill=(0, 0, 0), width=max(3, resolution // 48), joint="curve")
            draw.line(coordinates, fill=(0, 255, 255), width=max(1, resolution // 96), joint="curve")
    return render


class OverpassTileSource:
    """Resumable Overpass tile cache for the exact OSM vector query used here."""

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
                raise FileNotFoundError("No cached OSM response for %s while --offline is set" % tile_id)
            query = overpass_query(south, west, north, east)
            last_error = None
            for attempt in range(self.retries + 1):
                try:
                    response = self.session.post(self.overpass_url, data={"data": query}, timeout=240)
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


def discover_spec_ids(data_root, author_email):
    """Discover persisted evaluation specs from their configuration entries."""
    root = Path(data_root) / author_email
    return sorted(path.parent.parent.name for path in root.glob("*/config~evaluation_spec/*.json"))


def leaf_evaluation_ranges(phone_details):
    """Yield evaluation-section leaves, avoiding duplicate parent trajectories."""
    for evaluation_index, evaluation_range in enumerate(phone_details.get("evaluation_ranges", [])):
        trip_ranges = evaluation_range.get("evaluation_trip_ranges") or [evaluation_range]
        for trip_index, trip_range in enumerate(trip_ranges):
            section_ranges = trip_range.get("evaluation_section_ranges") or [trip_range]
            for section_index, section_range in enumerate(section_ranges):
                yield evaluation_index, trip_index, section_index, section_range


def load_phone_trajectories(data_root, author_email, spec_ids, location_key):
    """Load all phone/setting evaluation trajectories through ``PhoneView``.

    Each yielded source is retained separately, rather than merging phones.
    Stratified sampling can consequently include every OS/phone configuration.
    """
    from emeval.input.phone_view import PhoneView
    from emeval.input.spec_details import FileSpecDetails

    trajectories = []
    for spec_id in spec_ids:
        spec_details = FileSpecDetails(str(data_root), author_email, spec_id)
        phone_view = PhoneView(spec_details).map()
        for phone_os, phones in phone_view.items():
            for phone_label, phone_details in phones.items():
                for evaluation_index, trip_index, section_index, section_range in leaf_evaluation_ranges(phone_details):
                    location_df = section_range.get(location_key)
                    if not isinstance(location_df, pd.DataFrame):
                        continue
                    required_columns = {"longitude", "latitude"}
                    if not required_columns.issubset(location_df.columns):
                        continue
                    coordinates = location_df[["longitude", "latitude"]].dropna().drop_duplicates()
                    if len(coordinates) < 2:
                        continue
                    source_id = "%s--%s--%s--e%d--t%d--s%d" % (
                        spec_id, phone_os, phone_label, evaluation_index, trip_index, section_index)
                    trajectories.append({
                        "source_id": source_id,
                        "spec_id": spec_id,
                        "phone_os": phone_os,
                        "phone_label": phone_label,
                        "evaluation_range_index": evaluation_index,
                        "evaluation_trip_range_index": trip_index,
                        "evaluation_section_range_index": section_index,
                        "coordinates": list(coordinates.itertuples(index=False, name=None)),
                    })
    return trajectories


def tiles_for_trajectories(trajectories, tiles_per_trajectory, tile_side_meters, seed):
    """Select a balanced random sample of north-up projected grid tiles."""
    randomizer = random.Random(seed)
    selected = []
    seen = set()
    shuffled_trajectories = list(trajectories)
    randomizer.shuffle(shuffled_trajectories)
    for trajectory in shuffled_trajectories:
        coordinates = trajectory["coordinates"]
        sample_count = min(tiles_per_trajectory, len(coordinates))
        for point_index in randomizer.sample(range(len(coordinates)), sample_count):
            longitude, latitude = coordinates[point_index]
            crs, forward, inverse = transformers_for(longitude, latitude)
            x, y = forward.transform(longitude, latitude)
            grid_x = math.floor(x / tile_side_meters)
            grid_y = math.floor(y / tile_side_meters)
            tile_id = "%s--epsg%d--%d--%d" % (trajectory["source_id"], crs.to_epsg(), grid_x, grid_y)
            if tile_id in seen:
                continue
            seen.add(tile_id)
            minimum_x = grid_x * tile_side_meters
            minimum_y = grid_y * tile_side_meters
            maximum_x = minimum_x + tile_side_meters
            maximum_y = minimum_y + tile_side_meters
            corners = [inverse.transform(x_coord, y_coord) for x_coord, y_coord in (
                (minimum_x, minimum_y), (minimum_x, maximum_y),
                (maximum_x, minimum_y), (maximum_x, maximum_y),
            )]
            longitudes, latitudes = zip(*corners)
            selected.append({
                **trajectory,
                "tile_id": tile_id,
                "epsg": crs.to_epsg(),
                "bounds_utm_m": (minimum_x, minimum_y, maximum_x, maximum_y),
                "bounds_wgs84": {
                    "south": min(latitudes), "west": min(longitudes),
                    "north": max(latitudes), "east": max(longitudes),
                },
            })
    randomizer.shuffle(selected)
    return selected


def diverse_preview_tiles(tiles, count, seed):
    """Choose preview samples from different phones, alternating phone OSes.

    The full build is already balanced because every trajectory contributes the
    same number of candidate tiles.  This extra selection only makes a small
    preview contact sheet demonstrate that Android and iOS settings are both
    represented when they are available.
    """
    randomizer = random.Random(seed)
    by_source = {}
    for tile in tiles:
        by_source.setdefault(tile["source_id"], []).append(tile)
    sources_by_os = {}
    for source_tiles in by_source.values():
        sources_by_os.setdefault(source_tiles[0]["phone_os"], []).append(source_tiles)
    for source_groups in sources_by_os.values():
        randomizer.shuffle(source_groups)
        for source_tiles in source_groups:
            randomizer.shuffle(source_tiles)

    selected = []
    source_positions = {phone_os: 0 for phone_os in sources_by_os}
    phone_oses = sorted(sources_by_os)
    while len(selected) < count:
        added = False
        for phone_os in phone_oses:
            position = source_positions[phone_os]
            if position >= len(sources_by_os[phone_os]) or len(selected) >= count:
                continue
            selected.append(sources_by_os[phone_os][position][0])
            source_positions[phone_os] += 1
            added = True
        if not added:
            break
    return selected


def projected_trajectory(coordinates, transformer):
    """Project a WGS84 coordinate sequence into its selected tile's UTM CRS."""
    return [transformer.transform(longitude, latitude) for longitude, latitude in coordinates]


def render_sample(tile, source, output_dir, resolution, render_engine="maplibre",
                  standard_source=None, standard_zoom=None):
    """Render one tile, return JSON-serialisable manifest metadata, or skip it."""
    forward = Transformer.from_crs("EPSG:4326", CRS.from_epsg(tile["epsg"]), always_xy=True)
    vectors = source.ways_for_tile(tile["tile_id"], **tile["bounds_wgs84"])
    projected_ways = []
    for way in vectors:
        geometry = feature_geometry(way, forward)
        if geometry is not None:
            projected_ways.append({**way, "geometry": geometry})
    trajectory_xy = projected_trajectory(tile["coordinates"], forward)
    mask = rasterize_tile(
        projected_ways,
        trajectory_xy,
        tile["bounds_utm_m"],
        resolution,
    )
    counts = Counter(mask.ravel().tolist())
    osm_pixels = sum(counts[class_id] for class_id, name in CLASS_DEFINITIONS if name in OSM_CLASS_NAMES)
    osm_classes = sorted(name for class_id, name in CLASS_DEFINITIONS
                         if name in OSM_CLASS_NAMES and counts[class_id])
    if osm_pixels < tile["min_osm_pixels"] or len(osm_classes) < tile["min_osm_classes"]:
        return None

    output_dir = Path(output_dir)
    sample_name = tile["tile_id"] + ".png"
    images = {}
    if render_engine == "python":
        atomic_image(output_dir / "images" / "vivid" / sample_name, colorize_mask(mask, VIVID_PALETTE))
        atomic_image(output_dir / "images" / "ink" / sample_name, colorize_mask(mask, INK_PALETTE))
        atomic_image(output_dir / "images" / "mono" / sample_name, colorize_mask(mask, MONO_PALETTE, mode="L"))
        images.update({
            "vivid": (Path("images") / "vivid" / sample_name).as_posix(),
            "ink": (Path("images") / "ink" / sample_name).as_posix(),
            "mono": (Path("images") / "mono" / sample_name).as_posix(),
        })
    atomic_image(output_dir / "masks" / sample_name, Image.fromarray(mask, mode="L"))
    atomic_image(output_dir / "mask_previews" / sample_name, colorize_mask(mask, MASK_PALETTE))
    maplibre_input_path = Path("maplibre_inputs") / sample_name.replace(".png", ".json")
    atomic_json(output_dir / maplibre_input_path, {
        "tile_id": tile["tile_id"],
        "bounds_wgs84": tile["bounds_wgs84"],
        "bounds_utm_m": tile["bounds_utm_m"],
        "epsg": tile["epsg"],
        "ways": [
            {
                "osm_way_id": way["osm_way_id"],
                "class_name": way["class_name"],
                "tags": way.get("tags", {}),
                "nodes": way["nodes"],
            }
            for way in vectors
        ],
        "trajectory_coordinates": [list(coordinate) for coordinate in tile["coordinates"]],
    })
    manifest = {
        "tile_id": tile["tile_id"],
        "source_id": tile["source_id"],
        "spec_id": tile["spec_id"],
        "phone_os": tile["phone_os"],
        "phone_label": tile["phone_label"],
        "epsg": tile["epsg"],
        "bounds_utm_m": tile["bounds_utm_m"],
        "bounds_wgs84": tile["bounds_wgs84"],
        "resolution": resolution,
        "north_up": True,
        "osm_way_count": len(projected_ways),
        "osm_class_pixel_count": osm_pixels,
        "present_osm_classes": osm_classes,
        "class_pixel_counts": {CLASS_IDS[name]: counts[CLASS_IDS[name]] for name in CLASS_IDS},
        "render_engine": render_engine,
        "images": images,
        "mask": (Path("masks") / sample_name).as_posix(),
        "mask_preview": (Path("mask_previews") / sample_name).as_posix(),
        "maplibre_input": maplibre_input_path.as_posix(),
    }
    if standard_source is not None:
        standard_path = Path("images") / "standard" / sample_name
        atomic_image(
            output_dir / standard_path,
            standard_map_render(tile, trajectory_xy, resolution, standard_source, standard_zoom),
        )
        manifest["images"]["standard"] = standard_path.as_posix()
    return manifest


def write_contact_sheet(output_dir, manifests, resolution):
    """Write a labelled visual comparison: inputs, standard baseline, and mask."""
    if not manifests:
        return None
    tile_size = resolution
    label_height = 20
    available_columns = [
        style for style in ("maplibre", "vivid", "ink", "mono")
        if all(style in manifest["images"] for manifest in manifests)
    ]
    columns = available_columns
    if all("standard" in manifest["images"] for manifest in manifests):
        columns.append("standard + GPS")
    columns.append("mask")
    sheet = Image.new("RGB", (tile_size * len(columns), (tile_size + label_height) * len(manifests)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, manifest in enumerate(manifests):
        y = row * (tile_size + label_height)
        paths = [
            Path(output_dir) / manifest["mask_preview"]
            if style == "mask"
            else Path(output_dir) / manifest["images"][style]
            for style in columns
        ]
        for column, (label, path) in enumerate(zip(columns, paths)):
            with Image.open(path) as image:
                sheet.paste(image.convert("RGB"), (column * tile_size, y))
            draw.text((column * tile_size + 2, y + tile_size + 2), label, fill="black")
    contact_sheet = Path(output_dir) / "preview" / "contact_sheet.png"
    atomic_image(contact_sheet, sheet)
    return contact_sheet


def write_manifests(output_dir, manifests, args):
    """Persist samples as JSONL plus class and build metadata."""
    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as output_file:
        for manifest in manifests:
            output_file.write(json.dumps(manifest, sort_keys=True) + "\n")
    atomic_json(output_dir / "dataset.json", {
        "class_definitions": [{"id": class_id, "name": name} for class_id, name in CLASS_DEFINITIONS],
        "osm_priority": list(OSM_CLASS_NAMES),
        "draw_order": list(reversed(OSM_CLASS_NAMES)) + ["gps_trajectory"],
        "tile_side_meters": args.tile_side_meters,
        "resolution": args.resolution,
        "meters_per_pixel": args.tile_side_meters / args.resolution,
        "north_up": True,
        "seed": args.seed,
        "render_engine": args.render_engine,
        "render_options": {
            "maplibre": "MapLibre GL JS WebGL renderer using the configured style",
            "vivid": "legacy Python three-channel high-contrast class palette",
            "ink": "legacy Python black/white map with cyan GPS",
            "mono": "legacy Python single-channel high-contrast intensity map",
            **({"standard": "OpenStreetMap-standard raster baseline with high-contrast cyan GPS overlay"}
               if args.include_standard_render else {}),
        },
        **({"standard_render_attribution": STANDARD_RENDER_ATTRIBUTION,
            "standard_tile_url": args.standard_tile_url,
            "standard_tile_zoom": args.standard_tile_zoom}
           if args.include_standard_render else {}),
        "sample_count": len(manifests),
    })


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                        help="Root containing <author-email>/<spec-id> data (default: %(default)s)")
    parser.add_argument("--author-email", default=DEFAULT_AUTHOR_EMAIL,
                        help="Persisted datastore author directory (default: %(default)s)")
    parser.add_argument("--spec-id", action="append", dest="spec_ids",
                        help="Evaluation spec to include; repeat to select multiple (default: discover all)")
    parser.add_argument("--location-key", default="location_df",
                        help="PhoneView range dataframe to use (default: %(default)s)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Dataset root (default: %(default)s)")
    parser.add_argument("--render-engine", choices=("maplibre", "python"), default="maplibre",
                        help="Authoritative input renderer (default: maplibre; python is compatibility mode)")
    parser.add_argument("--maplibre-renderer-dir", type=Path,
                        help="MapLibre package directory (default: repository/maplibre_renderer)")
    parser.add_argument("--osm-cache-dir", type=Path,
                        help="Raw Overpass tile cache (default: <output-dir>/osm_tiles)")
    parser.add_argument("--overpass-url", default=DEFAULT_OVERPASS_URL,
                        help="Overpass interpreter endpoint (default: %(default)s)")
    parser.add_argument("--include-standard-render", action="store_true",
                        help="Add cached OpenStreetMap-standard comparison inputs with a cyan GPS overlay")
    parser.add_argument("--standard-tile-url", default=DEFAULT_STANDARD_TILE_URL,
                        help="Slippy tile URL with {z}, {x}, {y} placeholders (default: OpenStreetMap Standard)")
    parser.add_argument("--standard-cache-dir", type=Path,
                        help="Standard map tile cache (default: <output-dir>/standard_map_tiles)")
    parser.add_argument("--standard-tile-zoom", type=int, default=16,
                        help="Slippy-map zoom for standard-render comparisons (default: %(default)s)")
    parser.add_argument("--offline", action="store_true",
                        help="Require cached Overpass responses; do not make network requests")
    parser.add_argument("--retries", type=int, default=3,
                        help="Retries per uncached Overpass request (default: %(default)s)")
    parser.add_argument("--tile-side-meters", type=int, default=TILE_SIDE_METERS,
                        help="North-up projected tile edge length (default: %(default)s)")
    parser.add_argument("--resolution", type=int, default=RESOLUTION,
                        help="Square image/mask resolution (default: %(default)s)")
    parser.add_argument("--tiles-per-trajectory", type=int, default=8,
                        help="Balanced random samples from each phone trajectory (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=20260717,
                        help="Random sampling seed (default: %(default)s)")
    parser.add_argument("--min-osm-pixels", type=int, default=64,
                        help="Discard tiles with fewer OSM-labelled pixels (default: %(default)s)")
    parser.add_argument("--min-osm-classes", type=int, default=1,
                        help="Discard tiles with fewer present OSM classes (default: %(default)s)")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true", help="Render the complete selected dataset")
    action.add_argument("--preview-count", type=int, default=0,
                        help="Render this many samples and a comparison contact sheet")
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.tile_side_meters <= 0 or args.resolution <= 0 or args.tiles_per_trajectory <= 0:
        raise ValueError("Tile size, resolution, and tiles per trajectory must be positive")
    if (args.min_osm_pixels < 0 or args.min_osm_classes < 0 or args.preview_count < 0 or
            args.retries < 0 or args.standard_tile_zoom < 0):
        raise ValueError("Filter thresholds, preview count, and retries cannot be negative")
    if args.osm_cache_dir is None:
        args.osm_cache_dir = args.output_dir / "osm_tiles"
    if args.standard_cache_dir is None:
        args.standard_cache_dir = args.output_dir / "standard_map_tiles"
    if args.maplibre_renderer_dir is None:
        args.maplibre_renderer_dir = REPOSITORY_ROOT / "maplibre_renderer"
    if args.render_engine == "maplibre" and args.include_standard_render:
        raise ValueError("--include-standard-render is a legacy Python comparison; use the MapLibre style for the "
                         "authoritative render or run with --render-engine python")
    if args.execute and args.include_standard_render and args.standard_tile_url == DEFAULT_STANDARD_TILE_URL:
        raise ValueError("OpenStreetMap Standard is limited to small comparisons; use it with --preview-count or "
                         "provide a provider-approved --standard-tile-url for --execute")
    spec_ids = args.spec_ids or discover_spec_ids(args.data_root, args.author_email)
    if not spec_ids:
        raise ValueError("No evaluation specs discovered; supply --spec-id and verify --data-root/--author-email")

    trajectories = load_phone_trajectories(args.data_root, args.author_email, spec_ids, args.location_key)
    if not trajectories:
        raise ValueError("No usable %s trajectories were found" % args.location_key)
    tiles = tiles_for_trajectories(trajectories, args.tiles_per_trajectory, args.tile_side_meters, args.seed)
    print("Discovered %d separate phone trajectories across %d spec(s); selected %d grid tiles." %
          (len(trajectories), len(spec_ids), len(tiles)))
    if not args.execute and not args.preview_count:
        print("Dry run only. Use --preview-count N to compare renders or --execute to build the full dataset.")
        return

    if args.preview_count:
        tiles = diverse_preview_tiles(tiles, args.preview_count, args.seed)
    for tile in tiles:
        tile["min_osm_pixels"] = args.min_osm_pixels
        tile["min_osm_classes"] = args.min_osm_classes
    source = OverpassTileSource(args.osm_cache_dir, args.overpass_url, args.offline, args.retries)
    standard_source = None
    if args.include_standard_render:
        standard_source = StandardMapTileSource(
            args.standard_cache_dir, args.standard_tile_url, args.offline, args.retries)
    manifests = []
    skipped = 0
    for index, tile in enumerate(tiles, start=1):
        print("Rendering %d/%d: %s" % (index, len(tiles), tile["tile_id"]))
        manifest = render_sample(
            tile, source, args.output_dir, args.resolution, args.render_engine,
            standard_source=standard_source, standard_zoom=args.standard_tile_zoom,
        )
        if manifest is None:
            skipped += 1
        else:
            manifests.append(manifest)
    if args.render_engine == "maplibre" and manifests:
        cli_path = args.maplibre_renderer_dir / "src" / "cli.js"
        if not cli_path.is_file():
            raise ValueError("MapLibre renderer CLI was not found: %s" % cli_path)
        subprocess.run([
            "node", str(cli_path),
            "--input-dir", str(args.output_dir / "maplibre_inputs"),
            "--output-dir", str(args.output_dir / "images" / "maplibre"),
            "--resolution", str(args.resolution),
        ], check=True)
        for manifest in manifests:
            sample_name = Path(manifest["maplibre_input"]).with_suffix(".png").name
            manifest["images"]["maplibre"] = (Path("images") / "maplibre" / sample_name).as_posix()
    write_manifests(args.output_dir, manifests, args)
    if args.preview_count:
        contact_sheet = write_contact_sheet(args.output_dir, manifests, args.resolution)
        print("Preview contact sheet: %s" % contact_sheet)
    print("Wrote %d retained samples; filtered %d low-OSM samples." % (len(manifests), skipped))


if __name__ == "__main__":
    main()