"""Cached OpenStreetMap-standard raster tile source."""

import io
import os
import math
import time
from pathlib import Path

import requests
from PIL import Image


DEFAULT_STANDARD_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
STANDARD_RENDER_ATTRIBUTION = "© OpenStreetMap contributors"


def atomic_image(path, image):
    """Write an image atomically, preserving its PNG mode."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.%d.tmp.png" % (path.stem, os.getpid()))
    image.save(temporary, format="PNG")
    os.replace(temporary, path)


def pixel_coordinates(coordinates, bounds, resolution):
    """Map projected coordinates to north-up image coordinates."""
    minimum_x, minimum_y, maximum_x, maximum_y = bounds
    scale_x = resolution / (maximum_x - minimum_x)
    scale_y = resolution / (maximum_y - minimum_y)
    return [
        ((x - minimum_x) * scale_x, (maximum_y - y) * scale_y)
        for x, y in coordinates
    ]


def web_mercator_world_pixel(longitude, latitude, zoom, tile_size=256):
    """Return north-up global slippy-map pixels for a WGS84 coordinate."""
    latitude = max(min(latitude, 85.05112878), -85.05112878)
    world_size = tile_size * (2 ** zoom)
    latitude_radians = math.radians(latitude)
    return (
        (longitude + 180.0) / 360.0 * world_size,
        (1.0 - math.asinh(math.tan(latitude_radians)) / math.pi) / 2.0 * world_size,
    )


class StandardMapTileSource:
    """Small cached OpenStreetMap-standard tile client for comparisons."""

    def __init__(self, cache_dir, url_template, offline=False, retries=3):
        self.cache_dir = Path(cache_dir)
        self.url_template = url_template
        self.offline = offline
        self.retries = retries
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "OpenPATH rendered-trajectory viewer/1.0"})

    def cache_path(self, zoom, x, y):
        return self.cache_dir / str(zoom) / str(x) / (str(y) + ".png")

    def tile(self, zoom, x, y):
        """Load one standard map tile from cache or the configured endpoint."""
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
            raise FileNotFoundError(
                "No cached standard map tile %d/%d/%d while --offline is set" % (zoom, x, y))
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
        raise RuntimeError("Standard map tile request failed for %d/%d/%d: %s" %
                           (zoom, x, y, last_error))


def trajectory_color(timestamp, start_timestamp, end_timestamp):
    """Return the requested blue-to-red color for a trajectory timestamp."""
    duration = end_timestamp - start_timestamp
    if duration == 0:
        progress = 0.0
    else:
        progress = min(1.0, max(0.0, (timestamp - start_timestamp) / duration))
    return (round(255 * progress), 0, round(255 * (1.0 - progress)))


def draw_timestamped_trajectory(draw, trajectory_xyt, bounds, resolution, width):
    """Draw timestamped projected samples with linearly interpolated colors."""
    if len(trajectory_xyt) < 2:
        return
    coordinates = [
        (min(resolution - 1, max(0, x)), min(resolution - 1, max(0, y)))
        for x, y in pixel_coordinates(
            [(x, y) for x, y, _ in trajectory_xyt], bounds, resolution)
    ]
    for index, ((start_x, start_y), (end_x, end_y)) in enumerate(zip(coordinates, coordinates[1:])):
        segment_start_timestamp = trajectory_xyt[index][2]
        segment_end_timestamp = trajectory_xyt[index + 1][2]
        segment_count = max(1, math.ceil(math.hypot(end_x - start_x, end_y - start_y)))
        for segment_index in range(segment_count):
            start_fraction = segment_index / segment_count
            end_fraction = (segment_index + 1) / segment_count
            start_point = (
                start_x + (end_x - start_x) * start_fraction,
                start_y + (end_y - start_y) * start_fraction,
            )
            end_point = (
                start_x + (end_x - start_x) * end_fraction,
                start_y + (end_y - start_y) * end_fraction,
            )
            timestamp = (segment_start_timestamp +
                         (segment_end_timestamp - segment_start_timestamp) * start_fraction)
            draw.line((start_point, end_point),
                      fill=trajectory_color(timestamp, trajectory_xyt[0][2], trajectory_xyt[-1][2]),
                      width=width)
    radius = max(1, width // 2)
    for (x, y), (_, _, timestamp) in zip(coordinates, trajectory_xyt):
        draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                     fill=trajectory_color(timestamp, trajectory_xyt[0][2], trajectory_xyt[-1][2]))


def standard_map_render(tile, trajectory_xyt, resolution, source, zoom):
    """Render a north-up standard map baseline with a timestamped GPS gradient.

    ``trajectory_xyt`` contains projected ``(x, y, timestamp)`` tuples in the
    same CRS as ``tile["bounds_utm_m"]``. The first sample is blue and the
    final sample is red; each intervening segment is linearly interpolated.
    """
    bounds = tile["bounds_wgs84"]
    left, top = web_mercator_world_pixel(bounds["west"], bounds["north"], zoom)
    right, bottom = web_mercator_world_pixel(bounds["east"], bounds["south"], zoom)
    minimum_tile_x, maximum_tile_x = math.floor(left / 256), math.floor((right - 1) / 256)
    minimum_tile_y, maximum_tile_y = math.floor(top / 256), math.floor((bottom - 1) / 256)
    mosaic = Image.new("RGB", ((maximum_tile_x - minimum_tile_x + 1) * 256,
                               (maximum_tile_y - minimum_tile_y + 1) * 256), "white")
    for tile_y in range(minimum_tile_y, maximum_tile_y + 1):
        for tile_x in range(minimum_tile_x, maximum_tile_x + 1):
            mosaic.paste(source.tile(zoom, tile_x, tile_y),
                         ((tile_x - minimum_tile_x) * 256,
                          (tile_y - minimum_tile_y) * 256))
    crop = mosaic.crop((round(left - minimum_tile_x * 256), round(top - minimum_tile_y * 256),
                        round(right - minimum_tile_x * 256), round(bottom - minimum_tile_y * 256)))
    render = crop.resize((resolution, resolution), Image.Resampling.LANCZOS)
    if len(trajectory_xyt) >= 2:
        from PIL import ImageDraw
        draw_timestamped_trajectory(
            ImageDraw.Draw(render), trajectory_xyt, tile["bounds_utm_m"], resolution,
            width=max(1, resolution // 96),
        )
    return render
