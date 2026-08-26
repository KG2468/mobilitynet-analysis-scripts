"""Phone trajectory discovery, loading, and tile sampling."""

import math
import random
from pathlib import Path

import pandas as pd
from pyproj import CRS, Transformer


DEFAULT_TILE_SIDE_METERS = 640
DEFAULT_DATA_ROOT = Path("bin/data")
DEFAULT_AUTHOR_EMAIL = "shankari@eecs.berkeley.edu"


def transformers_for(longitude, latitude):
    """Return local UTM CRS plus WGS84-to-UTM and inverse transformers."""
    zone = int((longitude + 180) / 6) + 1
    epsg = (32600 if latitude >= 0 else 32700) + zone
    crs = CRS.from_epsg(epsg)
    return (
        crs,
        Transformer.from_crs("EPSG:4326", crs, always_xy=True),
        Transformer.from_crs(crs, "EPSG:4326", always_xy=True),
    )


class TrajectoryTileSampler:
    """Select reproducible, balanced tiles from phone trajectories."""

    def __init__(self, tiles_per_trajectory, tile_side_meters, seed,
                 data_root=DEFAULT_DATA_ROOT, author_email=DEFAULT_AUTHOR_EMAIL):
        self.tiles_per_trajectory = tiles_per_trajectory
        self.tile_side_meters = tile_side_meters
        self.seed = seed
        self.data_root = Path(data_root)
        self.author_email = author_email

    @staticmethod
    def discover_spec_ids(data_root, author_email):
        """Discover persisted evaluation specs from their configuration entries."""
        root = Path(data_root) / author_email
        return sorted(path.parent.parent.name for path in root.glob("*/config~evaluation_spec/*.json"))

    @staticmethod
    def leaf_evaluation_ranges(phone_details):
        """Yield evaluation-section leaves, avoiding duplicate parent trajectories."""
        for evaluation_index, evaluation_range in enumerate(phone_details.get("evaluation_ranges", [])):
            trip_ranges = evaluation_range.get("evaluation_trip_ranges") or [evaluation_range]
            for trip_index, trip_range in enumerate(trip_ranges):
                section_ranges = trip_range.get("evaluation_section_ranges") or [trip_range]
                for section_index, section_range in enumerate(section_ranges):
                    yield evaluation_index, trip_index, section_index, section_range

    def load_phone_trajectories(self, spec_ids, location_key,
                                data_root=None, author_email=None):
        """Load phone/setting evaluation trajectories through ``PhoneView``."""
        from emeval.input.phone_view import PhoneView
        from emeval.input.spec_details import FileSpecDetails

        data_root = self.data_root if data_root is None else Path(data_root)
        author_email = self.author_email if author_email is None else author_email
        trajectories = []
        for spec_id in spec_ids:
            spec_details = FileSpecDetails(str(data_root), author_email, spec_id)
            phone_view = PhoneView(spec_details).map()
            for phone_os, phones in phone_view.items():
                for phone_label, phone_details in phones.items():
                    for evaluation_index, trip_index, section_index, section_range in self.leaf_evaluation_ranges(phone_details):
                        location_df = section_range.get(location_key)
                        if not isinstance(location_df, pd.DataFrame):
                            continue
                        required_columns = {"longitude", "latitude", "ts"}
                        if not required_columns.issubset(location_df.columns):
                            continue
                        points = location_df[["longitude", "latitude", "ts"]].dropna().drop_duplicates()
                        if len(points) < 2:
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

                            "trajectory_points": list(points.itertuples(index=False, name=None)),
                        })
        return trajectories

    def find_tiles(self, trajectories):
        """Select tiles containing the largest nearby continuous trajectory run."""
        randomizer = random.Random(self.seed)
        selected = []
        seen = set()
        shuffled_trajectories = list(trajectories)
        randomizer.shuffle(shuffled_trajectories)
        for trajectory in shuffled_trajectories:
            points = trajectory["trajectory_points"]
            sample_count = min(self.tiles_per_trajectory, len(points))
            for sample_index in randomizer.sample(range(len(points)), sample_count):
                longitude, latitude, timestamp = points[sample_index]
                crs, forward, inverse = transformers_for(longitude, latitude)
                projected_points = [
                    (*forward.transform(point_longitude, point_latitude), point_timestamp)
                    for point_longitude, point_latitude, point_timestamp in points
                ]
                start_index, end_index, minimum_x, minimum_y, maximum_x, maximum_y = (
                    self.fitted_point_window(projected_points, sample_index))
                tile_points = points[start_index:end_index + 1]
                center_x = (minimum_x + maximum_x) / 2
                center_y = (minimum_y + maximum_y) / 2
                minimum_x = center_x - self.tile_side_meters / 2
                minimum_y = center_y - self.tile_side_meters / 2
                maximum_x = minimum_x + self.tile_side_meters
                maximum_y = minimum_y + self.tile_side_meters
                grid_x = math.floor(minimum_x)
                grid_y = math.floor(minimum_y)
                tile_id = "%s--epsg%d--%d--%d" % (
                    trajectory["source_id"], crs.to_epsg(), grid_x, grid_y)
                if tile_id in seen:
                    continue
                seen.add(tile_id)
                corners = [inverse.transform(x_coord, y_coord) for x_coord, y_coord in (
                    (minimum_x, minimum_y), (minimum_x, maximum_y),
                    (maximum_x, minimum_y), (maximum_x, maximum_y),
                )]
                longitudes, latitudes = zip(*corners)
                filtered_trajectory = {
                    key: trajectory[key]
                    for key in trajectory
                    if key != "trajectory_points"
                }
                selected.append({
                    **filtered_trajectory,
                    "tile_id": tile_id,
                    "sample_longitude": longitude,
                    "sample_latitude": latitude,
                    "sample_timestamp": timestamp,
                    "tile_trajectory_points": tile_points,
                    "epsg": crs.to_epsg(),
                    "bounds_utm_m": (minimum_x, minimum_y, maximum_x, maximum_y),
                    "bounds_wgs84": {
                        "south": min(latitudes), "west": min(longitudes),
                        "north": max(latitudes), "east": max(longitudes),
                    },
                })
        randomizer.shuffle(selected)
        return selected

    def fitted_point_window(self, projected_points, sample_index):
        """Greedily expand a contiguous point window while it fits in one tile."""
        left_index = sample_index
        right_index = sample_index
        minimum_x, minimum_y, _ = projected_points[sample_index]
        maximum_x = minimum_x
        maximum_y = minimum_y

        while True:
            candidates = []
            for point_index in (left_index - 1, right_index + 1):
                if point_index < 0 or point_index >= len(projected_points):
                    continue
                x, y, _ = projected_points[point_index]
                candidate_bounds = (
                    min(minimum_x, x), min(minimum_y, y),
                    max(maximum_x, x), max(maximum_y, y),
                )
                width = candidate_bounds[2] - candidate_bounds[0]
                height = candidate_bounds[3] - candidate_bounds[1]
                if width <= self.tile_side_meters and height <= self.tile_side_meters:
                    candidates.append((max(width, height), point_index, candidate_bounds))
            if not candidates:
                break
            _, point_index, (minimum_x, minimum_y, maximum_x, maximum_y) = min(candidates)
            if point_index < left_index:
                left_index = point_index
            else:
                right_index = point_index
        return left_index, right_index, minimum_x, minimum_y, maximum_x, maximum_y


    def select_preview_tiles(self, tiles, count):
        """Choose preview samples from different phones, alternating phone OSes."""
        randomizer = random.Random(self.seed)
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
