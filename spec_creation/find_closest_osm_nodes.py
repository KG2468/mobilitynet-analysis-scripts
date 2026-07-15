#!/usr/bin/env python3
"""Build an OSM way/node dataset for one ground-truth ``route_coords`` path.

For every route coordinate, the script calls Nominatim's Reverse API to obtain
the OSM object selected for that location. Each returned way is independently
looked up in Overpass so its tags and *complete ordered node list* can be
saved. Only ways tagged ``highway=*`` or rail ways with ``railway`` equal to
``subway``, ``rail``, or ``tram`` (and not ``service=yard|siding``) are accepted.

Invalid Nominatim results remain in ``review_required`` and are deliberately
excluded from the ``ways`` dataset. The output also records every transition
between successive accepted ways and every interrupted/unmatched route point.

The public Nominatim service requires an identifying User-Agent and permits at
most one request per second. Provide an email address and leave the default
delay in place unless pointing at a private Nominatim service.

Example (run from mobilitynet-analysis-scripts)::

    conda run -n emissioneval python spec_creation/find_closest_osm_nodes.py \
      --spec bin/data/shankari@eecs.berkeley.edu/car_scooter_brex_san_jose/config~evaluation_spec/0_9223372036854775807.json \
      --trip 'bus trip with e-scooter access' --leg city_escooter \
      --email you@example.org

By default the output is written below ``osm/``. Use ``--output`` to select a
different JSON file within that directory.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests


DEFAULT_NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
DEFAULT_OVERPASS_URL = "https://overpass-api.de/api/interpreter"
ALLOWED_RAILWAYS = {"subway", "rail", "tram"}
DISALLOWED_RAIL_SERVICES = {"yard", "siding"}


def load_label(spec_path):
    """Load an evaluation-spec wrapper or a bare evaluation-spec label."""
    with spec_path.open() as spec_file:
        spec = json.load(spec_file)
    return spec.get("data", {}).get("label", spec)


def get_route_feature(label, trip_id, leg_id, route_index):
    """Return the selected ``route_coords`` GeoJSON feature."""
    trip = next((trip for trip in label["evaluation_trips"] if trip["id"] == trip_id), None)
    if trip is None:
        raise ValueError("Trip %r was not found" % trip_id)

    leg = next((leg for leg in trip["legs"] if leg["id"] == leg_id), None)
    if leg is None:
        raise ValueError("Leg %r was not found in trip %r" % (leg_id, trip_id))
    if "route_coords" not in leg:
        raise ValueError("Leg %r has no route_coords geometry" % leg_id)

    route_coords = leg["route_coords"]
    if isinstance(route_coords, list):
        if not 0 <= route_index < len(route_coords):
            raise ValueError("--route-index %d is outside the %d available route versions" %
                             (route_index, len(route_coords)))
        route_coords = route_coords[route_index]
    if route_coords.get("geometry", {}).get("type") != "LineString":
        raise ValueError("route_coords for %r is not a GeoJSON LineString" % leg_id)
    return route_coords


def safe_name(value):
    """Return a filesystem-safe label while retaining enough human context."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def default_output_path(output_dir, label, trip_id, leg_id, route_index):
    spec_id = safe_name(label.get("id", "evaluation_spec"))
    trip = safe_name(trip_id)
    leg = safe_name(leg_id)
    return output_dir / spec_id / trip / ("%s_route_%d_osm_ways.json" % (leg, route_index))


def request_json(session, url, *, params=None, data=None, retries=3, timeout=180):
    """Request JSON with bounded retries for transient public-service failures."""
    last_error = None
    for attempt in range(retries + 1):
        try:
            response = session.get(url, params=params, timeout=timeout) if params is not None else \
                session.post(url, data=data, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as error:
            last_error = error
            if attempt < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError("Request to %s failed after %d attempt(s): %s" % (url, retries + 1, last_error))


def reverse_geocode(session, nominatim_url, longitude, latitude, email, retries):
    """Return the Nominatim reverse result selected for a route coordinate."""
    return request_json(
        session,
        nominatim_url,
        params={
            "format": "jsonv2",
            "lat": "%.7f" % latitude,
            "lon": "%.7f" % longitude,
            "zoom": 17,
            "layer": "address,railway",
            "addressdetails": 0,
            "email": email,
        },
        retries=retries,
        timeout=60,
    )


def get_way_with_nodes(session, overpass_url, way_id, retries):
    """Fetch one way's tags and its complete, ordered OSM node sequence."""
    query = "[out:json][timeout:60];way(%d);out body;>;out body;" % way_id
    payload = request_json(session, overpass_url, data={"data": query}, retries=retries)
    way = next((element for element in payload.get("elements", []) if element.get("type") == "way"), None)
    if way is None:
        raise RuntimeError("Overpass did not return OSM way %d" % way_id)

    nodes_by_id = {
        element["id"]: element
        for element in payload["elements"]
        if element.get("type") == "node"
    }
    missing_node_ids = [node_id for node_id in way.get("nodes", []) if node_id not in nodes_by_id]
    if missing_node_ids:
        raise RuntimeError("OSM way %d is missing %d requested node(s)" % (way_id, len(missing_node_ids)))

    return {
        "osm_way_id": way_id,
        "tags": way.get("tags", {}),
        "nodes": [
            {
                "osm_node_id": node_id,
                "longitude": nodes_by_id[node_id]["lon"],
                "latitude": nodes_by_id[node_id]["lat"],
            }
            for node_id in way.get("nodes", [])
        ],
    }


def validate_way_tags(tags):
    """Return an acceptance boolean and specific reason for the way tags."""
    if "highway" in tags:
        return True, "highway=%s" % tags["highway"]

    railway = tags.get("railway")
    service = tags.get("service")
    if railway in ALLOWED_RAILWAYS and service not in DISALLOWED_RAIL_SERVICES:
        return True, "railway=%s" % railway
    if railway in ALLOWED_RAILWAYS:
        return False, "railway=%s has excluded service=%s" % (railway, service)
    return False, "requires highway=* or railway=subway|rail|tram without service=yard|siding"


def result_summary(result):
    """Keep the Nominatim fields needed to audit a route-point assignment."""
    return {
        "osm_type": result.get("osm_type"),
        "osm_id": result.get("osm_id"),
        "category": result.get("category"),
        "type": result.get("type"),
        "display_name": result.get("display_name"),
    }


def build_dataset(args, points):
    """Reverse-geocode route points and assemble accepted ways, nodes, and transitions."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "OpenPATH ground-truth OSM mapper/1.0 (%s)" % args.email,
        "Accept-Language": "en",
    })

    ways_by_id = {}
    route_point_matches = []
    review_required = []
    transitions = []
    previous_way_id = None
    last_request_started = None

    def interrupt_previous_way(route_coordinate_index, reason):
        """Record when a review item breaks an otherwise accepted way sequence."""
        nonlocal previous_way_id
        if previous_way_id is not None:
            transitions.append({
                "at_route_coordinate_index": route_coordinate_index,
                "from_osm_way_id": previous_way_id,
                "to_osm_way_id": None,
                "transition_type": "interrupted_by_review_required",
                "reason": reason,
            })
        previous_way_id = None

    for index, (longitude, latitude) in enumerate(points):
        if last_request_started is not None:
            remaining_delay = args.nominatim_delay_seconds - (time.monotonic() - last_request_started)
            if remaining_delay > 0:
                time.sleep(remaining_delay)
        print("Reverse-geocoding route point %d/%d" % (index + 1, len(points)), file=sys.stderr)
        last_request_started = time.monotonic()

        base_match = {
            "route_coordinate_index": index,
            "route_longitude": longitude,
            "route_latitude": latitude,
        }
        try:
            result = reverse_geocode(session, args.nominatim_url, longitude, latitude, args.email, args.retries)
        except RuntimeError as error:
            match = {**base_match, "status": "review_required", "reason": "Nominatim request failed: %s" % error}
            route_point_matches.append(match)
            review_required.append(match)
            interrupt_previous_way(index, match["reason"])
            continue

        summary = result_summary(result)
        if result.get("osm_type") != "way" or not str(result.get("osm_id", "")).isdigit():
            match = {
                **base_match,
                "status": "review_required",
                "reason": "Nominatim did not return an OSM way",
                "nominatim": summary,
            }
            route_point_matches.append(match)
            review_required.append(match)
            interrupt_previous_way(index, match["reason"])
            continue

        way_id = int(result["osm_id"])
        if way_id not in ways_by_id:
            try:
                way = get_way_with_nodes(session, args.overpass_url, way_id, args.retries)
            except RuntimeError as error:
                match = {
                    **base_match,
                    "status": "review_required",
                    "reason": "Cannot retrieve returned OSM way: %s" % error,
                    "nominatim": summary,
                }
                route_point_matches.append(match)
                review_required.append(match)
                interrupt_previous_way(index, match["reason"])
                continue
            accepted, reason = validate_way_tags(way["tags"])
            ways_by_id[way_id] = {**way, "accepted": accepted, "validation_reason": reason}

        way = ways_by_id[way_id]
        if not way["accepted"]:
            match = {
                **base_match,
                "status": "review_required",
                "osm_way_id": way_id,
                "way_tags": way["tags"],
                "reason": "Returned way %d failed validation: %s" % (way_id, way["validation_reason"]),
                "nominatim": summary,
            }
            route_point_matches.append(match)
            review_required.append(match)
            interrupt_previous_way(index, match["reason"])
            continue

        match = {
            **base_match,
            "status": "accepted",
            "osm_way_id": way_id,
            "nominatim": summary,
        }
        route_point_matches.append(match)
        if previous_way_id is not None and way_id != previous_way_id:
            transitions.append({
                "at_route_coordinate_index": index,
                "from_osm_way_id": previous_way_id,
                "to_osm_way_id": way_id,
                "transition_type": "accepted_way_change",
            })
        previous_way_id = way_id

    accepted_way_ids = {match["osm_way_id"] for match in route_point_matches if match["status"] == "accepted"}
    ways = []
    for way_id in sorted(accepted_way_ids):
        way = ways_by_id[way_id]
        ways.append({
            "osm_way_id": way_id,
            "tags": way["tags"],
            "nodes": way["nodes"],
            "matched_route_coordinate_indices": [
                match["route_coordinate_index"]
                for match in route_point_matches
                if match.get("osm_way_id") == way_id and match["status"] == "accepted"
            ],
        })
    return ways, route_point_matches, transitions, review_required


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path, help="Evaluation-spec JSON file")
    parser.add_argument("--trip", required=True, help="evaluation_trips[].id")
    parser.add_argument("--leg", required=True, help="evaluation_trips[].legs[].id")
    parser.add_argument("--email", required=True,
                        help="Contact address included in the Nominatim User-Agent")
    parser.add_argument("--output", type=Path,
                        help="Output JSON path; defaults to osm/<spec>/<trip>/<leg>_route_<n>_osm_ways.json")
    parser.add_argument("--output-dir", type=Path, default=Path("osm"),
                        help="Root for the default output path (default: osm)")
    parser.add_argument("--route-index", type=int, default=0,
                        help="Index for time-versioned route_coords lists (default: 0)")
    parser.add_argument("--nominatim-url", default=DEFAULT_NOMINATIM_URL,
                        help="Nominatim reverse API endpoint")
    parser.add_argument("--overpass-url", default=DEFAULT_OVERPASS_URL,
                        help="Overpass interpreter endpoint for way-node lookup")
    parser.add_argument("--nominatim-delay-seconds", type=float, default=1.1,
                        help="Minimum time between Nominatim calls (default: 1.1)")
    parser.add_argument("--retries", type=int, default=3,
                        help="Retries for a failed HTTP request (default: 3)")
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.nominatim_delay_seconds < 1.0 and "nominatim.openstreetmap.org" in args.nominatim_url:
        raise ValueError("Public Nominatim requires at least one second between requests")
    if args.retries < 0:
        raise ValueError("--retries cannot be negative")

    label = load_label(args.spec)
    route = get_route_feature(label, args.trip, args.leg, args.route_index)
    points = route["geometry"]["coordinates"]
    if not points:
        raise ValueError("The selected route has no coordinates")

    ways, route_point_matches, transitions, review_required = build_dataset(args, points)
    output = {
        "dataset_type": "ground_truth_osm_way_node_mapping",
        "spec_id": label.get("id"),
        "trip_id": args.trip,
        "leg_id": args.leg,
        "route_index": args.route_index,
        "source": {
            "spec": str(args.spec),
            "nominatim_reverse_url": args.nominatim_url,
            "overpass_url": args.overpass_url,
            "nominatim_delay_seconds": args.nominatim_delay_seconds,
        },
        "route_point_count": len(points),
        "accepted_route_point_count": sum(match["status"] == "accepted" for match in route_point_matches),
        "review_required_count": len(review_required),
        "ways": ways,
        "route_point_matches": route_point_matches,
        "transitions": transitions,
        "review_required": review_required,
    }
    output_path = args.output or default_output_path(
        args.output_dir, label, args.trip, args.leg, args.route_index)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as output_file:
        json.dump(output, output_file, indent=2)
        output_file.write("\n")
    print("Wrote %d accepted ways, %d transitions, and %d review item(s) to %s" %
          (len(ways), len(transitions), len(review_required), output_path))


if __name__ == "__main__":
    main()