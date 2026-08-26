#!/usr/bin/env python3
"""Benchmark Nominatim/Overpass timing for a ground-truth route.

This is a diagnostic tool only: it does not modify route datasets and it does
not interact with an already-running mapping sweep. It measures four sources of
time independently:

* Nominatim reverse-API response time;
* local Nominatim result parsing;
* one batched Overpass way-and-node response time; and
* local Nominatim response processing time; and
* local tag validation/node processing time.

``--mode both`` first runs the current sequential workflow and then runs an
asynchronous pipeline. The pipeline has one producer thread that submits a new
Nominatim request every ``--nominatim-delay-seconds`` without waiting for prior
responses. Completed responses are enqueued immediately; separate processes
dequeue and parse them in true parallel. Overpass timings remain a separately
reported phase, so the benchmark can identify whether response latency, local
processing, or Overpass lookup is the bottleneck. The benchmark always uses a
cache-free batched Overpass query so it measures the network improvement; the
production mapper additionally persists its successful way/node results.

The default is deliberately a small, evenly distributed sample of ten route
coordinates. Increase ``--max-points`` only when the target service permits it.
"""

import argparse
import json
import multiprocessing
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

try:
    import find_closest_osm_nodes as mapper
except ModuleNotFoundError:
    from spec_creation import find_closest_osm_nodes as mapper


DEFAULT_OUTPUT_DIR = Path("osm/benchmarks")
PUBLIC_NOMINATIM_HOST = "nominatim.openstreetmap.org"


class StartRateLimiter:
    """Limit the interval between request starts across worker threads."""

    def __init__(self, interval_seconds):
        self.interval_seconds = interval_seconds
        self.next_start = 0.0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            start_at = max(now, self.next_start)
            self.next_start = start_at + self.interval_seconds
        delay = start_at - now
        if delay > 0:
            time.sleep(delay)


def is_public_nominatim(url):
    return PUBLIC_NOMINATIM_HOST in url.lower()


def evenly_sample(points, max_points):
    """Return evenly spaced ``(original_index, point)`` entries."""
    if max_points >= len(points):
        return list(enumerate(points))
    indices = sorted({round(index * (len(points) - 1) / (max_points - 1)) for index in range(max_points)})
    return [(index, points[index]) for index in indices]


def new_session(email):
    session = requests.Session()
    session.headers.update({
        "User-Agent": "OpenPATH OSM timing benchmark/1.0 (%s)" % email,
        "Accept-Language": "en",
    })
    return session


def reverse_timed(session, args, route_index, point, limiter):
    """Reverse-geocode a point while separating HTTP from local processing time."""
    longitude, latitude = point
    limiter.wait()
    api_start = time.monotonic()
    try:
        result = mapper.reverse_geocode(
            session, args.nominatim_url, longitude, latitude, args.email, args.retries)
        api_seconds = time.monotonic() - api_start
    except RuntimeError as error:
        return {
            "route_coordinate_index": route_index,
            "route_longitude": longitude,
            "route_latitude": latitude,
            "status": "request_failed",
            "error": str(error),
            "nominatim_api_seconds": time.monotonic() - api_start,
            "nominatim_processing_seconds": 0.0,
        }

    processing_start = time.monotonic()
    summary = mapper.result_summary(result)
    way_id = int(result["osm_id"]) if result.get("osm_type") == "way" and \
        str(result.get("osm_id", "")).isdigit() else None
    return {
        "route_coordinate_index": route_index,
        "route_longitude": longitude,
        "route_latitude": latitude,
        "status": "way_found" if way_id is not None else "not_a_way",
        "osm_way_id": way_id,
        "nominatim": summary,
        "nominatim_api_seconds": api_seconds,
        "nominatim_processing_seconds": time.monotonic() - processing_start,
    }


def reverse_request_timed(email, nominatim_url, retries, route_index, point):
    """Perform only the Nominatim request; a process will handle its response."""
    longitude, latitude = point
    session = new_session(email)
    api_start = time.monotonic()
    try:
        result = mapper.reverse_geocode(session, nominatim_url, longitude, latitude, email, retries)
        return {
            "route_coordinate_index": route_index,
            "route_longitude": longitude,
            "route_latitude": latitude,
            "status": "response_received",
            "nominatim_result": result,
            "nominatim_api_seconds": time.monotonic() - api_start,
            "response_received_monotonic": time.monotonic(),
        }
    except RuntimeError as error:
        return {
            "route_coordinate_index": route_index,
            "route_longitude": longitude,
            "route_latitude": latitude,
            "status": "request_failed",
            "error": str(error),
            "nominatim_api_seconds": time.monotonic() - api_start,
        }


def process_reverse_response(request_record):
    """Parse an asynchronous Nominatim reply in a separate process."""
    if request_record["status"] == "request_failed":
        return {**request_record, "nominatim_processing_seconds": 0.0,
                "response_queue_seconds": 0.0}

    processing_start = time.monotonic()
    result = request_record.pop("nominatim_result")
    response_received = request_record.pop("response_received_monotonic")
    summary = mapper.result_summary(result)
    way_id = int(result["osm_id"]) if result.get("osm_type") == "way" and \
        str(result.get("osm_id", "")).isdigit() else None
    return {
        **request_record,
        "status": "way_found" if way_id is not None else "not_a_way",
        "osm_way_id": way_id,
        "nominatim": summary,
        "response_queue_seconds": processing_start - response_received,
        "nominatim_processing_seconds": time.monotonic() - processing_start,
    }


def response_processor(input_queue, output_queue):
    """Continuously dequeue completed API replies for multiprocessing parsing."""
    while True:
        request_record = input_queue.get()
        if request_record is None:
            return
        try:
            output_queue.put(process_reverse_response(request_record))
        except Exception as error:  # Never leave the parent waiting for a result.
            output_queue.put({
                "route_coordinate_index": request_record.get("route_coordinate_index"),
                "status": "processing_failed",
                "error": "%s: %s" % (type(error).__name__, error),
                "nominatim_api_seconds": request_record.get("nominatim_api_seconds", 0.0),
                "response_queue_seconds": 0.0,
                "nominatim_processing_seconds": 0.0,
            })


def way_timed(session, args, way_id):
    """Look up a returned way and isolate network timing from tag validation."""
    api_start = time.monotonic()
    try:
        way = mapper.get_way_with_nodes(session, args.overpass_url, way_id, args.retries)
        api_seconds = time.monotonic() - api_start
    except RuntimeError as error:
        return {
            "osm_way_id": way_id,
            "status": "request_failed",
            "error": str(error),
            "overpass_api_seconds": time.monotonic() - api_start,
            "overpass_processing_seconds": 0.0,
        }

    processing_start = time.monotonic()
    accepted, validation_reason = mapper.validate_way_tags(way["tags"])
    return {
        "osm_way_id": way_id,
        "status": "accepted" if accepted else "review_required",
        "tags": way["tags"],
        "node_count": len(way["nodes"]),
        "validation_reason": validation_reason,
        "overpass_api_seconds": api_seconds,
        "overpass_processing_seconds": time.monotonic() - processing_start,
    }


def ways_batched_timed(session, args, way_ids):
    """Fetch all benchmark ways with a cache-free batched Overpass request."""
    api_start = time.monotonic()
    ways_by_id, failures = mapper.get_ways_with_nodes_batched(
        session, args.overpass_url, way_ids, args.retries,
        cache_dir=None, batch_size=args.overpass_batch_size)
    batch_record = {
        "requested_way_count": len(way_ids),
        "returned_way_count": len(ways_by_id),
        "failed_way_count": len(failures),
        "overpass_api_seconds": time.monotonic() - api_start,
    }
    way_records = []
    for way_id in way_ids:
        if way_id in failures:
            way_records.append({
                "osm_way_id": way_id,
                "status": "request_failed",
                "error": failures[way_id],
                "overpass_processing_seconds": 0.0,
            })
            continue
        processing_start = time.monotonic()
        way = ways_by_id[way_id]
        accepted, validation_reason = mapper.validate_way_tags(way["tags"])
        way_records.append({
            "osm_way_id": way_id,
            "status": "accepted" if accepted else "review_required",
            "tags": way["tags"],
            "node_count": len(way["nodes"]),
            "validation_reason": validation_reason,
            "overpass_processing_seconds": time.monotonic() - processing_start,
        })
    return batch_record, way_records


def summarize_seconds(records, field):
    values = [record[field] for record in records if field in record]
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "total_seconds": sum(values),
        "mean_seconds": statistics.mean(values),
        "median_seconds": statistics.median(values),
        "max_seconds": max(values),
    }


def run_sequential(args, sampled_points):
    """Benchmark the same serial request order used by the mapper."""
    session = new_session(args.email)
    limiter = StartRateLimiter(args.nominatim_delay_seconds)
    point_records = []
    started = time.monotonic()

    for route_index, point in sampled_points:
        point_record = reverse_timed(session, args, route_index, point, limiter)
        point_records.append(point_record)
    way_ids = sorted({record["osm_way_id"] for record in point_records if record.get("osm_way_id") is not None})
    overpass_batches, way_records = [], []
    if way_ids:
        overpass_batch, way_records = ways_batched_timed(session, args, way_ids)
        overpass_batches.append(overpass_batch)
    return format_result("sequential", started, point_records, way_records, overpass_batches)


def run_threaded(args, sampled_points):
    """Benchmark one paced async producer feeding a multiprocessing reply queue."""
    context = multiprocessing.get_context("spawn")
    response_queue = context.Queue()
    processed_queue = context.Queue()
    processors = [
        context.Process(target=response_processor, args=(response_queue, processed_queue))
        for _ in range(args.processor_workers)
    ]
    for processor in processors:
        processor.start()

    limiter = StartRateLimiter(args.nominatim_delay_seconds)
    started = time.monotonic()

    def producer():
        """Submit on the schedule; callbacks enqueue replies immediately on completion."""
        def enqueue_response(future):
            try:
                response_queue.put(future.result())
            except Exception as error:
                response_queue.put({
                    "route_coordinate_index": None,
                    "status": "request_failed",
                    "error": "%s: %s" % (type(error).__name__, error),
                    "nominatim_api_seconds": 0.0,
                })

        with ThreadPoolExecutor(max_workers=args.request_workers,
                                thread_name_prefix="nominatim-request") as executor:
            for route_index, point in sampled_points:
                limiter.wait()
                future = executor.submit(
                    reverse_request_timed, args.email, args.nominatim_url, args.retries, route_index, point)
                future.add_done_callback(enqueue_response)
        for _ in processors:
            response_queue.put(None)

    producer_thread = threading.Thread(target=producer, name="nominatim-producer")
    producer_thread.start()
    point_records = [processed_queue.get() for _ in sampled_points]
    producer_thread.join()
    for processor in processors:
        processor.join()

    way_ids = sorted({record["osm_way_id"] for record in point_records if record.get("osm_way_id") is not None})
    overpass_batches, way_records = [], []
    if way_ids:
        overpass_batch, way_records = ways_batched_timed(new_session(args.email), args, way_ids)
        overpass_batches.append(overpass_batch)
    return format_result("threaded", started, point_records, way_records, overpass_batches)


def format_result(mode, started, point_records, way_records, overpass_batches):
    """Return timing totals and raw measurements for a single benchmark mode."""
    return {
        "mode": mode,
        "wall_seconds": time.monotonic() - started,
        "route_point_results": point_records,
        "way_results": way_records,
        "overpass_batches": overpass_batches,
        "summary": {
            "nominatim_api": summarize_seconds(point_records, "nominatim_api_seconds"),
            "response_queue": summarize_seconds(point_records, "response_queue_seconds"),
            "nominatim_processing": summarize_seconds(point_records, "nominatim_processing_seconds"),
            "overpass_api": summarize_seconds(overpass_batches, "overpass_api_seconds"),
            "overpass_processing": summarize_seconds(way_records, "overpass_processing_seconds"),
            "ways_returned": len(way_records),
            "overpass_batch_count": len(overpass_batches),
            "accepted_ways": sum(record["status"] == "accepted" for record in way_records),
            "review_required_ways": sum(record["status"] == "review_required" for record in way_records),
        },
    }


def print_comparison(results):
    """Print a concise timing comparison suitable for terminal review."""
    for result in results:
        summary = result["summary"]
        print("%s: wall=%.3fs; Nominatim total=%.3fs; response-queue total=%.3fs; "
              "local processing total=%.3fs; Overpass total=%.3fs; "
              "Nominatim median=%.3fs; Overpass batch median=%.3fs; unique ways=%d" % (
                  result["mode"], result["wall_seconds"],
                  summary["nominatim_api"].get("total_seconds", 0.0),
                  summary["response_queue"].get("total_seconds", 0.0),
                  summary["nominatim_processing"].get("total_seconds", 0.0),
                  summary["overpass_api"].get("total_seconds", 0.0),
                  summary["nominatim_api"].get("median_seconds", 0.0),
                  summary["overpass_api"].get("median_seconds", 0.0),
                  summary["ways_returned"]))


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path, help="Evaluation-spec JSON file")
    parser.add_argument("--trip", required=True, help="evaluation_trips[].id")
    parser.add_argument("--leg", required=True, help="evaluation_trips[].legs[].id")
    parser.add_argument("--email", default="govilkinjal@gmail.com", help="Nominatim contact email")
    parser.add_argument("--route-index", type=int, default=0, help="Time-versioned route index")
    parser.add_argument("--max-points", type=int, default=10, help="Evenly sampled route points (default: 10)")
    parser.add_argument("--mode", choices=("sequential", "threaded", "both"), default="both",
                        help="Benchmark mode (default: both)")
    parser.add_argument("--nominatim-url", default=mapper.DEFAULT_NOMINATIM_URL,
                        help="Nominatim reverse API endpoint")
    parser.add_argument("--overpass-url", default=mapper.DEFAULT_OVERPASS_URL,
                        help="Overpass API endpoint")
    parser.add_argument("--overpass-batch-size", type=int, default=100,
                        help="Uncached OSM ways per batched Overpass query (default: 100)")
    parser.add_argument("--request-workers", type=int, default=4,
                        help="Maximum simultaneously in-flight Nominatim HTTP requests (default: 4)")
    parser.add_argument("--processor-workers", type=int, default=2,
                        help="Processes that dequeue and parse completed Nominatim responses (default: 2)")
    parser.add_argument("--nominatim-delay-seconds", type=float, default=1.1,
                        help="Minimum interval between Nominatim request starts")
    parser.add_argument("--retries", type=int, default=1, help="HTTP retries per request")
    parser.add_argument("--output", type=Path, help="Benchmark JSON output path")
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.max_points < 2:
        raise ValueError("--max-points must be at least 2")
    if args.request_workers < 1 or args.processor_workers < 1 or args.overpass_batch_size < 1:
        raise ValueError("worker counts must be positive")
    if args.retries < 0 or args.nominatim_delay_seconds < 0:
        raise ValueError("retries and delay must not be negative")
    if is_public_nominatim(args.nominatim_url):
        if args.nominatim_delay_seconds < 1.0:
            raise ValueError("Public Nominatim requires at least one second between request starts")

    label = mapper.load_label(args.spec)
    route = mapper.get_route_feature(label, args.trip, args.leg, args.route_index)
    sampled_points = evenly_sample(route["geometry"]["coordinates"], args.max_points)
    results = []
    if args.mode in {"sequential", "both"}:
        results.append(run_sequential(args, sampled_points))
    if args.mode in {"threaded", "both"}:
        results.append(run_threaded(args, sampled_points))

    output = {
        "benchmark_type": "ground_truth_osm_mapping_timing",
        "spec_id": label.get("id"),
        "trip_id": args.trip,
        "leg_id": args.leg,
        "route_index": args.route_index,
        "route_point_count": len(route["geometry"]["coordinates"]),
        "sampled_route_coordinate_indices": [index for index, _ in sampled_points],
        "configuration": {
            "nominatim_url": args.nominatim_url,
            "overpass_url": args.overpass_url,
            "request_workers": args.request_workers,
            "processor_workers": args.processor_workers,
            "overpass_batch_size": args.overpass_batch_size,
            "nominatim_delay_seconds": args.nominatim_delay_seconds,
            "retries": args.retries,
        },
        "results": results,
    }
    output_path = args.output or DEFAULT_OUTPUT_DIR / (
        "%s_%s_route_%d_timing.json" % (mapper.safe_name(args.trip), mapper.safe_name(args.leg), args.route_index))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as output_file:
        json.dump(output, output_file, indent=2)
        output_file.write("\n")
    print_comparison(results)
    print("Wrote benchmark results to %s" % output_path)


if __name__ == "__main__":
    main()