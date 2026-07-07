import scipy.interpolate as sci
import geopandas as gpd
import shapely as shp
import numpy as np
import random as random
import math
import arrow
import pandas as pd
import functools
import traceback

import emeval.metrics.dist_calculations as emd
import emeval.input.spec_details as eisd
import emeval.metrics.DTW as dtw

def interpolate_points_along_linestring(linestring, time_interval, points_per_second=1.0, with_time=False, time=None):
    """
    Interpolate points along a linestring at regular intervals
    
    Args:
        linestring (shapely.geometry.LineString): Input linestring
        time_interval (float): Time interval between interpolated points in seconds
        points_per_second (float): Number of points per second
    Returns:
        list: List of shapely.Point objects which contains both the original points from the linestring
        and the interpolated points at regular intervals, ensuring there at minimum points_per_second
        points per second while also maintaining all the detail from the original linestring
    """
    lat = linestring.coords[0][1]
    points = [shp.geometry.Point(coord[0] / math.cos(math.radians(lat)), coord[1]) for coord in list(linestring.coords)]
    # points = [shp.geometry.Point(coord[0], coord[1]) for coord in list(linestring.coords)]
    adjusted_linestring = shp.geometry.LineString(points)
    current_distance = 0.0
    total_length = adjusted_linestring.length
    interval = total_length / (points_per_second * time_interval)
    if with_time:
        current_time = time[0]
        time_step = 1.0 / points_per_second
        end_time = time[-1]
        while current_time < end_time:
            points.append(adjusted_linestring.interpolate(current_distance))
            time.append(current_time)
            current_time += time_step
            current_distance += interval
        time.sort()
    else:
        while current_distance < total_length:
            # Get point at current distance along the linestring
            point = adjusted_linestring.interpolate(current_distance)
            points.append(point)
            current_distance += interval
    points.sort(key=lambda p: adjusted_linestring.line_locate_point(p))
    # unadjusted_points = points
    unadjusted_points = [shp.geometry.Point(point.x * math.cos(math.radians(lat)), point.y) for point in points]
    if with_time:
        return unadjusted_points, time
    return unadjusted_points

random.seed(1)

####
# BEGIN: Building blocks of the final implementations
####

####
# BEGIN: NORMALIZATION
####

# In addition to filtering the sensed values in the polygons, we should also
# really filter the ground truth values in the polygons, since there is no
# ground truth within the polygon However, ground truth points are not known to
# be dense, and in some cases (e.g. commuter_rail_aboveground), there is a
# small gap between the polygon border and the first point outside it. We
# currently ignore this distance

def fill_gt_linestring(e):
    section_gt_shapes = gpd.GeoSeries(eisd.SpecDetails.get_shapes_for_leg(e["ground_truth"]["leg"]))
    e["ground_truth"]["gt_shapes"] = section_gt_shapes
    e["ground_truth"]["linestring"] = emd.filter_ground_truth_linestring(e["ground_truth"]["gt_shapes"])
    e["ground_truth"]["utm_gt_shapes"] = section_gt_shapes.apply(lambda s: shp.ops.transform(emd.to_utm_coords, s))
    e["ground_truth"]["utm_linestring"] = emd.filter_ground_truth_linestring(e["ground_truth"]["utm_gt_shapes"])

def to_gpdf(location_df):
    return gpd.GeoDataFrame(
        location_df, geometry=location_df.apply(
            lambda lr: shp.geometry.Point(lr.longitude, lr.latitude), axis=1))

def get_int_aligned_trajectory(location_df, tz="UTC", filter=False, spaced=False,
                              speed_threshold=60, acceleration_threshold=20,
                              jerk_threshold=5, max_clear_time=None,
                              refill_percentage=0.0):
    #check size of location_df
    if len(location_df) == 0:
        return gpd.GeoDataFrame({
            "ts": [],
            "fmt_time": [],
            "longitude": [],
            "latitude": [],
            "geometry": []
        })
    # lat_fn = sci.interp1d(x=location_df.ts, y=location_df.latitude)
    # lon_fn = sci.interp1d(x=location_df.ts, y=location_df.longitude)
    # # In order to avoid extrapolation, we use ceil for the first int and floor
    # # for the last int
    # first_int_ts = math.ceil(location_df.ts.iloc[0])
    # last_int_ts = math.floor(location_df.ts.iloc[-1])
    # new_ts_range = [float(ts) for ts in range(first_int_ts, last_int_ts, 1)]
    prev_loc = location_df.geometry.iloc[0]
    prev_ts = location_df.ts.iloc[0]
    start_ts = prev_ts
    new_points = [prev_loc]
    new_times = [prev_ts]
    for i in range(1, len(location_df)):
        loc = location_df.geometry.iloc[i]
        ts = location_df.ts.iloc[i]
        if ts-prev_ts > 1:
            points, times = interpolate_points_along_linestring(shp.geometry.LineString([prev_loc, loc]), ts - prev_ts, with_time=True, time=[prev_ts, ts])
            new_points.extend(points)
            new_times.extend(times)
        else:
            new_points.append(loc)
            new_times.append(ts)
        if ts - start_ts > len(new_points):
            print(ts)
        prev_loc = loc
        prev_ts = ts            

    # Align timestamps to integer seconds so that trajectories from different
    # devices (which have different sub-second offsets) share a common time
    # grid. Without this, merging on `ts` finds no matching keys and an inner
    # join collapses to zero rows.
    if not spaced:
        new_times = [float(round(ts)) for ts in new_times]
    new_fmt_time_range = [arrow.get(ts).to(tz) for ts in new_times]
    new_lat = [p.y for p in new_points]
    new_lng = [p.x for p in new_points]
    new_gpdf = gpd.GeoDataFrame({
        "latitude": new_lat,
        "longitude": new_lng,
        "ts": new_times,
        "fmt_time": new_fmt_time_range,
        "geometry": new_points
    })
    new_gpdf = new_gpdf.drop_duplicates()
    # Rounding to the integer grid can map two nearby raw/interpolated points to
    # the same second; keep one row per timestamp so the merge stays one-to-one.
    new_gpdf = new_gpdf.drop_duplicates(subset="ts", keep="first")
    if filter:
        speed_acceleration_jerk(new_gpdf)
        # Keep the points that satisfy all three physical-plausibility
        # thresholds; everything else is a candidate for removal. We retain the
        # removed rows so that, if filtering opens up a large temporal gap, we
        # can selectively re-add some of them below.
        keep_mask = (
            (new_gpdf.speed < speed_threshold)
            & (new_gpdf.acceleration < acceleration_threshold)
            & (new_gpdf.jerk < jerk_threshold)
        )
        kept_gpdf = new_gpdf[keep_mask]
        removed_gpdf = new_gpdf[~keep_mask]

        # If filtering left a temporal gap longer than `max_clear_time`, re-add a
        # fraction (`refill_percentage`) of the points that were removed inside
        # that gap. Every removed point is, by definition, "bad", so we cannot
        # cherry-pick the genuinely good ones. Instead we rank the removed points
        # into three tiers by how many of the harshest constraints they still
        # satisfy and randomize the pick within each tier, only dropping to a
        # worse tier when a better one cannot supply enough points:
        #   1. meets both the acceleration and jerk targets
        #   2. meets only the jerk target
        #   3. meets neither target
        if (max_clear_time is not None and refill_percentage > 0
                and len(removed_gpdf) > 0 and len(kept_gpdf) > 1):
            kept_sorted = kept_gpdf.sort_values(by="ts")
            kept_ts = list(kept_sorted.ts)
            refill_indices = []
            for i in range(1, len(kept_ts)):
                prev_ts = kept_ts[i - 1]
                curr_ts = kept_ts[i]
                if curr_ts - prev_ts <= max_clear_time:
                    continue
                gap_removed = removed_gpdf[
                    (removed_gpdf.ts > prev_ts) & (removed_gpdf.ts < curr_ts)]
                removed_count = len(gap_removed)
                if removed_count == 0:
                    continue
                n_to_add = int(round(refill_percentage * removed_count))
                if n_to_add < 1:
                    continue
                n_to_add = min(n_to_add, removed_count)

                meets_both = gap_removed[
                    (gap_removed.acceleration < acceleration_threshold)
                    & (gap_removed.jerk < jerk_threshold)]
                meets_jerk = gap_removed[
                    (gap_removed.acceleration >= acceleration_threshold)
                    & (gap_removed.jerk < jerk_threshold)]
                meets_none = gap_removed[
                    gap_removed.jerk >= jerk_threshold]

                chosen = []
                # Tier 1: prefer points that meet both targets.
                if n_to_add <= len(meets_both):
                    chosen.extend(random.sample(list(meets_both.index), n_to_add))
                else:
                    chosen.extend(list(meets_both.index))
                    remaining = n_to_add - len(meets_both)
                    # Tier 2: fall back to points that meet only the jerk target.
                    if remaining <= len(meets_jerk):
                        chosen.extend(random.sample(list(meets_jerk.index), remaining))
                    else:
                        chosen.extend(list(meets_jerk.index))
                        remaining -= len(meets_jerk)
                        # Tier 3: last resort, points that meet no target.
                        remaining = min(remaining, len(meets_none))
                        chosen.extend(random.sample(list(meets_none.index), remaining))
                refill_indices.extend(chosen)
            if refill_indices:
                kept_gpdf = pd.concat(
                    [kept_gpdf, removed_gpdf.loc[refill_indices]])
        new_gpdf = kept_gpdf.sort_values(by="ts").reset_index(drop=True)
        # Re-adding points changes the neighbor spacing, so the speed,
        # acceleration and jerk computed on the dense set are now stale; recompute
        # them on the final, filtered-and-refilled trajectory.
        if len(new_gpdf) > 3:
            speed_acceleration_jerk(new_gpdf)
    return new_gpdf



####
# END: NORMALIZATION
####

####
# BEGIN: DISTANCE CALCULATION
####

def add_gt_error_projection(location_gpdf, gt_linestring):
    location_gpdf["gt_distance"] = location_gpdf.distance(gt_linestring)
    location_gpdf["gt_projection"] = location_gpdf.geometry.apply(
        lambda p: gt_linestring.project(p))

def add_t_error(location_gpdf_a, location_gpdf_b):
    location_gpdf_a["t_distance"] = location_gpdf_a.distance(location_gpdf_b)
    location_gpdf_b["t_distance"] = location_gpdf_a.t_distance

def add_self_project(location_gpdf_a):
    loc_linestring = shp.geometry.LineString(coordinates=list(zip(
        location_gpdf.longitude, location_gdpf.latitude)))
    location_gpdf["s_projection"] = location_gpdf.geometry.apply(
        lambda p: loc_linestring.project(p))

####
# END: DISTANCE CALCULATION
####

####
# BEGIN: MERGE
####

# Assumes both entries exist
def b_merge_midpoint(loc_row):
    # print("merging %s" % loc_row)
    assert not pd.isnull(loc_row.geometry_i) and not pd.isnull(loc_row.geometry_a)
    midpoint = shp.geometry.LineString(coordinates=[loc_row.geometry_a, loc_row.geometry_i]).interpolate(0.5, normalized=True)
    # print(midpoint)
    final_geom = (midpoint, "midpoint")
    return final_geom

def b_merge_random(loc_row):
    # print("merging %s" % loc_row)
    assert not pd.isnull(loc_row.geometry_i) and not pd.isnull(loc_row.geometry_a)
    r_idx = random.choice(["geometry_a","geometry_i"])
    rp = loc_row[r_idx]
    # print(midpoint)
    final_geom = (rp, r_idx)
    return final_geom

def b_merge_closer_gt_dist(loc_row):
    # print("merging %s" % loc_row)
    assert not pd.isnull(loc_row.geometry_i) and not pd.isnull(loc_row.geometry_a)
    if loc_row.gt_distance_a < loc_row.gt_distance_i:
        final_geom = (loc_row.geometry_a, "android")
    else:
        final_geom = (loc_row.geometry_i, "ios")
    return final_geom

def b_merge_closer_gt_proj(loc_row):
    # print("merging %s" % loc_row)
    assert not pd.isnull(loc_row.geometry_i) and not pd.isnull(loc_row.geometry_a)
    if loc_row.gt_projection_a < loc_row.gt_projection_i:
        final_geom = (loc_row.geometry_a, "android")
    else:
        final_geom = (loc_row.geometry_i, "ios")
    return final_geom

def collapse_inner_join(loc_row, b_merge_fn):
    """
    Collapse a merged row. The merge was through inner join so both sides are
    known to exist
    """
    final_geom, source = b_merge_fn(loc_row)
    return {
        "ts": loc_row.ts,
        "longitude": final_geom.x,
        "latitude": final_geom.y,
        "geometry": final_geom,
        "source": source,
        "matching": [loc_row.geometry_a, loc_row.geometry_i]
    }

def collapse_outer_join_stateless(loc_row, b_merge_fn):
    """
    Collapse a merged row through outer join. This means that we can have
    either the left side or the right side, or both.
    - If only one side exists, we use it.
    - If both sides exist, we merge using `b_merge_fn`
    """
    source = None
    if pd.isnull(loc_row.geometry_i):
        assert not pd.isnull(loc_row.geometry_a)
        final_geom = loc_row.geometry_a
        source = "android"
    elif pd.isnull(loc_row.geometry_a):
        assert not pd.isnull(loc_row.geometry_i)
        final_geom = loc_row.geometry_i
        source = "ios"
    else:
        final_geom, source = b_merge_fn(loc_row)
    return {
        "ts": loc_row.ts,
        "longitude": final_geom.x,
        "latitude": final_geom.y,
        "geometry": final_geom,
        "source": source
    }

def make_collapse_outer_join_dist_so_far(more_details_fn = None):
    """
    Build a stateful collapse function for an outer-join merge that tracks the
    distance travelled so far along the ground truth linestring. The running
    `distance_so_far` is kept in a closure instead of a module-level global so
    that multiple reference trajectories can be constructed concurrently (e.g.
    on separate threads) without corrupting each other's progress state.

    The returned callable has the same semantics as the previous
    `collapse_outer_join_dist_so_far`: it collapses a merged row through outer
    join, preferring whichever side keeps the trajectory progressing forward
    along the ground truth.
    """
    state = {"distance_so_far": 0}

    def collapse_outer_join_dist_so_far(loc_row):
        source = None
        more_details = False
        EMPTY_POINT = shp.geometry.Point()

        if more_details_fn is not None and more_details_fn(loc_row):
            more_details = True

        distance_so_far = state["distance_so_far"]

        if more_details:
            print(loc_row.gt_projection_a, loc_row.gt_projection_i)
        if pd.isnull(loc_row.geometry_i):
            assert not pd.isnull(loc_row.geometry_a)
            if loc_row.gt_projection_a > distance_so_far:
                final_geom = loc_row.geometry_a
                source = "android"
            else:
                final_geom = EMPTY_POINT
        elif pd.isnull(loc_row.geometry_a):
            assert not pd.isnull(loc_row.geometry_i)
            if loc_row.gt_projection_i > distance_so_far:
                final_geom = loc_row.geometry_i
                source = "ios"
            else:
                final_geom = EMPTY_POINT
        else:
            assert not pd.isnull(loc_row.geometry_i) and not pd.isnull(loc_row.geometry_a)
            choice_series = gpd.GeoSeries([loc_row.geometry_a, loc_row.geometry_i])
            gt_projection_line_series = pd.Series([loc_row.gt_projection_a, loc_row.gt_projection_i])
            if more_details:
                print("gt_projection_line = %s" % gt_projection_line_series)
            distance_from_last_series = gt_projection_line_series.apply(lambda d: d - distance_so_far)
            if more_details:
                print("distance_from_last_series = %s" % distance_from_last_series)

            # assert not (distance_from_last_series < 0).all(), "distance_so_far = %s, distance_from_last = %s" % (distance_so_far, distance_from_last_series)
            if (distance_from_last_series < 0).all():
                if more_details:
                    print("all distances are negative, skipping...")
                final_geom = EMPTY_POINT
            else:
                if (distance_from_last_series < 0).any():
                    # avoid going backwards along the linestring (wonder how this works with San Jose u-turn)
                    closer_idx = distance_from_last_series.idxmax()
                    if more_details:
                        print("one distance is going backwards, found closer_idx = %d" % closer_idx)

                else:
                    distance_from_gt_series = pd.Series([loc_row.gt_distance_a, loc_row.gt_distance_i])
                    if more_details:
                        print("distance_from_gt_series = %s" % distance_from_gt_series)
                    closer_idx = distance_from_gt_series.idxmin()
                    if more_details:
                        print("both distances are positive, found closer_idx = %d" % closer_idx)

                if closer_idx == 0:
                    source = "android"
                else:
                    source = "ios"
                final_geom = choice_series.loc[closer_idx]

        if final_geom != EMPTY_POINT:
            if source == "android":
                state["distance_so_far"] = loc_row.gt_projection_a
            else:
                assert source == "ios"
                state["distance_so_far"] = loc_row.gt_projection_i

        if more_details:
            print("final_geom = %s, new_distance_so_far = %s" % (final_geom, state["distance_so_far"]))
        if final_geom == EMPTY_POINT:
            return {
                "ts": loc_row.ts,
                "longitude": np.nan,
                "latitude": np.nan,
                "geometry": EMPTY_POINT,
                "source": source
            }
        else:
            return {
                "ts": loc_row.ts,
                "longitude": final_geom.x,
                "latitude": final_geom.y,
                "geometry": final_geom,
                "source": source
            }

    return collapse_outer_join_dist_so_far

def group_points(mapping, options=-1):
    """
    Options: -1, 0, 1, 
    -1 for group for either element in each pair
    0 for group for only first element in each pair
    1 for group for only second element in each pair
    """
    groups = []
    current_group = [mapping[0]]
    current_dom = -1
    
    for pair in mapping[1:]:
        last = current_group[-1]
        # if the first element matches OR the second element matches, add to the current group
        if current_dom == -1:
            if pair[0] == last[0]:
                current_group.append(pair)
                current_dom = 0
            elif pair[1] == last[1]:
                current_group.append(pair)
                current_dom = 1
            else:
                groups.append(current_group)
                current_group = [pair]
                current_dom = -1
        elif current_dom == 0:
            if pair[0] == last[0]:
                current_group.append(pair)
            else:
                groups.append(current_group)
                current_group = [pair]
                current_dom = -1
        elif current_dom == 1:
            if pair[1] == last[1]:
                current_group.append(pair)
            else:
                groups.append(current_group)
                current_group = [pair]
                current_dom = -1
    
    groups.append(current_group)
    return groups

####
# END: MERGE
####

####
# END: Building blocks of the final implementations
####

####
# BEGIN: Combining into actual reference constructions
####

def ref_ends(e, dist_threshold, tz="UTC"):
    # This is only called from ref_ct_general and ref_gt_general, so 
    # the emd filter method adds the `outside_polygon` field to the input dataframe.
    # when we look for separate input and output polygons, we don't want to
    # mess up the other `outside_polygon` fields, so let's make copies
    # everywhere

    utm_gt_linestring = e["ground_truth"]["utm_linestring"]
    section_gt_shapes = e["ground_truth"]["gt_shapes"]

    def _get_filtered_loc(gt_key):
        unfiltered_loc_a_df = emd.to_geo_df(e["temporal_control"]["android"]["location_df"]).copy()
        emd.filter_geo_df(unfiltered_loc_a_df, section_gt_shapes.filter([gt_key]))
        loc_df_a = unfiltered_loc_a_df.query("outside_polygons==False")

        unfiltered_loc_b_df = emd.to_geo_df(e["temporal_control"]["ios"]["location_df"]).copy()
        emd.filter_geo_df(unfiltered_loc_b_df, section_gt_shapes.filter([gt_key]))
        loc_df_b = unfiltered_loc_b_df.query("outside_polygons==False")

        print(f"START_END: for threshold {dist_threshold}, before merging, for key {gt_key}, android: {len(unfiltered_loc_a_df)=} -> {len(loc_df_a)=}, ios: {len(unfiltered_loc_b_df)=} -> {len(loc_df_b)=}")

        return (loc_df_a, loc_df_b)

    start_loc_df_a, start_loc_df_b = _get_filtered_loc("start_loc")
    end_loc_df_a, end_loc_df_b = _get_filtered_loc("end_loc")
        
    merge_fn = functools.partial(collapse_inner_join, b_merge_fn=b_merge_midpoint)

    def _match_single_to_gt(filtered_loc_df, dist_threshold):
        new_location_df = get_int_aligned_trajectory(filtered_loc_df, tz)

        new_location_df_u = emd.to_utm_df(new_location_df)

        add_gt_error_projection(new_location_df_u, utm_gt_linestring)

        new_location_df["gt_distance"] = new_location_df_u.gt_distance
        new_location_df["gt_projection"] = new_location_df_u.gt_projection

        filtered_location_df = new_location_df.query("gt_distance < @dist_threshold")
        filtered_location_df['source'] = ['match_gt'] * len(filtered_location_df)
        # filtered_location_df.drop(columns=["gt_distance", "])
        return gpd.GeoDataFrame(filtered_location_df)

    def _align_and_merge(loc_df_a, loc_df_b, dist_threshold):
        # if this is exactly one, 
        # bus trip with e-scooter access city_escooter 3 and include_ends=True
        # fails with
        # x and y arrays must have at least 2 entries
        if len(loc_df_a) > 1 and len(loc_df_b) > 1:
            new_location_df_a = get_int_aligned_trajectory(loc_df_a, tz)
            new_location_df_i = get_int_aligned_trajectory(loc_df_b, tz)

            merged_df = pd.merge(new_location_df_a, new_location_df_i, on="ts",
                how="inner", suffixes=("_a", "_i")).sort_values(by="ts", axis="index")
            merged_df["t_distance"] = emd.to_utm_series(gpd.GeoSeries(merged_df.geometry_a)).distance(emd.to_utm_series(gpd.GeoSeries(merged_df.geometry_i)))
            filtered_merged_df = merged_df.query("t_distance < @dist_threshold")
            print("START_END: After filtering the merged dataframe, retained %d of %d (%s)" %
                  (len(filtered_merged_df), max(len(new_location_df_a), len(new_location_df_i)),
                    (len(filtered_merged_df)/max(len(new_location_df_a), len(new_location_df_i)))))
            ret_val = gpd.GeoDataFrame(list(filtered_merged_df.apply(merge_fn, axis=1)))
            if len(filtered_merged_df) == 0:
                print(f"CHECKME: {len(merged_df)=}, {len(filtered_merged_df)=}, START_END: after merging, {merged_df.head()=}")
                return gpd.GeoDataFrame([])
            else:
                return ret_val
#         elif len(loc_df_a) > 0 and len(loc_df_b) == 0:
#             return _match_single_to_gt(loc_df_a, dist_threshold)
#         elif len(loc_df_a) == 0 and len(loc_df_b) > 0:
#             return _match_single_to_gt(loc_df_b, dist_threshold)
        else:
            return gpd.GeoDataFrame([])

    start_initial_ends_gpdf = _align_and_merge(start_loc_df_a, start_loc_df_b, dist_threshold)
    end_initial_ends_gpdf = _align_and_merge(end_loc_df_a, end_loc_df_b, dist_threshold)

    return [start_initial_ends_gpdf, end_initial_ends_gpdf]

def ref_ct_general(e, b_merge_fn, dist_threshold, tz="UTC", include_ends=False):
    fill_gt_linestring(e)
    section_gt_shapes = e["ground_truth"]["gt_shapes"]
    # print("In ref_ct_general, %s" % section_gt_shapes.filter(items=["start_loc","end_loc"]))
    filtered_loc_df_a = emd.filter_geo_df(
        emd.to_geo_df(e["temporal_control"]["android"]["location_df"]),
        section_gt_shapes.filter(["start_loc","end_loc"]))
    filtered_loc_df_b = emd.filter_geo_df(
        emd.to_geo_df(e["temporal_control"]["ios"]["location_df"]),
        section_gt_shapes.filter(["start_loc","end_loc"]))
    print(f"MATCH TRAJECTORY: {len(filtered_loc_df_a)=}, {len(filtered_loc_df_b)=}")
    new_location_df_a = get_int_aligned_trajectory(filtered_loc_df_a, tz)
    new_location_df_i = get_int_aligned_trajectory(filtered_loc_df_b, tz)
    print (f"MATCH TRAJECTORY: after interpolation, {len(new_location_df_a)=}, {len(new_location_df_i)=}")
    print(f"MATCH TRAJECTORY: andriod after interpolation, {new_location_df_a.head()=}, {new_location_df_a.columns=}")
    print(f"MATCH TRAJECTORY: ios after interpolation, {new_location_df_i.head()=}, {new_location_df_i.columns=}")
    merged_df = pd.merge(new_location_df_a, new_location_df_i, on="ts",
        how="inner", suffixes=("_a", "_i")).sort_values(by="ts", axis="index")
    print(f"MATCH TRAJECTORY: after merging, {len(merged_df)=}")
    merged_df["t_distance"] = emd.to_utm_series(gpd.GeoSeries(merged_df.geometry_a)).distance(emd.to_utm_series(gpd.GeoSeries(merged_df.geometry_i)))
    print("t_distance stats before filtering: %s" % merged_df.t_distance.describe())
    filtered_merged_df = merged_df.query("t_distance < @dist_threshold")
    print("After filtering, retained %d of %d (%s)" %
          (len(filtered_merged_df), max(len(new_location_df_a), len(new_location_df_i)),
            (len(filtered_merged_df)/max(len(new_location_df_a), len(new_location_df_i)))))

    merge_fn = functools.partial(collapse_inner_join, b_merge_fn=b_merge_fn)
    initial_reference_gpdf = gpd.GeoDataFrame(list(filtered_merged_df.apply(merge_fn, axis=1)))
    if include_ends:
        [start_initial_ends_gpdf, end_initial_ends_gpdf] = ref_ends(e, dist_threshold, tz)
        print(f"CONCAT: {include_ends=}, before concatenating {len(start_initial_ends_gpdf)=}, {len(initial_reference_gpdf)=}, {len(end_initial_ends_gpdf)=}")
        initial_reference_gpdf = pd.concat([start_initial_ends_gpdf, initial_reference_gpdf, end_initial_ends_gpdf], axis=0).sort_values(by="ts").reset_index(drop=True)
        print(f"CONCAT: {include_ends=}, after concatenating {len(initial_reference_gpdf)=}")
    # print(end_initial_ends_gpdf)
    print(initial_reference_gpdf.columns)
    # print(initial_reference_gpdf[initial_reference_gpdf.ts.isna()])
    if len(initial_reference_gpdf.columns) > 1:
        initial_reference_gpdf["fmt_time"] = initial_reference_gpdf.ts.apply(lambda ts: arrow.get(ts).to(tz))
        assert len(initial_reference_gpdf[initial_reference_gpdf.latitude.isnull()]) == 0, "Found %d null entries out of %d total" % (len(initial_reference_gpdf.latitude.isnull()), len(initial_reference_gpdf))
        # print(initial_reference_gpdf.head())
        return initial_reference_gpdf
    else:
        return gpd.GeoDataFrame()
    
def ref_dtw_no_gt_with_ends_general(e, tz="UTC"):
    fill_gt_linestring(e)
    # print("In ref_ct_general, %s" % section_gt_shapes.filter(items=["start_loc","end_loc"]))
    a_pts = emd.to_geo_df(e["temporal_control"]["android"]["location_df"])
    i_pts = emd.to_geo_df(e["temporal_control"]["ios"]["location_df"])
    a_pts_seq = a_pts["geometry"].to_list()
    i_pts_seq = i_pts["geometry"].to_list()

    d = dtw.Dtw(a_pts_seq, i_pts_seq, dtw.calDistance)
    d.calculate()
    mapping = d.get_path()

    groups = group_points(mapping)

    print("After DTW, retained %d of %d (%s)" %
          (len(groups), max(len(a_pts), len(i_pts)),
            (len(groups)/max(len(a_pts), len(i_pts)))))
    #Average the postions and time stampsof each unique element in each group

    points = []
    timestamps = []
    for group in groups:
        #Get unique elements
        unique_elements_a = set()
        unique_elements_i = set()
        for pair in group:
            unique_elements_a.add(pair[0])
            unique_elements_i.add(pair[1])
        a_df = a_pts.iloc[list(unique_elements_a)]
        i_df = i_pts.iloc[list(unique_elements_i)]
        centroid = shp.geometry.MultiPoint(a_df["geometry"].to_list() + i_df["geometry"].to_list()).centroid
        ts = np.mean(a_df["ts"].to_list() + i_df["ts"].to_list())
        # Store points and timestamps in lists
        points.append(centroid)
        timestamps.append(ts)
        #Average the postions and time stamps
    
    # Create DataFrame from collected points and timestamps
    if len(points) == 0:
        return gpd.GeoDataFrame()
    
    gpdf = gpd.GeoDataFrame(
        data={'ts': timestamps},
        geometry=points
    )
    gpdf['longitude'] = gpdf.geometry.x
    gpdf['latitude'] = gpdf.geometry.y
    gpdf["fmt_time"] = gpdf.ts.apply(lambda ts: arrow.get(ts).to(tz))
    assert len(gpdf[gpdf.geometry.isnull()]) == 0, "Found %d null entries out of %d total" % (len(gpdf.geometry.isnull()), len(gpdf))
    return gpdf 

def ref_gt_general(e, b_merge_fn, dist_threshold, tz="UTC", include_ends=False):
    fill_gt_linestring(e)
    utm_gt_linestring = e["ground_truth"]["utm_linestring"]
    section_gt_shapes = e["ground_truth"]["gt_shapes"]
    filtered_loc_df_a = emd.filter_geo_df(
        emd.to_geo_df(e["temporal_control"]["android"]["location_df"]),
        section_gt_shapes.filter(["start_loc","end_loc"]))
    filtered_loc_df_b = emd.filter_geo_df(
        emd.to_geo_df(e["temporal_control"]["ios"]["location_df"]),
        section_gt_shapes.filter(["start_loc","end_loc"]))
    new_location_df_a = get_int_aligned_trajectory(filtered_loc_df_a, tz)
    new_location_df_i = get_int_aligned_trajectory(filtered_loc_df_b, tz)

    new_location_df_ua = emd.to_utm_df(new_location_df_a)
    new_location_df_ui = emd.to_utm_df(new_location_df_i)

    add_gt_error_projection(new_location_df_ua, utm_gt_linestring)
    add_gt_error_projection(new_location_df_ui, utm_gt_linestring)

    new_location_df_a["gt_distance"] = new_location_df_ua.gt_distance
    new_location_df_a["gt_projection"] = new_location_df_ua.gt_projection

    new_location_df_i["gt_distance"] = new_location_df_ui.gt_distance
    new_location_df_i["gt_projection"] = new_location_df_ui.gt_projection

    filtered_location_df_a = new_location_df_a.query("gt_distance < @dist_threshold")
    filtered_location_df_i = new_location_df_i.query("gt_distance < @dist_threshold")
    print("After filtering, %d of %d (%s) for android and %d of %d (%s) for ios" %
          (len(filtered_location_df_a), len(new_location_df_a), (len(filtered_location_df_a)/len(new_location_df_a)),
           len(filtered_location_df_i), len(new_location_df_i), (len(filtered_location_df_i)/len(new_location_df_i))))
    merged_df = pd.merge(filtered_location_df_a, filtered_location_df_i, on="ts",
        how="outer", suffixes=("_a", "_i")).sort_values(by="ts", axis="index")
    merge_fn = functools.partial(collapse_outer_join_stateless, b_merge_fn=b_merge_fn)
    initial_reference_gpdf = gpd.GeoDataFrame(list(merged_df.apply(merge_fn, axis=1)))
    if include_ends:
        [start_initial_ends_gpdf, end_initial_ends_gpdf] = ref_ends(e, dist_threshold, tz)
        initial_reference_gpdf = pd.concat([start_initial_ends_gpdf, initial_reference_gpdf, end_initial_ends_gpdf], axis=0).sort_values(by="ts").reset_index(drop=True)
    if len(initial_reference_gpdf.columns) > 1:
        initial_reference_gpdf["fmt_time"] = initial_reference_gpdf.ts.apply(lambda ts: arrow.get(ts).to(tz))
        print("After merging, found %d of android %d (%s), ios %d (%s)" %
              (len(initial_reference_gpdf), len(new_location_df_a), (len(initial_reference_gpdf)/len(new_location_df_a)),
               len(new_location_df_i), (len(initial_reference_gpdf)/len(new_location_df_i))))
        assert len(initial_reference_gpdf[initial_reference_gpdf.latitude.isnull()]) == 0, "Found %d null entries out of %d total" % (len(initial_reference_gpdf.latitude.isnull()), len(initial_reference_gpdf))
        return initial_reference_gpdf
    else:
        return gpd.GeoDataFrame()
 

def _solve_const_accel_offsets(fractions, v0, v1, T):
    """
    Map each distance fraction in (0, 1] to a time offset in (0, T] under a
    constant acceleration that carries the speed from `v0` to `v1` across the
    window of length `T`. A point that is a fraction `f` of the way along the
    (fixed) path is placed at the time when the constant-acceleration model has
    covered fraction `f` of its own travelled distance, which keeps the exit at
    exactly `T` while following the v0 -> v1 speed shape.
    """
    a = (v1 - v0) / T
    offsets = []
    for f in fractions:
        if abs(a) < 1e-9:
            offsets.append(f * T)
        else:
            disc = max(v0 * v0 * (1.0 - f) + f * v1 * v1, 0.0)
            offsets.append((-v0 + math.sqrt(disc)) / a)
    return offsets, a


# Toggle from the notebook with
#   emeval.metrics.reference_trajectory.DEBUG_SECONDPASS = True
# to get a detailed trace of every second-pass timestamp assignment. Set
# DEBUG_SECONDPASS_ENTRY to a ground-truth firstpass index to only print the run
# that starts right after that index (None prints every run).
DEBUG_SECONDPASS = False
DEBUG_SECONDPASS_ENTRY = None


def _sp_debug(active, msg):
    if active:
        print("SECONDPASS_DBG: " + msg)


def _assign_secondpass_timestamps(entry_pos, run_positions, exit_pos,
                                  t_entry, t_exit, v0, a0, v1, a1, jerk_limit,
                                  debug_tag=None):
    """
    Assign timestamps to the fixed `run_positions` that bridge the gap between
    two firstpass points. The positions are not moved; only their timestamps are
    chosen so the implied motion is a constant-acceleration transition from the
    entry speed `v0` to the exit speed `v1` over the fixed window
    [`t_entry`, `t_exit`].

    The only acceleration discontinuities are at the two ends: from the entry
    point's acceleration `a0` into the run's constant acceleration, and from the
    run back out to the exit point's acceleration `a1`. If either transition
    would exceed `jerk_limit`, the corresponding boundary interval is lengthened
    to the smallest duration that keeps the jerk within the limit (the single
    "adjusting step") and the interior is rescaled into the remaining window so
    the curve still lands on the exit point at `t_exit`.

    Pass `debug_tag` (e.g. the bracketing firstpass ground-truth index) to label
    the diagnostic printouts emitted when `DEBUG_SECONDPASS` is enabled.
    """
    N = len(run_positions)
    T = t_exit - t_entry

    active = DEBUG_SECONDPASS and (
        DEBUG_SECONDPASS_ENTRY is None or DEBUG_SECONDPASS_ENTRY == debug_tag)
    if active:
        _sp_debug(active, "==== run tag=%s N=%d ====" % (debug_tag, N))
        _sp_debug(active, "window: t_entry=%.6f t_exit=%.6f T=%.6f" %
                  (t_entry, t_exit, T))
        _sp_debug(active, "boundary kinematics: v0=%s a0=%s v1=%s a1=%s "
                  "jerk_limit=%s" % (v0, a0, v1, a1, jerk_limit))

    if N == 0:
        return []
    # Every list returned by this function holds offsets relative to `t_entry`
    # (i.e. in [0, T]), never absolute timestamps. Keeping the offsets local
    # preserves their fine (sub-millisecond) structure; adding them onto the
    # ~1.5e9 absolute Unix-epoch base at construction time would quantize them
    # onto a ~2.4e-7 s grid and inject the noise that the finite-difference
    # acceleration/jerk then amplify by orders of magnitude.
    uniform = [(n + 1) * (T / (N + 1)) for n in range(N)]
    return uniform #bypass dynamic second-pass timestamp assignment
    if T <= 0 or not np.all(np.isfinite([v0, a0, v1, a1])):
        _sp_debug(active, "FALLBACK uniform: T<=0 or non-finite kinematics "
                  "(T=%s, finite=%s) -> %s" %
                  (T, np.all(np.isfinite([v0, a0, v1, a1])), uniform))
        return uniform

    seq = [entry_pos] + list(run_positions) + [exit_pos]
    cum = [0.0]
    for k in range(1, len(seq)):
        cum.append(cum[-1] + dtw.calDistance(seq[k - 1], seq[k]))
    D = cum[-1]
    if D <= 0:
        _sp_debug(active, "FALLBACK uniform: total path length D=%s <= 0 -> %s" %
                  (D, uniform))
        return uniform

    fractions = [cum[k] / D for k in range(1, N + 1)]
    base_offsets, a = _solve_const_accel_offsets(fractions, v0, v1, T)
    bounds = [0.0] + base_offsets + [T]
    durations = [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]
    if active:
        _sp_debug(active, "path: D=%.6f cum=%s" % (D, _fmt_list(cum)))
        _sp_debug(active, "fractions=%s" % _fmt_list(fractions))
        _sp_debug(active, "const-accel a=%.6g base_offsets=%s" %
                  (a, _fmt_list(base_offsets)))
        _sp_debug(active, "bounds=%s" % _fmt_list(bounds))
        _sp_debug(active, "raw durations=%s (min=%.6g max=%.6g)" %
                  (_fmt_list(durations), min(durations), max(durations)))
    if any(d <= 0 for d in durations):
        _sp_debug(active, "FALLBACK uniform: non-positive duration in %s -> %s" %
                  (_fmt_list(durations), uniform))
        return uniform

    # Lengthen the first/last interval only when its jerk exceeds the limit.
    entry_jerk = abs((a - a0) / durations[0])
    entry_min = 0.0
    # if entry_jerk > jerk_limit:
        # entry_min = abs(a - a0) / jerk_limit
    exit_jerk = abs((a1 - a) / durations[-1])
    exit_min = 0.0
    # if exit_jerk > jerk_limit:
        # exit_min = abs(a1 - a) / jerk_limit
    if active:
        _sp_debug(active, "entry jerk=%.6g (limit=%s) -> entry_min=%.6g" %
                  (entry_jerk, jerk_limit, entry_min))
        _sp_debug(active, "exit  jerk=%.6g (limit=%s) -> exit_min=%.6g" %
                  (exit_jerk, jerk_limit, exit_min))

    new_first = max(durations[0], entry_min)
    new_last = max(durations[-1], exit_min)
    extra = (new_first - durations[0]) + (new_last - durations[-1])
    if extra > 0:
        interior = durations[1:-1]
        interior_sum = sum(interior)
        if interior_sum > 0 and extra < interior_sum:
            scale = (interior_sum - extra) / interior_sum
            durations = [new_first] + [d * scale for d in interior] + [new_last]
            _sp_debug(active, "stretch: extra=%.6g scaled interior by %.6g" %
                      (extra, scale))
        else:
            # Not enough interior slack (or no interior point): keep the stretch
            # request and let the renormalization below rescale to the window, so
            # the jerk limit is then honored only approximately.
            durations = [new_first] + interior + [new_last]
            _sp_debug(active, "stretch: extra=%.6g but interior_sum=%.6g "
                      "insufficient; jerk limit only approximate" %
                      (extra, interior_sum))

    durations = [max(d, 1e-6) for d in durations]
    total = sum(durations)
    durations = [d * (T / total) for d in durations]
    if active:
        _sp_debug(active, "final durations=%s (min=%.6g max=%.6g sum=%.6g)" %
                  (_fmt_list(durations), min(durations), max(durations),
                   sum(durations)))

    offs = []
    acc = 0.0
    for d in durations[:-1]:
        acc += d
        offs.append(acc)

    if active:
        _sp_debug(active, "offsets (run-local, rel. t_entry)=%s" % _fmt_list(offs))
        # Report on the precise local time axis ([0, T]); this is what the
        # downstream kinematics actually see now that timestamps keep their
        # local precision, so the acceleration here should be ~constant.
        _report_secondpass_kinematics(active, seq, [0.0] + offs + [T])
    return offs


def _fmt_list(values, max_items=None):
    """Format a numeric list for debug printing, showing all values."""
    def f(x):
        try:
            return "%.6g" % x
        except (TypeError, ValueError):
            return str(x)
    return "[" + ", ".join(f(v) for v in values) + "]"


def _report_secondpass_kinematics(active, seq, stamps):
    """Recompute the per-segment speed / acceleration / jerk implied by the
    assigned `stamps` for the ordered points `seq` (entry, run..., exit) and
    print the peak magnitudes plus a per-step table, so the caller can see
    exactly where the spikes originate."""
    if not active:
        return
    n = len(seq)
    if n != len(stamps) or n < 2:
        _sp_debug(active, "kinematics: cannot report (len(seq)=%d len(stamps)=%d)"
                  % (n, len(stamps)))
        return
    dists = [dtw.calDistance(seq[k - 1], seq[k]) for k in range(1, n)]
    dts = [stamps[k] - stamps[k - 1] for k in range(1, n)]
    speeds = []
    for d, dt in zip(dists, dts):
        speeds.append(d / dt if dt > 0 else float("inf"))
    accels = []
    for k in range(1, len(speeds)):
        dt = stamps[k + 1] - stamps[k - 1]
        accels.append((speeds[k] - speeds[k - 1]) / dt if dt > 0
                      else float("inf"))
    jerks = []
    for k in range(1, len(accels)):
        dt = stamps[k + 2] - stamps[k]
        jerks.append((accels[k] - accels[k - 1]) / dt if dt > 0
                     else float("inf"))

    def peak(xs):
        finite = [abs(x) for x in xs if np.isfinite(x)]
        return max(finite) if finite else float("inf")

    _sp_debug(active, "kinematics peaks: |speed|max=%.6g |accel|max=%.6g "
              "|jerk|max=%.6g" % (peak(speeds), peak(accels), peak(jerks)))
    _sp_debug(active, "  dt   =%s" % _fmt_list(dts))
    _sp_debug(active, "  dist =%s" % _fmt_list(dists))
    _sp_debug(active, "  speed=%s" % _fmt_list(speeds))
    _sp_debug(active, "  accel=%s" % _fmt_list(accels))
    _sp_debug(active, "  jerk =%s" % _fmt_list(jerks))


def _constant_velocity_fill(start_pos, run_positions, t_start, t_stop):
    """
    Fill an end-of-trajectory run (which has no following firstpass point) by
    spreading the fixed `run_positions` between `t_start` and `t_stop` at
    constant velocity, i.e. each timestamp is proportional to that point's
    cumulative distance from `start_pos`. The final run point lands on `t_stop`.
    """
    N = len(run_positions)
    span = t_stop - t_start
    # Offsets are returned relative to `t_start` (i.e. in [0, span]) so the
    # caller can add them to a small local time base and avoid the catastrophic
    # float cancellation that occurs when tiny offsets are added directly onto a
    # ~1.5e9 absolute Unix-epoch base.
    uniform = [(n + 1) * (span / (N + 1)) for n in range(N)]
    return uniform #bypass dyanmic assignment
    if N == 0:
        return []
    seq = [start_pos] + list(run_positions)
    cum = [0.0]
    for k in range(1, len(seq)):
        cum.append(cum[-1] + dtw.calDistance(seq[k - 1], seq[k]))
    D = cum[-1]
    if D <= 0 or span <= 0:
        return uniform
    return [(cum[k] / D) * span for k in range(1, N + 1)]


def ref_dtw_gt_with_ends_general(e, tz="UTC", points_per_second=1, interp=2, time_threshold=300, jerk_limit=5,
                                 speed_threshold=60, acceleration_threshold=55,
                                 jerk_threshold=15, max_clear_time=None,
                                 refill_percentage=0.0):
    fill_gt_linestring(e)
    a_pts = emd.to_geo_df(e["temporal_control"]["android"]["location_df"])
    i_pts = emd.to_geo_df(e["temporal_control"]["ios"]["location_df"])
    if interp >= 1:
        new_a_pts = get_int_aligned_trajectory(a_pts, tz, True, True,
                                               speed_threshold=speed_threshold,
                                               acceleration_threshold=acceleration_threshold,
                                               jerk_threshold=jerk_threshold,
                                               max_clear_time=max_clear_time,
                                               refill_percentage=refill_percentage)
        new_i_pts = get_int_aligned_trajectory(i_pts, tz, True, True,
                                               speed_threshold=speed_threshold,
                                               acceleration_threshold=acceleration_threshold,
                                               jerk_threshold=jerk_threshold,
                                               max_clear_time=max_clear_time,
                                               refill_percentage=refill_percentage)
    else:
        new_a_pts = a_pts
        new_i_pts = i_pts
    a_pts_seq = new_a_pts["geometry"].to_list()
    i_pts_seq = new_i_pts["geometry"].to_list()

    start_ts = min(new_a_pts["ts"].iloc[0], new_i_pts["ts"].iloc[0])
    end_ts = max(new_a_pts["ts"].iloc[-1], new_i_pts["ts"].iloc[-1])


    # Get points at 1 second intervals along the ground truth linestring
    if interp == 0 or interp == 2:
        gt_pts = interpolate_points_along_linestring(e["ground_truth"]["linestring"], time_interval=(end_ts-start_ts), points_per_second=points_per_second)
    else:
        gt_pts = [shp.geometry.Point(coord) for coord in list(e["ground_truth"]["linestring"].coords)]
    
    # print("In ref_ct_general, %s" % section_gt_shapes.filter(items=["start_loc","end_loc"]))
    
    d_a = dtw.Dtw(gt_pts, a_pts_seq, dtw.calDistance)
    d_a.calculate()
    mapping_a = d_a.get_path()

    d_i = dtw.Dtw(gt_pts, i_pts_seq, dtw.calDistance)
    d_i.calculate()
    mapping_i = d_i.get_path()

    groups_a = []
    a_idx = len(mapping_a) - 1   
    groups_i = []
    i_idx = len(mapping_i) - 1
    match_streak = 0
    firstpass_idxes = []
    secondpass_idxes = []
    for idx in range(len(gt_pts)):
        group_a = []
        group_i = []
        while a_idx >= 0 and mapping_a[a_idx][0] == idx:
            group_a.append(mapping_a[a_idx][1])
            a_idx -= 1
        while i_idx >= 0 and mapping_i[i_idx][0] == idx:
            group_i.append(mapping_i[i_idx][1])
            i_idx -= 1
        if len(groups_a) > 0 and groups_a[-1] == group_a and groups_i[-1] == group_i:
            if match_streak == 0:
                secondpass_idxes.append([idx])
            else:
                secondpass_idxes[-1].append(idx)
            match_streak += 1
        else:
            firstpass_idxes.append(idx)
            match_streak = 0
        groups_a.append(group_a)
        groups_i.append(group_i)
    
    # print("After DTW, retained %d of %d (%s) for android and %d of %d (%s) for ios" %
    #       (len(groups_a), max(len(a_pts), len(i_pts)),
    #         (len(groups_a)/max(len(a_pts), len(i_pts))),
    #         len(groups_i), max(len(a_pts), len(i_pts)),
    #         (len(groups_i)/max(len(a_pts), len(i_pts)))))
    #Average the postions and time stampsof each unique element in each group

    
    
    def get_centriod_and_ts(idx, timeseries_id):
        if timeseries_id == 0:
            centroid_a, ts_a = get_centriod_and_ts(idx, 1)
            centroid_i, ts_i = get_centriod_and_ts(idx, 2)
            return shp.geometry.MultiPoint([centroid_a, centroid_i]).centroid, (ts_a + ts_i)/2
        if timeseries_id == 1:
            groups = groups_a
            pts = new_a_pts
        elif timeseries_id == 2:
            groups = groups_i
            pts = new_i_pts

        unique_elements = set()
        for pt in groups[idx]:
            unique_elements.add(pt)
        df = pts.iloc[list(unique_elements)]
        matched_points = df["geometry"].to_list()
        matched_ts = df["ts"].to_list()
        matched_ts_mean = np.mean(matched_ts)
        matched_points_centroid = shp.geometry.MultiPoint(matched_points).centroid
        return matched_points_centroid, matched_ts_mean

    # Helper to check for timestamp collisions between successive firstpass points
    # as they are assigned, avoiding any risk of index misalignment between lists.
    def _check_fp_collision(cur_idx, prev_idx, offset, prev_offset):
        if prev_idx >= 0 and timestamps[-1] - timestamps[-2] <= 0:
            gt_prev = prev_idx
            gt_cur = cur_idx
            _, prev_a_time = get_centriod_and_ts(gt_prev, 1)
            _, prev_i_time = get_centriod_and_ts(gt_prev, 2)
            _, cur_a_time = get_centriod_and_ts(gt_cur, 1)
            _, cur_i_time = get_centriod_and_ts(gt_cur, 2)
            print("FP_COLLISION_DBG: firstpass collision found inside assignment loop!")
            print("  fp_pos %d->%d  gt_idx %d->%d  ts %.6f->%.6f (d=%.6g) offset=%.6g prev_offset=%.6g"
                  % (len(timestamps) - 2, len(timestamps) - 1, gt_prev, gt_cur,
                     timestamps[-2], timestamps[-1],
                     timestamps[-1] - timestamps[-2], offset, prev_offset))
            print("     groups_a_prev[%d]=%s  groups_a_prev_t=%.6f  groups_a[%d]=%s  groups_a_t=%.6f  (equal=%s)"
                  % (gt_prev, groups_a[gt_prev], prev_a_time, gt_cur, groups_a[gt_cur], cur_a_time,
                     groups_a[gt_prev] == groups_a[gt_cur]))
            print("     groups_i_prev[%d]=%s  groups_i_prev_t=%.6f  groups_i[%d]=%s  groups_i_t=%.6f  (equal=%s)"
                  % (gt_prev, groups_i[gt_prev], prev_i_time, gt_cur, groups_i[gt_cur], cur_i_time,
                     groups_i[gt_prev] == groups_i[gt_cur]))
            # Also show the gt indices skipped between the two firstpass points,
            # i.e. the second-pass run (if any) sitting between them.
            skipped = list(range(gt_prev + 1, gt_cur))
            print("     gt indices between them (second-pass run): %s" % skipped)

    #First pass
    points = []
    timestamps = []
    timeseries_ids = []
    # ranges = []
    offset = 0
    timeseries_id = 0 # 0 for dtw, 1 for android, 2 for ios
    prev_idx = -1
    prev_offset = 0
    for idx in firstpass_idxes:
        #Get unique elements
        matched_points_centroid_a, matched_ts_mean_a = get_centriod_and_ts(idx, 1)
        matched_points_centroid_i, matched_ts_mean_i = get_centriod_and_ts(idx, 2)

        #Outlier removal
        # ranges.append(max(matched_ts_a + matched_ts_i)-min(matched_ts_a + matched_ts_i))
        # if len(matched_points) == 2:
        # Decide which stream drives this firstpass point: the spatially closer
        # of android (1) / ios (2) when the two streams disagree in time by more
        # than the threshold, otherwise the combined dtw average (0).
        if abs(matched_ts_mean_a - matched_ts_mean_i) > time_threshold:
            if dtw.calDistance(gt_pts[idx], matched_points_centroid_a) > dtw.calDistance(gt_pts[idx], matched_points_centroid_i):
                chosen_id = 2
            else:
                chosen_id = 1
        else:
            chosen_id = 0

        # A concrete stream (android=1 / ios=2) is "constant" at this index when
        # its DTW match set is identical to the previous ground-truth index's,
        # i.e. it did not advance (a cone in that stream). Pinning a timestamp to
        # a constant stream through the continuity offset collapses it onto the
        # previous point's timestamp and produces a zero gap.
        def _series_constant(sid):
            return sid in (1, 2) and idx > 0 and (
                (groups_i if sid == 2 else groups_a)[idx] ==
                (groups_i if sid == 2 else groups_a)[idx - 1])

        # Guard 1: do not switch onto a stream that is constant here; stay on the
        # current (still forward-moving) stream instead.
        if chosen_id != timeseries_id and _series_constant(chosen_id):
            chosen_id = timeseries_id

        # Guard 2: if the selected stream is itself constant here, force the
        # combined dtw average (0), which keeps advancing as long as either
        # stream is still moving.
        if _series_constant(chosen_id):
            chosen_id = 0

        # Apply the continuity offset whenever the active stream changes.
        if chosen_id != timeseries_id:
            burn, prev_new = get_centriod_and_ts(idx - 1, chosen_id)
            burn, prev_old = get_centriod_and_ts(idx - 1, timeseries_id)
            offset += prev_old - prev_new
            timeseries_id = chosen_id

        if chosen_id == 1:
            ts = matched_ts_mean_a
        elif chosen_id == 2:
            ts = matched_ts_mean_i
        else:
            ts = (matched_ts_mean_a + matched_ts_mean_i) / 2
        # points.append(centroid)
        points.append(gt_pts[idx])
        timestamps.append(ts + offset)
        timeseries_ids.append(chosen_id)
        _check_fp_collision(idx, prev_idx, offset, prev_offset)
        prev_idx = idx
        prev_offset = offset
        #Average the postions and time stamps
    # print(np.histogram(ranges, bins=10))

    # High-precision local time axis, in seconds since `start_ts`. Kept in
    # lock-step with `timestamps` through the second pass. Differencing this
    # small-magnitude axis (instead of the ~1.5e9 absolute epoch) is what keeps
    # the finite-difference speed/acceleration/jerk free of roundoff noise.
    ts_local = [t - start_ts for t in timestamps]

    # Speed and acceleration at each firstpass point, used to shape the
    # second-pass transition curves. Computed now, while `points`/`timestamps`
    # still hold only the (mutually aligned) firstpass entries, and keyed by
    # ground-truth index through `firstpass_idxes`.
    fp_speed = {}
    if len(points) > 3:
        try:
            fp_gpdf = gpd.GeoDataFrame(data={"ts_local": list(ts_local)},
                                       geometry=list(points))
            speed_acceleration_jerk(fp_gpdf)
            for pos, gt_idx in enumerate(firstpass_idxes):
                fp_speed[gt_idx] = (float(fp_gpdf.speed.iloc[pos]),
                                    float(fp_gpdf.acceleration.iloc[pos]))
        except Exception as exp_fp:
            print("Could not compute firstpass speed/accel for second-pass "
                  "shaping, falling back to uniform: %s" % exp_fp)
            fp_speed = {}

    #Second pass
    for idx_list in secondpass_idxes:
        first_idx = idx_list[0]
        timeseries_id = timeseries_ids[first_idx-1]

        if first_idx >= len(timestamps):
            # End-of-trajectory run: there is no firstpass point after it. Fill
            # from the last placed point to `end_ts` at constant velocity. If
            # there is no room left (end_ts <= last stamp), fall back to the
            # legacy timestamps[-2] uniform scheme.
            start_stamp = timestamps[-1]
            if end_ts > start_stamp:
                start_pos = points[-1]
                rel_offsets = _constant_velocity_fill(
                    start_pos, [gt_pts[i] for i in idx_list], start_stamp, end_ts)
                base_local = start_stamp - start_ts
                for n in range(len(idx_list)):
                    points.insert(idx_list[n], gt_pts[idx_list[n]])
                    timestamps.insert(idx_list[n], start_stamp + rel_offsets[n])
                    ts_local.insert(idx_list[n], base_local + rel_offsets[n])
                    timeseries_ids.insert(idx_list[n], timeseries_id)
            else:
                first_stamp = timestamps[-2]
                last_stamp = timestamps[-1]
                step = (last_stamp - first_stamp) / (len(idx_list) + 1)
                if step == 0:
                    print("ZERO_STEP_DBG: end-of-trajectory else-branch: "
                          "timestamps[-2]=%.6f == timestamps[-1]=%.6f, "
                          "len(idx_list)=%d -> step=0; inserted points will "
                          "share a timestamp and cause ZeroDivisionError in "
                          "speed_acceleration_jerk" %
                          (first_stamp, last_stamp, len(idx_list)))
                timestamps[-1] = first_stamp + step
                ts_local[-1] = (first_stamp - start_ts) + step
                base_local = (first_stamp - start_ts) + step
                first_stamp += step
                for n in range(len(idx_list)):
                    points.insert(idx_list[n], gt_pts[idx_list[n]])
                    timestamps.insert(idx_list[n], first_stamp + ((n + 1) * step))
                    ts_local.insert(idx_list[n], base_local + ((n + 1) * step))
                    timeseries_ids.insert(idx_list[n], timeseries_id)
            continue

        # Interior run: build a constant-acceleration transition curve between
        # the bracketing firstpass points, with jerk-limited adjusting steps at
        # the entry and exit.
        first_stamp = timestamps[first_idx - 1]
        last_stamp = timestamps[first_idx]
        entry_pos = points[first_idx - 1]
        exit_pos = points[first_idx]
        v0, a0 = fp_speed.get(first_idx - 1, (np.nan, np.nan))
        v1, a1 = fp_speed.get(idx_list[-1] + 1, (np.nan, np.nan))
        # INTERIOR_WINDOW_DBG: the run is placed inside [first_stamp, last_stamp],
        # the two bracketing firstpass timestamps. If that window is zero/negative
        # the whole run collapses onto one instant. Report it so we can trace a
        # zero gap back to its bracket.
        if last_stamp - first_stamp <= 0:
            print("INTERIOR_WINDOW_DBG: run idx_list=%s bracketed by "
                  "timestamps[%d]=%.6f and timestamps[%d]=%.6f -> T=%.6g "
                  "(v0=%s v1=%s); run will collapse to a single timestamp"
                  % (idx_list, first_idx - 1, first_stamp, first_idx,
                     last_stamp, last_stamp - first_stamp, v0, v1))
        rel_offsets = _assign_secondpass_timestamps(
            entry_pos, [gt_pts[i] for i in idx_list], exit_pos,
            first_stamp, last_stamp, v0, a0, v1, a1, jerk_limit,
            debug_tag=first_idx)
        base_local = first_stamp - start_ts
        for n in range(len(idx_list)):
            points.insert(idx_list[n], gt_pts[idx_list[n]])
            timestamps.insert(idx_list[n], first_stamp + rel_offsets[n])
            ts_local.insert(idx_list[n], base_local + rel_offsets[n])
            timeseries_ids.insert(idx_list[n], timeseries_id)
            
            
    # Create DataFrame from collected points and timestamps
    if len(points) == 0:
        return gpd.GeoDataFrame()

    gpdf = gpd.GeoDataFrame(
        data={'ts': timestamps, 'ts_local': ts_local},
        # data={"ts": ts_fake},
        geometry=points
        # geometry=gt_pts
    )

    # Pre-flight check: scan ts_local for zero (or negative) consecutive gaps
    # that would cause ZeroDivisionError inside speed_acceleration_jerk. Raw
    # sensor time is strictly increasing at >=1s spacing, so any zero gap here
    # is manufactured by the second-pass insertion logic, not the input data.
    zero_gap_idxs = [i for i in range(1, len(ts_local))
                     if ts_local[i] - ts_local[i - 1] <= 0]
    if zero_gap_idxs:
        print("ZERO_GAP_DBG: found %d zero/negative ts_local gaps before "
              "speed_acceleration_jerk; indices (and surrounding values):" %
              len(zero_gap_idxs))
        for zi in zero_gap_idxs:
            lo = max(0, zi - 2)
            hi = min(len(ts_local), zi + 3)
            print("  gap at idx=%d: ts_local[%d:%d]=%s  ts[%d:%d]=%s" %
                  (zi, lo, hi, ts_local[lo:hi], lo, hi, timestamps[lo:hi]))

    speed_acceleration_jerk(gpdf)
    matching = []


    # sensor_coneness = [0, 0]
    # gt_coneness = [0, 0]
    # prev_matched_a = []
    # a_streak = 1
    # a_count_gt = 0
    # a_count_sens = 0
    # prev_matched_i = []
    # i_streak = 1
    # i_count_gt = 0
    # i_count_sens = 0
    # for idx in range(len(firstpass_idxes)):
    #     match = [gt_pts[firstpass_idxes[idx]]]
    #     if timeseries_ids[idx] == 1 or timeseries_ids[idx] == 0:
    #         a_count_gt += 1
    #         matched_a = [a_pts_seq[a_pts] for a_pts in groups_a[firstpass_idxes[idx]]]
    #         a_count_sens += len(matched_a)
    #         match += matched_a
    #         sensor_coneness[0] += len(matched_a)**2
    #         if matched_a == prev_matched_a:
    #             a_streak += 1
    #         else:
    #             gt_coneness[0] += a_streak**2
    #             a_streak = 1
    #         prev_matched_a = matched_a
    #     if timeseries_ids[idx] == 2 or timeseries_ids[idx] == 0:
    #         i_count_gt += 1
    #         matched_i = [i_pts_seq[i_pts] for i_pts in groups_i[firstpass_idxes[idx]]]
    #         i_count_sens += len(matched_i)
    #         match += matched_i
    #         sensor_coneness[1] += len(matched_i)**2
    #         if matched_i == prev_matched_i:
    #             i_streak += 1
    #         else:
    #             gt_coneness[1] += i_streak**2
    #             i_streak = 1
    #         prev_matched_i = matched_i

    # The second pass collapses runs of consecutive ground truth points that DTW
    # mapped to the identical sensor point(s); those run members duplicate the
    # cone fan of the firstpass point that precedes them. Collapse the matching
    # so the cone is represented once: emit the fan only on the owning firstpass
    # point and give the redundant run members just their own ground truth point.
    secondpass_set = set()
    for idx_list in secondpass_idxes:
        secondpass_set.update(idx_list)

    for m in range(len(gt_pts)):
        match = [gt_pts[m]]
        if m not in secondpass_set:
            if timeseries_ids[m] == 1 or timeseries_ids[m] == 0:
                match += [a_pts_seq[a_pts] for a_pts in groups_a[m]]
            if timeseries_ids[m] == 2 or timeseries_ids[m] == 0:
                match += [i_pts_seq[i_pts] for i_pts in groups_i[m]]
        matching.append(match)
    gpdf['matching'] = matching
    gpdf['timeseries_id'] = timeseries_ids
    gpdf['longitude'] = gpdf.geometry.x
    gpdf['latitude'] = gpdf.geometry.y
    gpdf["fmt_time"] = gpdf.ts.apply(lambda ts: arrow.get(ts).to(tz))
    assert len(gpdf[gpdf.geometry.isnull()]) == 0, "Found %d null entries out of %d total" % (len(gpdf.geometry.isnull()), len(gpdf))
    return gpdf

def ref_dtw_gt_with_ends_no_second_pass(e, tz="UTC", points_per_second=1, interp=2):
    fill_gt_linestring(e)
    a_pts = emd.to_geo_df(e["temporal_control"]["android"]["location_df"])
    i_pts = emd.to_geo_df(e["temporal_control"]["ios"]["location_df"])
    if interp >= 1:
        new_a_pts = get_int_aligned_trajectory(a_pts, tz, True, True)
        new_i_pts = get_int_aligned_trajectory(i_pts, tz, True, True)
    else:
        new_a_pts = a_pts
        new_i_pts = i_pts
    a_pts_seq = new_a_pts["geometry"].to_list()
    i_pts_seq = new_i_pts["geometry"].to_list()

    start_ts = min(new_a_pts["ts"].iloc[0], new_i_pts["ts"].iloc[0])
    end_ts = max(new_a_pts["ts"].iloc[-1], new_i_pts["ts"].iloc[-1])


    # Get points at 1 second intervals along the ground truth linestring
    if interp == 0 or interp == 2:
        gt_pts = interpolate_points_along_linestring(e["ground_truth"]["linestring"], time_interval=(end_ts-start_ts), points_per_second=points_per_second)
    else:
        gt_pts = [shp.geometry.Point(coord) for coord in list(e["ground_truth"]["linestring"].coords)]
    
    # print("In ref_ct_general, %s" % section_gt_shapes.filter(items=["start_loc","end_loc"]))
    
    d_a = dtw.Dtw(gt_pts, a_pts_seq, dtw.calDistance)
    d_a.calculate()
    mapping_a = d_a.get_path()

    d_i = dtw.Dtw(gt_pts, i_pts_seq, dtw.calDistance)
    d_i.calculate()
    mapping_i = d_i.get_path()

    groups_a = []
    a_idx = len(mapping_a) - 1   
    groups_i = []
    i_idx = len(mapping_i) - 1
    for idx in range(len(gt_pts)):
        group_a = []
        group_i = []
        while a_idx >= 0 and mapping_a[a_idx][0] == idx:
            group_a.append(mapping_a[a_idx][1])
            a_idx -= 1
        while i_idx >= 0 and mapping_i[i_idx][0] == idx:
            group_i.append(mapping_i[i_idx][1])
            i_idx -= 1
        groups_a.append(group_a)
        groups_i.append(group_i)
    
    # print("After DTW, retained %d of %d (%s) for android and %d of %d (%s) for ios" %
    #       (len(groups_a), max(len(a_pts), len(i_pts)),
    #         (len(groups_a)/max(len(a_pts), len(i_pts))),
    #         len(groups_i), max(len(a_pts), len(i_pts)),
    #         (len(groups_i)/max(len(a_pts), len(i_pts)))))
    #Average the postions and time stampsof each unique element in each group

    
    
    def get_centriod_and_ts(idx, timeseries_id):
        if timeseries_id == 0:
            centroid_a, ts_a = get_centriod_and_ts(idx, 1)
            centroid_i, ts_i = get_centriod_and_ts(idx, 2)
            return shp.geometry.MultiPoint([centroid_a, centroid_i]).centroid, (ts_a + ts_i)/2
        if timeseries_id == 1:
            groups = groups_a
            pts = new_a_pts
        elif timeseries_id == 2:
            groups = groups_i
            pts = new_i_pts

        unique_elements = set()
        for pt in groups[idx]:
            unique_elements.add(pt)
        df = pts.iloc[list(unique_elements)]
        matched_points = df["geometry"].to_list()
        matched_ts = df["ts"].to_list()
        matched_ts_mean = np.mean(matched_ts)
        matched_points_centroid = shp.geometry.MultiPoint(matched_points).centroid
        return matched_points_centroid, matched_ts_mean

    #First pass
    points = []
    timestamps = []
    timeseries_ids = []
    # ranges = []
    offset = 0
    timeseries_id = 0 # 0 for dtw, 1 for android, 2 for ios
    for idx in range(len(gt_pts)):
        #Get unique elements
        matched_points_centroid_a, matched_ts_mean_a = get_centriod_and_ts(idx, 1)
        matched_points_centroid_i, matched_ts_mean_i = get_centriod_and_ts(idx, 2)

        #Outlier removal
        # ranges.append(max(matched_ts_a + matched_ts_i)-min(matched_ts_a + matched_ts_i))
        # if len(matched_points) == 2:
        if abs(matched_ts_mean_a - matched_ts_mean_i) > 300:
            if dtw.calDistance(gt_pts[idx], matched_points_centroid_a) > dtw.calDistance(gt_pts[idx], matched_points_centroid_i):
                # points.append(matched_points_i_centroid)
                if timeseries_id != 2:
                    burn, prev_new = get_centriod_and_ts(idx-1, 2)
                    burn, prev_old = get_centriod_and_ts(idx-1, timeseries_id)
                    offset += prev_old - prev_new
                    timeseries_id = 2

                points.append(gt_pts[idx])
                timestamps.append(matched_ts_mean_i + offset)
                timeseries_ids.append(2)
                continue
            else:
                # points.append(matched_points_a_centroid)
                if timeseries_id != 1:
                    burn, prev_new = get_centriod_and_ts(idx-1, 1)
                    burn, prev_old = get_centriod_and_ts(idx-1, timeseries_id)
                    offset += prev_old - prev_new
                    timeseries_id = 1
                points.append(gt_pts[idx])
                timestamps.append(matched_ts_mean_a + offset)
                timeseries_ids.append(1)
                continue
        # if len(matched_points) > 2:
        #     first = np.percentile(matched_ts, 25)
        #     third = np.percentile(matched_ts, 75)
        #     iqr = third - first
        #     outliers = [pt for pt in matched_ts if pt < (first - iqr) or pt > (third + iqr)]
        #     for pt in outliers:
        #         matched_points.remove(matched_points[matched_ts.index(pt)])
        #         matched_ts.remove(pt)
            
            
        

        #Average remaining points
        if timeseries_id != 0:
            burn, prev_new = get_centriod_and_ts(idx-1, 0)
            burn, prev_old = get_centriod_and_ts(idx-1, timeseries_id)
            offset += prev_old - prev_new
            timeseries_id = 0
        # centroid = shp.geometry.MultiPoint([matched_points_centroid_a, matched_points_centroid_i]).centroid
        ts = (matched_ts_mean_a + matched_ts_mean_i) / 2
        # points.append(centroid)
        points.append(gt_pts[idx])
        timestamps.append(ts + offset)
        timeseries_ids.append(0)
        #Average the postions and time stamps
    # print(np.histogram(ranges, bins=10))

    # Create DataFrame from collected points and timestamps
    if len(points) == 0:
        return gpd.GeoDataFrame()

    gpdf = gpd.GeoDataFrame(
        data={'ts': timestamps},
        # data={"ts": ts_fake},
        geometry=points
        # geometry=gt_pts
    )

    speed_acceleration_jerk(gpdf)
    matching = []

    # Because there is no second pass, the cone runs (consecutive ground truth
    # points whose DTW matches converge on the same sensor point(s)) are never
    # collapsed. Detect those runs and fan every ground truth point in a run out
    # to ALL sensor points shared across the entire run, so the un-collapsed cone
    # is represented in the matching as a dense fan rather than a per-point match.
    def cone_run_unions(groups):
        unions = [set(groups[idx]) for idx in range(len(groups))]

        def flush(run):
            if run is None:
                return
            union = set().union(*(set(groups[j]) for j in run))
            for j in run:
                unions[j] = union

        run = None
        for idx in range(len(groups)):
            if len(groups[idx]) == 0:
                flush(run)
                run = None
                continue
            if run is not None and (set(groups[idx]) & set(groups[run[-1]])):
                run.append(idx)
            else:
                flush(run)
                run = [idx]
        flush(run)
        return unions

    a_cone_unions = cone_run_unions(groups_a)
    i_cone_unions = cone_run_unions(groups_i)

    for m in range(len(gt_pts)):
        match = [gt_pts[m]]
        if timeseries_ids[m] == 1 or timeseries_ids[m] == 0:
            match += [a_pts_seq[a_pts] for a_pts in sorted(a_cone_unions[m])]
        if timeseries_ids[m] == 2 or timeseries_ids[m] == 0:
            match += [i_pts_seq[i_pts] for i_pts in sorted(i_cone_unions[m])]
        matching.append(match)
    gpdf['matching'] = matching
    gpdf['timeseries_id'] = timeseries_ids
    gpdf['longitude'] = gpdf.geometry.x
    gpdf['latitude'] = gpdf.geometry.y
    gpdf["fmt_time"] = gpdf.ts.apply(lambda ts: arrow.get(ts).to(tz))
    assert len(gpdf[gpdf.geometry.isnull()]) == 0, "Found %d null entries out of %d total" % (len(gpdf.geometry.isnull()), len(gpdf))
    return gpdf

def speed_acceleration_jerk(gpdf):
    """
    Calculate and add speed, acceleration, and jerk for a given GeoDataFrame.

    Time differences are taken on a small-magnitude local axis rather than on
    the raw ~1.5e9 absolute Unix-epoch `ts`. Differencing the absolute epoch
    directly quantizes sub-second offsets onto a ~2.4e-7 s grid, and the finite
    differences then amplify that roundoff by orders of magnitude (visible as
    spurious acceleration wobble and exploding jerk). If a high-precision
    `ts_local` column is present it is used as-is; otherwise the absolute `ts`
    is shifted by its first value, which at least avoids re-quantizing here.
    """
    if "ts_local" in gpdf.columns:
        t = list(gpdf["ts_local"])
    else:
        ts = list(gpdf.ts)
        t0 = ts[0] if len(ts) else 0.0
        t = [x - t0 for x in ts]
    skip = 0
    speed = []
    acceleration = []
    jerk = []
    for idx in range(1, len(gpdf)):
        dist = dtw.calDistance(gpdf.iloc[idx].geometry, gpdf.iloc[idx-1].geometry)
        speed.append(dist / (t[idx] - t[idx-1]))
        if skip > 0:
            acceleration.append((speed[idx-1] - speed[idx-2]) / (t[idx-1] - t[idx-2]))
            if skip > 1:
                jerk.append((acceleration[idx-2] - acceleration[idx-3]) / (t[idx-2] - t[idx-3]))
            else:
                skip += 1
        else:
            skip += 1
    speed.append(0)
    acceleration.append((speed[-1] - speed[-2]) / (t[-1] - t[-2]))
    acceleration.append(0)
    jerk.append((acceleration[-2] - acceleration[-3]) / (t[-2] - t[-3]))
    jerk.append((acceleration[-1] - acceleration[-2]) / (t[-1] - t[-2]))
    jerk.append(0)
    gpdf["speed"] = speed
    gpdf["acceleration"] = acceleration
    gpdf["jerk"] = jerk


def ref_travel_forward(e, dist_threshold, tz="UTC", include_ends=False):
    fill_gt_linestring(e)
    section_gt_shapes = e["ground_truth"]["gt_shapes"]
    # print(f"GEO_DF: before filtering, {len(e['temporal_control']['android']['location_df'])=} and {len(e['temporal_control']['ios']['location_df'])=}")
    filtered_utm_loc_df_a = emd.filter_geo_df(
        emd.to_geo_df(e["temporal_control"]["android"]["location_df"]),
        section_gt_shapes.filter(["start_loc","end_loc"]))
    filtered_utm_loc_df_b = emd.filter_geo_df(
        emd.to_geo_df(e["temporal_control"]["ios"]["location_df"]),
        section_gt_shapes.filter(["start_loc","end_loc"]))
    # print(f"GEO_DF: after filtering, {len(filtered_utm_loc_df_a)=} and {len(filtered_utm_loc_df_b)=}")
    new_location_df_a = get_int_aligned_trajectory(filtered_utm_loc_df_a, tz)
    new_location_df_i = get_int_aligned_trajectory(filtered_utm_loc_df_b, tz)

    utm_gt_linestring = e["ground_truth"]["utm_linestring"]

    new_location_df_ua = emd.to_utm_df(new_location_df_a)
    new_location_df_ui = emd.to_utm_df(new_location_df_i)

    add_gt_error_projection(new_location_df_ua, utm_gt_linestring)
    add_gt_error_projection(new_location_df_ui, utm_gt_linestring)

    new_location_df_a["gt_distance"] = new_location_df_ua.gt_distance
    new_location_df_a["gt_projection"] = new_location_df_ua.gt_projection

    new_location_df_i["gt_distance"] = new_location_df_ui.gt_distance
    new_location_df_i["gt_projection"] = new_location_df_ui.gt_projection

    new_location_df_a["gt_cum_proj"] = new_location_df_a.gt_projection.cumsum()
    new_location_df_i["gt_cum_proj"] = new_location_df_i.gt_projection.cumsum()

    filtered_location_df_a = new_location_df_a.query("gt_distance < @dist_threshold")
    filtered_location_df_i = new_location_df_i.query("gt_distance < @dist_threshold")
    print("After filtering, %d of %d (%s) for android and %d of %d (%s) for ios" %
          (len(filtered_location_df_a), len(new_location_df_a), (len(filtered_location_df_a)/len(new_location_df_a)),
           len(filtered_location_df_i), len(new_location_df_i), (len(filtered_location_df_i)/len(new_location_df_i))))
    merged_df = pd.merge(filtered_location_df_a, filtered_location_df_i, on="ts",
        how="outer", suffixes=("_a", "_i")).sort_values(by="ts", axis="index")
    merge_fn = make_collapse_outer_join_dist_so_far(more_details_fn = None)
    initial_reference_gpdf = gpd.GeoDataFrame(list(merged_df.apply(merge_fn, axis=1)))
    if include_ends:
        [start_initial_ends_gpdf, end_initial_ends_gpdf] = ref_ends(e, dist_threshold, tz)
        print(f"CONCAT: {include_ends=}, before concatenating {len(start_initial_ends_gpdf)=}, {len(initial_reference_gpdf)=}, {len(end_initial_ends_gpdf)=}")
        initial_reference_gpdf = pd.concat([start_initial_ends_gpdf, initial_reference_gpdf, end_initial_ends_gpdf], axis=0).sort_values(by="ts").reset_index(drop=True)
        print(f"CONCAT: {include_ends=}, after concatenating {len(initial_reference_gpdf)=}")
    if len(initial_reference_gpdf.columns) > 1:
        initial_reference_gpdf["fmt_time"] = initial_reference_gpdf.ts.apply(lambda ts: arrow.get(ts).to(tz))
        reference_gpdf = initial_reference_gpdf[initial_reference_gpdf.latitude.notnull()]
        print("After merging, found %d / %d of android %d (%s), ios %d (%s)" %
              (len(reference_gpdf), len(initial_reference_gpdf), len(new_location_df_a), (len(reference_gpdf)/len(new_location_df_a)),
               len(new_location_df_i), (len(reference_gpdf)/len(new_location_df_i))))
        assert len(reference_gpdf[reference_gpdf.latitude.isnull()]) == 0, "Found %d null entries out of %d total" % (len(reference_gpdf[reference_gpdf.latitude.isnull()]), len(initial_reference_gpdf))
        return reference_gpdf
    else:
        return gpd.GeoDataFrame()


####
# BEGIN: Single-stream reference constructions
# These build a reference from a SINGLE device stream (android or ios) instead
# of merging both. They are the single-stream analogues of `ref_travel_forward`
# and `ref_dtw_gt_with_ends_general`, plus a raw (no-interpolation) reference.
####

def _forward_progress_filter(location_df):
    """
    Keep only the points whose projection along the ground truth linestring is
    strictly increasing, i.e. the single-stream analogue of the forward-progress
    check in `make_collapse_outer_join_dist_so_far`. This drops points that
    would make the reference travel backwards along the ground truth.
    """
    distance_so_far = 0
    keep = []
    for i in range(len(location_df)):
        proj = location_df.gt_projection.iloc[i]
        if proj > distance_so_far:
            keep.append(True)
            distance_so_far = proj
        else:
            keep.append(False)
    return location_df[pd.Series(keep, index=location_df.index)]

def ref_ends_single(e, dist_threshold, device, tz="UTC"):
    """
    Single-stream analogue of `ref_ends`: match the start/end location points of
    a single device to the ground truth, retaining only those within
    `dist_threshold` of the ground truth linestring.
    """
    utm_gt_linestring = e["ground_truth"]["utm_linestring"]
    section_gt_shapes = e["ground_truth"]["gt_shapes"]

    def _get_filtered_loc(gt_key):
        unfiltered = emd.to_geo_df(e["temporal_control"][device]["location_df"]).copy()
        emd.filter_geo_df(unfiltered, section_gt_shapes.filter([gt_key]))
        return unfiltered.query("outside_polygons==False")

    def _match_single_to_gt(filtered_loc_df):
        if len(filtered_loc_df) < 2:
            return gpd.GeoDataFrame([])
        new_location_df = get_int_aligned_trajectory(filtered_loc_df, tz)
        new_location_df_u = emd.to_utm_df(new_location_df)
        add_gt_error_projection(new_location_df_u, utm_gt_linestring)
        new_location_df["gt_distance"] = new_location_df_u.gt_distance
        new_location_df["gt_projection"] = new_location_df_u.gt_projection
        filtered = new_location_df.query("gt_distance < @dist_threshold")
        if len(filtered) == 0:
            return gpd.GeoDataFrame([])
        filtered = gpd.GeoDataFrame(filtered).copy()
        filtered["source"] = device
        return filtered

    start_ends = _match_single_to_gt(_get_filtered_loc("start_loc"))
    end_ends = _match_single_to_gt(_get_filtered_loc("end_loc"))
    return [start_ends, end_ends]

def ref_travel_forward_single(e, dist_threshold, device, tz="UTC", include_ends=False):
    """
    Single-stream travel-forward reference. Uses only `device`'s trajectory,
    filters to points close to the ground truth, and keeps only points that
    make forward progress along the ground truth.
    """
    fill_gt_linestring(e)
    section_gt_shapes = e["ground_truth"]["gt_shapes"]
    filtered_loc_df = emd.filter_geo_df(
        emd.to_geo_df(e["temporal_control"][device]["location_df"]),
        section_gt_shapes.filter(["start_loc", "end_loc"]))
    new_location_df = get_int_aligned_trajectory(filtered_loc_df, tz)

    utm_gt_linestring = e["ground_truth"]["utm_linestring"]
    new_location_df_u = emd.to_utm_df(new_location_df)
    add_gt_error_projection(new_location_df_u, utm_gt_linestring)
    new_location_df["gt_distance"] = new_location_df_u.gt_distance
    new_location_df["gt_projection"] = new_location_df_u.gt_projection

    filtered_location_df = new_location_df.query("gt_distance < @dist_threshold")
    print("SINGLE TF (%s): after gt_distance filter, retained %d of %d" %
          (device, len(filtered_location_df), len(new_location_df)))

    reference_gpdf = gpd.GeoDataFrame(_forward_progress_filter(filtered_location_df)).copy()
    reference_gpdf["source"] = device

    if include_ends:
        [start_ends, end_ends] = ref_ends_single(e, dist_threshold, device, tz)
        reference_gpdf = pd.concat([start_ends, reference_gpdf, end_ends], axis=0).sort_values(by="ts").reset_index(drop=True)

    reference_gpdf = reference_gpdf[reference_gpdf.geometry.notnull()]
    if len(reference_gpdf) > 1 and len(reference_gpdf.columns) > 1:
        reference_gpdf = reference_gpdf.drop_duplicates(subset="ts", keep="first").sort_values(by="ts").reset_index(drop=True)
        reference_gpdf["fmt_time"] = reference_gpdf.ts.apply(lambda ts: arrow.get(ts).to(tz))
        return gpd.GeoDataFrame(reference_gpdf)
    else:
        return gpd.GeoDataFrame()

def _spread_equal_timestamps(timestamps):
    """
    Make a non-decreasing timestamp list strictly increasing by linearly
    spreading runs of identical timestamps up to the next distinct value. This
    avoids division-by-zero when computing speed/acceleration/jerk for
    single-stream DTW references where many ground truth points can map to the
    same device point (and therefore share a timestamp).
    """
    ts = list(timestamps)
    n = len(ts)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ts[j + 1] == ts[i]:
            j += 1
        if j > i:
            nxt = ts[j + 1] if j + 1 < n else (ts[i] + (j - i + 1))
            start = ts[i]
            step = (nxt - start) / (j - i + 1)
            for k in range(i, j + 1):
                ts[k] = start + (k - i) * step
        i = j + 1
    return ts

def ref_dtw_gt_single(e, device, tz="UTC", points_per_second=1, interp=2):
    """
    Single-stream DTW reference. Runs DTW between ground truth points and a
    single device's trajectory, then for each ground truth point uses the mean
    timestamp of the matched device points. The reference geometry follows the
    ground truth points (consistent with `ref_dtw_gt_with_ends_general`).
    """
    fill_gt_linestring(e)
    pts = emd.to_geo_df(e["temporal_control"][device]["location_df"])
    if interp >= 1:
        new_pts = get_int_aligned_trajectory(pts, tz, True, True)
    else:
        new_pts = pts
    pts_seq = new_pts["geometry"].to_list()
    if len(pts_seq) < 2:
        return gpd.GeoDataFrame()

    start_ts = new_pts["ts"].iloc[0]
    end_ts = new_pts["ts"].iloc[-1]

    if interp == 0 or interp == 2:
        gt_pts = interpolate_points_along_linestring(e["ground_truth"]["linestring"], time_interval=(end_ts - start_ts), points_per_second=points_per_second)
    else:
        gt_pts = [shp.geometry.Point(coord) for coord in list(e["ground_truth"]["linestring"].coords)]

    d = dtw.Dtw(gt_pts, pts_seq, dtw.calDistance)
    d.calculate()
    mapping = d.get_path()

    groups = []
    m_idx = len(mapping) - 1
    for idx in range(len(gt_pts)):
        group = []
        while m_idx >= 0 and mapping[m_idx][0] == idx:
            group.append(mapping[m_idx][1])
            m_idx -= 1
        groups.append(group)

    points = []
    timestamps = []
    matching = []
    for idx in range(len(gt_pts)):
        unique_elements = sorted(set(groups[idx]))
        if len(unique_elements) == 0:
            continue
        df = new_pts.iloc[unique_elements]
        matched_ts_mean = float(np.mean(df["ts"].to_list()))
        points.append(gt_pts[idx])
        timestamps.append(matched_ts_mean)
        matching.append([gt_pts[idx]] + [pts_seq[p] for p in unique_elements])

    if len(points) == 0:
        return gpd.GeoDataFrame()

    timestamps = _spread_equal_timestamps(timestamps)

    gpdf = gpd.GeoDataFrame(data={'ts': timestamps}, geometry=points)
    speed_acceleration_jerk(gpdf)
    gpdf['matching'] = matching
    gpdf['longitude'] = gpdf.geometry.x
    gpdf['latitude'] = gpdf.geometry.y
    gpdf['source'] = device
    gpdf["fmt_time"] = gpdf.ts.apply(lambda ts: arrow.get(ts).to(tz))
    assert len(gpdf[gpdf.geometry.isnull()]) == 0, "Found %d null entries out of %d total" % (len(gpdf.geometry.isnull()), len(gpdf))
    return gpdf

def raw_reference(e, device, tz="UTC"):
    """
    Use a single device's completely raw GPS trajectory (no interpolation, no
    ground-truth filtering) as the reference. This is the baseline against which
    the constructed references are compared.
    """
    fill_gt_linestring(e)
    raw_df = emd.to_geo_df(e["temporal_control"][device]["location_df"]).copy()
    if len(raw_df) < 2:
        return gpd.GeoDataFrame()
    gpdf = gpd.GeoDataFrame(raw_df).sort_values(by="ts").reset_index(drop=True)
    gpdf["longitude"] = gpdf.geometry.x
    gpdf["latitude"] = gpdf.geometry.y
    gpdf["source"] = device
    gpdf["fmt_time"] = gpdf.ts.apply(lambda ts: arrow.get(ts).to(tz))
    return gpdf

def spatio_temporal_error(two_stream_df, other_df):
    """
    Spatio-temporal error of `other_df` relative to the 2-stream (DTW) reference
    `two_stream_df`. For each point in `two_stream_df`, find the closest point in
    `other_df` using

        sqrt(spatial_dist^2 + (speed_at_point * time_diff)^2)

    where `spatial_dist` is in meters, `speed_at_point` is the speed of the
    2-stream reference point being iterated, and `time_diff = |ts_ref - ts_other|`.
    The per-point minimums are averaged to produce the final error.
    """
    if two_stream_df is None or other_df is None or len(two_stream_df) == 0 or len(other_df) == 0:
        return np.nan

    if "speed" not in two_stream_df.columns:
        two_stream_df = two_stream_df.copy()
        speed_acceleration_jerk(two_stream_df)

    ref_utm = emd.to_utm_series(gpd.GeoSeries(list(two_stream_df.geometry)))
    other_utm = emd.to_utm_series(gpd.GeoSeries(list(other_df.geometry)))
    other_x = np.array([p.x for p in other_utm])
    other_y = np.array([p.y for p in other_utm])
    other_ts = np.array(list(other_df.ts), dtype=float)

    ref_ts = list(two_stream_df.ts)
    ref_speed = list(two_stream_df.speed)

    errors = []
    for k, p in enumerate(ref_utm):
        spatial = np.sqrt((other_x - p.x) ** 2 + (other_y - p.y) ** 2)
        dt = np.abs(other_ts - float(ref_ts[k]))
        speed_k = ref_speed[k]
        if pd.isnull(speed_k):
            speed_k = 0.0
        combined = np.sqrt(spatial ** 2 + (speed_k * dt) ** 2)
        errors.append(combined.min())
    return float(np.mean(errors))

####
# END: Single-stream reference constructions
####


####
# END: Combining into actual reference constructions
####


####
# BEGIN: Final ensemble reference construction that uses ground truth
# - if the ground truth is simple, use the `travel_forward`
# - if the ground truth is complex, use trajectory-only with midpoint
# - we leave the threshold as a parameter, defaulting to 25, which seems to
# work pretty well in the evaluation
####

coverage_density = lambda df, sr: len(df)/(sr["end_ts"] - sr["start_ts"])
coverage_time = lambda df, sr: (df.ts.iloc[-1] - df.ts.iloc[0])/(sr["end_ts"] - sr["start_ts"])
coverage_max_gap = lambda df, sr: df.ts.diff().max()/(sr["end_ts"] - sr["start_ts"])
max_jerk = lambda df, sr: df.jerk.abs().max()
max_acceleration = lambda df, sr: df.acceleration.abs().max()
max_speed = lambda df, sr: df.speed.abs().max()
median_jerk = lambda df, sr: df.jerk.abs().median()
mean_median_jerk_ratio = lambda df, sr: df.jerk.abs().median()/df.jerk.abs().mean()



def final_ref_ensemble(e, dist_threshold=25, tz="UTC", include_ends=False):
    fill_gt_linestring(e)
    gt_linestring = e["ground_truth"]["linestring"]
    try:
        tf_ref_df = ref_travel_forward(e, dist_threshold, tz, include_ends)
        speed_acceleration_jerk(tf_ref_df)
        tf_stats = {
            "coverage_density": coverage_density(tf_ref_df, e),
            "coverage_time": coverage_time(tf_ref_df, e),
            "coverage_max_gap": coverage_max_gap(tf_ref_df, e),
            "max_jerk": max_jerk(tf_ref_df, e),
            "max_acceleration": max_acceleration(tf_ref_df, e),
            "max_speed": max_speed(tf_ref_df, e),
            "median_jerk": median_jerk(tf_ref_df, e),
            "mean_median_jerk_ratio": mean_median_jerk_ratio(tf_ref_df, e)
        }
        print("Validated tf, stats are %s" % tf_stats)
    except Exception as exp_tf:
        print("Found exception %s while computing tf_ref_df, skipping" % exp_tf)
        traceback.print_exc()
        tf_stats = None

    try:
        ct_ref_df = ref_ct_general(e, b_merge_midpoint, dist_threshold, tz, include_ends)
        speed_acceleration_jerk(ct_ref_df)
        ct_stats = {
            "coverage_density": coverage_density(ct_ref_df, e),
            "coverage_time": coverage_time(ct_ref_df, e),
            "coverage_max_gap": coverage_max_gap(ct_ref_df, e),
            "max_jerk": max_jerk(ct_ref_df, e),
            "max_acceleration": max_acceleration(ct_ref_df, e),
            "max_speed": max_speed(ct_ref_df, e),
            "median_jerk": median_jerk(ct_ref_df, e),
            "mean_median_jerk_ratio": mean_median_jerk_ratio(ct_ref_df, e)
        }
        print("Validated ct, stats are %s" % ct_stats)
    except Exception as exp_ct:
        print("Found exception %s while computing ct_ref_df, skipping" % exp_ct)
        traceback.print_exc()
        ct_stats = None

    # try:
    #     dtw_ref_df = ref_dtw_gt_with_ends_general(e, tz)
    #     dtw_stats = {
    #         "coverage_density": coverage_density(dtw_ref_df, e),
    #         "coverage_time": coverage_time(dtw_ref_df, e),
    #         "coverage_max_gap": coverage_max_gap(dtw_ref_df, e),
    #         "max_jerk": max_jerk(dtw_ref_df, e),
    #         "max_acceleration": max_acceleration(dtw_ref_df, e),
    #         "max_speed": max_speed(dtw_ref_df, e),
    #         "median_jerk": median_jerk(dtw_ref_df, e),
    #         "mean_median_jerk_ratio": mean_median_jerk_ratio(dtw_ref_df, e)
    #     }
    #     print("Validated dtw, stats are %s" % dtw_stats)
    # except Exception as exp_dtw:
    #     print("Found exception %s while computing dtw_ref_df, skipping" % exp_dtw)
    #     traceback.print_exc()
    #     dtw_stats = None

    if tf_stats is None and ct_stats is None:
        assert False, "Neither method works!"
    elif tf_stats is None and ct_stats is not None:
        return ("ct", ct_ref_df)
    elif tf_stats is not None and ct_stats is None:
        return ("tf", tf_ref_df)

    assert tf_stats is not None and ct_stats is not None

    if tf_stats["coverage_max_gap"] > ct_stats["coverage_max_gap"] and\
        tf_stats["coverage_density"] < ct_stats["coverage_density"]:
        print("max_gap for tf = %s > ct = %s and density %s < %s, returning ct len = %d not tf len = %d" %
            (tf_stats["coverage_max_gap"], ct_stats["coverage_max_gap"],
             tf_stats["coverage_density"], ct_stats["coverage_density"],
             len(ct_ref_df), len(tf_ref_df)))
        return ("ct", ct_ref_df)
    else:
        print("for tf = %s v/s ct = %s, density %s v/s %s, returning tf len = %d not cf len = %d" %
            (tf_stats["coverage_max_gap"], ct_stats["coverage_max_gap"],
             tf_stats["coverage_density"], ct_stats["coverage_density"],
             len(tf_ref_df), len(ct_ref_df)))
        return ("tf", tf_ref_df)

####
# END: Final ensemble reference construction that uses ground truth
####

def ref_and_stats(e, function, dist_threshold=25, tz="UTC", include_ends=False, device=None, time_threshold=300):
    fill_gt_linestring(e)
    gt_linestring = e["ground_truth"]["linestring"]

    def stats_gen(ref_df, e):
        speed_acceleration_jerk(ref_df)
        stats = {
            "coverage_density": coverage_density(ref_df, e),
            "coverage_time": coverage_time(ref_df, e),
            "coverage_max_gap": coverage_max_gap(ref_df, e),
            "max_jerk": max_jerk(ref_df, e),
            "max_acceleration": max_acceleration(ref_df, e),
            "max_speed": max_speed(ref_df, e),
            "median_jerk": median_jerk(ref_df, e),
            "mean_median_jerk_ratio": mean_median_jerk_ratio(ref_df, e)
        }
        # Spatial (ground truth) error: average distance of the reference from
        # the ground truth linestring. Moved here from the notebook so every
        # reference variant gets a consistent `gt_error` in its stats.
        try:
            stats["gt_error"] = emd.dist_using_projection_adjusted(ref_df, gt_linestring)
        except Exception as exp_gt:
            print("Found exception %s while computing gt_error" % exp_gt)
            stats["gt_error"] = np.nan
        return stats
    
    if function == 'tf':
        try:
            ref_df = ref_travel_forward(e, dist_threshold, tz, include_ends)
            stats = stats_gen(ref_df, e)
        except Exception as exp_tf:
            print("Found exception %s while computing tf_ref_df, skipping" % exp_tf)
            traceback.print_exc()
            stats = None
    elif function == 'ct':
        try:
            ref_df = ref_ct_general(e, b_merge_midpoint, dist_threshold, tz, include_ends)
            stats = stats_gen(ref_df, e)
        except Exception as exp_ct:
            print("Found exception %s while computing ct_ref_df, skipping" % exp_ct)
            traceback.print_exc()
            stats = None
    elif function == 'dtw':
        try:
            ref_df = ref_dtw_gt_with_ends_general(e, tz, time_threshold=time_threshold)
            stats = stats_gen(ref_df, e)
        except Exception as exp_dtw:
            print("Found exception %s while computing dtw_ref_df, skipping" % exp_dtw)
            traceback.print_exc()
            stats = None
    elif function == 'dtw_no_collapse':
        try:
            ref_df = ref_dtw_gt_with_ends_no_second_pass(e, tz)
            stats = stats_gen(ref_df, e)
        except Exception as exp_dtw:
            print("Found exception %s while computing dtw_no_collapse_ref_df, skipping" % exp_dtw)
            traceback.print_exc()
            stats = None
    elif function == 'tf_single':
        assert device is not None, "tf_single requires a device ('android' or 'ios')"
        try:
            ref_df = ref_travel_forward_single(e, dist_threshold, device, tz, include_ends)
            stats = stats_gen(ref_df, e)
        except Exception as exp_tfs:
            print("Found exception %s while computing tf_single_ref_df (%s), skipping" % (exp_tfs, device))
            traceback.print_exc()
            stats = None
    elif function == 'dtw_single':
        assert device is not None, "dtw_single requires a device ('android' or 'ios')"
        try:
            ref_df = ref_dtw_gt_single(e, device, tz)
            stats = stats_gen(ref_df, e)
        except Exception as exp_dtws:
            print("Found exception %s while computing dtw_single_ref_df (%s), skipping" % (exp_dtws, device))
            traceback.print_exc()
            stats = None
    elif function == 'raw':
        assert device is not None, "raw requires a device ('android' or 'ios')"
        try:
            ref_df = raw_reference(e, device, tz)
            stats = stats_gen(ref_df, e)
        except Exception as exp_raw:
            print("Found exception %s while computing raw_ref_df (%s), skipping" % (exp_raw, device))
            traceback.print_exc()
            stats = None
    else:
        assert False, "Unknown function %s" % function
    
    if stats is not None:
        print("Validated %s, stats are %s" % (function, stats))
    else:
        return None

    return ref_df, stats