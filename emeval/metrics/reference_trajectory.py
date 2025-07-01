import scipy.interpolate as sci
import geopandas as gpd
import shapely as shp
import numpy as np
import random as random
import math
import arrow
import pandas as pd
import functools

import emeval.metrics.dist_calculations as emd
import emeval.input.spec_details as eisd
import emeval.metrics.DTW as dtw

def interpolate_points_along_linestring(linestring, time_interval, points_per_second=1.0):
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
    
    while current_distance < total_length:
        # Get point at current distance along the linestring
        point = adjusted_linestring.interpolate(current_distance)
        points.append(point)
        current_distance += interval
    points.sort(key=lambda p: adjusted_linestring.line_locate_point(p))
    # unadjusted_points = points
    unadjusted_points = [shp.geometry.Point(point.x * math.cos(math.radians(lat)), point.y) for point in points]
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

def get_int_aligned_trajectory(location_df, tz="UTC"):
    #check size of location_df
    if len(location_df) == 0:
        return gpd.GeoDataFrame({
            "ts": [],
            "fmt_time": [],
            "longitude": [],
            "latitude": [],
            "geometry": []
        })
    lat_fn = sci.interp1d(x=location_df.ts, y=location_df.latitude)
    lon_fn = sci.interp1d(x=location_df.ts, y=location_df.longitude)
    # In order to avoid extrapolation, we use ceil for the first int and floor
    # for the last int
    first_int_ts = math.ceil(location_df.ts.iloc[0])
    last_int_ts = math.floor(location_df.ts.iloc[-1])
    new_ts_range = [float(ts) for ts in range(first_int_ts, last_int_ts, 1)]
    new_fmt_time_range = [arrow.get(ts).to(tz) for ts in new_ts_range]
    new_lat = lat_fn(new_ts_range)
    new_lng = lon_fn(new_ts_range)
    new_gpdf = gpd.GeoDataFrame({
        "latitude": new_lat,
        "longitude": new_lng,
        "ts": new_ts_range,
        "fmt_time": new_fmt_time_range,
        "geometry": [shp.geometry.Point(x, y) for x, y in zip(new_lng, new_lat)]
    })
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
        "source": source
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

def collapse_outer_join_dist_so_far(loc_row, more_details_fn = None):
    """
    Collapse a merged row through outer join. This means that we can have
    either the left side or the right side, or both. In this case, we also
    want to make sure that the trajectory state is "progressing". In this only
    current implementation, we check that the distance along the ground truth
    trajectory is progressively increasing.  Since this can be complex to debug,
    the `more_details` function returns `True` for rows for which we need more
    details of the computation.
    """
    global distance_so_far

    source = None
    more_details = False
    EMPTY_POINT = shp.geometry.Point()

    if more_details_fn is not None and more_details_fn(loc_row):
        more_details = True

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
            distance_so_far = loc_row.gt_projection_a
        else:
            assert source == "ios"
            distance_so_far = loc_row.gt_projection_i
        
    if more_details:
        print("final_geom = %s, new_distance_so_far = %s" % (final_geom, distance_so_far))
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

def ref_ct_general(e, b_merge_fn, dist_threshold, tz="UTC"):
    fill_gt_linestring(e)
    section_gt_shapes = e["ground_truth"]["gt_shapes"]
    # print("In ref_ct_general, %s" % section_gt_shapes.filter(items=["start_loc","end_loc"]))
    filtered_loc_df_a = emd.filter_geo_df(
        emd.to_geo_df(e["temporal_control"]["android"]["location_df"]),
        section_gt_shapes.filter(["start_loc","end_loc"]))
    filtered_loc_df_b = emd.filter_geo_df(
        emd.to_geo_df(e["temporal_control"]["ios"]["location_df"]),
        section_gt_shapes.filter(["start_loc","end_loc"]))
    new_location_df_a = get_int_aligned_trajectory(filtered_loc_df_a, tz)
    new_location_df_i = get_int_aligned_trajectory(filtered_loc_df_b, tz)
    merged_df = pd.merge(new_location_df_a, new_location_df_i, on="ts",
        how="inner", suffixes=("_a", "_i")).sort_values(by="ts", axis="index")
    merged_df["t_distance"] = emd.to_utm_series(gpd.GeoSeries(merged_df.geometry_a)).distance(emd.to_utm_series(gpd.GeoSeries(merged_df.geometry_i)))
    filtered_merged_df = merged_df.query("t_distance < @dist_threshold")
    print("After filtering, retained %d of %d (%s)" %
          (len(filtered_merged_df), max(len(new_location_df_a), len(new_location_df_i)),
            (len(filtered_merged_df)/max(len(new_location_df_a), len(new_location_df_i)))))
    merge_fn = functools.partial(collapse_inner_join, b_merge_fn=b_merge_fn)
    initial_reference_gpdf = gpd.GeoDataFrame(list(filtered_merged_df.apply(merge_fn, axis=1)))
    # print(initial_reference_gpdf.columns)
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

def ref_gt_general(e, b_merge_fn, dist_threshold, tz="UTC"):
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
    if len(initial_reference_gpdf.columns) > 1:
        initial_reference_gpdf["fmt_time"] = initial_reference_gpdf.ts.apply(lambda ts: arrow.get(ts).to(tz))
        print("After merging, found %d of android %d (%s), ios %d (%s)" %
              (len(initial_reference_gpdf), len(new_location_df_a), (len(initial_reference_gpdf)/len(new_location_df_a)),
               len(new_location_df_i), (len(initial_reference_gpdf)/len(new_location_df_i))))
        assert len(initial_reference_gpdf[initial_reference_gpdf.latitude.isnull()]) == 0, "Found %d null entries out of %d total" % (len(initial_reference_gpdf.latitude.isnull()), len(initial_reference_gpdf))
        return initial_reference_gpdf
    else:
        return gpd.GeoDataFrame()
    
def ref_dtw_gt_with_ends_general(e, tz="UTC", points_per_second=1, interp=2):
    fill_gt_linestring(e)
    a_pts = emd.to_geo_df(e["temporal_control"]["android"]["location_df"])
    i_pts = emd.to_geo_df(e["temporal_control"]["ios"]["location_df"])
    if interp >= 1:
        new_a_pts = get_int_aligned_trajectory(a_pts, tz)
        new_i_pts = get_int_aligned_trajectory(i_pts, tz)
    else:
        new_a_pts = a_pts
        new_i_pts = i_pts
    a_pts_seq = new_a_pts["geometry"].to_list()
    i_pts_seq = new_i_pts["geometry"].to_list()

    start_ts = min(new_a_pts["ts"].iloc[0], new_i_pts["ts"].iloc[0])
    end_ts = max(new_a_pts["ts"].iloc[-1], new_i_pts["ts"].iloc[-1])


    # Get points at 1-meter intervals along the ground truth linestring
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

    points = []
    timestamps = []
    ranges = []
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
    
    gpdf['longitude'] = gpdf.geometry.x
    gpdf['latitude'] = gpdf.geometry.y
    gpdf["fmt_time"] = gpdf.ts.apply(lambda ts: arrow.get(ts).to(tz))
    assert len(gpdf[gpdf.geometry.isnull()]) == 0, "Found %d null entries out of %d total" % (len(gpdf.geometry.isnull()), len(gpdf))
    return gpdf

def speed_acceleration_jerk(gpdf):
    skip = 0
    speed = []
    acceleration = []
    jerk = []
    for idx in range(1, len(gpdf)):
        dist = dtw.calDistance(gpdf.iloc[idx].geometry, gpdf.iloc[idx-1].geometry)
        speed.append(dist / (gpdf.iloc[idx].ts - gpdf.iloc[idx-1].ts))
        if skip > 0:
            acceleration.append((speed[idx-1] - speed[idx-2]) / (gpdf.iloc[idx-1].ts - gpdf.iloc[idx-2].ts))
            if skip > 1:
                jerk.append((acceleration[idx-2] - acceleration[idx-3]) / (gpdf.iloc[idx-2].ts - gpdf.iloc[idx-3].ts))
            else:
                skip += 1
        else:
            skip += 1
    speed.append(0)
    acceleration.append((speed[-1] - speed[-2]) / (gpdf.iloc[-1].ts - gpdf.iloc[-2].ts))
    acceleration.append(0)
    jerk.append((acceleration[-2] - acceleration[-3]) / (gpdf.iloc[-2].ts - gpdf.iloc[-3].ts))
    jerk.append((acceleration[-1] - acceleration[-2]) / (gpdf.iloc[-1].ts - gpdf.iloc[-2].ts))
    jerk.append(0)
    gpdf["speed"] = speed
    gpdf["acceleration"] = acceleration
    gpdf["jerk"] = jerk


def ref_travel_forward(e, dist_threshold, tz="UTC"):
    # This function needs a global variable
    global distance_so_far
    distance_so_far = 0
    fill_gt_linestring(e)
    section_gt_shapes = e["ground_truth"]["gt_shapes"]
    filtered_utm_loc_df_a = emd.filter_geo_df(
        emd.to_geo_df(e["temporal_control"]["android"]["location_df"]),
        section_gt_shapes.filter(["start_loc","end_loc"]))
    filtered_utm_loc_df_b = emd.filter_geo_df(
        emd.to_geo_df(e["temporal_control"]["ios"]["location_df"]),
        section_gt_shapes.filter(["start_loc","end_loc"]))
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
    merge_fn = functools.partial(collapse_outer_join_dist_so_far, more_details_fn = None)
    initial_reference_gpdf = gpd.GeoDataFrame(list(merged_df.apply(merge_fn, axis=1)))
    
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

def final_ref_ensemble(e, dist_threshold=25, tz="UTC"):
    fill_gt_linestring(e)
    gt_linestring = e["ground_truth"]["linestring"]
    tf_ref_df = ref_travel_forward(e, dist_threshold, tz)
    ct_ref_df = ref_ct_general(e, b_merge_midpoint, dist_threshold, tz)
    tf_stats = {
        "coverage_density": coverage_density(tf_ref_df, e),
        "coverage_time": coverage_time(tf_ref_df, e),
        "coverage_max_gap": coverage_max_gap(tf_ref_df, e)
    }
    ct_stats = {
        "coverage_density": coverage_density(ct_ref_df, e),
        "coverage_time": coverage_time(ct_ref_df, e),
        "coverage_max_gap": coverage_max_gap(ct_ref_df, e)
    }
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
