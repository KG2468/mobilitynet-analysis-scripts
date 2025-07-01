import pickle
import emeval.metrics.movie as emm
import emeval.metrics.reference_trajectory as emr
import emeval.metrics.dist_calculations as emd

with open('deep_dive_dataset.pkl', 'rb') as f:
    deep_dive_dataset = pickle.load(f)

idx = 0
for e in deep_dive_dataset.values():
    # with open('deep_dive_dataset' + str(idx) + 'android_ts.txt', 'w') as f:
    #     f.write(emr.get_int_aligned_trajectory(emd.to_geo_df(e["temporal_control"]["android"]["location_df"]), tz="America/Los_Angeles")["ts"].to_csv(index=False))
    # with open('deep_dive_dataset' + str(idx) + 'ios_ts.txt', 'w') as f:
    #     f.write(emr.get_int_aligned_trajectory(emd.to_geo_df(e["temporal_control"]["ios"]["location_df"]), tz="America/Los_Angeles")["ts"].to_csv(index=False))
    # with open('deep_dive_dataset' + str(idx) + 'dtw_ts.txt', 'w') as f:
    #     f.write(e["dtw_gt"]["ts"].to_csv(index=False))
    # if idx == 2:
        # emr.ref_dtw_gt_with_ends_general(e, points_per_second=1, interp=2, tz="America/Los_Angeles")
    emm.create_route_animation(e["dtw_gt"], timestamp_col="ts", fps=10, output_folder="dtw_gt_frames" + str(idx), output_filename="dtw_gt" + str(idx) + ".mp4")
        # emm.create_route_animation(emd.to_geo_df(e["temporal_control"]["android"]["location_df"]), timestamp_col="ts", fps=10, output_folder="android_frames" + str(idx), output_filename="android" + str(idx) + ".mp4")
        # emm.create_route_animation(emd.to_geo_df(e["temporal_control"]["ios"]["location_df"]), timestamp_col="ts", fps=10, output_folder="ios_frames" + str(idx), output_filename="ios" + str(idx) + ".mp4")
    idx += 1