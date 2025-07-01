import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import contextily as ctx
import pandas as pd
from datetime import datetime, timedelta
import os
from moviepy.editor import ImageSequenceClip
import glob
from typing import Optional
import geopandas as gpd
from shapely.geometry import Point

SCALING = 100

def create_route_animation(
    gdf: gpd.GeoDataFrame,
    timestamp_col: str = 'ts',
    fps: int = 10,
    output_folder: str = "map_animation_frames",
    output_filename: str = "route_animation.mp4",
    clean_up: bool = True
) -> None:
    """
    Create an animated video of a route from a GeoDataFrame with timestamps.

    Parameters:
    -----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame containing Point geometries and timestamps
    timestamp_col : str, optional
        Name of the column containing timestamp data, by default 'ts'
    fps : int, optional
        Frames per second for the output video, by default 10
    output_folder : str, optional
        Folder to store temporary animation frames, by default "map_animation_frames"
    output_filename : str, optional
        Name of the output video file, by default "route_animation.mp4"
    clean_up : bool, optional
        Whether to clean up temporary frame files, by default True
    
    Returns:
    --------
    None
        Saves the animation to the specified output file
    """
    # Create a copy to avoid modifying the input
    df = gdf.copy()
    
    # Extract coordinates from geometry
    df['lon'] = df.geometry.x
    df['lat'] = df.geometry.y
    
    # Ensure the DataFrame is sorted by timestamp
    df = df.sort_values(by=timestamp_col).reset_index(drop=True)
    
    # Determine the bounding box for the map with some padding
    min_lon, max_lon = df['lon'].min() - 0.05, df['lon'].max() + 0.05
    min_lat, max_lat = df['lat'].min() - 0.05, df['lat'].max() + 0.05

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        # Clean up previous frames if they exist
        files = glob.glob(os.path.join(output_folder, "*.png"))
        for f in files:
            os.remove(f)

    # Calculate the duration of the animation based on timestamps
    start_time = df[timestamp_col].min()
    end_time = df[timestamp_col].max()
    total_duration_seconds = end_time - start_time

    # Determine the time step for each frame
    num_frames = int(total_duration_seconds * fps / SCALING)
    time_steps = [start_time + (t * SCALING / fps) for t in range(num_frames + 1)]

    # Generate Frames
    image_files = []
    fig, ax = plt.subplots(figsize=(10, 10))

    # Initialize Basemap
    m = Basemap(
        llcrnrlon=min_lon, urcrnrlon=max_lon,
        llcrnrlat=min_lat, urcrnrlat=max_lat,
        resolution='i',
        projection='merc',
        ax=ax
    )

    # Draw map features
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()
    m.fillcontinents(color='lightgray', lake_color='aqua')
    m.drawmapboundary(fill_color='aqua')
    m.drawparallels(range(int(min_lat), int(max_lat) + 2, 1), labels=[1,0,0,0])
    m.drawmeridians(range(int(min_lon), int(max_lon) + 2, 1), labels=[0,0,0,1])

    # ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

    # Convert all original coordinates to map projection once
    x_all, y_all = m(df['lon'].values, df['lat'].values)
    m.plot(x_all, y_all, 'k--', latlon=False, linewidth=1, alpha=0.5, label='Route')

    # Animation loop
    point_handle = None
    time_text_handle = None

    for i, current_time in enumerate(time_steps):
        # Find the two closest points in time
        lower_bound_idx = df[df[timestamp_col] <= current_time].index.max()
        upper_bound_idx = df[df[timestamp_col] >= current_time].index.min()

        if pd.isna(lower_bound_idx):  # Before first point
            current_lon, current_lat = df.loc[0, ['lon', 'lat']]
            current_speed = 0
        elif pd.isna(upper_bound_idx):  # After last point
            current_lon, current_lat = df.loc[len(df) - 1, ['lon', 'lat']]
            current_speed = 0
        elif lower_bound_idx == upper_bound_idx:  # Exactly on a timestamped point
            current_lon, current_lat = df.loc[lower_bound_idx, ['lon', 'lat']]
            current_speed = df.loc[lower_bound_idx, 'speed']
        else:
            # Interpolate between points
            p1 = df.loc[lower_bound_idx]
            p2 = df.loc[upper_bound_idx]
            time_diff = (p2[timestamp_col] - p1[timestamp_col])
            
            if time_diff == 0:
                current_lon, current_lat = p1['lon'], p1['lat']
                current_speed = p1['speed']
            else:
                elapsed = (current_time - p1[timestamp_col])
                factor = elapsed / time_diff
                current_lon = p1['lon'] + (p2['lon'] - p1['lon']) * factor
                current_lat = p1['lat'] + (p2['lat'] - p1['lat']) * factor
                current_speed = p1['speed'] + (p2['speed'] - p1['speed']) * factor

        # Plot current position
        x_current, y_current = m(current_lon, current_lat)
        if point_handle:
            point_handle.remove()
        point_handle, = m.plot(x_current, y_current, 'ro', latlon=False, markersize=8, label='Current Position')

        # Update time text
        if time_text_handle:
            time_text_handle.remove()
        time_text_handle = ax.text(
            0.02, 0.98, 
            f"Time: {current_time} \n Speed: {round(current_speed*2.2369362920544, 2)} mph",
            transform=ax.transAxes, 
            fontsize=12,
            verticalalignment='top', 
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1, alpha=0.7)
        )

        # Save frame
        frame_path = os.path.join(output_folder, f"frame_{i:04d}.png")
        plt.savefig(frame_path, dpi=150, bbox_inches='tight')
        image_files.append(frame_path)
        print(f"Generated frame {i+1}/{len(time_steps)}", end='\r')

    plt.close(fig)

    # Create video
    print(f"\nCompiling {len(image_files)} frames into a video...")
    clip = ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(output_filename, codec="libx264")
    print(f"Video saved as {output_filename}")

    # Clean up
    if clean_up and image_files:
        for f in image_files:
            os.remove(f)
        os.rmdir(output_folder)
        print("Cleaned up temporary files.")


# Example usage:
if __name__ == "__main__":
    # Example data creation
    import geopandas as gpd
    from shapely.geometry import Point
    import pandas as pd
    
    # Create sample GeoDataFrame
    data = [
        {"timestamp": pd.Timestamp("2025-06-25 10:00:00"), "geometry": Point(-122.4194, 37.7749)},
        {"timestamp": pd.Timestamp("2025-06-25 10:00:10"), "geometry": Point(-122.4100, 37.7800)},
        {"timestamp": pd.Timestamp("2025-06-25 10:00:20"), "geometry": Point(-122.4000, 37.7850)},
        {"timestamp": pd.Timestamp("2025-06-25 10:00:30"), "geometry": Point(-122.3900, 37.7900)},
        {"timestamp": pd.Timestamp("2025-06-25 10:00:40"), "geometry": Point(-122.3800, 37.7950)},
        {"timestamp": pd.Timestamp("2025-06-25 10:00:50"), "geometry": Point(-122.3700, 37.8000)},
    ]
    
    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
    
    # Generate animation
    create_route_animation(
        gdf=gdf,
        timestamp_col='timestamp',
        fps=10,
        output_filename="example_route_animation.mp4"
    )