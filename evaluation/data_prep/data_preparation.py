# data_preparation.py

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import glob
import os
from tqdm import tqdm
import folium
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
import seaborn as sns
from modules.utils.gridding import CityGridding
import osmnx as ox # <-- Import osmnx

# (Keep load_and_combine_ved, assign_points_to_grid, plot_activity_heatmap, plot_weekly_grid_coverage functions as they are)
def load_and_combine_ved(data_dir: str) -> pd.DataFrame:
    """Loads, combines, and adds a 'week' column to all VED CSVs."""
    csv_files = glob.glob(os.path.join(data_dir, "VED_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No VED CSV files found in directory: {data_dir}")

    print(f"Found {len(csv_files)} VED files to combine.")
    df_list = []
    for f in tqdm(csv_files, desc="Loading VED files"):
        try:
            df = pd.read_csv(f, usecols=['Timestamp(ms)', 'Latitude[deg]', 'Longitude[deg]', 'VehId'])
            week_str = os.path.basename(f).split('_')[1]
            df['week'] = pd.to_datetime(week_str, format='%y%m%d')
            df_list.append(df)
        except Exception as e:
            print(f"Warning: Could not process file {f}. Error: {e}")

    if not df_list:
        raise ValueError("No data could be loaded from the CSV files.")
        
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df.rename(columns={
        'Timestamp(ms)': 'timestamp', 
        'Latitude[deg]': 'lat',
        'Longitude[deg]': 'lon'
    }, inplace=True)
    
    # CRITICAL FIX: Ensure coordinate columns are numeric
    print("Converting coordinates to numeric format...")
    combined_df['lat'] = pd.to_numeric(combined_df['lat'], errors='coerce')
    combined_df['lon'] = pd.to_numeric(combined_df['lon'], errors='coerce')
    
    # Drop any rows with invalid coordinates
    initial_count = len(combined_df)
    combined_df = combined_df.dropna(subset=['lat', 'lon'])
    final_count = len(combined_df)
    if initial_count != final_count:
        print(f"Removed {initial_count - final_count} rows with invalid coordinates")
    
    # Additional validation: ensure coordinates are within reasonable bounds
    combined_df = combined_df[
        (combined_df['lat'] >= -90) & (combined_df['lat'] <= 90) &
        (combined_df['lon'] >= -180) & (combined_df['lon'] <= 180)
    ]
    
    # Normalize timestamps to simulation-friendly range
    print("Normalizing timestamps for simulation...")
    combined_df['timestamp'] = combined_df['timestamp'] / 1000.0  # Convert ms to seconds
    
    # Normalize timestamps within each vehicle's trajectory to 0-3600s (1 hour) range
    def normalize_vehicle_timestamps(group):
        if len(group) > 1:
            min_ts = group['timestamp'].min()
            max_ts = group['timestamp'].max()
            if max_ts > min_ts:
                # Scale to 0-3600 seconds (1 hour) for simulation
                group['timestamp'] = (group['timestamp'] - min_ts) / (max_ts - min_ts) * 3600
            else:
                group['timestamp'] = 0
        else:
            # Single point trajectory - set timestamp to 0
            group['timestamp'] = 0
        return group
    
    # Apply normalization with proper error handling
    try:
        combined_df = combined_df.groupby('VehId').apply(normalize_vehicle_timestamps).reset_index(drop=True)
    except Exception as e:
        print(f"Warning: Error in timestamp normalization: {e}")
        # Fallback: set all timestamps to 0
        combined_df['timestamp'] = 0
    
    # CRITICAL FIX: Ensure VehId is string type for consistent handling
    combined_df['VehId'] = combined_df['VehId'].astype(str)
    
    # CRITICAL FIX: Ensure all required columns exist and are properly typed
    required_columns = ['VehId', 'lat', 'lon', 'timestamp', 'week']
    for col in required_columns:
        if col not in combined_df.columns:
            print(f"Warning: Missing required column {col}")
        else:
            print(f"✅ Column {col} present with {len(combined_df[col].dropna())} valid values")
    
    print("Finished combining and normalizing files.")
    print(f"Final dataset: {len(combined_df)} rows, {combined_df['VehId'].nunique()} unique vehicles")
    return combined_df

def assign_points_to_grid(df: pd.DataFrame, grid_cells: list) -> gpd.GeoDataFrame:
    """Assigns each data point to a grid cell via a spatial join."""
    print("Creating GeoDataFrame from raw data points...")
    if len(df) > 2_000_000:
        print(f"Downsampling from {len(df)} to 2,000,000 points for faster spatial join.")
        df = df.sample(n=2_000_000, random_state=42)

    geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
    gdf_points = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    print("Creating GeoDataFrame for grid cells...")
    grid_polygons = []
    for cell in grid_cells:
        corners_lon = [c[0] for c in cell['corners']]
        corners_lat = [c[1] for c in cell['corners']]
        polygon_geom = gpd.GeoSeries([Point(lon, lat) for lon, lat in zip(corners_lon, corners_lat)]).unary_union.convex_hull
        grid_polygons.append(polygon_geom)
        
    gdf_grid = gpd.GeoDataFrame({'cell_id': [c['cell_id'] for c in grid_cells]}, geometry=grid_polygons, crs="EPSG:4326")

    print("Performing spatial join to assign points to grids...")
    joined_gdf = gpd.sjoin(gdf_points, gdf_grid, how="inner", predicate='within')
    return joined_gdf

def plot_activity_heatmap(grid_counts: pd.Series, grid_cells: list, output_file: str):
    """Generates an HTML heatmap showing data point density."""
    print(f"Generating heatmap and saving to {output_file}...")
    grid_centers = {cell['cell_id']: (cell['center_lat'], cell['center_lon']) for cell in grid_cells}
    heat_data = [[grid_centers[cid][0], grid_centers[cid][1], float(count)] for cid, count in grid_counts.items() if cid in grid_centers]

    all_lats = [cell['center_lat'] for cell in grid_cells]
    all_lons = [cell['center_lon'] for cell in grid_cells]
    map_center = [sum(all_lats) / len(all_lats), sum(all_lons) / len(all_lons)] if all_lats else [42.28, -83.74]
    
    hmap = folium.Map(location=map_center, zoom_start=12)
    HeatMap(heat_data, radius=15).add_to(hmap)
    folium.TileLayer('cartodbdark_matter').add_to(hmap)

    for cell in grid_cells:
        corners = [(c[1], c[0]) for c in cell['corners']]
        folium.Polygon(locations=corners, color='blue', weight=1, fill=False, fill_opacity=0.1).add_to(hmap)

    hmap.save(output_file)

def plot_weekly_grid_coverage(df: pd.DataFrame, total_grids: int, output_file: str):
    """Plots the percentage of grids that have data points for each week."""
    print(f"Generating weekly coverage plot and saving to {output_file}...")
    coverage_by_week = df.groupby('week')['cell_id'].nunique().reset_index()
    coverage_by_week['coverage_pct'] = (coverage_by_week['cell_id'] / total_grids) * 100

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 9))
    sns.barplot(data=coverage_by_week.sort_values('week'), x='week', y='coverage_pct', color='dodgerblue', ax=ax)

    ax.set_title('Percentage of Grids with VED Data Coverage Per Week', fontsize=20, pad=20, weight='bold')
    ax.set_ylabel('Grid Coverage (%)', fontsize=16)
    ax.set_xlabel('Week Starting On', fontsize=16)
    
    date_labels = coverage_by_week.sort_values('week')['week'].dt.strftime('%Y-%m-%d')
    ax.set_xticklabels(date_labels, rotation=75, ha='right')
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', fontsize=10, padding=3)

    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Weekly coverage plot saved.")

# --- Main Execution ---
if __name__ == "__main__":
    CITY_NAME_STR = "Ann Arbor, Michigan, USA"
    # --- Directory Configuration ---
    DATA_RAW_DIR = "./data/raw/"           # Raw weekly VED CSVs
    PROCESSED_DIR = "./data/validation/"   # Output location for processed parquet
    MAP_OUTPUT_DIR = "./evaluation/"        # Visualisation outputs (heatmaps, plots)
    CITY_NET_DIR = "./generated_files/city_network/"  # City-wide OSM/SUMO network files

    # Create required directories if they do not exist
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MAP_OUTPUT_DIR, exist_ok=True)
    os.makedirs(CITY_NET_DIR, exist_ok=True)

    try:
        # 1. Load and combine all VED data
        combined_df_full = load_and_combine_ved(DATA_RAW_DIR)

        # 2. Generate the city grid system (for data assignment and visualization)
        print(f"Generating city grid for {CITY_NAME_STR}...")
        gridder = CityGridding(
            primary_grid_size_km=1.0, 
            fetch_osm_features=False,  # Speed up grid creation
            debug_mode=False,          # Reduce logging for cleaner output
            cache_dir="cache"          # Use centralized cache
        )
        ann_arbor_coords = (42.2808, -83.7430)
        city_grids = gridder.create_city_grid(CITY_NAME_STR, coordinates=ann_arbor_coords)
        
        # Ensure grids follow standard format
        city_grids = gridder._standardize_grid_format(city_grids)

        # ------------------- ARCHITECTURAL CHANGE -------------------
        # 3. Generate ONE city-wide OSM file for high-fidelity routing
        city_osm_path = os.path.join(CITY_NET_DIR, "ann_arbor.osm.xml")
        if not os.path.exists(city_osm_path):
            print(f"Generating city-wide OSM network file at: {city_osm_path}")
            # Use osmnx to get the graph for the whole city
            graph = ox.graph_from_place(CITY_NAME_STR, network_type='drive')
            ox.save_graph_xml(graph, filepath=city_osm_path)
            print("City-wide OSM file saved.")
        else:
            print("City-wide OSM file already exists. Skipping generation.")
        # The SUMO network (.net.xml) will be generated from this file by the SUMONetwork class later.
        # We no longer generate OSM files for each grid.
        # -----------------------------------------------------------

        # 4. Assign data points to grids
        gridded_df = assign_points_to_grid(combined_df_full, city_grids)
        print(f"Successfully assigned {len(gridded_df)} data points to {gridded_df['cell_id'].nunique()} grids.")

        # 5. Save the processed data for future use
        processed_file = os.path.join(PROCESSED_DIR, "ved_processed_with_grids.parquet")
        print(f"Saving processed data to {processed_file}...")
        gridded_df.to_parquet(processed_file)

        # 6. Generate Overall Activity Heatmap
        heatmap_file = os.path.join(MAP_OUTPUT_DIR, "grid_activity_heatmap.html")
        grid_point_counts = gridded_df['cell_id'].value_counts()
        plot_activity_heatmap(grid_point_counts, city_grids, heatmap_file)

        # 7. Generate Weekly Coverage Plot
        coverage_plot_file = os.path.join(MAP_OUTPUT_DIR, "weekly_grid_coverage.png")
        plot_weekly_grid_coverage(gridded_df, len(city_grids), coverage_plot_file)

        print("\nEDA and Network Preparation Complete. Check the 'evaluation' and 'generated_files/city_network' directories.")

    except (FileNotFoundError, ValueError) as e:
        print(f"\nERROR: A required file or directory was not found or was empty. {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()