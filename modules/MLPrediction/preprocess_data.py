#!/usr/bin/env python3
"""
Data Preprocessing Script for EV Demand Prediction
=================================================

This script handles the complete data preprocessing workflow, from raw charging
data to a feature-rich training set ready for the ML pipeline.

It performs the following steps:
1.  Calculates demand scores from raw UrbanEV or ST-EVCDP datasets.
2.  Creates a geographic grid based on the locations of the charging stations.
3.  Maps the charging stations to the grid cells and aggregates demand scores.
4.  Extracts a comprehensive set of urban features from OpenStreetMap for each grid cell.
5.  Merges the demand scores and OSM features into a final training CSV.

Usage:
------
.. code-block:: bash

    # Preprocess the UrbanEV dataset
    python -m modules.MLPrediction.preprocess_data --dataset-type urbanev --city-name "Shenzhen, China"

    # Preprocess the ST-EVCDP dataset and save to a custom path
    python -m modules.MLPrediction.preprocess_data --dataset-type st-evcdp --city-name "Shenzhen, China" --output-file "results/st_evcdp_training_data.csv"
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from typing import List

# Add project root to path to allow relative imports
sys.path.append(str(Path(__file__).resolve().parents[2]))

from modules.MLPrediction.demand_score_calculator import DemandScoreCalculator
from modules.MLPrediction.portable_osm_extractor import PortableOSMExtractor
from modules.utils.gridding import CityGridding
from modules.utils.log_configs import setup_logging

logger = logging.getLogger(__name__)

def extract_osm_features_once(dataset_type: str, city_name: str, data_dir: str = "data",
                             enhanced_gridding: bool = True, num_random_variations: int = 3, 
                             scales: List[float] = None):
    """
    Extract OSM features once and cache them for reuse across multiple demand variations.
    
    Returns:
        Tuple of (osm_features_df, grid_cells, inf_data, raw_data_dir)
    """
    data_dir = Path(data_dir)
    logger.info(f"🗺️ Extracting OSM features once for '{dataset_type}' dataset in {city_name}")

    # Load infrastructure data
    if dataset_type == "urbanev":
        raw_data_dir = data_dir / "training" / "UrbanEV-main"
        inf_data = pd.read_csv(raw_data_dir / "inf.csv").rename(columns={'TAZID': 'station_id'})
    elif dataset_type == "st-evcdp":
        raw_data_dir = data_dir / "training" / "ST-EVCDP-main"
        inf_data = pd.read_csv(raw_data_dir / "information.csv").rename(columns={'grid': 'station_id', 'la': 'latitude', 'lon': 'longitude'})
    else:
        raise ValueError("Invalid dataset type. Choose 'urbanev' or 'st-evcdp'.")

    # Create geographic grid (same as before)
    valid_stations = inf_data[
        inf_data['latitude'].notna() & 
        inf_data['longitude'].notna() &
        (inf_data['latitude'].abs() > 0.1) &
        (inf_data['longitude'].abs() > 0.1)
    ].copy()

    if valid_stations.empty:
        raise ValueError("No valid station coordinates found to create a grid.")

    logger.info(f"Found {len(valid_stations)} stations with valid coordinates to define the grid area.")
    
    min_lat, max_lat = valid_stations['latitude'].min(), valid_stations['latitude'].max()
    min_lon, max_lon = valid_stations['longitude'].min(), valid_stations['longitude'].max()

    gridder = CityGridding(primary_grid_size_km=1.0)
    coverage_polygon = box(min_lon, min_lat, max_lon, max_lat)
    
    if enhanced_gridding:
        logger.info("🎯 Using enhanced gridding with random and multiscale variations...")
        if scales is None:
            scales = [0.5, 0.75, 1.0, 1.5, 2.0]
        
        grid_cells = gridder.create_comprehensive_grid_dataset(
            base_polygon=coverage_polygon,
            include_random=True,
            include_multiscale=True,
            num_random_variations=num_random_variations,
            scales=scales,
            grid_name=f"{dataset_type}_{city_name.replace(' ', '_').replace(',', '')}"
        )
        logger.info(f"✅ Enhanced gridding created {len(grid_cells)} total grid cells")
    else:
        logger.info("📐 Using standard uniform gridding...")
        grid_cells_gdf = gridder.create_grid_for_polygon(coverage_polygon)
        grid_cells = grid_cells_gdf.to_dict('records')
        for i, cell in enumerate(grid_cells):
            if 'grid_id' not in cell:
                cell['grid_id'] = f"grid_{i}"
            cell['grid_type'] = 'base'
            cell['grid_variation'] = 0
            cell['grid_scale'] = 1.0
        logger.info(f"✅ Standard gridding created {len(grid_cells)} grid cells")

    # Extract OSM features once with advanced caching
    from modules.utils.cache_utils import CacheManager, create_cache_key
    
    cache_manager = CacheManager(cache_dir="cache")
    cache_key_data = {
        'dataset_type': dataset_type,
        'city_name': city_name,
        'enhanced_gridding': enhanced_gridding,
        'num_random_variations': num_random_variations,
        'scales': scales,
        'total_grids': len(grid_cells),
        'step': 'osm_features_only'
    }
    osm_cache_key = create_cache_key(cache_key_data)
    
    # Try to load cached OSM features
    cached_osm_data = cache_manager.load("osm_features", osm_cache_key)
    if cached_osm_data and 'osm_features' in cached_osm_data:
        osm_features = cached_osm_data['osm_features']
        logger.info(f"✅ Loaded {len(osm_features)} cached OSM features")
    else:
        logger.info("🗺️ Extracting OSM features with progressive caching...")
        osm_extractor = PortableOSMExtractor()
        
        # Get all grid IDs for OSM extraction
        relevant_grid_ids = [cell['grid_id'] for cell in grid_cells]
        
        cache_file_path = cache_manager.get_cache_path("osm_features", osm_cache_key)
        
        # Extract OSM features for all grids
        osm_features = osm_extractor.extract_features_for_grids_with_caching(
            grid_cells=grid_cells,
            city_name=city_name,
            relevant_grid_ids=relevant_grid_ids,
            cache_file=str(cache_file_path),
            save_every=100
        )
        
        # Final cache save
        osm_cache_data = {
            'osm_features': osm_features,
            'cache_key_data': cache_key_data,
            'timestamp': pd.Timestamp.now().isoformat(),
            'total_grids': len(relevant_grid_ids)
        }
        cache_manager.save("osm_features", osm_cache_key, osm_cache_data)
    
    logger.info(f"✅ OSM features extracted for {len(osm_features)} grids and cached for reuse")
    
    return osm_features, grid_cells, inf_data, raw_data_dir


def preprocess_data(dataset_type: str, city_name: str, output_file: str, data_dir: str = "data", 
                   enhanced_gridding: bool = True, num_random_variations: int = 3, 
                   scales: List[float] = None, demand_score_variations: bool = True):
    """
    Runs the complete data preprocessing pipeline with enhanced gridding options.

    Args:
        dataset_type: The type of dataset to process ('urbanev' or 'st-evcdp').
        city_name: The name of the city for OSM feature extraction (e.g., "Shenzhen, China").
        output_file: Path to save the final training CSV file.
        data_dir: The root directory where the training data is located.
        enhanced_gridding: Whether to use enhanced gridding with random and multiscale variations.
        num_random_variations: Number of random grid variations to create.
        scales: List of scale factors for multiscale grids (default: [0.5, 0.75, 1.0, 1.5, 2.0]).
        demand_score_variations: Whether to generate multiple demand score variations for experiments.
    """
    data_dir = Path(data_dir)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting data preprocessing for '{dataset_type}' dataset.")

    # Step 0: Extract OSM features once (cached for reuse)
    # ===================================================
    logger.info("🗺️ Step 0: Extracting/Loading OSM features (cached for reuse)...")
    osm_features, grid_cells, inf_data, raw_data_dir = extract_osm_features_once(
        dataset_type, city_name, data_dir, enhanced_gridding, num_random_variations, scales
    )

    # Step 1: Calculate Demand Scores with Multiple Variations  
    # ========================================================
    demand_calculator = DemandScoreCalculator()

    # Define demand score variations for experiments
    if demand_score_variations:
        demand_variations = [
            (0.0, 1.0, "hours_only"),      # 0*kwh + 1*hrs
            (0.25, 0.75, "kwh25_hrs75"),   # 0.25*kwh + 0.75*hrs  
            (0.5, 0.5, "balanced"),        # 0.5*kwh + 0.5*hrs
            (0.75, 0.25, "kwh75_hrs25"),   # 0.75*kwh + 0.25*hrs
            (1.0, 0.0, "kwh_only")         # 1*kwh + 0*hrs
        ]
        logger.info(f"🔬 Generating {len(demand_variations)} demand score variations for experiments")
    else:
        # Default balanced approach
        demand_variations = [(0.5, 0.5, "balanced")]
        logger.info("📊 Using single balanced demand score (0.5*kwh + 0.5*hrs)")
    
    # Calculate demand scores for each variation
    all_demand_scores = {}
    for kwh_weight, hrs_weight, variation_name in demand_variations:
        logger.info(f"  → Calculating '{variation_name}' (kwh_weight={kwh_weight}, hrs_weight={hrs_weight})")
        
        if dataset_type == "urbanev":
            variation_scores = demand_calculator.calculate_demand_scores_from_urban_ev(
                str(raw_data_dir), kwh_weight=kwh_weight, hrs_weight=hrs_weight
            )
        else:  # st-evcdp
            variation_scores = demand_calculator.calculate_demand_scores_from_st_evcdp(
                str(raw_data_dir), kwh_weight=kwh_weight, hrs_weight=hrs_weight
            )
        
        # Rename demand_score column to include variation
        variation_scores = variation_scores.rename(columns={'demand_score': f'demand_score_{variation_name}'})
        all_demand_scores[variation_name] = variation_scores
    
    # Merge all variations together
    variation_keys = list(all_demand_scores.keys())
    
    if not variation_keys:
        logger.warning("No demand score variations were calculated.")
        base_demand = pd.DataFrame() # Or handle as error
    else:
        # Start with the full DataFrame from the first variation
        base_demand = all_demand_scores[variation_keys[0]].copy()
        
        # For subsequent variations, merge only the demand score column
        for variation_name in variation_keys[1:]:
            scores_to_merge = all_demand_scores[variation_name][['station_id', f'demand_score_{variation_name}']]
            base_demand = base_demand.merge(scores_to_merge, on='station_id', how='outer')
    
    logger.info(f"✅ Calculated demand score variations for {len(base_demand)} stations.")

    # Step 2: Map Demand to Grid with Spatial Interpolation (with Caching)
    # =====================================================================
    from modules.utils.cache_utils import CacheManager, create_cache_key
    
    # Initialize cache manager
    cache_manager = CacheManager(cache_dir="cache")
    
    # Create cache key based on dataset parameters
    cache_key_data = {
        'dataset_type': dataset_type,
        'city_name': city_name,
        'enhanced_gridding': enhanced_gridding,
        'num_random_variations': num_random_variations,
        'scales': scales,
        'total_grids': len(grid_cells),
        'stations_hash': str(hash(str(base_demand.values.tobytes())))[:8],
        'demand_variations': len(demand_variations)
    }
    demand_cache_key = create_cache_key(cache_key_data)
    
    # Try to load cached demand scores
    cached_demand_data = cache_manager.load("demand_scores", demand_cache_key)
    if cached_demand_data:
        grid_demand_scores = cached_demand_data['grid_demand_scores']
        logger.info(f"✅ Loaded {len(grid_demand_scores)} cached grid demand scores")
    else:
        logger.info("🔮 Generating demand scores with spatial interpolation...")
        
        # The base_demand dataframe already contains station coordinates and all necessary info.
        # The previous merge with inf_data was redundant and caused column name collisions.
        station_data_with_demand = base_demand
        
        # OPTIMIZED: Interpolate all demand variations at once
        logger.info("🔮 Processing optimized grid aggregation for ALL variations simultaneously...")
        
        # Create a single comprehensive interpolation for all grid cells
        logger.info("🎯 Creating optimized spatial interpolation for all variations...")
        grid_demand_scores = demand_calculator.interpolate_all_demand_variations_at_once(
            all_demand_variations=all_demand_scores,
            grid_cells=grid_cells,
            station_data_with_demand=station_data_with_demand
        )
        
        # Cache the results
        cache_data = {
            'grid_demand_scores': grid_demand_scores,
            'cache_key_data': cache_key_data,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        cache_manager.save("demand_scores", demand_cache_key, cache_data)
    
    logger.info(f"✅ Generated demand estimates for {len(grid_demand_scores)} total grid cells.")

    # Step 3: Merge and Finalize
    # ==========================
    training_data = pd.merge(osm_features, grid_demand_scores, on="grid_id", how="inner")
    
    # Add latitude and longitude of the grid cell center for context
    if enhanced_gridding:
        # For enhanced gridding, extract center coordinates from grid_cells list
        grid_centers_data = []
        for cell in grid_cells:
            grid_centers_data.append({
                'grid_id': cell['grid_id'],
                'latitude': cell.get('center_lat', (cell.get('min_lat', 0) + cell.get('max_lat', 0)) / 2),
                'longitude': cell.get('center_lon', (cell.get('min_lon', 0) + cell.get('max_lon', 0)) / 2),
                'grid_type': cell.get('grid_type', 'unknown'),
                'grid_scale': cell.get('grid_scale', 1.0),
                'grid_variation': cell.get('grid_variation', 0),
                'actual_size_km': cell.get('actual_size_km', 1.0),
                'size_scale': cell.get('size_scale', 1.0),
                'offset_x_km': cell.get('offset_x_km', 0.0),
                'offset_y_km': cell.get('offset_y_km', 0.0)
            })
        grid_centers = pd.DataFrame(grid_centers_data)
    else:
        # For standard gridding, extract coordinates from grid_cells list
        grid_centers_data = []
        for cell in grid_cells:
            grid_centers_data.append({
                'grid_id': cell['grid_id'],
                'latitude': cell.get('center_lat', (cell.get('min_lat', 0) + cell.get('max_lat', 0)) / 2),
                'longitude': cell.get('center_lon', (cell.get('min_lon', 0) + cell.get('max_lon', 0)) / 2),
                'grid_type': 'base',
                'grid_scale': 1.0,
                'grid_variation': 0
            })
        grid_centers = pd.DataFrame(grid_centers_data)
    
    training_data = pd.merge(training_data, grid_centers, on="grid_id", how="left")

    training_data.to_csv(output_path, index=False)
    logger.info(f"✅ Successfully created training data with {len(training_data)} samples.")
    
    # Log dataset composition if enhanced gridding was used
    if enhanced_gridding and 'grid_type' in training_data.columns:
        logger.info("📊 Dataset composition by grid type:")
        composition = training_data['grid_type'].value_counts()
        for grid_type, count in composition.items():
            percentage = (count / len(training_data)) * 100
            logger.info(f"   {grid_type}: {count} samples ({percentage:.1f}%)")
    
    logger.info(f"Final training data saved to: {output_path}")


def main():
    """Main function to parse arguments and run the script."""
    parser = argparse.ArgumentParser(
        description="Preprocess raw EV charging data to create a training-ready dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        required=True,
        choices=['urbanev', 'st-evcdp'],
        help="The type of raw dataset to process."
    )
    parser.add_argument(
        "--city-name",
        type=str,
        required=True,
        help="The name of the city for OSM feature extraction (e.g., 'Shenzhen, China')."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="results/shenzhen_urbanev_training_data.csv",
        help="Path to save the final training CSV file."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Root directory where the training data is located."
    )
    parser.add_argument(
        "--enhanced-gridding",
        action="store_true",
        default=True,
        help="Use enhanced gridding with random and multiscale variations."
    )
    parser.add_argument(
        "--no-enhanced-gridding",
        dest="enhanced_gridding",
        action="store_false",
        help="Disable enhanced gridding and use standard uniform gridding."
    )
    parser.add_argument(
        "--num-random-variations",
        type=int,
        default=3,
        help="Number of random grid variations to create (default: 3)."
    )
    parser.add_argument(
        "--scales",
        nargs="+",
        type=float,
        default=None,
        help="Scale factors for multiscale grids (default: [0.5, 0.75, 1.0, 1.5, 2.0])."
    )
    parser.add_argument(
        "--demand-score-variations",
        action="store_true",
        default=False,
        help="Generate multiple demand score variations for experiments (0*kwh+1*hrs, 0.25+0.75, 0.5+0.5, 0.75+0.25, 1*kwh+0*hrs)."
    )
    parser.add_argument(
        "--osm-only",
        action="store_true",
        default=False,
        help="Extract OSM features only (no demand processing) for caching. Useful for preparing data before running experiments."
    )
    
    args = parser.parse_args()
    
    setup_logging()
    
    if args.osm_only:
        logger = logging.getLogger(__name__)
        logger.info("🗺️ OSM-only mode: Extracting and caching OSM features...")
        osm_features, grid_cells, inf_data, raw_data_dir = extract_osm_features_once(
            args.dataset_type, args.city_name, args.data_dir, 
            args.enhanced_gridding, args.num_random_variations, args.scales
        )
        logger.info(f"✅ OSM features extracted and cached for {len(osm_features)} grids")
        logger.info("🎯 Now you can run preprocessing with --demand-score-variations quickly!")
    else:
        preprocess_data(
            dataset_type=args.dataset_type,
            city_name=args.city_name,
            output_file=args.output_file,
            data_dir=args.data_dir,
            enhanced_gridding=args.enhanced_gridding,
            num_random_variations=args.num_random_variations,
            scales=args.scales,
            demand_score_variations=args.demand_score_variations
        )

if __name__ == "__main__":
    main()
