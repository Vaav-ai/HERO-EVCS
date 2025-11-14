import osmnx as ox
import geopandas as gpd
from shapely.geometry import Polygon, box, Point, MultiPolygon
from shapely.ops import unary_union
import numpy as np
from pyproj import Transformer, CRS
import folium
from typing import List, Dict, Tuple, Optional
import logging
from tqdm import tqdm
import unittest
import sys
from datetime import datetime
import pickle
import hashlib
from pathlib import Path
import os
import time
from retry import retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from .cache_utils import CacheManager, create_cache_key
import pandas as pd

class CityGridding:
    """
    Unified, robust city gridding system with comprehensive caching and progress tracking.
    
    Provides consistent grid cell format across all modules in the EV placement system.
    """
    
    # Standard grid cell format - all modules should use this structure
    STANDARD_GRID_FIELDS = [
        'grid_id',      # Unique identifier (replaces cell_id for consistency)
        'cell_id',      # Legacy compatibility field (maps to grid_id)
        'min_lat', 'max_lat', 'min_lon', 'max_lon',  # Boundaries
        'center_lat', 'center_lon',                   # Center coordinates
        'area_km2',                                   # Area in square kilometers
        'corners',                                    # List of corner coordinates
    ]
    
    def __init__(self, primary_grid_size_km: float = 1.0, debug_mode: bool = False,
                 water_threshold: float = 0.7, urban_threshold: float = 0.3,
                 fetch_osm_features: bool = False, cache_dir: str = "cache"):
        """
        Initialize with enhanced urban area detection and caching.
        
        Args:
            primary_grid_size_km: Size of primary grid cells in kilometers
            debug_mode: Enable detailed debugging output
            water_threshold: Maximum allowed water coverage (0-1) for a cell
            urban_threshold: Minimum required urban area coverage (0-1) for a cell
            fetch_osm_features: When True, query OSM for urban / water layers to filter
                                cells. Set to False to greatly speed up grid creation
            cache_dir: Directory for caching grid data and OSM downloads
        
        Warning:
            Setting fetch_osm_features=True can cause hanging due to OSM API timeouts.
            For reliable operation, keep fetch_osm_features=False (default).
        """
        self.primary_grid_size_km = primary_grid_size_km
        self.debug_mode = debug_mode
        self.water_threshold = water_threshold
        self.urban_threshold = urban_threshold
        self.fetch_osm_features = fetch_osm_features
        self.logger = self._setup_logger()
        self.debug_info = {'warnings': []}
        
        # Initialize cache manager
        self.cache_manager = CacheManager(cache_dir)
        
        # OSM settings for robust downloading - reduced timeouts to prevent hanging
        ox.settings.use_cache = True
        ox.settings.log_console = False
        ox.settings.requests_timeout = 60  # Reduced from 120 to prevent hanging
        ox.settings.overpass_settings = '[out:json][timeout:60]'  # Reduced timeout
        
        # Additional settings to prevent API issues
        if hasattr(ox.settings, 'max_query_area_size'):
            ox.settings.max_query_area_size = 50 * 1000 * 1000  # 50 km² max
        
        # Progress tracking
        self.current_progress = None


    def _setup_logger(self) -> logging.Logger:
        """Set up logging with file and console handlers."""
        logger = logging.getLogger(f"CityGridding_{id(self)}")
        logger.setLevel(logging.DEBUG if self.debug_mode else logging.INFO)

        # Prevent duplicate handlers
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

            # File handler for debugging
            if self.debug_mode:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                file_handler = logging.FileHandler(f'gridding_debug_{timestamp}.log')
                file_handler.setLevel(logging.DEBUG)
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                file_handler.setFormatter(file_formatter)
                logger.addHandler(file_handler)

        return logger
    def create_city_grid(self, city_name: str, coordinates: tuple = None, 
                        force_regenerate: bool = False) -> List[Dict]:
        """Create grid with comprehensive caching and progress tracking.
        
        Args:
            city_name: Name of the city
            coordinates: Optional tuple of (latitude, longitude) coordinates
            force_regenerate: If True, ignore cached data and regenerate
            
        Returns:
            List of standardized grid cell dictionaries
        """
        # Create cache key
        cache_key = self._create_cache_key(city_name, coordinates)
        
        # Check for cached grid
        if not force_regenerate:
            cached_grid = self.cache_manager.load("grid", cache_key)
            if cached_grid:
                self.logger.info(f"✅ Loaded cached grid for {city_name} ({len(cached_grid)} cells)")
                # Ensure cached data follows standard format
                return self._standardize_grid_format(cached_grid)
        
        # Create new grid with progress tracking
        self.logger.info(f"🏗️ Creating new grid for {city_name}...")
        
        try:
            # Step 1: Get city boundary
            self.logger.info("🗺️ Step 1/6: Fetching city boundary...")
            start_time = time.time()
            city_gdf = self._get_precise_boundary(city_name, coordinates)
            self.debug_info['boundary_fetch_time'] = time.time() - start_time
            
            if city_gdf is None or city_gdf.empty:
                raise ValueError(f"Invalid boundary for {city_name}")
            
            # Step 2: Convert to UTM
            self.logger.info("🌐 Step 2/6: Converting to UTM coordinates...")
            utm_crs = self._get_utm_crs(city_gdf)
            city_utm = city_gdf.to_crs(utm_crs)
            
            # Step 3: Create primary grid
            self.logger.info("🔲 Step 3/6: Creating primary grid cells...")
            start_time = time.time()
            primary_cells = self._create_primary_grid(city_utm)
            
            # Step 4: Handle remaining areas
            self.logger.info("🔄 Step 4/6: Processing remaining areas...")
            all_cells = self._handle_remaining_areas(city_utm, primary_cells)
            self.debug_info['grid_creation_time'] = time.time() - start_time
            
            # Step 5: Convert to geographic coordinates
            self.logger.info("🌍 Step 5/6: Converting to geographic coordinates...")
            start_time = time.time()
            final_grid = self._convert_to_geographic(all_cells, utm_crs)
            self.debug_info['conversion_time'] = time.time() - start_time
            
            # Step 6: Standardize and cache
            self.logger.info("💾 Step 6/6: Finalizing and caching...")
            standardized_grid = self._standardize_grid_format(final_grid)
            self._update_cell_statistics(standardized_grid)
            
            # Cache the result
            if standardized_grid:
                self.cache_manager.save("grid", cache_key, standardized_grid)
            
            self.logger.info(f"✅ Grid creation completed: {len(standardized_grid)} cells")
            return standardized_grid
            
        except Exception as e:
            self.logger.error(f"❌ Grid creation failed: {str(e)}", exc_info=self.debug_mode)
            return []

    def create_grid_for_polygon(self, polygon: Polygon, grid_name: str = "custom_polygon_grid") -> gpd.GeoDataFrame:
        """
        Creates a vectorized grid from a Shapely Polygon with optimizations.

        This method is optimized to avoid slow iterative operations by using
        vectorized spatial joins.

        Args:
            polygon (Polygon): The polygon to grid.
            grid_name (str): A name for caching purposes.

        Returns:
            gpd.GeoDataFrame: A GeoDataFrame containing the grid cells.
        """
        boundary_gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
        
        # Add a check for unusually large polygons which may indicate coordinate errors
        bounds = boundary_gdf.total_bounds
        lon_span = bounds[2] - bounds[0]
        lat_span = bounds[3] - bounds[1]
        if lon_span > 20 or lat_span > 20: # A span >20 degrees is continent-sized and likely an error
            self.logger.warning(
                f"Input polygon has a very large geographic span ({lon_span:.1f}° lon, {lat_span:.1f}° lat). "
                "This can lead to an extremely large grid and high memory usage. "
                "Please check the input data for invalid coordinates (e.g., (0,0))."
            )

        cache_key = create_cache_key({
            'polygon_wkt': polygon.wkt,
            'grid_size': self.primary_grid_size_km,
            'type': 'polygon_grid_v2' # New version
        })

        cached_grid = self.cache_manager.load("grid_gdf", cache_key)
        if cached_grid is not None:
            self.logger.info(f"✅ Loaded cached polygon grid ({len(cached_grid)} cells)")
            return cached_grid
        
        self.logger.info("🏗️ Creating grid from polygon (optimized method)...")
        
        # 1. Convert to UTM for accurate meter-based grid
        utm_crs = self._get_utm_crs(boundary_gdf)
        boundary_utm = boundary_gdf.to_crs(utm_crs)
        
        # 2. Create a grid of squares that covers the bounding box of the polygon
        bounds = boundary_utm.total_bounds
        grid_size_m = self.primary_grid_size_km * 1000
        
        xmin, ymin, xmax, ymax = bounds
        
        # Calculate number of cells
        nx = int(np.ceil((xmax - xmin) / grid_size_m))
        ny = int(np.ceil((ymax - ymin) / grid_size_m))
        
        self.logger.info(f"Creating a {nx}x{ny} grid to cover the area...")

        polygons = []
        # No tqdm here, as it can be too verbose for the log file
        for i in range(nx):
            for j in range(ny):
                x0 = xmin + i * grid_size_m
                y0 = ymin + j * grid_size_m
                polygons.append(box(x0, y0, x0 + grid_size_m, y0 + grid_size_m))

        grid_gdf_utm = gpd.GeoDataFrame({'geometry': polygons}, crs=utm_crs)
        
        # 3. Spatially join the grid with the boundary to get only intersecting cells
        self.logger.info("Performing spatial join to find intersecting cells...")
        intersecting_grid = gpd.sjoin(grid_gdf_utm, boundary_utm, how="inner", predicate="intersects")
        
        # 4. Clip the intersecting cells to the exact boundary for a clean fit
        self.logger.info("Clipping grid to exact boundary...")
        # Use a temporary GeoDataFrame to avoid potential sjoin artifacts
        intersecting_geoms = gpd.GeoDataFrame(geometry=intersecting_grid.geometry)
        clipped_grid = gpd.clip(intersecting_geoms, boundary_utm)
        
        # 5. Convert back to geographic coordinates
        final_grid_list = self._convert_to_geographic(clipped_grid.geometry.tolist(), utm_crs)
        
        if not final_grid_list:
            self.logger.warning("No grid cells were generated after processing.")
            return gpd.GeoDataFrame()
            
        # 6. Standardize the format of the dictionaries
        standardized_grid_list = self._standardize_grid_format(final_grid_list)
        
        # 7. Create the final GeoDataFrame correctly
        df = pd.DataFrame(standardized_grid_list)
        df['geometry'] = df['corners'].apply(lambda corners: Polygon(corners))
        final_gdf = gpd.GeoDataFrame(df, crs="EPSG:4326", geometry='geometry')

        # Cache result
        if not final_gdf.empty:
            self.cache_manager.save("grid_gdf", cache_key, final_gdf)
        
        self.logger.info(f"✅ Created polygon grid with {len(final_gdf)} cells")
        return final_gdf

    def _create_cache_key(self, city_name: str, coordinates: tuple = None) -> str:
        """Create a unique cache key for the city and parameters."""
        key_data = {
            'city_name': city_name.lower().replace(' ', '_'),
            'coordinates': coordinates,
            'grid_size': self.primary_grid_size_km,
            'water_threshold': self.water_threshold,
            'urban_threshold': self.urban_threshold,
            'fetch_features': self.fetch_osm_features
        }
        key_string = str(sorted(key_data.items()))
        return hashlib.md5(key_string.encode()).hexdigest()[:16]

    def _create_city_grid_impl(self, city_name: str, coordinates: tuple = None) -> List[Dict]:
        """Original grid creation implementation (rename the current create_city_grid method)"""
        start_time = datetime.now()
        self.logger.info(f"Starting grid creation for {city_name}")
        
        try:
            # Get and validate city boundary
            boundary_start = datetime.now()
            city_gdf = self._get_precise_boundary(city_name, coordinates)
            self.debug_info['boundary_fetch_time'] = (datetime.now() - boundary_start).total_seconds()
            
            if city_gdf is None or city_gdf.empty:
                raise ValueError(f"Invalid boundary for {city_name}")
                
            if self.debug_mode:
                self._validate_geometry(city_gdf.geometry.iloc[0], "City Boundary")
            
            # Convert to UTM
            utm_crs = self._get_utm_crs(city_gdf)
            city_utm = city_gdf.to_crs(utm_crs)
            
            # Create grids
            grid_start = datetime.now()
            primary_cells = self._create_primary_grid(city_utm)
            all_cells = self._handle_remaining_areas(city_utm, primary_cells)
            self.debug_info['grid_creation_time'] = (datetime.now() - grid_start).total_seconds()
            
            # Validate grid coverage
            if self.debug_mode:
                coverage = self._calculate_coverage(city_utm, all_cells)
                self.logger.debug(f"Grid coverage: {coverage:.2f}%")
                self._check_grid_quality(all_cells)
            
            # Convert to geographic coordinates
            conv_start = datetime.now()
            final_grid = self._convert_to_geographic(all_cells, utm_crs)
            self.debug_info['conversion_time'] = (datetime.now() - conv_start).total_seconds()
            
            # Update statistics
            self._update_cell_statistics(final_grid)
            
            total_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Grid creation completed in {total_time:.2f} seconds")
            
            return final_grid
            
        except Exception as e:
            if not hasattr(self.debug_info, 'error_log'):
                self.debug_info['error_log'] = []
            self.debug_info['error_log'].append(str(e))
            self.logger.error(f"Error in grid creation: {str(e)}", exc_info=self.debug_mode)
            return []
    def _validate_geometry(self, geom, name: str) -> None:
        """Validate geometry and log issues."""
        if not geom.is_valid:
            self.logger.warning(f"Invalid {name} geometry detected")
            self.debug_info['warnings'].append(f"Invalid {name} geometry")
            return False
        return True
    def _update_cell_statistics(self, grid_cells: List[Dict]) -> None:
        """Update cell statistics for debugging."""
        areas = [cell['area_km2'] for cell in grid_cells]
        self.debug_info['cell_statistics'] = {
            'total_cells': len(grid_cells),
            'primary_cells': sum(1 for cell in grid_cells if cell['area_km2'] >= self.primary_grid_size_km),
            'flexible_cells': sum(1 for cell in grid_cells if cell['area_km2'] < self.primary_grid_size_km),
            'min_area': min(areas) if areas else 0,
            'max_area': max(areas) if areas else 0,
            'mean_area': np.mean(areas) if areas else 0,
            'std_area': np.std(areas) if areas else 0
        }
    def _check_grid_quality(self, cells: List[Polygon]) -> None:
        """Check grid quality metrics."""
        try:
            # Check for overlaps
            for i, cell1 in enumerate(cells):
                for j, cell2 in enumerate(cells[i+1:], i+1):
                    if cell1.overlaps(cell2):
                        overlap_area = cell1.intersection(cell2).area
                        if overlap_area > 1e-6:  # Small threshold to account for floating point errors
                            self.logger.warning(f"Overlap detected between cells {i} and {j}")
                            self.debug_info['warnings'].append(f"Cell overlap: {i}-{j}")

            # Check for gaps
            union = unary_union(cells)
            if not union.is_valid:
                self.logger.warning("Invalid union of grid cells")
                self.debug_info['warnings'].append("Invalid grid union")

        except Exception as e:
            self.logger.error(f"Error in grid quality check: {str(e)}")
    def _calculate_coverage(self, city_utm: gpd.GeoDataFrame, cells: List[Polygon]) -> float:
        """Calculate grid coverage percentage."""
        try:
            city_area = city_utm.geometry.iloc[0].area
            grid_union = unary_union(cells)
            intersection_area = city_utm.geometry.iloc[0].intersection(grid_union).area
            return (intersection_area / city_area) * 100
        except Exception as e:
            self.logger.error(f"Error calculating coverage: {str(e)}")
            return 0.0

    def _get_utm_crs(self, gdf: gpd.GeoDataFrame) -> CRS:
        """
        Get appropriate UTM CRS for the city's location.
        """
        # Get centroid of the city
        centroid = gdf.geometry.iloc[0].centroid
        lon, lat = centroid.x, centroid.y
        
        # Calculate UTM zone
        zone_number = int((lon + 180) / 6) + 1
        zone_letter = 'N' if lat >= 0 else 'S'
        
        return CRS.from_string(f"+proj=utm +zone={zone_number} {zone_letter} +ellps=WGS84")

    def _create_primary_grid(self, city_utm: gpd.GeoDataFrame) -> List[Polygon]:
        """
        Create primary grid cells with enhanced filtering.
        
        Args:
            city_utm: GeoDataFrame with city boundary in UTM coordinates
            
        Returns:
            List of Polygon objects representing grid cells
        """
        bounds = city_utm.total_bounds
        grid_size_m = self.primary_grid_size_km * 1000
        
        # Minimal padding
        padding = grid_size_m / 4  # Reduced padding
        bounds = (
            bounds[0] - padding,
            bounds[1] - padding,
            bounds[2] + padding,
            bounds[3] + padding
        )
        
        nx = int(np.ceil((bounds[2] - bounds[0]) / grid_size_m))
        ny = int(np.ceil((bounds[3] - bounds[1]) / grid_size_m))
        
        grid_cells = []
        city_geom = city_utm.geometry.iloc[0]
        
        # Optionally fetch urban / water features
        urban_union = water_union = None
        if self.fetch_osm_features:
            try:
                # Convert UTM geometry back to WGS84 for OSM query
                city_wgs84 = city_utm.to_crs("EPSG:4326")

                urban_tags = {
                    'landuse': ['residential', 'commercial', 'industrial', 'retail'],
                    'building': True
                }

                # Overpass queries can be slow; set a modest timeout to prevent hanging
                original_timeout = ox.settings.requests_timeout
                ox.settings.requests_timeout = 30  # Aggressive timeout for grid creation
                ox.settings.use_cache = True

                urban_gdf = ox.features_from_polygon(
                    city_wgs84.geometry.iloc[0],
                    tags=urban_tags
                ).to_crs(city_utm.crs)

                water_gdf = ox.features_from_polygon(
                    city_wgs84.geometry.iloc[0],
                    tags={'natural': ['water', 'bay'], 'water': True}
                ).to_crs(city_utm.crs)

                urban_union = unary_union(urban_gdf.geometry) if not urban_gdf.empty else None
                water_union = unary_union(water_gdf.geometry) if not water_gdf.empty else None
            except Exception as e:
                self.logger.warning(f"OSM feature fetch failed or timed out, proceeding without filters: {str(e)}")
                urban_union = None
                water_union = None
            finally:
                # Restore original timeout
                if 'original_timeout' in locals():
                    ox.settings.requests_timeout = original_timeout
        
        for i in range(nx):
            for j in range(ny):
                x0 = bounds[0] + i * grid_size_m
                y0 = bounds[1] + j * grid_size_m
                
                cell = box(x0, y0, x0 + grid_size_m, y0 + grid_size_m)
                
                if cell.intersects(city_geom):
                    include_cell = True
                    
                    # Check water coverage
                    if water_union is not None:
                        water_coverage = cell.intersection(water_union).area / cell.area
                        if water_coverage > self.water_threshold:
                            include_cell = False
                    
                    # Check urban coverage
                    if urban_union is not None and include_cell:
                        urban_coverage = cell.intersection(urban_union).area / cell.area
                        if urban_coverage < self.urban_threshold:
                            include_cell = False
                    
                    if include_cell:
                        grid_cells.append(cell)
        
        return grid_cells
    def _handle_remaining_areas(self, city_utm: gpd.GeoDataFrame, 
                              primary_cells: List[Polygon]) -> List[Polygon]:
        """
        Handle remaining areas with stricter filtering.
        """
        if not primary_cells:
            return []
            
        primary_union = unary_union(primary_cells)
        remaining = city_utm.geometry.iloc[0].difference(primary_union)
        
        if remaining.is_empty:
            return primary_cells
        
        # Use stricter thresholds for remaining areas
        remaining_cells = self._create_flexible_grid(
            remaining,
            urban_threshold=self.urban_threshold * 1.2,  # 20% stricter
            water_threshold=self.water_threshold * 0.8   # 20% stricter
        )
        
        return primary_cells + remaining_cells

    def _create_flexible_grid(self, geometry: Polygon, 
                            urban_threshold: float = None,
                            water_threshold: float = None) -> List[Polygon]:
        """
        Create flexible grid with custom thresholds.
        """
        if geometry.is_empty:
            return []
        
        urban_threshold = urban_threshold or self.urban_threshold
        water_threshold = water_threshold or self.water_threshold
        
        flexible_size = (self.primary_grid_size_km * 1000) / 2
        bounds = geometry.bounds
        cells = []
        
        urban_union = water_union = None
        if self.fetch_osm_features:
            try:
                # Get urban and water features for remaining areas
                urban_gdf = ox.features_from_polygon(
                    geometry,
                    tags={'landuse': ['residential', 'commercial', 'industrial']}
                )
                water_gdf = ox.features_from_polygon(
                    geometry,
                    tags={'natural': ['water', 'bay'], 'water': True}
                )

                urban_union = unary_union(urban_gdf.geometry)
                water_union = unary_union(water_gdf.geometry)
            except Exception:
                urban_union = None
                water_union = None
        
        x = bounds[0]
        while x < bounds[2]:
            y = bounds[1]
            while y < bounds[3]:
                cell = box(x, y, x + flexible_size, y + flexible_size)
                if cell.intersects(geometry):
                    include_cell = True
                    
                    # Apply stricter filtering
                    if water_union is not None:
                        water_coverage = cell.intersection(water_union).area / cell.area
                        if water_coverage > water_threshold:
                            include_cell = False
                    
                    if urban_union is not None and include_cell:
                        urban_coverage = cell.intersection(urban_union).area / cell.area
                        if urban_coverage < urban_threshold:
                            include_cell = False
                    
                    if include_cell:
                        cells.append(cell)
                y += flexible_size
            x += flexible_size
        
        return cells

    def _convert_to_geographic(self, cells: List[Polygon], 
                             utm_crs: CRS) -> List[Dict]:
        """
        Convert grid cells to geographic coordinates with properties.
        """
        gdf = gpd.GeoDataFrame(geometry=cells, crs=utm_crs)
        gdf_geo = gdf.to_crs("EPSG:4326")
        
        final_grid = []
        for idx, cell in enumerate(gdf_geo.geometry.tolist()):
            bounds = cell.bounds
            final_grid.append({
                'cell_id': f"cell_{idx}",
                'min_lat': bounds[1],
                'max_lat': bounds[3],
                'min_lon': bounds[0],
                'max_lon': bounds[2],
                'center_lat': cell.centroid.y,
                'center_lon': cell.centroid.x,
                'corners': list(cell.exterior.coords)[:-1],
                'area_km2': gdf.geometry.iloc[idx].area / 1_000_000
            })
            
        return final_grid
    def generate_grid_osm_files(self, grid_cells: List[Dict], output_dir: str,
                               max_workers: int = 4, retry_attempts: int = 3) -> List[str]:
        """
        Robustly download OSM data for each grid cell with caching and progress tracking.

        Args:
            grid_cells: The list of grid cell dicts from create_city_grid()
            output_dir: The directory where the OSM files will be saved
            max_workers: Number of parallel download workers
            retry_attempts: Number of retry attempts for failed downloads

        Returns:
            A list of paths to the successfully created OSM files
        """
        self.logger.info(f"🗺️ Generating OSM files for {len(grid_cells)} grids...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Check for existing files and create work list
        work_items = []
        existing_files = []
        
        for cell in grid_cells:
            grid_id = self._get_consistent_grid_id(cell)
            file_path = os.path.join(output_dir, f"{grid_id}.osm.xml")
            
            # Check cache and existing files
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                existing_files.append(file_path)
                continue
                
            work_items.append((cell, file_path))
        
        self.logger.info(f"📁 Found {len(existing_files)} existing OSM files")
        self.logger.info(f"⬬ Need to download {len(work_items)} OSM files")
        
        if not work_items:
            return existing_files
        
        # Download with progress tracking and parallel processing
        successful_downloads = []
        failed_downloads = []
        
        self.logger.info(f"📥 Starting OSM downloads for {len(work_items)} grids...")
        
        # Simple progress tracking without tqdm to avoid display issues
        total_items = len(work_items)
        completed = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all download tasks
            future_to_item = {
                executor.submit(self._download_grid_osm, cell, file_path, retry_attempts): 
                (cell, file_path) for cell, file_path in work_items
            }
            
            # Process completed downloads
            for future in as_completed(future_to_item):
                cell, file_path = future_to_item[future]
                grid_id = self._get_consistent_grid_id(cell)
                completed += 1
                
                try:
                    success = future.result()
                    if success:
                        successful_downloads.append(file_path)
                        status = "✅"
                    else:
                        failed_downloads.append(grid_id)
                        status = "❌"
                except Exception as e:
                    failed_downloads.append(grid_id)
                    self.logger.error(f"Download failed for {grid_id}: {e}")
                    status = "❌"
                
                # Log progress every 10 downloads or at the end
                if completed % 10 == 0 or completed == total_items:
                    self.logger.info(f"📊 OSM Download Progress: {completed}/{total_items} ({(completed/total_items)*100:.1f}%) - ✅{len(successful_downloads)} ❌{len(failed_downloads)}")
        
        # Summary
        total_files = existing_files + successful_downloads
        self.logger.info(f"📊 OSM Download Summary:")
        self.logger.info(f"  ✅ Existing files: {len(existing_files)}")
        self.logger.info(f"  ⬬ New downloads: {len(successful_downloads)}")
        self.logger.info(f"  ❌ Failed downloads: {len(failed_downloads)}")
        self.logger.info(f"  📁 Total available: {len(total_files)}")
        
        if failed_downloads:
            self.logger.warning(f"⚠️ Failed to download OSM data for: {failed_downloads}")
        
        return total_files

    def create_grid_from_bounds(self, min_lat: float, min_lon: float, 
                               max_lat: float, max_lon: float, 
                               grid_name: str = "custom_grid") -> List[Dict]:
        """
        Create grid from explicit boundary coordinates.
        
        Args:
            min_lat, min_lon, max_lat, max_lon: Boundary coordinates
            grid_name: Name for caching purposes
            
        Returns:
            List of standardized grid cell dictionaries
        """
        # Create a simple polygon from bounds
        from shapely.geometry import box
        import geopandas as gpd
        
        boundary_polygon = box(min_lon, min_lat, max_lon, max_lat)
        boundary_gdf = gpd.GeoDataFrame(geometry=[boundary_polygon], crs="EPSG:4326")
        
        # Use existing grid creation pipeline
        cache_key = create_cache_key({
            'bounds': (min_lat, min_lon, max_lat, max_lon),
            'grid_size': self.primary_grid_size_km,
            'type': 'bounds_grid'
        })
        
        # Check cache first
        cached_grid = self.cache_manager.load("grid", cache_key)
        if cached_grid:
            self.logger.info(f"✅ Loaded cached bounds grid ({len(cached_grid)} cells)")
            return self._standardize_grid_format(cached_grid)
        
        # Create new grid
        self.logger.info(f"🏗️ Creating grid from bounds: ({min_lat:.4f}, {min_lon:.4f}) to ({max_lat:.4f}, {max_lon:.4f})")
        
        # Convert to UTM and create grid
        utm_crs = self._get_utm_crs(boundary_gdf)
        boundary_utm = boundary_gdf.to_crs(utm_crs)
        
        # Create grid cells
        primary_cells = self._create_primary_grid(boundary_utm)
        all_cells = self._handle_remaining_areas(boundary_utm, primary_cells)
        
        # Convert back to geographic coordinates
        final_grid = self._convert_to_geographic(all_cells, utm_crs)
        standardized_grid = self._standardize_grid_format(final_grid)
        
        # Cache result
        if standardized_grid:
            self.cache_manager.save("grid", cache_key, standardized_grid)
        
        self.logger.info(f"✅ Created bounds grid with {len(standardized_grid)} cells")
        return standardized_grid

    @retry(tries=3, delay=2, backoff=2)
    def _download_grid_osm(self, cell: Dict, file_path: str, retry_attempts: int = 3) -> bool:
        """
        Download OSM data for a single grid cell with retry logic.
        
        Args:
            cell: Grid cell dictionary
            file_path: Output file path
            retry_attempts: Number of retry attempts
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            grid_id = self._get_consistent_grid_id(cell)
            
            # Check if already cached
            cache_key = create_cache_key({
                'grid_bounds': (cell['min_lat'], cell['min_lon'], cell['max_lat'], cell['max_lon']),
                'type': 'osm_graph'
            })
            
            cached_graph = self.cache_manager.load("osm_graph", cache_key)
            
            if cached_graph is None:
                # Download new graph
                bbox = (cell['max_lat'], cell['min_lat'], cell['max_lon'], cell['min_lon'])
                
                # Use smaller timeout for individual grids
                original_timeout = ox.settings.requests_timeout
                ox.settings.requests_timeout = 60
                
                try:
                    graph = ox.graph_from_bbox(
                        *bbox, 
                        network_type='drive', 
                        simplify=True, 
                        truncate_by_edge=True,
                        clean_periphery=True
                    )
                    
                    # Cache the graph for future use
                    self.cache_manager.save("osm_graph", cache_key, graph)
                    
                finally:
                    ox.settings.requests_timeout = original_timeout
            else:
                graph = cached_graph
            
            # Save as OSM XML file
            ox.save_graph_xml(graph, filepath=file_path)
            
            # Verify file was created and has content
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                return True
            else:
                self.logger.warning(f"OSM file for {grid_id} was created but is empty")
                return False
                
        except Exception as e:
            self.logger.debug(f"OSM download failed for {grid_id}: {e}")
            return False

    def _get_consistent_grid_id(self, cell: Dict) -> str:
        """Get consistent grid ID from cell dict, handling both old and new formats."""
        return cell.get('grid_id', cell.get('cell_id', f"grid_{hash(str(cell))%10000}"))

    def _standardize_grid_format(self, grid_cells: List[Dict]) -> List[Dict]:
        """
        Ensure all grid cells follow the standard format with consistent field names.
        
        Args:
            grid_cells: List of grid cell dictionaries (potentially in old format)
            
        Returns:
            List of standardized grid cell dictionaries
        """
        standardized = []
        
        for i, cell in enumerate(grid_cells):
            # Create standardized cell
            std_cell = {}
            
            # Handle ID fields - ensure both grid_id and cell_id are present
            if 'grid_id' in cell:
                std_cell['grid_id'] = cell['grid_id']
                std_cell['cell_id'] = cell['grid_id']  # For backward compatibility
            elif 'cell_id' in cell:
                std_cell['grid_id'] = cell['cell_id']
                std_cell['cell_id'] = cell['cell_id']
            else:
                # Generate consistent ID
                grid_id = f"grid_{i:04d}"
                std_cell['grid_id'] = grid_id
                std_cell['cell_id'] = grid_id
            
            # Copy all standard fields
            for field in self.STANDARD_GRID_FIELDS:
                if field in cell:
                    std_cell[field] = cell[field]
                elif field in ['grid_id', 'cell_id']:
                    # Already handled above
                    continue
                else:
                    # Set defaults for missing fields
                    if field == 'area_km2' and 'area_km2' not in cell:
                        # Calculate area from bounds if missing
                        std_cell['area_km2'] = self._calculate_cell_area_from_bounds(cell)
                    elif field in ['center_lat', 'center_lon'] and field not in cell:
                        # Calculate center from bounds
                        if field == 'center_lat':
                            std_cell['center_lat'] = (cell.get('min_lat', 0) + cell.get('max_lat', 0)) / 2
                        else:
                            std_cell['center_lon'] = (cell.get('min_lon', 0) + cell.get('max_lon', 0)) / 2
                    elif field == 'corners' and field not in cell:
                        # Generate corners from bounds
                        std_cell['corners'] = [
                            [cell.get('min_lon', 0), cell.get('min_lat', 0)],
                            [cell.get('min_lon', 0), cell.get('max_lat', 0)],
                            [cell.get('max_lon', 0), cell.get('max_lat', 0)],
                            [cell.get('max_lon', 0), cell.get('min_lat', 0)]
                        ]
            
            # Copy any additional fields that might be useful
            for key, value in cell.items():
                if key not in std_cell:
                    std_cell[key] = value
            
            standardized.append(std_cell)
        
        return standardized

    def _calculate_cell_area_from_bounds(self, cell: Dict) -> float:
        """Calculate approximate area of a grid cell from its bounds."""
        try:
            if all(key in cell for key in ['min_lat', 'max_lat', 'min_lon', 'max_lon']):
                lat_diff = cell['max_lat'] - cell['min_lat']
                lon_diff = cell['max_lon'] - cell['min_lon']
                
                # Rough approximation using spherical Earth
                avg_lat = np.radians((cell['min_lat'] + cell['max_lat']) / 2)
                lat_km = lat_diff * 111.32  # km per degree latitude
                lon_km = lon_diff * 111.32 * np.cos(avg_lat)  # km per degree longitude
                
                return lat_km * lon_km
            else:
                return self.primary_grid_size_km ** 2  # Default fallback
        except Exception:
            return self.primary_grid_size_km ** 2

    def visualize_grid(self, grid_cells: List[Dict], output_file: str = "city_grid.html") -> None:
        """
        Create a reliable visualization of grid cells on a map background.

        Args:
            grid_cells: List of grid cell dictionaries containing coordinates and metadata
            output_file: Name of the output HTML file
        """
        if not grid_cells:
            self.logger.error("No grid cells to visualize")
            return

        try:
            # Ensure grid cells are standardized
            std_cells = self._standardize_grid_format(grid_cells)
            
            # Create a folium map centered on the first cell's centroid
            first_cell = std_cells[0]
            map_center = [first_cell['center_lat'], first_cell['center_lon']]
            city_map = folium.Map(location=map_center, zoom_start=12)

            # Add progress bar for large numbers of cells
            cells_to_plot = std_cells
            if len(std_cells) > 100:
                cells_to_plot = tqdm(std_cells, desc="Adding grid cells to map")

            # Add grid cells to the map
            for cell in cells_to_plot:
                grid_id = self._get_consistent_grid_id(cell)
                corners = [(lat, lon) for lon, lat in cell['corners']]  # Reverse coordinates for folium
                
                # Create popup with comprehensive info
                popup_text = f"""
                <b>Grid ID:</b> {grid_id}<br>
                <b>Area:</b> {cell['area_km2']:.2f} km²<br>
                <b>Center:</b> ({cell['center_lat']:.4f}, {cell['center_lon']:.4f})<br>
                <b>Bounds:</b> [{cell['min_lat']:.4f}, {cell['min_lon']:.4f}] to [{cell['max_lat']:.4f}, {cell['max_lon']:.4f}]
                """
                
                folium.Polygon(
                    locations=corners,
                    color='blue',
                    fill=True,
                    fill_opacity=0.4,
                    popup=popup_text,
                    tooltip=f"Grid {grid_id}"
                ).add_to(city_map)

            # Add summary information
            total_area = sum(cell['area_km2'] for cell in std_cells)
            summary_html = f"""
            <div style="position: fixed; top: 10px; left: 50px; width: 300px; height: 90px; 
                        background-color: white; border:2px solid grey; z-index:9999; 
                        font-size:14px; padding: 10px">
            <b>Grid Summary</b><br>
            Total Cells: {len(std_cells)}<br>
            Total Area: {total_area:.2f} km²<br>
            Avg Cell Size: {total_area/len(std_cells):.2f} km²
            </div>
            """
            city_map.get_root().html.add_child(folium.Element(summary_html))

            # Save the map to an HTML file
            city_map.save(output_file)
            self.logger.info(f"📊 Grid visualization saved to {output_file}")
            self.logger.info(f"   • {len(std_cells)} cells displayed")
            self.logger.info(f"   • Total area: {total_area:.2f} km²")

        except Exception as e:
            self.logger.error(f"Error during visualization: {str(e)}", exc_info=self.debug_mode)


    def _get_precise_boundary(self, city_name: str, coordinates: tuple = None) -> gpd.GeoDataFrame:
        """
        Get precise city boundary focusing on urban areas.
        
        Args:
            city_name: Name of the city
            coordinates: Optional tuple of (latitude, longitude) coordinates
            
        Returns:
            GeoDataFrame with city boundary
        """
        try:
            # Get administrative boundary - using coordinates if provided
            if coordinates:
                lat, lon = coordinates
                # Ensure lon, lat order for Point creation
                point = Point(lon, lat)
                buffer_dist = 0.05  # ~5km buffer, adjust as needed
                buffered_geom = point.buffer(buffer_dist)
                gdf = gpd.GeoDataFrame(geometry=[buffered_geom], crs="EPSG:4326")
                self.logger.info(f"Using provided coordinates ({lat}, {lon}) with buffer")
            else:
                gdf = ox.geocode_to_gdf(city_name)

            if gdf.empty:
                return None

            # If not fetching features, return the basic boundary
            if not self.fetch_osm_features:
                self.logger.info("Skipping OSM feature fetching for boundary refinement.")
                return gdf

            # Proceed with urban area refinement only if fetch_osm_features is True
            query_area = gdf.geometry.iloc[0]
            if not coordinates:
                query_area = city_name # Use name for place-based query

            # Get urban areas and landuse
            try:
                urban_tags = {
                    'landuse': ['residential', 'commercial', 'industrial', 'retail'],
                    'place': ['city', 'town', 'suburb', 'neighbourhood']
                }
                
                # Fetch features from the specified area
                urban_gdf = ox.features_from_place(query_area, tags=urban_tags) if isinstance(query_area, str) else ox.features_from_polygon(query_area, tags=urban_tags)
                water_gdf = ox.features_from_place(query_area, tags={'natural': ['water', 'bay'], 'water': True}) if isinstance(query_area, str) else ox.features_from_polygon(query_area, tags={'natural': ['water', 'bay'], 'water': True})
                
                if not urban_gdf.empty:
                    # Convert to same CRS
                    urban_gdf = urban_gdf.to_crs(gdf.crs)
                    water_gdf = water_gdf.to_crs(gdf.crs) if not water_gdf.empty else None
                    
                    # Create unions
                    urban_union = unary_union(urban_gdf.geometry)
                    water_union = unary_union(water_gdf.geometry) if water_gdf is not None and not water_gdf.empty else None
                    
                    # Get primary geometry and refine
                    geometry = gdf.geometry.iloc[0]
                    if isinstance(geometry, MultiPolygon):
                        geometry = max(geometry.geoms, key=lambda p: p.area)
                    
                    result = geometry.intersection(urban_union)
                    if water_union is not None:
                        result = result.difference(water_union)
                    
                    # Buffer and simplify for cleaner geometry
                    buffered = result.buffer(0.005)
                    simplified = buffered.simplify(0.001)
                    
                    return gpd.GeoDataFrame(geometry=[simplified], crs="EPSG:4326")

            except Exception as e:
                self.logger.warning(f"Error fetching or processing urban/water data: {str(e)}. Using original boundary.")
            
            # Fallback to the original GDF if feature extraction fails
            return gdf
                    
        except Exception as e:
            self.logger.warning(f"Failed to get boundary for '{city_name}': {str(e)}")
            return None


    def run_comprehensive_test(self, city_name: str) -> Dict:
        """
        Run comprehensive testing and evaluation for a city.
        
        Args:
            city_name: Name of the city to test
            
        Returns:
            Dictionary containing test results and metrics
        """
        test_results = {
            'city': city_name,
            'timestamp': datetime.now().isoformat(),
            'metrics': {},
            'errors': [],
            'warnings': [],
            'performance': {}
        }
        
        try:
            # Time the grid creation
            start_time = datetime.now()
            grid_cells = self.create_city_grid(city_name)
            creation_time = (datetime.now() - start_time).total_seconds()
            
            if not grid_cells:
                test_results['errors'].append("Failed to create grid cells")
                return test_results
                
            # Calculate metrics
            test_results['metrics'] = {
                'total_cells': len(grid_cells),
                'primary_cells': sum(1 for cell in grid_cells if cell['area_km2'] >= self.primary_grid_size_km),
                'flexible_cells': sum(1 for cell in grid_cells if cell['area_km2'] < self.primary_grid_size_km),
                'min_area': min(cell['area_km2'] for cell in grid_cells),
                'max_area': max(cell['area_km2'] for cell in grid_cells),
                'avg_area': np.mean([cell['area_km2'] for cell in grid_cells]),
                'area_std': np.std([cell['area_km2'] for cell in grid_cells])
            }
            
            # Check cell quality
            for i, cell1 in enumerate(grid_cells):
                # Check for valid coordinates
                if any(not (-90 <= lat <= 90) for lat in [cell1['min_lat'], cell1['max_lat']]):
                    test_results['warnings'].append(f"Invalid latitude in cell {cell1['cell_id']}")
                if any(not (-180 <= lon <= 180) for lon in [cell1['min_lon'], cell1['max_lon']]):
                    test_results['warnings'].append(f"Invalid longitude in cell {cell1['cell_id']}")
                
                # Check for missing properties
                required_props = ['cell_id', 'area_km2', 'corners', 'center_lat', 'center_lon']
                missing_props = [prop for prop in required_props if prop not in cell1]
                if missing_props:
                    test_results['warnings'].append(f"Missing properties in cell {cell1['cell_id']}: {missing_props}")
            
            # Performance metrics
            test_results['performance'] = {
                'total_time': creation_time,
                'avg_time_per_cell': creation_time / len(grid_cells) if grid_cells else 0,
                'memory_usage': sys.getsizeof(grid_cells) / (1024 * 1024)  # MB
            }
            
            # Create visualization with test overlay
            try:
                test_output = f"test_{city_name.lower()}_grid.html"
                self.visualize_grid(grid_cells, test_output)
                test_results['visualization'] = test_output
            except Exception as e:
                test_results['warnings'].append(f"Visualization error: {str(e)}")
            
        except Exception as e:
            test_results['errors'].append(f"Test execution error: {str(e)}")
            
        return test_results

    def create_random_grids(self, 
                           base_polygon: Polygon, 
                           num_variations: int = 5,
                           size_variation_range: Tuple[float, float] = (0.5, 2.0),
                           offset_variation_km: float = 0.5,
                           grid_name: str = "random_grids") -> List[List[Dict]]:
        """
        Create multiple random grid variations for robust dataset generation.
        
        This method generates multiple grid layouts with random offsets and sizes
        to create diverse training samples that improve model generalization.
        
        Args:
            base_polygon: The base area to grid
            num_variations: Number of different grid variations to create
            size_variation_range: Tuple of (min_scale, max_scale) for grid size variation
            offset_variation_km: Maximum random offset in kilometers
            grid_name: Name for caching purposes
            
        Returns:
            List of grid cell lists, each representing a different random variation
        """
        self.logger.info(f"🎲 Creating {num_variations} random grid variations...")
        
        random_grids = []
        original_size = self.primary_grid_size_km
        
        # Set random seed for reproducibility if needed
        np.random.seed(42)
        
        for i in range(num_variations):
            self.logger.info(f"Creating random grid variation {i+1}/{num_variations}")
            
            # Random size variation
            size_scale = np.random.uniform(*size_variation_range)
            varied_size = original_size * size_scale
            
            # Random offset
            offset_x = np.random.uniform(-offset_variation_km, offset_variation_km)
            offset_y = np.random.uniform(-offset_variation_km, offset_variation_km)
            
            # Apply offset to polygon
            offset_polygon = self._apply_offset_to_polygon(base_polygon, offset_x, offset_y)
            
            # Create grid with varied size
            old_size = self.primary_grid_size_km
            self.primary_grid_size_km = varied_size
            
            try:
                grid_gdf = self.create_grid_for_polygon(
                    offset_polygon, 
                    grid_name=f"{grid_name}_var_{i}"
                )
                
                if not grid_gdf.empty:
                    grid_cells = grid_gdf.to_dict('records')
                    
                    # Add variation metadata
                    for cell in grid_cells:
                        cell['grid_variation'] = i
                        cell['size_scale'] = size_scale
                        cell['offset_x_km'] = offset_x
                        cell['offset_y_km'] = offset_y
                        if 'grid_id' not in cell:
                            cell['grid_id'] = f"var{i}_{cell.get('cell_id', f'grid_{len(grid_cells)}')}"
                    
                    random_grids.append(grid_cells)
                    self.logger.info(f"✅ Variation {i+1}: {len(grid_cells)} cells (size={varied_size:.2f}km)")
                
            finally:
                # Restore original size
                self.primary_grid_size_km = old_size
        
        self.logger.info(f"🎯 Created {len(random_grids)} random grid variations")
        return random_grids

    def create_multiscale_grids(self, 
                               base_polygon: Polygon,
                               scales: List[float] = None,
                               grid_name: str = "multiscale_grids") -> List[List[Dict]]:
        """
        Create multiple grid scales for multi-resolution analysis.
        
        This generates grids at different scales to capture features at various
        spatial resolutions, improving model robustness across different urban densities.
        
        Args:
            base_polygon: The base area to grid
            scales: List of scale factors relative to primary_grid_size_km
            grid_name: Name for caching purposes
            
        Returns:
            List of grid cell lists, each representing a different scale
        """
        if scales is None:
            scales = [0.5, 0.75, 1.0, 1.5, 2.0]  # Half to double the base size
            
        self.logger.info(f"📏 Creating multiscale grids with {len(scales)} scales: {scales}")
        
        multiscale_grids = []
        original_size = self.primary_grid_size_km
        
        for i, scale in enumerate(scales):
            scaled_size = original_size * scale
            self.logger.info(f"Creating scale {i+1}/{len(scales)}: {scaled_size:.2f}km")
            
            # Temporarily change grid size
            self.primary_grid_size_km = scaled_size
            
            try:
                grid_gdf = self.create_grid_for_polygon(
                    base_polygon, 
                    grid_name=f"{grid_name}_scale_{scale}"
                )
                
                if not grid_gdf.empty:
                    grid_cells = grid_gdf.to_dict('records')
                    
                    # Add scale metadata
                    for cell in grid_cells:
                        cell['grid_scale'] = scale
                        cell['actual_size_km'] = scaled_size
                        if 'grid_id' not in cell:
                            cell['grid_id'] = f"scale{scale}_{cell.get('cell_id', f'grid_{len(grid_cells)}')}"
                    
                    multiscale_grids.append(grid_cells)
                    self.logger.info(f"✅ Scale {scale}: {len(grid_cells)} cells")
                
            finally:
                # Restore original size
                self.primary_grid_size_km = original_size
        
        self.logger.info(f"🎯 Created {len(multiscale_grids)} multiscale grids")
        return multiscale_grids

    def create_comprehensive_grid_dataset(self,
                                        base_polygon: Polygon,
                                        include_random: bool = True,
                                        include_multiscale: bool = True,
                                        num_random_variations: int = 3,
                                        scales: List[float] = None,
                                        grid_name: str = "comprehensive_dataset") -> List[Dict]:
        """
        Create a comprehensive grid dataset combining base, random, and multiscale grids.
        
        This is the main method for creating robust training datasets with diverse
        grid configurations that improve model generalization.
        
        Args:
            base_polygon: The base area to grid
            include_random: Whether to include random grid variations
            include_multiscale: Whether to include multiscale grids
            num_random_variations: Number of random variations to create
            scales: List of scale factors for multiscale grids
            grid_name: Name for caching purposes
            
        Returns:
            Flattened list of all grid cells from all variations
        """
        self.logger.info("🏗️ Creating comprehensive grid dataset for robust ML training...")
        
        all_grid_cells = []
        
        # 1. Base grid (original uniform grid)
        self.logger.info("📐 Creating base uniform grid...")
        base_grid_gdf = self.create_grid_for_polygon(base_polygon, grid_name=f"{grid_name}_base")
        
        if not base_grid_gdf.empty:
            base_grid_cells = base_grid_gdf.to_dict('records')
            for cell in base_grid_cells:
                cell['grid_type'] = 'base'
                cell['grid_variation'] = 0
                cell['grid_scale'] = 1.0
                if 'grid_id' not in cell:
                    cell['grid_id'] = f"base_{cell.get('cell_id', f'grid_{len(base_grid_cells)}')}"
            
            all_grid_cells.extend(base_grid_cells)
            self.logger.info(f"✅ Base grid: {len(base_grid_cells)} cells")
        
        # 2. Random variations
        if include_random:
            self.logger.info("🎲 Adding random grid variations...")
            random_grids = self.create_random_grids(
                base_polygon, 
                num_variations=num_random_variations,
                grid_name=f"{grid_name}_random"
            )
            
            for grid_cells in random_grids:
                for cell in grid_cells:
                    cell['grid_type'] = 'random'
                all_grid_cells.extend(grid_cells)
        
        # 3. Multiscale grids
        if include_multiscale:
            self.logger.info("📏 Adding multiscale grids...")
            multiscale_grids = self.create_multiscale_grids(
                base_polygon, 
                scales=scales,
                grid_name=f"{grid_name}_multiscale"
            )
            
            for grid_cells in multiscale_grids:
                for cell in grid_cells:
                    cell['grid_type'] = 'multiscale'
                all_grid_cells.extend(grid_cells)
        
        self.logger.info(f"🎯 Comprehensive dataset created: {len(all_grid_cells)} total grid cells")
        self.logger.info(f"   📊 Dataset composition:")
        
        # Log composition statistics
        grid_types = {}
        for cell in all_grid_cells:
            grid_type = cell.get('grid_type', 'unknown')
            grid_types[grid_type] = grid_types.get(grid_type, 0) + 1
        
        for grid_type, count in grid_types.items():
            percentage = (count / len(all_grid_cells)) * 100
            self.logger.info(f"      {grid_type}: {count} cells ({percentage:.1f}%)")
        
        return all_grid_cells

    def _apply_offset_to_polygon(self, polygon: Polygon, offset_x_km: float, offset_y_km: float) -> Polygon:
        """
        Apply a spatial offset to a polygon in kilometers.
        
        Args:
            polygon: Input polygon
            offset_x_km: Offset in kilometers along x-axis (longitude direction)
            offset_y_km: Offset in kilometers along y-axis (latitude direction)
            
        Returns:
            Offset polygon
        """
        # Convert km offsets to approximate degree offsets
        # This is a rough approximation; for precise work, use proper projection
        lat_offset = offset_y_km / 111.32  # ~111.32 km per degree latitude
        
        # Longitude offset depends on latitude
        bounds = polygon.bounds
        avg_lat = (bounds[1] + bounds[3]) / 2
        lon_offset = offset_x_km / (111.32 * np.cos(np.radians(avg_lat)))
        
        # Apply offset using shapely's translate method
        from shapely.affinity import translate
        offset_polygon = translate(polygon, xoff=lon_offset, yoff=lat_offset)
        
        return offset_polygon

    def get_debug_info(self) -> Dict:
        """Get debugging information."""
        return self.debug_info

    def run_self_test(self) -> bool:
        """Run self-test to verify functionality."""
        test_cities = ["Amsterdam", "Paris", "London"]
        all_passed = True
        
        for city in test_cities:
            try:
                self.logger.info(f"Testing with {city}...")
                grid_cells = self.create_city_grid(city)
                
                if not grid_cells:
                    self.logger.error(f"Failed to create grid for {city}")
                    all_passed = False
                    continue
                
                # Verify cell properties
                for cell in grid_cells:
                    if not all(key in cell for key in ['cell_id', 'area_km2', 'corners']):
                        self.logger.error(f"Missing required cell properties in {city}")
                        all_passed = False
                        break
                
                self.logger.info(f"Successfully tested {city}")
                
            except Exception as e:
                self.logger.error(f"Test failed for {city}: {str(e)}")
                all_passed = False
                
        return all_passed

# Quick smoke-test (skip heavy OSM feature queries for speed)
if __name__ == "__main__":
    gridder = CityGridding(primary_grid_size_km=1.0, debug_mode=True, fetch_osm_features=False)
    
    # Test cities
    test_cities = ["Amsterdam", "San Francisco", "Paris"]
    
    print("Running comprehensive tests...")
    for city in test_cities:
        print(f"\nTesting {city}...")
        results = gridder.run_comprehensive_test(city)
        
        # Print test results
        print(f"\nResults for {city}:")
        print("Metrics:")
        for key, value in results['metrics'].items():
            print(f"  {key}: {value}")
            
        print("\nPerformance:")
        for key, value in results['performance'].items():
            print(f"  {key}: {value:.3f}")
            
        if results['warnings']:
            print("\nWarnings:")
            for warning in results['warnings']:
                print(f"  - {warning}")
                
        if results['errors']:
            print("\nErrors:")
            for error in results['errors']:
                print(f"  - {error}")