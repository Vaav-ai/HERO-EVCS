#!/usr/bin/env python3
"""
Visualization Module for EV Charging Station Placement

This module provides comprehensive visualization capabilities for EV charging station
placement results, including city maps with station locations, allocation analysis,
and performance comparisons.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import json
import logging

# Publication-quality matplotlib settings
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, MaxNLocator

# Set publication-quality global parameters
mpl.rcParams.update({
    'pdf.fonttype': 42,  # For Adobe Illustrator compatibility
    'ps.fonttype': 42,   # For Adobe Illustrator compatibility
    'font.family': 'Arial',  # Academic standard font
    'font.size': 12,         # Base font size
    'axes.labelsize': 14,    # Axis labels
    'axes.titlesize': 16,    # Title size
    'xtick.labelsize': 12,   # X-axis tick labels
    'ytick.labelsize': 12,   # Y-axis tick labels
    'legend.fontsize': 12,   # Legend font size
    'figure.titlesize': 18,  # Figure title size
    'text.usetex': False,    # Use matplotlib's mathtext instead of LaTeX for portability
    'figure.dpi': 300,       # High resolution
    'savefig.dpi': 300,      # High resolution for saved figures
    'savefig.bbox': 'tight', # Tight bounding box
    'axes.linewidth': 1.2,   # Thicker axes lines
    'lines.linewidth': 2.0,  # Thicker plot lines
    'grid.alpha': 0.3,      # Grid transparency
    'grid.linewidth': 0.8,  # Grid line width
})
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.abspath('.'))

try:
    import folium
    import geopandas as gpd
    from shapely.geometry import Point, Polygon
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    print("⚠️ Folium not available. Interactive maps will be disabled.")

try:
    import contextily as ctx
    CONTEXTILY_AVAILABLE = True
except ImportError:
    CONTEXTILY_AVAILABLE = False
    print("⚠️ Contextily not available. Background maps will be disabled.")


class ChargingStationVisualizer:
    """
    Comprehensive visualization class for EV charging station placement results.
    
    This class provides various visualization methods including:
    - Interactive city maps with station locations
    - Allocation analysis plots
    - Performance comparison charts
    - Grid-based visualization
    """
    
    def __init__(self, logger=None, city_name: str = "Unknown City"):
        """
        Initialize the visualization class.
        
        Args:
            logger: Logger instance for logging
            city_name: Name of the city for dynamic bounds calculation
        """
        self.logger = logger or logging.getLogger(__name__)
        self.output_dir = "./test_results/visualizations"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Dynamic city bounds - will be calculated from actual data
        self.city_name = city_name
        self.city_bounds = None  # Will be calculated dynamically
        
        self.logger.info("🎨 ChargingStationVisualizer initialized")
        self.logger.info(f"   City: {self.city_name}")
        self.logger.info(f"   Output directory: {self.output_dir}")
        self.logger.info(f"   Folium available: {FOLIUM_AVAILABLE}")
        self.logger.info(f"   Contextily available: {CONTEXTILY_AVAILABLE}")
    
    def _calculate_dynamic_bounds(self, all_stations: List[Dict], city_grids_data: List[Dict]) -> Dict[str, float]:
        """
        Calculate dynamic city bounds from station and grid data.
        
        Args:
            all_stations: List of station dictionaries with lat/lon coordinates
            city_grids_data: List of grid cell data with coordinates
            
        Returns:
            Dictionary with min_lat, max_lat, min_lon, max_lon
        """
        self.logger.info("📐 Calculating dynamic city bounds from data...")
        
        # Collect all coordinates
        all_lats = []
        all_lons = []
        
        # Add station coordinates
        for station in all_stations:
            lat = station.get('lat')
            lon = station.get('lon')
            if lat is not None and lon is not None:
                all_lats.append(lat)
                all_lons.append(lon)
        
        # Add grid coordinates
        for grid in city_grids_data:
            all_lats.extend([grid.get('min_lat'), grid.get('max_lat')])
            all_lons.extend([grid.get('min_lon'), grid.get('max_lon')])
        
        if not all_lats or not all_lons:
            # Fallback to default bounds if no data
            self.logger.warning("⚠️ No valid coordinates found, using default bounds")
            return {
                'min_lat': 42.0,
                'max_lat': 43.0,
                'min_lon': -84.0,
                'max_lon': -83.0
            }
        
        # Calculate bounds with padding
        lat_range = max(all_lats) - min(all_lats)
        lon_range = max(all_lons) - min(all_lons)
        
        # Add 10% padding to bounds
        lat_padding = lat_range * 0.1
        lon_padding = lon_range * 0.1
        
        bounds = {
            'min_lat': min(all_lats) - lat_padding,
            'max_lat': max(all_lats) + lat_padding,
            'min_lon': min(all_lons) - lon_padding,
            'max_lon': max(all_lons) + lon_padding
        }
        
        self.logger.info(f"📍 Calculated bounds: lat [{bounds['min_lat']:.4f}, {bounds['max_lat']:.4f}], lon [{bounds['min_lon']:.4f}, {bounds['max_lon']:.4f}]")
        return bounds
    
    def _calculate_smart_text_offset(self, lon: float, lat: float, existing_positions: List[Tuple[float, float]]) -> Tuple[float, float]:
        """
        Calculate smart text offset to avoid overlaps with existing text positions.
        
        Args:
            lon: Longitude of the station
            lat: Latitude of the station
            existing_positions: List of existing text positions
            
        Returns:
            Tuple of (x_offset, y_offset) in points
        """
        # Base offset positions to try (in order of preference)
        base_offsets = [
            (8, 8),    # Top-right
            (8, -8),    # Bottom-right
            (-8, 8),    # Top-left
            (-8, -8),   # Bottom-left
            (12, 0),    # Right
            (-12, 0),   # Left
            (0, 12),    # Top
            (0, -12),   # Bottom
        ]
        
        # Convert existing positions to approximate pixel coordinates
        # This is a rough approximation for overlap detection
        min_distance = 20  # Minimum distance in points
        
        for offset in base_offsets:
            # Calculate potential text position
            potential_x = lon + offset[0] * 0.0001  # Rough conversion
            potential_y = lat + offset[1] * 0.0001
            
            # Check distance from existing positions
            too_close = False
            for existing_x, existing_y in existing_positions:
                distance = ((potential_x - existing_x) ** 2 + (potential_y - existing_y) ** 2) ** 0.5
                if distance < min_distance * 0.0001:  # Convert to coordinate units
                    too_close = True
                    break
            
            if not too_close:
                return offset
        
        # If all positions are too close, use the first one
        return base_offsets[0]
    
    def visualize_exact_station_locations(self, 
                                         all_results: Dict[str, Dict],
                                         city_grids_data: List[Dict],
                                         title: str = "Exact EV Charging Station Locations",
                                         save_path: Optional[str] = None) -> str:
        """
        Create a visualization showing exact lat/lon coordinates of charging stations
        from the best performing method for each grid.
        
        Args:
            all_results: Dictionary containing results for all grids
            city_grids_data: List of grid cell data with coordinates
            title: Title for the visualization
            save_path: Optional path to save the visualization
            
        Returns:
            Path to saved visualization file
        """
        self.logger.info(f"📍 Creating exact station location visualization...")
        
        # Extract best placements from each grid
        all_stations = []
        grid_stations = {}
        
        self.logger.info(f"🔍 Processing {len(all_results)} grids for station extraction...")
        
        for grid_id, grid_data in all_results.items():
            if 'results' not in grid_data:
                self.logger.debug(f"Grid {grid_id} has no 'results' field")
                continue
                
            # Find the best performing method for this grid
            best_method = None
            best_reward = -float('inf')
            
            for method_name, method_result in grid_data['results'].items():
                if 'error' not in method_result:
                    # Try different ways to get the reward
                    reward = method_result.get('reward', 0)
                    if reward == 0:
                        # Try to get reward from metrics
                        metrics = method_result.get('metrics', {})
                        reward = metrics.get('best_reward', 0)
                    if reward == 0:
                        # Try to get reward from simulation evaluation
                        sim_eval = method_result.get('simulation_evaluation', {})
                        reward = sim_eval.get('simulation_reward', 0)
                    
                    if reward > best_reward:
                        best_reward = reward
                        best_method = method_name
            
            if best_method and best_method in grid_data['results']:
                method_result = grid_data['results'][best_method]
                placements = method_result.get('placements', [])
                
                # If no placements found, try to extract from best_placement (for hybrid methods)
                if not placements:
                    best_placement = method_result.get('best_placement', [])
                    if isinstance(best_placement, list) and best_placement:
                        placements = best_placement
                        self.logger.debug(f"Extracted {len(placements)} placements from best_placement for {best_method}")
                    else:
                        self.logger.warning(f"No placements found for {best_method} in grid {grid_id}")
                        continue
                
                self.logger.debug(f"Grid {grid_id}: best method={best_method}, reward={best_reward}, placements={len(placements)}")
                
                grid_stations[grid_id] = {
                    'method': best_method,
                    'reward': best_reward,
                    'stations': placements
                }
                
                # Add stations to overall list
                for i, station in enumerate(placements):
                    station_copy = station.copy()
                    station_copy['grid_id'] = grid_id
                    station_copy['method'] = best_method
                    station_copy['reward'] = best_reward
                    station_copy['station_id'] = f"{grid_id}_{i+1}"
                    all_stations.append(station_copy)
                    
                    # Debug log each station
                    self.logger.debug(f"Added station {station_copy['station_id']}: lat={station_copy.get('lat')}, lon={station_copy.get('lon')}")
            else:
                self.logger.debug(f"Grid {grid_id}: no valid method found")
        
        self.logger.info(f"📊 Station extraction summary: {len(all_stations)} total stations from {len(grid_stations)} grids")
        
        if not all_stations:
            self.logger.warning("No station placements found for visualization")
            return None
        
        # Calculate dynamic bounds from actual data
        self.city_bounds = self._calculate_dynamic_bounds(all_stations, city_grids_data)
        
        # Create matplotlib visualization with enhanced background (better for papers)
        self.logger.info("📊 Creating static PNG visualization with map background...")
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(title, fontsize=20, fontweight='bold')
        
        # 1. Exact Station Locations Map
        self._plot_exact_station_map(ax1, all_stations, city_grids_data)
        
        # 2. Station Distribution by Method
        self._plot_method_distribution(ax2, all_stations)
        
        # 3. Grid Performance Analysis
        self._plot_grid_performance(ax3, grid_stations)
        
        # 4. Station Statistics
        self._plot_station_statistics(ax4, all_stations)
        
        plt.tight_layout()
        
        # Save the visualization
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(self.output_dir, f"exact_station_locations_{timestamp}.png")
        
        # Save as both PDF (for publication) and PNG (for quick viewing)
        pdf_path = save_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✅ Exact station location visualization saved to: {save_path}")
        return save_path
    
    def _plot_exact_station_map(self, ax, all_stations: List[Dict], city_grids_data: List[Dict]):
        """Plot exact station locations on city map with static map background."""
        ax.set_title("Exact EV Charging Station Locations", fontsize=16, fontweight='bold')
        
        # Try to add static map background using contextily with robust approach
        map_background_added = False
        
        try:
            import contextily as ctx
            import requests
            
            # Test internet connectivity first
            try:
                requests.get('https://tile.openstreetmap.org', timeout=3)
                internet_available = True
            except:
                internet_available = False
                self.logger.warning("⚠️ No internet connection - using enhanced background")
            
            if internet_available:
                # FIXED: Use correct contextily providers (only available ones)
                map_sources = [
                    ('OpenStreetMap Mapnik', ctx.providers.OpenStreetMap.Mapnik),
                    ('CartoDB Positron', ctx.providers.CartoDB.Positron),
                    ('CartoDB DarkMatter', ctx.providers.CartoDB.DarkMatter),
                ]
                
                # Add Stamen providers if available (they might not be in all contextily versions)
                try:
                    map_sources.extend([
                        ('Stamen Terrain', ctx.providers.Stamen.Terrain),
                        ('Stamen TonerLite', ctx.providers.Stamen.TonerLite),
                    ])
                except AttributeError:
                    self.logger.debug("Stamen providers not available in this contextily version")
                
                # Set the plot bounds first
                ax.set_xlim(self.city_bounds['min_lon'], self.city_bounds['max_lon'])
                ax.set_ylim(self.city_bounds['min_lat'], self.city_bounds['max_lat'])
                
                for source_name, source in map_sources:
                    try:
                        # Add map background with better error handling
                        ctx.add_basemap(ax, crs='EPSG:4326', source=source, 
                                    alpha=0.6, zorder=0, attribution=False)
                        self.logger.info(f"✅ Added map background using {source_name}")
                        map_background_added = True
                        break
                    except Exception as e:
                        error_msg = str(e)[:100] if len(str(e)) > 100 else str(e)
                        self.logger.warning(f"⚠️ Could not add map background: {source_name} - {error_msg}")
                        continue
            
            if not map_background_added:
                raise Exception("All map sources failed")
                
        except ImportError:
            self.logger.warning("⚠️ Contextily not available. Using enhanced background.")
        except Exception as e:
            self.logger.debug(f"Map background error: {e}")
            self.logger.info("🔄 Using enhanced fallback background")
        
        # Enhanced fallback background if no map was added
        if not map_background_added:
            # Create a more realistic map-like background
            ax.set_facecolor('#e6f3ff')
            
            # Add a subtle gradient effect to simulate terrain
            import numpy as np
            x = np.linspace(self.city_bounds['min_lon'], self.city_bounds['max_lon'], 100)
            y = np.linspace(self.city_bounds['min_lat'], self.city_bounds['max_lat'], 100)
            X, Y = np.meshgrid(x, y)
            
            # Create a subtle terrain-like pattern
            Z = np.sin(X * 50) * np.cos(Y * 50) * 0.1
            ax.contourf(X, Y, Z, levels=20, alpha=0.1, cmap='terrain', zorder=0)
            
            # Add subtle grid lines
            ax.grid(True, alpha=0.2, color='#2E8B57', linestyle='-', linewidth=0.3)
            
            # Add map-like elements
            ax.text(0.02, 0.98, self.city_name, transform=ax.transAxes, 
                fontsize=14, alpha=0.9, verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.95, 
                        edgecolor='#2E8B57', linewidth=2))
            
            # Add coordinate labels
            ax.text(0.02, 0.02, 'Geographic Coordinates (WGS84)', transform=ax.transAxes, 
                fontsize=11, alpha=0.7, verticalalignment='bottom',
                style='italic', color='#2E8B57')
            
            # Add border
            for spine in ax.spines.values():
                spine.set_edgecolor('#2E8B57')
                spine.set_linewidth(2)
            
            # Add compass rose
            ax.text(0.95, 0.95, 'N', transform=ax.transAxes, fontsize=16, 
                fontweight='bold', color='#2E8B57', alpha=0.8)
            
            self.logger.info("✅ Using enhanced map-like background")        
        # Create grid DataFrame
        grid_df = pd.DataFrame(city_grids_data)
        
        # Plot grid cells as semi-transparent overlay
        for _, row in grid_df.iterrows():
            width = row['max_lon'] - row['min_lon']
            height = row['max_lat'] - row['min_lat']
            
            rect = patches.Rectangle(
                (row['min_lon'], row['min_lat']), 
                width, height,
                linewidth=1, edgecolor='blue', facecolor='lightblue', alpha=0.1
            )
            ax.add_patch(rect)
        
        # Plot exact station locations with enhanced styling
        method_colors = {
            'hybrid_ucb': '#FF4444',           # Bright red
            'hybrid_epsilon_greedy': '#4444FF', # Bright blue
            'hybrid_thompson_sampling': '#FF44FF', # Bright magenta/pink
            'kmeans': '#FF8844',               # Orange
            'random': '#8844FF',               # Purple
            'uniform': '#8B4513'               # Brown
        }
        
        method_markers = {
            'hybrid_ucb': 'o',
            'hybrid_epsilon_greedy': 's',
            'hybrid_thompson_sampling': '^',
            'kmeans': 'D',
            'random': 'v',
            'uniform': 'p'
        }
        
        # Plot stations with exact lat/lon positioning
        plotted_stations = 0
        skipped_stations = 0
        
        for station in all_stations:
            lat = station.get('lat')
            lon = station.get('lon')
            method = station.get('method', 'unknown')
            station_id = station.get('station_id', f"{station.get('grid_id', '')}")
            
            # Debug logging for station coordinates
            self.logger.debug(f"Station {station_id}: lat={lat}, lon={lon}, method={method}")
            
            if lat is not None and lon is not None:
                # Check if station is within city bounds
                if (self.city_bounds['min_lat'] <= lat <= self.city_bounds['max_lat'] and 
                    self.city_bounds['min_lon'] <= lon <= self.city_bounds['max_lon']):
                    
                    color = method_colors.get(method, '#000000')
                    marker = method_markers.get(method, 'o')
                    
                    # Enhanced marker styling for better visibility on map (smaller size for many stations)
                    ax.scatter(lon, lat, c=color, marker=marker, s=60, alpha=0.9, 
                              edgecolors='white', linewidth=1, zorder=10)
                    
                    # Skip text labels to avoid clutter - stations are identifiable by color/marker
                    # Users can hover over stations in interactive maps for details
                    
                    plotted_stations += 1
                else:
                    self.logger.warning(f"Station {station_id} outside city bounds: lat={lat}, lon={lon}")
                    skipped_stations += 1
            else:
                self.logger.warning(f"Station {station_id} has invalid coordinates: lat={lat}, lon={lon}")
                skipped_stations += 1
        
        self.logger.info(f"📊 Station plotting summary: {plotted_stations} plotted, {skipped_stations} skipped")
        
        # Set bounds and labels with proper scaling (only if no map background was added)
        if not map_background_added:
            ax.set_xlim(self.city_bounds['min_lon'], self.city_bounds['max_lon'])
            ax.set_ylim(self.city_bounds['min_lat'], self.city_bounds['max_lat'])
        
        ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
        ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
        
        # Remove default grid for cleaner map appearance
        ax.grid(False)
        
        # Add legend with smart positioning to avoid hiding stations
        legend_elements = []
        for method, color in method_colors.items():
            marker = method_markers.get(method, 'o')
            legend_elements.append(plt.Line2D([0], [0], marker=marker, color='w', 
                                           markerfacecolor=color, markersize=8, 
                                           markeredgecolor='white', markeredgewidth=1,
                                           label=method.replace('_', ' ').title()))
        
        # Place legend completely outside the plot area to avoid any overlap
        legend = ax.legend(handles=legend_elements, loc='center left', fontsize=9,
                          frameon=True, fancybox=True, shadow=True,
                          bbox_to_anchor=(1.02, 0.5))  # Position outside right edge
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.95)
        legend.get_frame().set_edgecolor('gray')
        legend.get_frame().set_linewidth(1)
    
    
    def create_static_map_with_folium(self, all_stations: List[Dict], city_grids_data: List[Dict], 
                                    title: str = "Exact EV Charging Station Locations") -> Optional[str]:
        """Create a static map using folium with exact station locations."""
        try:
            import folium
            
            # Calculate center point
            lats = [station['lat'] for station in all_stations]
            lons = [station['lon'] for station in all_stations]
            center_lat = sum(lats) / len(lats)
            center_lon = sum(lons) / len(lons)
            
            # Create folium map
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=13,
                tiles='OpenStreetMap'
            )
            
            # Add stations with different colors for each method
            method_colors = {
                'hybrid_ucb': 'red',
                'hybrid_epsilon_greedy': 'blue', 
                'hybrid_thompson_sampling': 'magenta',
                'kmeans': 'orange',
                'random': 'purple',
                'uniform': 'brown'
            }
            
            for station in all_stations:
                color = method_colors.get(station['method'], 'black')
                folium.CircleMarker(
                    location=[station['lat'], station['lon']],
                    radius=5,
                    popup=f"{station['station_id']} ({station['method']})",
                    color='white',
                    weight=1,
                    fillColor=color,
                    fillOpacity=0.8
                ).add_to(m)
            
            # Add grid boundaries
            for grid in city_grids_data:
                folium.Rectangle(
                    bounds=[[grid['min_lat'], grid['min_lon']], 
                           [grid['max_lat'], grid['max_lon']]],
                    color='blue',
                    weight=2,
                    fillOpacity=0.1,
                    fillColor='lightblue'
                ).add_to(m)
            
            # Add legend
            legend_html = '''
            <div style="position: fixed; 
                        bottom: 50px; left: 50px; width: 200px; height: 120px; 
                        background-color: white; border:2px solid grey; z-index:9999; 
                        font-size:14px; padding: 10px">
            <p><b>Charging Station Methods</b></p>
            <p><i class="fa fa-circle" style="color:red"></i> Hybrid UCB</p>
            <p><i class="fa fa-circle" style="color:blue"></i> Hybrid Epsilon-Greedy</p>
            <p><i class="fa fa-circle" style="color:magenta"></i> Hybrid Thompson Sampling</p>
            <p><i class="fa fa-circle" style="color:orange"></i> K-Means</p>
            <p><i class="fa fa-circle" style="color:purple"></i> Random</p>
            <p><i class="fa fa-circle" style="color:brown"></i> Uniform</p>
            </div>
            '''
            m.get_root().html.add_child(folium.Element(legend_html))
            
            # Save HTML
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            html_path = f"{self.output_dir}/exact_stations_map_{timestamp}.html"
            m.save(html_path)
            
            self.logger.info(f"✅ Interactive map with static background saved to: {html_path}")
            return html_path
            
        except ImportError:
            self.logger.warning("⚠️ Folium not available for static map creation")
            return None
        except Exception as e:
            self.logger.warning(f"⚠️ Could not create folium map: {e}")
            return None
    
    def _plot_method_distribution(self, ax, all_stations: List[Dict]):
        """Plot distribution of stations by method."""
        ax.set_title("Station Distribution by Method", fontsize=14, fontweight='bold')
        
        # Count stations by method
        method_counts = {}
        for station in all_stations:
            method = station.get('method', 'unknown')
            method_counts[method] = method_counts.get(method, 0) + 1
        
        if method_counts:
            methods = list(method_counts.keys())
            counts = list(method_counts.values())
            
            # Create bar chart
            bars = ax.bar(methods, counts, alpha=0.7, edgecolor='black')
            ax.set_ylabel('Number of Stations', fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{count}', ha='center', va='bottom', fontweight='bold')
            
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'No station data available', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_grid_performance(self, ax, grid_stations: Dict[str, Dict]):
        """Plot grid performance analysis."""
        ax.set_title("Grid Performance Analysis", fontsize=14, fontweight='bold')
        
        if not grid_stations:
            ax.text(0.5, 0.5, 'No grid data available', ha='center', va='center', transform=ax.transAxes)
            return
        
        # Extract performance data
        grid_ids = list(grid_stations.keys())
        rewards = [grid_stations[gid]['reward'] for gid in grid_ids]
        station_counts = [len(grid_stations[gid]['stations']) for gid in grid_ids]
        
        # Create scatter plot
        scatter = ax.scatter(station_counts, rewards, alpha=0.7, s=100, c=rewards, 
                           cmap='viridis', edgecolors='black')
        
        ax.set_xlabel('Number of Stations', fontsize=12)
        ax.set_ylabel('Best Reward', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Reward', fontsize=12)
        
        # Add grid ID labels
        for i, grid_id in enumerate(grid_ids):
            ax.annotate(grid_id, (station_counts[i], rewards[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    def _plot_station_statistics(self, ax, all_stations: List[Dict]):
        """Plot station statistics."""
        ax.set_title("Station Statistics", fontsize=14, fontweight='bold')
        
        if not all_stations:
            ax.text(0.5, 0.5, 'No station data available', ha='center', va='center', transform=ax.transAxes)
            return
        
        # Calculate statistics
        total_stations = len(all_stations)
        unique_grids = len(set(station.get('grid_id') for station in all_stations))
        method_counts = {}
        for station in all_stations:
            method = station.get('method', 'unknown')
            method_counts[method] = method_counts.get(method, 0) + 1
        
        # Create statistics text
        stats_text = f"""Total Stations: {total_stations}
Grids with Stations: {unique_grids}
Average per Grid: {total_stations/unique_grids:.2f}

Method Distribution:"""
        
        for method, count in method_counts.items():
            percentage = (count / total_stations) * 100
            stats_text += f"\n  {method.replace('_', ' ').title()}: {count} ({percentage:.1f}%)"
        
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=11, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    def visualize_all_stations_on_map(self, 
                                    station_allocations: Dict[str, int],
                                    city_grids_data: List[Dict],
                                    results_data: Optional[Dict] = None,
                                    title: str = "EV Charging Station Placement",
                                    save_path: Optional[str] = None) -> str:
        """
        Create a comprehensive visualization of all charging stations on a city map.
        
        Args:
            station_allocations: Dictionary mapping grid_id to number of stations
            city_grids_data: List of grid cell data with coordinates
            results_data: Optional results data for performance visualization
            title: Title for the visualization
            save_path: Optional path to save the visualization
            
        Returns:
            Path to saved visualization file
        """
        self.logger.info(f"🗺️ Creating comprehensive station placement visualization...")
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(title, fontsize=20, fontweight='bold')
        
        # 1. Station Allocation Map
        self._plot_station_allocation_map(ax1, station_allocations, city_grids_data)
        
        # 2. Station Count Distribution
        self._plot_station_distribution(ax2, station_allocations)
        
        # 3. Grid Coverage Analysis
        self._plot_grid_coverage(ax3, station_allocations, city_grids_data)
        
        # 4. Performance Summary (if results available)
        if results_data:
            self._plot_performance_summary(ax4, results_data)
        else:
            self._plot_allocation_statistics(ax4, station_allocations)
        
        plt.tight_layout()
        
        # Save the visualization
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(self.output_dir, f"station_placement_comprehensive_{timestamp}.png")
        
        # Save as both PDF (for publication) and PNG (for quick viewing)
        pdf_path = save_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✅ Comprehensive visualization saved to: {save_path}")
        return save_path
    
    def _plot_station_allocation_map(self, ax, station_allocations: Dict[str, int], city_grids_data: List[Dict]):
        """Plot station allocation on city map."""
        ax.set_title("Station Allocation by Grid", fontsize=14, fontweight='bold')
        
        # Ensure city_bounds is set
        if self.city_bounds is None:
            # Calculate bounds from grid data
            all_lats = []
            all_lons = []
            for grid in city_grids_data:
                all_lats.extend([grid.get('min_lat'), grid.get('max_lat')])
                all_lons.extend([grid.get('min_lon'), grid.get('max_lon')])
            
            if all_lats and all_lons:
                lat_range = max(all_lats) - min(all_lats)
                lon_range = max(all_lons) - min(all_lons)
                lat_padding = lat_range * 0.1
                lon_padding = lon_range * 0.1
                
                self.city_bounds = {
                    'min_lat': min(all_lats) - lat_padding,
                    'max_lat': max(all_lats) + lat_padding,
                    'min_lon': min(all_lons) - lon_padding,
                    'max_lon': max(all_lons) + lon_padding
                }
            else:
                # Fallback bounds
                self.city_bounds = {
                    'min_lat': 42.0,
                    'max_lat': 43.0,
                    'min_lon': -84.0,
                    'max_lon': -83.0
                }
        
        # Create grid DataFrame
        grid_df = pd.DataFrame(city_grids_data)
        
        # Add station counts
        grid_df['stations'] = grid_df['grid_id'].map(station_allocations).fillna(0)
        
        # Create color map based on station count
        max_stations = max(station_allocations.values()) if station_allocations else 1
        colors = plt.cm.Reds(np.linspace(0.3, 1.0, max_stations + 1))
        
        # Plot each grid cell
        for _, row in grid_df.iterrows():
            # Create rectangle for grid cell
            width = row['max_lon'] - row['min_lon']
            height = row['max_lat'] - row['min_lat']
            
            # Color based on station count
            station_count = int(row['stations'])
            color = colors[min(station_count, max_stations)]
            
            # Add rectangle
            rect = patches.Rectangle(
                (row['min_lon'], row['min_lat']), 
                width, height,
                linewidth=1, edgecolor='black', facecolor=color, alpha=0.7
            )
            ax.add_patch(rect)
            
            # Add station count label
            if station_count > 0:
                center_lon = (row['min_lon'] + row['max_lon']) / 2
                center_lat = (row['min_lat'] + row['max_lat']) / 2
                ax.text(center_lon, center_lat, str(station_count), 
                       ha='center', va='center', fontweight='bold', fontsize=10)
        
        # Set bounds and labels
        ax.set_xlim(self.city_bounds['min_lon'], self.city_bounds['max_lon'])
        ax.set_ylim(self.city_bounds['min_lat'], self.city_bounds['max_lat'])
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=max_stations))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Number of Stations', fontsize=12)
    
    def _plot_station_distribution(self, ax, station_allocations: Dict[str, int]):
        """Plot distribution of station counts."""
        ax.set_title("Station Count Distribution", fontsize=14, fontweight='bold')
        
        if not station_allocations:
            ax.text(0.5, 0.5, 'No station allocations', ha='center', va='center', transform=ax.transAxes)
            return
        
        station_counts = list(station_allocations.values())
        
        # Create histogram
        bins = range(0, max(station_counts) + 2)
        ax.hist(station_counts, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Add statistics
        mean_count = np.mean(station_counts)
        median_count = np.median(station_counts)
        
        ax.axvline(mean_count, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_count:.1f}')
        ax.axvline(median_count, color='green', linestyle='--', linewidth=2, label=f'Median: {median_count:.1f}')
        
        ax.set_xlabel('Stations per Grid', fontsize=12)
        ax.set_ylabel('Number of Grids', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_grid_coverage(self, ax, station_allocations: Dict[str, int], city_grids_data: List[Dict]):
        """Plot grid coverage analysis."""
        ax.set_title("Grid Coverage Analysis", fontsize=14, fontweight='bold')
        
        # Calculate coverage statistics
        total_grids = len(city_grids_data)
        grids_with_stations = len(station_allocations)
        coverage_percentage = (grids_with_stations / total_grids) * 100
        
        # Create pie chart
        labels = ['Grids with Stations', 'Grids without Stations']
        sizes = [grids_with_stations, total_grids - grids_with_stations]
        colors = ['lightgreen', 'lightcoral']
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        
        # Add statistics text
        stats_text = f'Total Grids: {total_grids}\nCovered Grids: {grids_with_stations}\nCoverage: {coverage_percentage:.1f}%'
        ax.text(1.3, 0.5, stats_text, transform=ax.transAxes, fontsize=11, 
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def _plot_performance_summary(self, ax, results_data: Dict):
        """Plot performance summary if results are available."""
        ax.set_title("Performance Summary", fontsize=14, fontweight='bold')
        
        # Extract performance metrics
        successful_grids = sum(1 for grid_data in results_data.values() 
                             if isinstance(grid_data, dict) and 'successful_methods' in grid_data)
        total_grids = len(results_data)
        
        # Create performance bar chart
        methods = ['Successful Grids', 'Failed Grids']
        counts = [successful_grids, total_grids - successful_grids]
        colors = ['lightgreen', 'lightcoral']
        
        bars = ax.bar(methods, counts, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{count}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Number of Grids', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_allocation_statistics(self, ax, station_allocations: Dict[str, int]):
        """Plot allocation statistics."""
        ax.set_title("Allocation Statistics", fontsize=14, fontweight='bold')
        
        if not station_allocations:
            ax.text(0.5, 0.5, 'No station allocations', ha='center', va='center', transform=ax.transAxes)
            return
        
        station_counts = list(station_allocations.values())
        total_stations = sum(station_counts)
        
        # Create statistics text
        stats_text = f"""Total Stations: {total_stations}
Grids with Stations: {len(station_allocations)}
Average per Grid: {np.mean(station_counts):.2f}
Max per Grid: {max(station_counts)}
Min per Grid: {min(station_counts)}
Standard Deviation: {np.std(station_counts):.2f}"""
        
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=12, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    def create_interactive_map(self, 
                             station_allocations: Dict[str, int],
                             city_grids_data: List[Dict],
                             results_data: Optional[Dict] = None) -> str:
        """
        Create an interactive map using Folium (if available).
        
        Args:
            station_allocations: Dictionary mapping grid_id to number of stations
            city_grids_data: List of grid cell data with coordinates
            results_data: Optional results data for popup information
            
        Returns:
            Path to saved HTML map file
        """
        if not FOLIUM_AVAILABLE:
            self.logger.warning("⚠️ Folium not available. Cannot create interactive map.")
            return None
        
        self.logger.info("🌐 Creating interactive map...")
        
        # Calculate center point
        center_lat = (self.city_bounds['min_lat'] + self.city_bounds['max_lat']) / 2
        center_lon = (self.city_bounds['min_lon'] + self.city_bounds['max_lon']) / 2
        
        # Create map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=13,
            tiles='OpenStreetMap'
        )
        
        # Add grid cells
        for grid_data in city_grids_data:
            grid_id = grid_data['grid_id']
            station_count = station_allocations.get(grid_id, 0)
            
            # Create polygon for grid cell
            polygon_coords = [
                [grid_data['min_lat'], grid_data['min_lon']],
                [grid_data['max_lat'], grid_data['min_lon']],
                [grid_data['max_lat'], grid_data['max_lon']],
                [grid_data['min_lat'], grid_data['max_lon']],
                [grid_data['min_lat'], grid_data['min_lon']]
            ]
            
            # Color based on station count
            if station_count == 0:
                color = 'lightgray'
                opacity = 0.3
            else:
                # Scale color intensity
                max_stations = max(station_allocations.values()) if station_allocations else 1
                intensity = station_count / max_stations
                color = f'red'
                opacity = 0.3 + (intensity * 0.5)
            
            # Add polygon
            folium.Polygon(
                locations=polygon_coords,
                color='black',
                weight=1,
                fillColor=color,
                fillOpacity=opacity,
                popup=f"Grid: {grid_id}<br>Stations: {station_count}"
            ).add_to(m)
            
            # Add station count label
            if station_count > 0:
                center_lat_grid = (grid_data['min_lat'] + grid_data['max_lat']) / 2
                center_lon_grid = (grid_data['min_lon'] + grid_data['max_lon']) / 2
                
                folium.Marker(
                    [center_lat_grid, center_lon_grid],
                    popup=f"Grid: {grid_id}<br>Stations: {station_count}",
                    icon=folium.DivIcon(
                        html=f'<div style="font-size: 12px; font-weight: bold; color: black; text-align: center;">{station_count}</div>',
                        icon_size=(20, 20),
                        icon_anchor=(10, 10)
                    )
                ).add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>Station Count Legend</b></p>
        <p><i class="fa fa-square" style="color:lightgray"></i> No Stations</p>
        <p><i class="fa fa-square" style="color:red"></i> With Stations</p>
        <p><i class="fa fa-circle" style="color:black"></i> Station Count</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Save map
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        map_path = os.path.join(self.output_dir, f"interactive_station_map_{timestamp}.html")
        m.save(map_path)
        
        self.logger.info(f"✅ Interactive map saved to: {map_path}")
        return map_path
    
    def visualize_single_grid_results(self, 
                                    grid_id: str,
                                    station_count: int,
                                    results_data: Dict,
                                    save_path: Optional[str] = None) -> str:
        """
        Create visualization for a single grid's results.
        
        Args:
            grid_id: Grid identifier
            station_count: Number of stations in this grid
            results_data: Results data for this grid
            save_path: Optional path to save the visualization
            
        Returns:
            Path to saved visualization file
        """
        self.logger.info(f"📊 Creating visualization for grid {grid_id}...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Grid {grid_id} Analysis ({station_count} stations)', fontsize=16, fontweight='bold')
        
        # 1. Method Performance Comparison
        if 'results' in results_data and results_data['results']:
            self._plot_method_comparison(ax1, results_data['results'])
        else:
            ax1.set_title("Method Performance Comparison", fontsize=12, fontweight='bold')
            ax1.text(0.5, 0.5, 'No method results available', ha='center', va='center', 
                    transform=ax1.transAxes, fontsize=12)
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
            ax1.axis('off')
        
        # 2. Station Count Distribution
        ax2.set_title("Station Allocation", fontsize=12, fontweight='bold')
        ax2.bar(['This Grid'], [station_count], color='skyblue', alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Number of Stations', fontsize=10)
        ax2.set_ylim(0, max(station_count * 1.2, 1))
        ax2.grid(True, alpha=0.3)
        
        # Add value label on bar
        ax2.text(0, station_count + 0.05, str(station_count), ha='center', va='bottom', 
                fontweight='bold', fontsize=12)
        
        # 3. Performance Metrics
        if 'comparison_metrics' in results_data and results_data['comparison_metrics']:
            self._plot_performance_metrics(ax3, results_data['comparison_metrics'])
        else:
            ax3.set_title("Performance Metrics", fontsize=12, fontweight='bold')
            ax3.text(0.5, 0.5, 'No performance metrics available', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)
            ax3.axis('off')
        
        # 4. Test Data Summary
        if 'test_data_summary' in results_data and results_data['test_data_summary']:
            self._plot_test_data_summary(ax4, results_data['test_data_summary'])
        else:
            ax4.set_title("Grid Information", fontsize=12, fontweight='bold')
            # Create a simple info display
            info_text = f"""Grid ID: {grid_id}
Stations: {station_count}
Status: Analysis Complete"""
            ax4.text(0.1, 0.9, info_text, transform=ax4.transAxes, fontsize=11, 
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(self.output_dir, f"grid_{grid_id}_analysis_{timestamp}.png")
        
        # Save as both PDF (for publication) and PNG (for quick viewing)
        pdf_path = save_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✅ Grid visualization saved to: {save_path}")
        return save_path
    
    def _plot_method_comparison(self, ax, results: Dict):
        """Plot method performance comparison."""
        ax.set_title("Method Performance Comparison", fontsize=12, fontweight='bold')
        
        methods = []
        rewards = []
        colors = []
        
        for method_name, result in results.items():
            if 'error' not in result and 'reward' in result:
                methods.append(method_name)
                rewards.append(result['reward'])
                colors.append('lightgreen' if result.get('success', False) else 'lightcoral')
        
        if methods:
            bars = ax.bar(methods, rewards, color=colors, alpha=0.7, edgecolor='black')
            ax.set_ylabel('Reward')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, reward in zip(bars, rewards):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{reward:.3f}', ha='center', va='bottom', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'No valid results', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_performance_metrics(self, ax, metrics: Dict):
        """Plot performance metrics."""
        ax.set_title("Performance Metrics", fontsize=12, fontweight='bold')
        
        # Extract key metrics
        metric_names = []
        metric_values = []
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                metric_names.append(key.replace('_', ' ').title())
                metric_values.append(value)
        
        if metric_names:
            bars = ax.bar(metric_names, metric_values, color='lightblue', alpha=0.7, edgecolor='black')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'No metrics available', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_test_data_summary(self, ax, summary: Dict):
        """Plot test data summary."""
        ax.set_title("Test Data Summary", fontsize=12, fontweight='bold')
        
        # Create summary text
        summary_text = f"""Trajectory Count: {summary.get('trajectory_count', 'N/A')}
Grid Bounds: {summary.get('grid_bounds', 'N/A')}"""
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')


def visualize_all_stations_placement(station_allocations: Dict[str, int],
                                   city_grids_data: List[Dict],
                                   results_data: Optional[Dict] = None,
                                   title: str = "EV Charging Station Placement",
                                   interactive: bool = True,
                                   city_name: str = "Unknown City") -> Dict[str, str]:
    """
    Convenience function to create comprehensive station placement visualizations.
    
    Args:
        station_allocations: Dictionary mapping grid_id to number of stations
        city_grids_data: List of grid cell data with coordinates
        results_data: Optional results data for performance visualization
        title: Title for the visualization
        interactive: Whether to create interactive map
        city_name: Name of the city for dynamic bounds calculation
        
    Returns:
        Dictionary with paths to created visualizations
    """
    visualizer = ChargingStationVisualizer(city_name=city_name)
    
    # Create comprehensive visualization
    comprehensive_path = visualizer.visualize_all_stations_on_map(
        station_allocations, city_grids_data, results_data, title
    )
    
    # Create interactive map if requested and available
    interactive_path = None
    if interactive and FOLIUM_AVAILABLE:
        interactive_path = visualizer.create_interactive_map(
            station_allocations, city_grids_data, results_data
        )
    
    return {
        'comprehensive': comprehensive_path,
        'interactive': interactive_path
    }


def visualize_exact_station_locations(all_results: Dict[str, Dict],
                                    city_grids_data: List[Dict],
                                    title: str = "Exact EV Charging Station Locations",
                                    city_name: str = "Unknown City") -> str:
    """
    Convenience function to create exact station location visualization.
    
    Args:
        all_results: Dictionary containing results for all grids
        city_grids_data: List of grid cell data with coordinates
        title: Title for the visualization
        city_name: Name of the city for dynamic bounds calculation
        
    Returns:
        Path to saved visualization file
    """
    visualizer = ChargingStationVisualizer(city_name=city_name)
    
    # Create exact station location visualization
    exact_path = visualizer.visualize_exact_station_locations(
        all_results, city_grids_data, title
    )
    
    return exact_path
