import sumolib
import os

class SUMONetwork:
    """Handles SUMO network file operations"""
    def __init__(self, network_file_path: str):
        self.network_file_path = network_file_path
        self.net_file_path = self._get_network_file()
        self.net = sumolib.net.readNet(self.net_file_path)

    def _get_network_file(self) -> str:
        """Get the appropriate SUMO network file"""
        # Check if the input file is already a .net.xml file
        if self.network_file_path.endswith('.net.xml'):
            if os.path.exists(self.network_file_path):
                print(f"Using existing SUMO network file: {self.network_file_path}")
                return self.network_file_path
            else:
                raise FileNotFoundError(f"Network file not found: {self.network_file_path}")
        
        # If it's an OSM file, convert it to .net.xml
        net_file_path = f'{os.path.splitext(self.network_file_path)[0]}.net.xml'
        
        # Check if network file already exists
        if os.path.exists(net_file_path):
            print(f"Network file already exists: {net_file_path}")
            return net_file_path
            
        print(f"Converting OSM to network: {self.network_file_path} -> {net_file_path}")
        
        # More robust netconvert command with proper error handling and better parameters
        cmd = (f'netconvert --osm-files {self.network_file_path} -o {net_file_path} '
               f'--geometry.remove --roundabouts.guess --ramps.guess '
               f'--junctions.join --tls.guess-signals --tls.discard-simple '
               f'--remove-edges.by-vclass rail_slow,rail_fast,bicycle,pedestrian '
               f'--keep-edges.by-vclass passenger,delivery,taxi,bus,coach,motorcycle '
               f'--remove-edges.isolated --junctions.corner-detail 5 '
               f'--output.street-names --output.original-names '
               f'--proj.utm --ignore-errors.edge-type')
        
        print(f"Running: {cmd}")
        result = os.system(cmd)
        
        if result != 0:
            print(f"Warning: netconvert returned code {result}, but continuing...")
        
        if not os.path.exists(net_file_path):
            raise FileNotFoundError(f"Net file not created: {net_file_path}")
        
        print(f"Network conversion completed: {net_file_path}")
        return net_file_path

    def get_edges(self):
        """Get all edges in the network"""
        return self.net.getEdges()
    
    def find_closest_edge(self, lat: float, lon: float, radius: float = 200.0):
        """Find the closest edge to given coordinates"""
        try:
            x, y = self.net.convertLonLat2XY(lon, lat)
            edges = self.net.getNeighboringEdges(x, y, radius)
            if edges:
                # Return the closest edge
                closest_edge = min(edges, key=lambda edge: edge[1])
                return closest_edge[0]
        except Exception as e:
            print(f"Warning: Could not find edge for coordinates ({lat}, {lon}): {e}")
        return None

    def getNeighboringEdges(self, x: float, y: float, r: float = 1000.0):
        """Get neighboring edges within radius r of given coordinates."""
        try:
            return self.net.getNeighboringEdges(x, y, r)
        except Exception as e:
            print(f"Warning: Could not get neighboring edges for coordinates ({x}, {y}): {e}")
            return []
    
    def convertLonLat2XY(self, lon: float, lat: float):
        """Convert lon/lat to network XY coordinates."""
        return self.net.convertLonLat2XY(lon, lat)
    
    def convertXY2LonLat(self, x: float, y: float):
        """Convert network XY coordinates to lon/lat."""
        return self.net.convertXY2LonLat(x, y)