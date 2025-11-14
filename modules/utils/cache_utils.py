import os
import shutil
import pickle
import hashlib
from pathlib import Path
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class CacheManager:
    """Utility for managing cache files across the EV placement system."""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_cache_path(self, cache_type: str, cache_key: str) -> Path:
        """Get the full path for a cache file."""
        return self.cache_dir / f"{cache_type}_{cache_key}.pkl"
    
    def exists(self, cache_type: str, cache_key: str) -> bool:
        """Check if a cache file exists."""
        return self.get_cache_path(cache_type, cache_key).exists()
    
    def load(self, cache_type: str, cache_key: str) -> Optional[object]:
        """Load data from cache."""
        cache_path = self.get_cache_path(cache_type, cache_key)
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
                logger.info(f"Loaded {cache_type} cache: {cache_key}")
                return data
        except Exception as e:
            logger.warning(f"Failed to load {cache_type} cache {cache_key}: {e}")
            return None
    
    def save(self, cache_type: str, cache_key: str, data: object) -> bool:
        """Save data to cache."""
        try:
            cache_path = self.get_cache_path(cache_type, cache_key)
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved {cache_type} cache: {cache_key}")
            return True
        except Exception as e:
            logger.warning(f"Failed to save {cache_type} cache {cache_key}: {e}")
            return False
    
    def clear_cache(self, cache_type: Optional[str] = None):
        """Clear cache files."""
        if cache_type:
            # Clear specific cache type
            pattern = f"{cache_type}_*.pkl"
            for cache_file in self.cache_dir.glob(pattern):
                cache_file.unlink()
            logger.info(f"Cleared {cache_type} cache files")
        else:
            # Clear all cache
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir()
            logger.info("Cleared all cache files")
    
    def get_cache_info(self) -> Dict:
        """Get information about cached data."""
        cache_info = {
            'total_files': 0,
            'total_size_mb': 0,
            'cache_types': {}
        }
        
        if not self.cache_dir.exists():
            return cache_info
        
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_info['total_files'] += 1
            cache_info['total_size_mb'] += cache_file.stat().st_size / (1024 * 1024)
            
            # Parse cache type from filename
            cache_type = cache_file.stem.split('_')[0]
            if cache_type not in cache_info['cache_types']:
                cache_info['cache_types'][cache_type] = {
                    'count': 0,
                    'size_mb': 0
                }
            cache_info['cache_types'][cache_type]['count'] += 1
            cache_info['cache_types'][cache_type]['size_mb'] += cache_file.stat().st_size / (1024 * 1024)
        
        return cache_info
    
    def list_cached_items(self, cache_type: Optional[str] = None) -> List[Dict]:
        """List cached items with details."""
        items = []
        
        if not self.cache_dir.exists():
            return items
        
        pattern = f"{cache_type}_*.pkl" if cache_type else "*.pkl"
        for cache_file in self.cache_dir.glob(pattern):
            stat = cache_file.stat()
            items.append({
                'name': cache_file.stem,
                'type': cache_file.stem.split('_')[0],
                'key': '_'.join(cache_file.stem.split('_')[1:]),
                'size_mb': stat.st_size / (1024 * 1024),
                'modified': stat.st_mtime
            })
        
        return items

def create_cache_key(data: Dict) -> str:
    """Create a unique cache key from a dictionary of parameters."""
    key_string = str(sorted(data.items()))
    return hashlib.md5(key_string.encode()).hexdigest()[:16]