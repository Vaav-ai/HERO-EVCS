#!/usr/bin/env python3
"""
Cache Management Utility for EV Placement Project
================================================

This script helps manage cache files for the preprocessing pipeline.

Usage:
------
.. code-block:: bash

    # Show cache status
    python -m modules.utils.manage_cache --status

    # Clear all cache
    python -m modules.utils.manage_cache --clear-all

    # Clear specific cache type
    python -m modules.utils.manage_cache --clear-type osm_features

    # List all cached items
    python -m modules.utils.manage_cache --list
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from modules.utils.cache_utils import CacheManager
from modules.utils.log_configs import setup_logging

def main():
    """Main function to handle cache management commands."""
    parser = argparse.ArgumentParser(
        description="Manage cache files for EV placement preprocessing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show cache status and statistics"
    )
    
    parser.add_argument(
        "--list",
        action="store_true", 
        help="List all cached items with details"
    )
    
    parser.add_argument(
        "--clear-all",
        action="store_true",
        help="Clear all cache files"
    )
    
    parser.add_argument(
        "--clear-type",
        type=str,
        help="Clear cache files of specific type (e.g., 'osm_features', 'demand_scores')"
    )
    
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="cache",
        help="Cache directory path"
    )
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Initialize cache manager
    cache_manager = CacheManager(cache_dir=args.cache_dir)
    
    if args.status:
        show_cache_status(cache_manager)
    elif args.list:
        list_cache_items(cache_manager)
    elif args.clear_all:
        clear_all_cache(cache_manager)
    elif args.clear_type:
        clear_cache_type(cache_manager, args.clear_type)
    else:
        parser.print_help()

def show_cache_status(cache_manager: CacheManager):
    """Show comprehensive cache status."""
    print("🗂️  Cache Status Report")
    print("=" * 50)
    
    info = cache_manager.get_cache_info()
    
    print(f"📊 Total Files: {info['total_files']}")
    print(f"💾 Total Size: {info['total_size_mb']:.2f} MB")
    print()
    
    if info['cache_types']:
        print("📂 Cache Types:")
        for cache_type, stats in info['cache_types'].items():
            print(f"   {cache_type:15} | {stats['count']:3} files | {stats['size_mb']:8.2f} MB")
    else:
        print("✨ No cache files found")
    
    print()

def list_cache_items(cache_manager: CacheManager):
    """List all cached items with details."""
    print("📋 Cached Items")
    print("=" * 80)
    
    items = cache_manager.list_cached_items()
    
    if not items:
        print("✨ No cache files found")
        return
    
    # Sort by type and then by size
    items.sort(key=lambda x: (x['type'], -x['size_mb']))
    
    print(f"{'Type':<15} | {'Key':<25} | {'Size (MB)':<10} | {'Modified'}")
    print("-" * 80)
    
    for item in items:
        import datetime
        modified_date = datetime.datetime.fromtimestamp(item['modified']).strftime('%Y-%m-%d %H:%M')
        print(f"{item['type']:<15} | {item['key'][:25]:<25} | {item['size_mb']:<10.2f} | {modified_date}")

def clear_all_cache(cache_manager: CacheManager):
    """Clear all cache files with confirmation."""
    print("⚠️  WARNING: This will delete ALL cache files!")
    
    info = cache_manager.get_cache_info()
    if info['total_files'] == 0:
        print("✨ No cache files to clear")
        return
    
    print(f"📊 Files to delete: {info['total_files']}")
    print(f"💾 Total size: {info['total_size_mb']:.2f} MB")
    
    confirm = input("\n🤔 Are you sure? (yes/no): ").lower().strip()
    if confirm in ['yes', 'y']:
        cache_manager.clear_cache()
        print("✅ All cache files cleared!")
    else:
        print("❌ Operation cancelled")

def clear_cache_type(cache_manager: CacheManager, cache_type: str):
    """Clear cache files of specific type with confirmation."""
    print(f"⚠️  WARNING: This will delete all '{cache_type}' cache files!")
    
    items = cache_manager.list_cached_items(cache_type)
    if not items:
        print(f"✨ No '{cache_type}' cache files found")
        return
    
    total_size = sum(item['size_mb'] for item in items)
    print(f"📊 Files to delete: {len(items)}")
    print(f"💾 Total size: {total_size:.2f} MB")
    
    confirm = input(f"\n🤔 Clear all '{cache_type}' cache? (yes/no): ").lower().strip()
    if confirm in ['yes', 'y']:
        cache_manager.clear_cache(cache_type)
        print(f"✅ All '{cache_type}' cache files cleared!")
    else:
        print("❌ Operation cancelled")

if __name__ == "__main__":
    main()


