#!/usr/bin/env python3
"""
Update Cache Script for Nigerian Ward Analysis

Adds ward-level data to geocoding cache for Nigerian coordinates using GRID3 boundaries.
Usage: python update_cache.py
"""

import json
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime
import shutil
from pathlib import Path

def backup_cache():
    """Create timestamped backup of current cache file"""
    cache_file = Path('data/cache/geocoding_cache.json')
    if not cache_file.exists():
        print("Error: Cache file not found")
        return False
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_file = Path(f'data/cache/geocoding_cache_backup_{timestamp}.json')
    shutil.copy2(cache_file, backup_file)
    return True

def load_cache():
    """Load existing geocoding cache"""
    cache_file = Path('data/cache/geocoding_cache.json')
    with open(cache_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_cache(cache):
    """Save updated cache back to file"""
    cache_file = Path('data/cache/geocoding_cache.json')
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

def load_boundaries():
    """Load GRID3 ward and LGA boundary files"""
    # Load ward boundaries
    ward_file = Path('data/shapefiles/Nigeria_-_Ward_Boundaries.geojson')
    if not ward_file.exists():
        print(f"Error: Ward file not found: {ward_file}")
        print("Available files:")
        for f in Path('data/shapefiles/').glob('*'):
            print(f"  {f.name}")
        return None, None
    
    # Load LGA boundaries
    lga_file = Path('data/shapefiles/GRID3_NGA_-_Operational_LGA_Boundaries.geojson')
    if not lga_file.exists():
        print(f"Error: LGA file not found: {lga_file}")
        print("Available files:")
        for f in Path('data/shapefiles/').glob('*'):
            print(f"  {f.name}")
        return None, None
    
    wards_gdf = gpd.read_file(ward_file)
    lgas_gdf = gpd.read_file(lga_file)
    
    return wards_gdf, lgas_gdf

def find_ward_and_lga(lat, lng, wards_gdf, lgas_gdf):
    """Find ward and LGA for given coordinates using point-in-polygon lookup"""
    point = Point(lng, lat)  # GeoJSON uses lng, lat order
    
    ward_name = None
    lga_name = None
    
    # Check which ward polygon contains this point
    ward_matches = wards_gdf[wards_gdf.geometry.contains(point)]
    if not ward_matches.empty:
        # Try common column name variations
        ward_cols = ['ward_name', 'name', 'Ward_Name', 'WARD_NAME', 'wardname', 'Ward', 'WARD']
        for col in ward_cols:
            if col in ward_matches.columns:
                ward_name = ward_matches.iloc[0][col]
                break
        
        # Fallback: use first non-geometry column
        if ward_name is None:
            non_geom_cols = [col for col in ward_matches.columns if col != 'geometry']
            if non_geom_cols:
                ward_name = ward_matches.iloc[0][non_geom_cols[0]]
    
    # Check which LGA polygon contains this point
    lga_matches = lgas_gdf[lgas_gdf.geometry.contains(point)]
    if not lga_matches.empty:
        # Try common column name variations
        lga_cols = ['lga_name', 'name', 'LGA_Name', 'LGA_NAME', 'lganame', 'LGA', 'lga']
        for col in lga_cols:
            if col in lga_matches.columns:
                lga_name = lga_matches.iloc[0][col]
                break
        
        # Fallback: use first non-geometry column
        if lga_name is None:
            non_geom_cols = [col for col in lga_matches.columns if col != 'geometry']
            if non_geom_cols:
                lga_name = lga_matches.iloc[0][non_geom_cols[0]]
    
    return ward_name, lga_name

def update_nigerian_entries(cache, wards_gdf, lgas_gdf, verbose=True):
    """Update cache entries for Nigerian coordinates with ward data"""
    nigerian_count = 0
    updated_count = 0
    
    for cache_key, entry in cache.items():
        if entry.get('country') == 'Nigeria':
            nigerian_count += 1
            
            # Parse coordinates from cache key (format: "lat,lng")
            try:
                lat_str, lng_str = cache_key.split(',')
                lat, lng = float(lat_str), float(lng_str)
            except:
                continue
            
            # Perform spatial lookup
            ward_name, lga_name = find_ward_and_lga(lat, lng, wards_gdf, lgas_gdf)
            
            # Add ward field if found
            if ward_name:
                entry['ward'] = ward_name
                updated_count += 1
                
                # Overwrite LGA with GRID3 data (more accurate than OSM)
                if lga_name and lga_name != entry.get('local_admin'):
                    entry['local_admin'] = lga_name
    
    if verbose:
        print(f"Processed {nigerian_count} Nigerian locations, added ward data to {updated_count}")
    
    return cache

def update_cache_from_grid3(create_backup=True, verbose=True):
    """Update cache with GRID3 ward data for Nigerian coordinates
    
    Args:
        create_backup: Whether to create a timestamped backup of the cache
        verbose: Whether to print progress information
        
    Returns:
        bool: True if successful, False otherwise
    """
    if verbose:
        print("Updating cache with GRID3 ward data...")
    
    # Backup existing cache
    if create_backup:
        if not backup_cache():
            return False
    
    # Load cache
    cache = load_cache()
    
    # Load GRID3 boundary files
    wards_gdf, lgas_gdf = load_boundaries()
    if wards_gdf is None or lgas_gdf is None:
        return False
    
    # Update Nigerian entries with ward data
    updated_cache = update_nigerian_entries(cache, wards_gdf, lgas_gdf, verbose)
    
    # Save updated cache
    save_cache(updated_cache)
    
    if verbose:
        print("Cache update complete")
    
    return True

def main():
    """Main execution function for command-line usage"""
    print("=" * 50)
    success = update_cache_from_grid3(create_backup=True, verbose=True)
    print("=" * 50)
    
    if not success:
        print("Cache update failed")

if __name__ == "__main__":
    main()