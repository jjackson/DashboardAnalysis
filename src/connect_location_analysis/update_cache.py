#!/usr/bin/env python3
"""
Update Cache Script for Nigerian Ward Analysis

This script updates the existing geocoding cache to add ward-level data for Nigerian coordinates.
It uses GRID3 GeoJSON boundary files to perform point-in-polygon lookups.

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
        print("❌ Cache file not found!")
        return False
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_file = Path(f'data/cache/geocoding_cache_backup_{timestamp}.json')
    
    shutil.copy2(cache_file, backup_file)
    print(f"✅ Cache backed up to: {backup_file}")
    return True

def load_cache():
    """Load existing geocoding cache"""
    cache_file = Path('data/cache/geocoding_cache.json')
    with open(cache_file, 'r', encoding='utf-8') as f:
        cache = json.load(f)
    print(f"📊 Loaded cache with {len(cache)} entries")
    return cache

def save_cache(cache):
    """Save updated cache back to file"""
    cache_file = Path('data/cache/geocoding_cache.json')
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved updated cache with {len(cache)} entries")

def load_boundaries():
    """Load GRID3 ward and LGA boundary files"""
    print("🗺️ Loading GRID3 boundary files...")
    
    # Load ward boundaries - using actual filename from screenshot
    ward_file = Path('data/shapefiles/Nigeria_-_Ward_Boundaries.geojson')
    if not ward_file.exists():
        print(f"❌ Ward file not found: {ward_file}")
        print("   Available files in data/shapefiles/:")
        for f in Path('data/shapefiles/').glob('*'):
            print(f"   - {f.name}")
        return None, None
    
    # Load LGA boundaries - using actual filename from screenshot
    lga_file = Path('data/shapefiles/GRID3_NGA_-_Operational_LGA_Boundaries.geojson')
    if not lga_file.exists():
        print(f"❌ LGA file not found: {lga_file}")
        print("   Available files in data/shapefiles/:")
        for f in Path('data/shapefiles/').glob('*'):
            print(f"   - {f.name}")
        return None, None
    
    wards_gdf = gpd.read_file(ward_file)
    lgas_gdf = gpd.read_file(lga_file)
    
    print(f"   ✅ Loaded {len(wards_gdf)} ward boundaries")
    print(f"   ✅ Loaded {len(lgas_gdf)} LGA boundaries")
    
    # Show column names for debugging
    print(f"   📋 Ward columns: {list(wards_gdf.columns)}")
    print(f"   📋 LGA columns: {list(lgas_gdf.columns)}")
    
    return wards_gdf, lgas_gdf

def find_ward_and_lga(lat, lng, wards_gdf, lgas_gdf):
    """Find ward and LGA for given coordinates using point-in-polygon lookup"""
    point = Point(lng, lat)  # Note: GeoJSON uses lng, lat order
    
    ward_name = None
    lga_name = None
    
    # Find ward
    ward_matches = wards_gdf[wards_gdf.geometry.contains(point)]
    if not ward_matches.empty:
        # Try common column names for ward
        ward_cols = ['ward_name', 'name', 'Ward_Name', 'WARD_NAME', 'wardname', 'Ward', 'WARD']
        for col in ward_cols:
            if col in ward_matches.columns:
                ward_name = ward_matches.iloc[0][col]
                break
        
        # If no standard column found, use first non-geometry column
        if ward_name is None:
            non_geom_cols = [col for col in ward_matches.columns if col != 'geometry']
            if non_geom_cols:
                ward_name = ward_matches.iloc[0][non_geom_cols[0]]
    
    # Find LGA
    lga_matches = lgas_gdf[lgas_gdf.geometry.contains(point)]
    if not lga_matches.empty:
        # Try common column names for LGA
        lga_cols = ['lga_name', 'name', 'LGA_Name', 'LGA_NAME', 'lganame', 'LGA', 'lga']
        for col in lga_cols:
            if col in lga_matches.columns:
                lga_name = lga_matches.iloc[0][col]
                break
        
        # If no standard column found, use first non-geometry column
        if lga_name is None:
            non_geom_cols = [col for col in lga_matches.columns if col != 'geometry']
            if non_geom_cols:
                lga_name = lga_matches.iloc[0][non_geom_cols[0]]
    
    return ward_name, lga_name

def update_nigerian_entries(cache, wards_gdf, lgas_gdf):
    """Update cache entries for Nigerian coordinates"""
    print("🇳🇬 Processing Nigerian coordinates...")
    
    nigerian_count = 0
    updated_count = 0
    
    for cache_key, entry in cache.items():
        if entry.get('country') == 'Nigeria':
            nigerian_count += 1
            
            # Extract coordinates from cache key
            try:
                lat_str, lng_str = cache_key.split(',')
                lat, lng = float(lat_str), float(lng_str)
            except:
                print(f"   ⚠️ Could not parse coordinates from key: {cache_key}")
                continue
            
            # Find ward and LGA
            ward_name, lga_name = find_ward_and_lga(lat, lng, wards_gdf, lgas_gdf)
            
            # Update entry
            if ward_name:
                entry['ward'] = ward_name
                updated_count += 1
                
                # Overwrite local_admin with LGA if we found one
                if lga_name and lga_name != entry.get('local_admin'):
                    old_local_admin = entry.get('local_admin', 'Unknown')
                    entry['local_admin'] = lga_name
                    if nigerian_count <= 10:  # Show first 10 for debugging
                        print(f"   📍 {lat:.4f},{lng:.4f}: {old_local_admin} → {lga_name} | Ward: {ward_name}")
            
            # Progress indicator
            if nigerian_count % 100 == 0:
                print(f"   Progress: {nigerian_count} Nigerian entries processed, {updated_count} updated")
    
    print(f"✅ Processed {nigerian_count} Nigerian entries")
    print(f"✅ Updated {updated_count} entries with ward data")
    
    return cache

def main():
    """Main execution function"""
    print("🚀 Starting Nigerian Ward Cache Update")
    print("=" * 50)
    
    # Step 1: Backup existing cache
    if not backup_cache():
        return
    
    # Step 2: Load cache
    cache = load_cache()
    
    # Step 3: Load boundary files
    wards_gdf, lgas_gdf = load_boundaries()
    if wards_gdf is None or lgas_gdf is None:
        print("❌ Could not load boundary files. Exiting.")
        return
    
    # Step 4: Update Nigerian entries
    updated_cache = update_nigerian_entries(cache, wards_gdf, lgas_gdf)
    
    # Step 5: Save updated cache
    save_cache(updated_cache)
    
    print("=" * 50)
    print("🎉 Cache update completed successfully!")
    print("\nNext steps:")
    print("1. Run your analysis script to see ward-level data")
    print("2. Check the dashboard for new ward filtering options")

if __name__ == "__main__":
    main()
