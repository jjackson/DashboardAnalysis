# ConnectLocationAnalysis

GPS location analysis dashboard with Nigerian ward-level granularity.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r ../../requirements.txt
   ```

2. **Download GRID3 boundary files:**
   - Go to https://grid3.org/geospatial-data-nigeria
   - Download these 3 GeoJSON files to `data/shapefiles/`:
     - Ward boundaries
     - LGA boundaries  
     - State boundaries

3. **Run analysis:**
   ```bash
   python run_analysis.py
   ```

4. **Update cache with ward data:**
   ```bash
   python update_cache.py
   ```

## Scripts

- **`run_analysis.py`** - Main dashboard generator. Creates timestamped HTML output with interactive maps and filtering. Will 
create JSON cache as it resolves locations for GPS and attemp to retreive cache before making API call to OSM.

- **`update_cache.py`** - Cache updater specific to Nigera. Adds Nigerian ward data to existing geocoding cache using GRID3 boundaries.  Also overwrites LGA if it disagrees with OSM's original
answer.

## Data Structure

- `data/cache/` - Geocoding cache (automatically managed)
- `data/shapefiles/` - GRID3 boundary files (manual download required)
- `output/` - Generated HTML dashboards

Dashboard auto-opens in browser with ward-level analysis for Nigerian locations.
