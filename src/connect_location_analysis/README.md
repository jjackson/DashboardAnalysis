# ConnectLocationAnalysis

GPS location analysis dashboard with Nigerian ward-level granularity.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r ../../requirements.txt
   ```

2. **Configure Superset access:**
   - Ensure your `.env` file has Superset credentials:
     ```
     SUPERSET_URL=your_superset_url
     SUPERSET_USERNAME=your_username
     SUPERSET_PASSWORD=your_password
     ```

3. **Download GRID3 boundary files (one-time setup):**
   - Go to https://grid3.org/geospatial-data-nigeria
   - Download these 3 GeoJSON files to `data/shapefiles/`:
     - Ward boundaries
     - LGA boundaries  
     - State boundaries

4. **Run analysis with fresh data:**
   ```bash
   python run_analysis.py --fetch-fresh
   ```

5. **Update cache with ward data:**
   ```bash
   python update_cache.py
   ```

### Alternative: Use existing data

```bash
python run_analysis.py
```

## Scripts

- **`run_analysis.py`** - Main dashboard generator. Creates timestamped HTML output with interactive maps and filtering. Automatically uses the most recent data file and supports `--fetch-fresh` flag to download fresh data from Superset.

- **`update_cache.py`** - Cache updater specific to Nigeria. Adds Nigerian ward data to existing geocoding cache using GRID3 boundaries. Also overwrites LGA if it disagrees with OSM's original answer.

## Data Structure

- `data/cache/` - Geocoding cache (automatically managed)
- `data/shapefiles/` - GRID3 boundary files (manual download required)
- `output/` - Generated HTML dashboards

Dashboard auto-opens in browser with ward-level analysis for Nigerian locations.
