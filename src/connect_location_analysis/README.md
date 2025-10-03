# ConnectLocationAnalysis

GPS location analysis dashboard with Nigerian ward-level granularity.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r ../../requirements.txt
   ```

2. **Configure Superset (in root `.env` file):**
   ```
   SUPERSET_URL=your_superset_url
   SUPERSET_USERNAME=your_username
   SUPERSET_PASSWORD=your_password
   ```

3. **Download GRID3 boundary files (one-time setup):**
   - Go to https://grid3.org/geospatial-data-nigeria
   - Download these GeoJSON files to `data/shapefiles/`:
     - Ward boundaries
     - LGA boundaries

4. **Run analysis:**
   ```bash
   python run_analysis.py --fetch-fresh
   ```

That's it! Dashboard auto-opens in your browser.

## Usage

**Fetch fresh data and create dashboard:**
```bash
python run_analysis.py --fetch-fresh
```

**Use cached data (no Superset call):**
```bash
python run_analysis.py
```

**Use only locations with existing geocode cache (no OSM API calls):**
```bash
python run_analysis.py --cached-only
```

## How It Works

1. Pulls GPS location data from Superset
2. Reverse geocodes coordinates to country/state/LGA using OpenStreetMap
3. Adds Nigerian ward data using GRID3 boundaries
4. Creates interactive dashboard with filtering by location hierarchy
5. All geocoding results are cached for fast subsequent runs

## Data Structure

- `data/cache/` - Geocoding cache (auto-managed)
- `data/shapefiles/` - GRID3 boundary files (manual download)
- `data/` - CSV data files from Superset
- `output/` - Generated HTML dashboards (timestamped)

## Scripts

- **`run_analysis.py`** - Main script. Fetches data from Superset, geocodes locations, adds ward data, and generates dashboard.
- **`update_cache.py`** - Manual utility to add GRID3 ward data to existing cache (normally called automatically by run_analysis.py).