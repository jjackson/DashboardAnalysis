import pandas as pd
import plotly.express as px
import webbrowser
import os
import json
import requests
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from geopy.geocoders import Nominatim
from collections import defaultdict
import time
import glob

def find_latest_data_file():
    """Find the most recent connect location analysis data file"""
    data_dir = Path('data')
    
    # Look for connect_location_analysis files first
    connect_files = list(data_dir.glob('connect_location_analysis_*.csv'))
    if connect_files:
        # Sort by modification time, newest first
        latest_file = max(connect_files, key=lambda x: x.stat().st_mtime)
        print(f"Using latest connect location data: {latest_file.name}")
        return latest_file
    
    # Fallback to old sqllab files
    sqllab_files = list(data_dir.glob('sqllab_untitled_query_*.csv'))
    if sqllab_files:
        latest_file = max(sqllab_files, key=lambda x: x.stat().st_mtime)
        print(f"Using fallback data file: {latest_file.name}")
        return latest_file
    
    # No data files found
    print("No data files found in data/ directory")
    print("Run with --fetch-fresh to download fresh data from Superset")
    return None

def load_data(cached_only=False, fetch_fresh=False):
    """Load GPS location data from CSV"""
    
    # Fetch fresh data if requested
    if fetch_fresh:
        print("Fetching fresh data from Superset...")
        import sys
        sys.path.append(str(Path(__file__).parent.parent / 'utils'))
        from superset_extractor import SupersetExtractor
        from sql_queries import SQL_CONNECT_LOCATION_ANALYSIS
        
        extractor = SupersetExtractor()
        extractor.authenticate()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"connect_location_analysis_{timestamp}"
        extractor.export_query_to_csv(SQL_CONNECT_LOCATION_ANALYSIS, output_filename, verbose=True)
        extractor.close()
    
    # Find the latest data file
    data_file = find_latest_data_file()
    if data_file is None:
        raise FileNotFoundError("No data files found. Run with --fetch-fresh first.")
    
    df = pd.read_csv(data_file)
    
    if cached_only:
        print(f"📊 Loading full dataset: {len(df)} rows")
        print(f"🎯 CACHED-ONLY MODE: Will only use locations with existing cache entries")
        
        # Load cache to filter data
        geocode_cache = load_geocoding_cache()
        
        # Filter to only rows that have cache entries
        cached_rows = []
        for idx, row in df.iterrows():
            lat, lng = row['latitude'], row['longitude']
            cache_key = f"{lat:.4f},{lng:.4f}"
            if cache_key in geocode_cache:
                cached_rows.append(idx)
        
        df = df.iloc[cached_rows].reset_index(drop=True)
        print(f"🎯 Found {len(df)} locations with cached geocoding data")
        
        # Convert date column to datetime (handle mixed formats)
        df['date'] = pd.to_datetime(df['date'], format='mixed')
        
        # Add geocoding data from cache (no API calls)
        df = add_geographic_boundaries_from_cache(df)
    else:
        # Use full dataset now that caching is working
        print(f"📊 Loading full dataset: {len(df)} rows")
        print(f"🎯 Ready to process all locations with smart caching")
        
        # Convert date column to datetime (handle mixed formats)
        df['date'] = pd.to_datetime(df['date'], format='mixed')
        
        # Add reverse geocoding for hierarchical filtering
        df = add_geographic_boundaries(df)
    
    return df

def load_geocoding_cache():
    """Load existing geocoding cache from JSON file"""
    cache_file = Path('data/cache/geocoding_cache.json')
    cache_file.parent.mkdir(exist_ok=True)
    
    if cache_file.exists():
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            print(f"📋 Loaded geocoding cache with {len(cache)} entries")
            return cache
        except Exception as e:
            print(f"⚠️ Error loading cache: {e}")
            return {}
    else:
        print("📋 No existing cache found, starting fresh")
        return {}

def save_geocoding_cache(cache):
    """Save geocoding cache to JSON file immediately"""
    cache_file = Path('data/cache/geocoding_cache.json')
    cache_file.parent.mkdir(exist_ok=True)
    
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️ Error saving cache: {e}")

def add_geographic_boundaries_from_cache(df):
    """Add geographic boundaries using only cached data (no API calls)"""
    print("🌍 Adding geographic boundaries from cache only...")
    
    # Load existing geocode cache
    geocode_cache = load_geocoding_cache()
    
    # Add columns for geographic hierarchy
    df['country'] = 'Unknown'
    df['state_province'] = 'Unknown' 
    df['local_admin'] = 'Unknown'
    df['ward'] = 'Unknown'
    
    cache_hits = 0
    
    print(f"   Processing {len(df)} cached locations...")
    
    for idx in range(len(df)):
        row = df.iloc[idx]
        lat, lng = row['latitude'], row['longitude']
        
        # Create cache key (rounded to ~100m precision to allow for GPS noise)
        cache_key = f"{lat:.4f},{lng:.4f}"
        
        if cache_key in geocode_cache:
            cached = geocode_cache[cache_key]
            df.at[idx, 'country'] = cached['country']
            df.at[idx, 'state_province'] = cached['state_province']
            df.at[idx, 'local_admin'] = cached['local_admin']
            df.at[idx, 'ward'] = cached.get('ward', 'Unknown')  # Ward data only available for some locations
            cache_hits += 1
    
    print(f"✅ Cache-only processing complete!")
    print(f"   📊 Cache hits: {cache_hits}")
    print(f"   🎯 Found {cache_hits} locations with geographic data")
    
    return df

def add_geographic_boundaries(df):
    """Add country, state/province, and local administrative boundaries with JSON caching"""
    print("🌍 Adding geographic boundaries...")
    
    # Load existing geocode cache
    geocode_cache = load_geocoding_cache()
    
    # Add columns for geographic hierarchy
    df['country'] = 'Unknown'
    df['state_province'] = 'Unknown' 
    df['local_admin'] = 'Unknown'
    df['ward'] = 'Unknown'
    
    # Track stats
    cache_hits = 0
    api_calls = 0
    
    # Initialize geocoder (only if needed)
    geolocator = None
    
    print(f"   Processing {len(df)} locations...")
    
    for idx in range(len(df)):
        row = df.iloc[idx]
        lat, lng = row['latitude'], row['longitude']
        
        # Create cache key (rounded to ~100m precision to allow for GPS noise)
        cache_key = f"{lat:.4f},{lng:.4f}"
        
        # Progress indicator
        if idx % 100 == 0 or idx < 20:
            print(f"   Processing location {idx + 1}/{len(df)}: {lat}, {lng}")
        elif idx % 10 == 0:
            print(f"   Progress: {idx + 1}/{len(df)} ({((idx + 1)/len(df)*100):.1f}%)")
        
        # Check cache first
        if cache_key in geocode_cache:
            cached = geocode_cache[cache_key]
            df.at[idx, 'country'] = cached['country']
            df.at[idx, 'state_province'] = cached['state_province']
            df.at[idx, 'local_admin'] = cached['local_admin']
            df.at[idx, 'ward'] = cached.get('ward', 'Unknown')  # Ward data only available for some locations
            cache_hits += 1
            if idx % 100 == 0 or idx < 20:
                print(f"      💾 Cache hit: {cached['country']}, {cached['state_province']}, {cached['local_admin']}")
        else:
            # Need to make API call
            if geolocator is None:
                geolocator = Nominatim(user_agent="connect_location_analysis")
                
            try:
                if idx % 100 == 0 or idx < 20:
                    print(f"      🌐 Making API call...")
                
                location = geolocator.reverse(f"{lat}, {lng}", language='en', exactly_one=True)
                
                if location and location.raw.get('address'):
                    addr = location.raw['address']
                    
                    # Extract hierarchical boundaries
                    country = addr.get('country', 'Unknown')
                    state = (addr.get('state') or addr.get('province') or 
                            addr.get('region') or addr.get('county') or 'Unknown')
                    local = (addr.get('city') or addr.get('town') or addr.get('village') or 
                            addr.get('municipality') or addr.get('suburb') or 'Unknown')
                    
                    # Update dataframe
                    df.at[idx, 'country'] = country
                    df.at[idx, 'state_province'] = state
                    df.at[idx, 'local_admin'] = local
                    
                    # Add to cache and save immediately
                    geocode_cache[cache_key] = {
                        'country': country,
                        'state_province': state,
                        'local_admin': local,
                        'full_address': str(location),
                        'cached_at': datetime.now().isoformat()
                    }
                    save_geocoding_cache(geocode_cache)
                    
                    api_calls += 1
                    if idx % 100 == 0 or idx < 20:
                        print(f"      ✅ Geocoded & cached: {country}, {state}, {local}")
                else:
                    print(f"      ❌ No geocoding result found")
                    # Cache the "not found" result too
                    geocode_cache[cache_key] = {
                        'country': 'Unknown',
                        'state_province': 'Unknown',
                        'local_admin': 'Unknown',
                        'full_address': 'Not found',
                        'cached_at': datetime.now().isoformat()
                    }
                    save_geocoding_cache(geocode_cache)
                    api_calls += 1
                    
                # Rate limiting - be nice to Nominatim
                time.sleep(1.1)
                    
            except Exception as e:
                print(f"      ⚠️ Error geocoding: {e}")
                continue
    
    geocoded_count = df[df['country'] != 'Unknown'].shape[0]
    print(f"✅ Geocoding complete!")
    print(f"   📊 Cache hits: {cache_hits}, API calls: {api_calls}")
    print(f"   🎯 Found {geocoded_count} locations with geographic data")
    print(f"   💾 Cache now contains {len(geocode_cache)} entries")
    
    return df

def create_charts(df):
    """Create interactive dashboard with Leaflet map and filtering"""
    
    # Prepare data for JavaScript
    locations_data = []
    for _, row in df.iterrows():
        locations_data.append({
            'flw_id': int(row['flw_id']),
            'opp_name': str(row['opp_name']),
            'date': row['date'].strftime('%Y-%m-%d %H:%M:%S'),
            'lat': float(row['latitude']),
            'lng': float(row['longitude']),
            'country': str(row['country']) if pd.notna(row['country']) else 'Unknown',
            'state_province': str(row['state_province']) if pd.notna(row['state_province']) else 'Unknown',
            'local_admin': str(row['local_admin']) if pd.notna(row['local_admin']) else 'Unknown',
            'ward': str(row['ward']) if pd.notna(row['ward']) else 'Unknown',
            'accuracy': float(row['accuracy_in_m'])
        })
    
    # Create summary statistics
    total_users = len(df)
    countries = df['country'].value_counts().to_dict()
    
    # Time range analysis - handle timezone-aware dates
    now = pd.Timestamp.now(tz='UTC')
    df['date'] = df['date'].dt.tz_convert('UTC') if df['date'].dt.tz is not None else df['date'].dt.tz_localize('UTC')
    df['days_since_visit'] = (now - df['date']).dt.days
    
    time_ranges = {
        'Last 7 days': len(df[df['days_since_visit'] <= 7]),
        'Last 30 days': len(df[df['days_since_visit'] <= 30]),
        'Last 90 days': len(df[df['days_since_visit'] <= 90]),
        'Last 365 days': len(df[df['days_since_visit'] <= 365]),
        'Over 1 year': len(df[df['days_since_visit'] > 365])
    }
    
    return create_dashboard_html(locations_data, total_users, countries, time_ranges)

def create_dashboard_html(locations_data, total_users, countries, time_ranges):
    """Generate the complete dashboard HTML with Leaflet map"""
    
    dashboard_html = f"""
    <div class="dashboard-container">
        <div class="controls-panel">
            <!-- <h2>Location Analysis</h2> -->
            
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>Total Users</h3>
                    <div class="stat-number">{total_users:,}</div>
                </div>
                <div class="stat-card">
                    <h3>Countries</h3>
                    <div class="stat-number">{len(countries)}</div>
                </div>
            </div>
            
            <div class="filter-section">
                <h3>Time Range Filter</h3>
                <select id="timeFilter" onchange="filterByTime()">
                    <option value="all">All Time</option>
                    <option value="7">Last 7 days ({time_ranges['Last 7 days']} users)</option>
                    <option value="30">Last 30 days ({time_ranges['Last 30 days']} users)</option>
                    <option value="90">Last 90 days ({time_ranges['Last 90 days']} users)</option>
                    <option value="365">Last 365 days ({time_ranges['Last 365 days']} users)</option>
                </select>
            </div>
            
            <div class="filter-section">
                <h3>Geographic Filters</h3>
                <div class="geo-filters">
                    <select id="countryFilter" onchange="filterByCountry()">
                        <option value="">Select Country...</option>
                    </select>
                    <select id="stateFilter" onchange="filterByState()" disabled>
                        <option value="">Select State/Province...</option>
                    </select>
                    <select id="localFilter" onchange="filterByLocal()" disabled>
                        <option value="">Select City/Local...</option>
                    </select>
                    <select id="wardFilter" onchange="filterByWard()" disabled>
                        <option value="">Select Ward (if available)...</option>
                    </select>
                </div>
            </div>
            
            <div class="filter-section">
                <button onclick="clearAllFilters()" class="clear-btn">Clear All Filters</button>
                <div id="activeFilters" class="active-filters"></div>
            </div>
            
            <div id="summaryStats" class="summary-stats"></div>
        </div>
        
        <div class="map-container">
            <div class="map-header">
                <h3>Connect Users</h3>
                <div id="mapCounter" class="map-counter">Showing 0 users on map</div>
            </div>
            <div id="map"></div>
        </div>
    </div>
    """
    
    # Add JavaScript functionality
    js_code = create_dashboard_js(locations_data)
    
    return [dashboard_html + js_code]

def create_dashboard_js(locations_data):
    """Generate JavaScript code for interactive functionality"""
    return f"""
    <script>
        // Global variables
        let map;
        let markersLayer;
        let allLocations = {json.dumps(locations_data)};
        let filteredLocations = [...allLocations];
        let activeFilters = {{}};
        
        // Initialize map
        function initMap() {{
            map = L.map('map').setView([0, 0], 2);
            
            L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                attribution: '© OpenStreetMap contributors'
            }}).addTo(map);
            
            markersLayer = L.layerGroup().addTo(map);
            
            // Add event listeners for map movement and zoom
            map.on('moveend', function() {{
                console.log('Map moved, updating counter...');
                updateVisibleMapCounter();
            }});
            
            map.on('zoomend', function() {{
                console.log('Map zoomed, updating counter...');
                updateVisibleMapCounter();
            }});
            
            // Populate country dropdown
            populateCountryFilter();
            
            // Show all locations initially
            updateMap();
            updateSummary();
            updateMapCounter();
        }}
        
        function populateCountryFilter() {{
            const countries = [...new Set(filteredLocations.map(loc => loc.country))].sort();
            const countrySelect = document.getElementById('countryFilter');
            
            // Clear existing options except the first one
            countrySelect.innerHTML = '<option value="">Select Country...</option>';
            
            countries.forEach(country => {{
                if (country !== 'Unknown') {{
                    const option = document.createElement('option');
                    option.value = country;
                    const count = filteredLocations.filter(l => l.country === country).length;
                    option.textContent = `${{country}} (${{count}})`;
                    countrySelect.appendChild(option);
                }}
            }});
        }}
        
        function filterByTime() {{
            const timeFilter = document.getElementById('timeFilter').value;
            activeFilters.time = timeFilter;
            
            if (timeFilter === 'all') {{
                delete activeFilters.time;
            }}
            
            applyFilters();
            
            // Update geographic filters with new counts but preserve selections
            const currentCountry = activeFilters.country;
            const currentState = activeFilters.state;
            const currentLocal = activeFilters.local;
            const currentWard = activeFilters.ward;
            
            // Repopulate country filter with updated counts
            populateCountryFilter();
            
            // Restore and reapply geographic filters if they still exist in filtered data
            if (currentCountry) {{
                const countrySelect = document.getElementById('countryFilter');
                const countryOption = Array.from(countrySelect.options).find(opt => opt.value === currentCountry);
                
                if (countryOption) {{
                    // Country still exists in filtered data, restore it
                    countrySelect.value = currentCountry;
                    activeFilters.country = currentCountry;
                    
                    // Repopulate state filter
                    populateStateFilter(currentCountry);
                    
                    if (currentState) {{
                        const stateSelect = document.getElementById('stateFilter');
                        const stateOption = Array.from(stateSelect.options).find(opt => opt.value === currentState);
                        
                        if (stateOption) {{
                            // State still exists, restore it
                            stateSelect.value = currentState;
                            activeFilters.state = currentState;
                            
                            // Repopulate local filter
                            populateLocalFilter(currentCountry, currentState);
                            
                            if (currentLocal) {{
                                const localSelect = document.getElementById('localFilter');
                                const localOption = Array.from(localSelect.options).find(opt => opt.value === currentLocal);
                                
                                if (localOption) {{
                                    // Local still exists, restore it
                                    localSelect.value = currentLocal;
                                    activeFilters.local = currentLocal;
                                    
                                    // Repopulate ward filter
                                    populateWardFilter(currentCountry, currentState, currentLocal);
                                    
                                    if (currentWard) {{
                                        const wardSelect = document.getElementById('wardFilter');
                                        const wardOption = Array.from(wardSelect.options).find(opt => opt.value === currentWard);
                                        
                                        if (wardOption) {{
                                            // Ward still exists, restore it
                                            wardSelect.value = currentWard;
                                            activeFilters.ward = currentWard;
                                        }} else {{
                                            // Ward no longer exists, clear it
                                            delete activeFilters.ward;
                                        }}
                                    }}
                                }} else {{
                                    // Local no longer exists, clear it and ward
                                    delete activeFilters.local;
                                    delete activeFilters.ward;
                                }}
                            }}
                        }} else {{
                            // State no longer exists, clear state, local, and ward
                            document.getElementById('stateFilter').innerHTML = '<option value="">Select State/Province...</option>';
                            document.getElementById('localFilter').innerHTML = '<option value="">Select City/Local...</option>';
                            document.getElementById('wardFilter').innerHTML = '<option value="">Select Ward (if available)...</option>';
                            document.getElementById('stateFilter').disabled = true;
                            document.getElementById('localFilter').disabled = true;
                            document.getElementById('wardFilter').disabled = true;
                            delete activeFilters.state;
                            delete activeFilters.local;
                            delete activeFilters.ward;
                        }}
                    }}
                }} else {{
                    // Country no longer exists, clear all geographic filters
                    document.getElementById('stateFilter').innerHTML = '<option value="">Select State/Province...</option>';
                    document.getElementById('localFilter').innerHTML = '<option value="">Select City/Local...</option>';
                    document.getElementById('wardFilter').innerHTML = '<option value="">Select Ward (if available)...</option>';
                    document.getElementById('stateFilter').disabled = true;
                    document.getElementById('localFilter').disabled = true;
                    document.getElementById('wardFilter').disabled = true;
                    delete activeFilters.country;
                    delete activeFilters.state;
                    delete activeFilters.local;
                    delete activeFilters.ward;
                }}
            }}
            
            // Reapply filters with preserved geographic selections
            applyFilters();
        }}
        
        function populateStateFilter(country) {{
            const stateSelect = document.getElementById('stateFilter');
            stateSelect.innerHTML = '<option value="">Select State/Province...</option>';
            
            // Populate state dropdown based on ALL locations for this country
            const states = [...new Set(
                allLocations
                    .filter(loc => loc.country === country)
                    .map(loc => loc.state_province)
            )].sort();
            
            states.forEach(state => {{
                if (state !== 'Unknown') {{
                    const option = document.createElement('option');
                    option.value = state;
                    const count = allLocations.filter(l => l.country === country && l.state_province === state).length;
                    option.textContent = `${{state}} (${{count}})`;
                    stateSelect.appendChild(option);
                }}
            }});
            
            stateSelect.disabled = false;
        }}
        
        function populateLocalFilter(country, state) {{
            const localSelect = document.getElementById('localFilter');
            localSelect.innerHTML = '<option value="">Select City/Local...</option>';
            
            // Populate local dropdown based on ALL locations for this country/state
            const locals = [...new Set(
                allLocations
                    .filter(loc => loc.country === country && loc.state_province === state)
                    .map(loc => loc.local_admin)
            )].sort();
            
            locals.forEach(local => {{
                if (local !== 'Unknown') {{
                    const option = document.createElement('option');
                    option.value = local;
                    const count = allLocations.filter(l => 
                        l.country === country && 
                        l.state_province === state && 
                        l.local_admin === local
                    ).length;
                    option.textContent = `${{local}} (${{count}})`;
                    localSelect.appendChild(option);
                }}
            }});
            
            localSelect.disabled = false;
        }}
        
        function filterByCountry() {{
            const country = document.getElementById('countryFilter').value;
            const stateSelect = document.getElementById('stateFilter');
            const localSelect = document.getElementById('localFilter');
            const wardSelect = document.getElementById('wardFilter');
            
            // Reset dependent filters
            stateSelect.innerHTML = '<option value="">Select State/Province...</option>';
            localSelect.innerHTML = '<option value="">Select City/Local...</option>';
            wardSelect.innerHTML = '<option value="">Select Ward (if available)...</option>';
            stateSelect.disabled = !country;
            localSelect.disabled = true;
            wardSelect.disabled = true;
            
            if (country) {{
                activeFilters.country = country;
                populateStateFilter(country);
            }} else {{
                delete activeFilters.country;
                delete activeFilters.state;
                delete activeFilters.local;
                delete activeFilters.ward;
            }}
            
            applyFilters();
        }}
        
        function filterByState() {{
            const state = document.getElementById('stateFilter').value;
            const localSelect = document.getElementById('localFilter');
            const wardSelect = document.getElementById('wardFilter');
            
            // Clear local and ward filters when state changes
            localSelect.innerHTML = '<option value="">Select City/Local...</option>';
            wardSelect.innerHTML = '<option value="">Select Ward (if available)...</option>';
            wardSelect.disabled = true;
            
            // Clear active filters for local and ward
            delete activeFilters.local;
            delete activeFilters.ward;
            
            if (state) {{
                activeFilters.state = state;
                // Populate local filter for the new state
                populateLocalFilter(activeFilters.country, state);
                // Apply filters after populating local options
                applyFilters();
            }} else {{
                delete activeFilters.state;
                localSelect.disabled = true;
                applyFilters();
            }}
        }}
        
        function filterByLocal() {{
            const local = document.getElementById('localFilter').value;
            const wardSelect = document.getElementById('wardFilter');
            
            // Clear ward filter when local changes
            wardSelect.innerHTML = '<option value="">Select Ward (if available)...</option>';
            wardSelect.disabled = true;
            delete activeFilters.ward;
            
            if (local) {{
                activeFilters.local = local;
                populateWardFilter(activeFilters.country, activeFilters.state, local);
            }} else {{
                delete activeFilters.local;
            }}
            
            applyFilters();
        }}
        
        function populateWardFilter(country, state, local) {{
            const wardSelect = document.getElementById('wardFilter');
            wardSelect.innerHTML = '<option value="">Select Ward (if available)...</option>';
            
            // Only show wards for Nigerian locations
            if (country !== 'Nigeria') {{
                wardSelect.disabled = true;
                return;
            }}
            
            const wards = [...new Set(
                allLocations
                    .filter(loc => loc.country === country && 
                                 loc.state_province === state && 
                                 loc.local_admin === local &&
                                 loc.ward !== 'Unknown')
                    .map(loc => loc.ward)
            )].sort();
            
            wards.forEach(ward => {{
                const option = document.createElement('option');
                option.value = ward;
                const count = allLocations.filter(l => 
                    l.country === country && 
                    l.state_province === state && 
                    l.local_admin === local &&
                    l.ward === ward
                ).length;
                option.textContent = `${{ward}} (${{count}})`;
                wardSelect.appendChild(option);
            }});
            
            wardSelect.disabled = wards.length === 0;
        }}
        
        function filterByWard() {{
            const ward = document.getElementById('wardFilter').value;
            
            if (ward) {{
                activeFilters.ward = ward;
            }} else {{
                delete activeFilters.ward;
            }}
            
            applyFilters();
        }}
        
        function applyFilters() {{
            filteredLocations = allLocations.filter(location => {{
                // Time filter
                if (activeFilters.time) {{
                    const daysSince = Math.floor((new Date() - new Date(location.date)) / (1000 * 60 * 60 * 24));
                    if (daysSince > parseInt(activeFilters.time)) return false;
                }}
                
                // Geographic filters
                if (activeFilters.country && location.country !== activeFilters.country) return false;
                if (activeFilters.state && location.state_province !== activeFilters.state) return false;
                if (activeFilters.local && location.local_admin !== activeFilters.local) return false;
                if (activeFilters.ward && location.ward !== activeFilters.ward) return false;
                
                return true;
            }});
            
            updateMap();
            updateSummary();
            updateActiveFiltersDisplay();
        }}
        
        function updateMapCounter() {{
            const counter = document.getElementById('mapCounter');
            const count = filteredLocations.length;
            counter.textContent = `Showing ${{count}} user${{count !== 1 ? 's' : ''}} on map`;
        }}
        
        function updateVisibleMapCounter() {{
            const counter = document.getElementById('mapCounter');
            
            if (filteredLocations.length === 0) {{
                counter.textContent = 'Showing 0 users on map';
                return;
            }}
            
            // Get current map bounds
            const bounds = map.getBounds();
            console.log('Map bounds:', bounds.toString());
            
            // Count markers within visible bounds
            let visibleCount = 0;
            filteredLocations.forEach(location => {{
                // Use the same lat/lng properties as in updateMap function
                const lat = parseFloat(location.lat);
                const lng = parseFloat(location.lng);
                if (bounds.contains([lat, lng])) {{
                    visibleCount++;
                }}
            }});
            
            console.log(`Visible: ${{visibleCount}}, Total: ${{filteredLocations.length}}`);
            
            const totalCount = filteredLocations.length;
            if (visibleCount === totalCount) {{
                counter.textContent = `Showing ${{totalCount}} user${{totalCount !== 1 ? 's' : ''}} on map`;
            }} else {{
                counter.textContent = `Showing ${{visibleCount}} of ${{totalCount}} users in view`;
            }}
        }}
        
        function updateMap() {{
            markersLayer.clearLayers();
            
            if (filteredLocations.length === 0) {{
                return;
            }}
            
            // Add markers
            filteredLocations.forEach(location => {{
                let locationText = `${{location.local_admin}}, ${{location.state_province}}, ${{location.country}}`;
                if (location.ward !== 'Unknown' && location.country === 'Nigeria') {{
                    locationText = `${{location.ward}} Ward, ` + locationText;
                }}
                
                const marker = L.marker([location.lat, location.lng])
                    .bindPopup(`
                        <strong>User ID:</strong> ${{location.flw_id}}<br>
                        <strong>Opportunity:</strong> ${{location.opp_name}}<br>
                        <strong>Last Visit:</strong> ${{location.date}}<br>
                        <strong>Location:</strong> ${{locationText}}<br>
                        <strong>Accuracy:</strong> ${{location.accuracy}}m
                    `);
                markersLayer.addLayer(marker);
            }});
            
            // Fit map to markers
            if (filteredLocations.length > 0) {{
                const group = new L.featureGroup(markersLayer.getLayers());
                map.fitBounds(group.getBounds().pad(0.1));
                
                // Update counter after map bounds are set
                setTimeout(() => {{
                    updateVisibleMapCounter();
                }}, 100);
            }}
        }}
        
        function updateSummary() {{
            const summary = document.getElementById('summaryStats');
            const count = filteredLocations.length;
            
            // Country breakdown
            const countryBreakdown = {{}};
            filteredLocations.forEach(loc => {{
                countryBreakdown[loc.country] = (countryBreakdown[loc.country] || 0) + 1;
            }});
            
            let html = `<h3>Filtered Results: ${{count}} users</h3>`;
            
            if (Object.keys(countryBreakdown).length > 0) {{
                html += '<div class="country-breakdown">';
                Object.entries(countryBreakdown)
                    .sort((a, b) => b[1] - a[1])
                    .forEach(([country, count]) => {{
                        html += `<div class="country-stat">${{country}}: ${{count}}</div>`;
                    }});
                html += '</div>';
            }}
            
            summary.innerHTML = html;
        }}
        
        function updateActiveFiltersDisplay() {{
            const container = document.getElementById('activeFilters');
            const filters = [];
            
            if (activeFilters.time) filters.push(`Time: Last ${{activeFilters.time}} days`);
            if (activeFilters.country) filters.push(`Country: ${{activeFilters.country}}`);
            if (activeFilters.state) filters.push(`State: ${{activeFilters.state}}`);
            if (activeFilters.local) filters.push(`Local: ${{activeFilters.local}}`);
            if (activeFilters.ward) filters.push(`Ward: ${{activeFilters.ward}}`);
            
            if (filters.length > 0) {{
                container.innerHTML = '<strong>Active Filters:</strong><br>' + filters.join('<br>');
            }} else {{
                container.innerHTML = '';
            }}
        }}
        
        function clearAllFilters() {{
            activeFilters = {{}};
            document.getElementById('timeFilter').value = 'all';
            document.getElementById('countryFilter').value = '';
            document.getElementById('stateFilter').innerHTML = '<option value="">Select State/Province...</option>';
            document.getElementById('localFilter').innerHTML = '<option value="">Select City/Local...</option>';
            document.getElementById('wardFilter').innerHTML = '<option value="">Select Ward (if available)...</option>';
            document.getElementById('stateFilter').disabled = true;
            document.getElementById('localFilter').disabled = true;
            document.getElementById('wardFilter').disabled = true;
            
            filteredLocations = [...allLocations];
            updateMap();
            updateSummary();
            updateActiveFiltersDisplay();
            populateCountryFilter();
        }}
        
        // Initialize when page loads
        document.addEventListener('DOMContentLoaded', initMap);
    </script>
    """

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Connect Location Analysis Dashboard')
    parser.add_argument('--cached-only', action='store_true', 
                       help='Only use locations that have cached geocoding data (no API calls)')
    parser.add_argument('--fetch-fresh', action='store_true',
                       help='Fetch fresh data from Superset before running analysis')
    args = parser.parse_args()
    
    df = load_data(cached_only=args.cached_only, fetch_fresh=args.fetch_fresh)
    
    # Update cache with GRID3 ward data for Nigerian coordinates
    # (Only creates backup if not in cached-only mode)
    if not args.cached_only:
        try:
            from update_cache import update_cache_from_grid3
            update_cache_from_grid3(create_backup=False, verbose=True)
        except Exception as e:
            print(f"⚠️ Warning: Could not update GRID3 ward data: {e}")
            print("   Dashboard will still work, but Nigerian wards may be missing")
    
    charts = create_charts(df)
    
    # Generate HTML with modern Tailwind-inspired styling
    html = f"""<!DOCTYPE html>
<html><head>
<title>Connect Location Analysis Dashboard</title>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
    /* Modern CSS Reset & Base */
    *, *::before, *::after {{ box-sizing: border-box; }}
    * {{ margin: 0; padding: 0; }}
    
    body {{ 
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        color: #1e293b;
        line-height: 1.6;
    }}
    
    /* Dashboard Layout */
    .dashboard-container {{ 
        display: flex; 
        height: 100vh; 
        background: #ffffff;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
    }}
    
    /* Controls Panel - Modern Sidebar */
    .controls-panel {{ 
        width: 320px; 
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border-right: 1px solid #e2e8f0;
        overflow-y: auto; 
        box-shadow: 4px 0 25px -5px rgba(0, 0, 0, 0.1);
        position: relative;
    }}
    

    
    /* Map Container */
    .map-container {{ 
        flex: 1; 
        display: flex; 
        flex-direction: column; 
        background: #ffffff;
    }}
    
    .map-header {{ 
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 12px 20px; 
        border-bottom: 1px solid #e2e8f0; 
        display: flex; 
        justify-content: space-between; 
        align-items: center;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }}
    
    .map-header h3 {{ 
        margin: 0; 
        color: #1e293b; 
        font-weight: 700; 
        font-size: 20px;
        display: flex;
        align-items: center;
        gap: 6px;
    }}
    
    .map-counter {{ 
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white; 
        padding: 6px 12px; 
        border-radius: 16px; 
        font-size: 12px; 
        font-weight: 600;
        box-shadow: 0 2px 8px 0 rgba(59, 130, 246, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }}
    
    #map {{ 
        flex: 1; 
        width: 100%; 
        border-radius: 0 0 12px 0;
    }}
    
    /* Typography & Headers */
    h2 {{ 
        margin: 0 0 20px 0; 
        color: #1e293b; 
        font-weight: 700; 
        font-size: 20px;
        padding: 20px 20px 0 20px;
        position: relative;
    }}
    

    
    h3 {{ 
        color: #374151; 
        margin: 0 0 12px 0; 
        font-weight: 600; 
        font-size: 14px;
        display: flex;
        align-items: center;
        gap: 6px;
    }}
    
    /* Stats Grid - Modern Cards */
    .stats-grid {{ 
        display: grid; 
        grid-template-columns: 1fr 1fr; 
        gap: 12px; 
        margin: 20px 20px 20px 20px;
    }}
    
    .stat-card {{ 
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 14px; 
        border-radius: 8px; 
        text-align: center; 
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 4px -1px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }}
    
    .stat-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #3b82f6 0%, #1d4ed8 100%);
    }}
    
    .stat-card:hover {{ 
        transform: translateY(-1px); 
        box-shadow: 0 6px 15px -3px rgba(0, 0, 0, 0.1);
    }}
    
    .stat-card h3 {{ 
        margin: 0 0 4px 0; 
        font-size: 11px; 
        color: #6b7280; 
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }}
    
    .stat-number {{ 
        font-size: 22px; 
        font-weight: 700; 
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    
    /* Filter Sections */
    .filter-section {{ 
        margin: 0 20px 18px 20px;
        background: #ffffff;
        padding: 16px;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }}
    
    /* Form Controls - Modern Inputs */
    .geo-filters select, #timeFilter {{ 
        width: 100%; 
        padding: 10px 12px; 
        margin-bottom: 8px; 
        border: 1px solid #e2e8f0; 
        border-radius: 6px;
        font-size: 13px;
        font-weight: 500;
        color: #374151;
        background: #ffffff;
        transition: all 0.2s ease;
        appearance: none;
        background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 20 20'%3e%3cpath stroke='%236b7280' stroke-linecap='round' stroke-linejoin='round' stroke-width='1.5' d='m6 8 4 4 4-4'/%3e%3c/svg%3e");
        background-position: right 10px center;
        background-repeat: no-repeat;
        background-size: 14px;
        padding-right: 32px;
    }}
    
    .geo-filters select:focus, #timeFilter:focus {{ 
        outline: none; 
        border-color: #3b82f6; 
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.1);
    }}
    
    .geo-filters select:disabled {{ 
        background: #f9fafb; 
        color: #9ca3af; 
        cursor: not-allowed;
        border-color: #e5e7eb;
    }}
    
    /* Clear Button - Modern Design */
    .clear-btn {{ 
        background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%);
        color: white; 
        border: none; 
        padding: 10px 16px; 
        border-radius: 6px; 
        cursor: pointer; 
        width: 100%;
        font-weight: 600;
        font-size: 13px;
        transition: all 0.2s ease;
        box-shadow: 0 2px 8px 0 rgba(107, 114, 128, 0.3);
    }}
    
    .clear-btn:hover {{ 
        background: linear-gradient(135deg, #4b5563 0%, #374151 100%);
        transform: translateY(-1px);
        box-shadow: 0 6px 20px 0 rgba(107, 114, 128, 0.4);
    }}
    
    .clear-btn:active {{ 
        transform: translateY(0);
    }}
    
    /* Active Filters - Modern Chips */
    .active-filters {{ 
        margin-top: 12px; 
        padding: 12px; 
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-radius: 6px; 
        font-size: 12px;
        border: 1px solid #93c5fd;
    }}
    
    /* Summary Stats */
    .summary-stats {{ 
        margin: 20px;
        background: #ffffff;
        padding: 16px;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }}
    
    .summary-stats h3 {{
        color: #374151;
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 12px;
        padding-bottom: 6px;
        border-bottom: 1px solid #e2e8f0;
    }}
    
    .country-breakdown {{ 
        margin-top: 8px; 
    }}
    
    .country-stat {{ 
        padding: 6px 0; 
        border-bottom: 1px solid #f3f4f6;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 13px;
        color: #374151;
    }}
    
    .country-stat:last-child {{
        border-bottom: none;
    }}
    
    /* Scrollbar Styling */
    .controls-panel::-webkit-scrollbar {{
        width: 6px;
    }}
    
    .controls-panel::-webkit-scrollbar-track {{
        background: #f1f5f9;
    }}
    
    .controls-panel::-webkit-scrollbar-thumb {{
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        border-radius: 3px;
    }}
    
    .controls-panel::-webkit-scrollbar-thumb:hover {{
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
    }}
    
    /* Responsive Design */
    @media (max-width: 768px) {{
        .dashboard-container {{
            flex-direction: column;
        }}
        
        .controls-panel {{
            width: 100%;
            height: auto;
            max-height: 40vh;
        }}
        
        .stats-grid {{
            grid-template-columns: 1fr;
        }}
        
        h2 {{
            font-size: 20px;
            padding: 20px 20px 0 20px;
        }}
        
        .filter-section {{
            margin: 0 20px 20px 20px;
            padding: 16px;
        }}
    }}
    
    /* Animation for smooth interactions */
    * {{
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }}
</style>
</head><body>
{''.join(charts)}
</body></html>"""
    
    # Save and open
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"output/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / "index.html"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html)
    
    webbrowser.open(f"file:///{os.path.abspath(filename).replace(os.sep, '/')}")
    print(f"✅ Saved: {filename}")

if __name__ == "__main__":
    main()
