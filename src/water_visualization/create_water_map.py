"""
Water Points Map Generator
Creates interactive Leaflet maps from CommCare water survey data with on-demand image loading.
"""
import os
import json
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import pandas as pd

from waterpoint import WaterPoint


def find_projects(water_data_path: str = "data") -> Dict[str, Dict[str, str]]:
    """Find all water survey projects in the water_data directory."""
    projects = {}
    
    water_data_dir = Path(water_data_path)
    if not water_data_dir.exists():
        return projects
    
    # Look for Excel files and matching image directories
    for excel_file in water_data_dir.glob("*.xlsx"):
        filename = excel_file.stem
        project_name = None
        
        # Handle new filename patterns: "{PROJECT} Final Waterbody Data.xlsx" or "{PROJECT} Final Waterbody data.xlsx"
        if "Final Waterbody" in filename:
            if "Final Waterbody Data" in filename:
                project_name = filename.split(" Final Waterbody Data")[0]
            elif "Final Waterbody data" in filename:
                project_name = filename.split(" Final Waterbody data")[0]
        # Handle old filename pattern: "{PROJECT} CCC Waterbody Survey - August 7.xlsx"
        elif "CCC Waterbody Survey" in filename:
            project_name = filename.split(" CCC Waterbody Survey")[0]
        
        if project_name:
            # Find matching image directory - try multiple patterns
            image_dirs = []
            
            # Try new pattern: "{PROJECT} Waterbody Survey/" or "{PROJECT} Waterbody survey/"
            image_dirs.extend(water_data_dir.glob(f"{project_name} Waterbody Survey*"))
            image_dirs.extend(water_data_dir.glob(f"{project_name} Waterbody survey*"))
            
            # Try old pattern: "{PROJECT} Pics*"
            if not image_dirs:
                image_dirs.extend(water_data_dir.glob(f"{project_name} Pics*"))
            
            if image_dirs:
                projects[project_name] = {
                    "excel_path": str(excel_file),
                    "images_path": str(image_dirs[0])
                }
    
    return projects


def load_water_points(project_name: str, water_data_path: str = "data") -> List[WaterPoint]:
    """Load water points from Excel file for a specific project."""
    projects = find_projects(water_data_path)
    
    if project_name not in projects:
        raise ValueError(f"Project '{project_name}' not found. Available projects: {list(projects.keys())}")
    
    project_info = projects[project_name]
    excel_path = project_info["excel_path"]
    images_path = project_info["images_path"]
    
    # Read Excel file
    df = pd.read_excel(excel_path)
    
    # Convert each row to WaterPoint
    water_points = []
    for _, row in df.iterrows():
        try:
            water_point = WaterPoint.from_excel_row(row, images_path)
            water_points.append(water_point)
        except Exception as e:
            print(f"Error processing row {row.get('number', '?')}: {e}")
            continue
    
    return water_points



def create_small_thumbnail(image_path: str, max_width: int = 60, cache_dir: str = None) -> Optional[str]:
    """Create a small base64-encoded thumbnail for popup preview with caching."""
    try:
        if not os.path.exists(image_path):
            return None
        
        # Create cache filename
        if cache_dir:
            cache_filename = f"thumb_{os.path.basename(image_path)}.b64"
            cache_path = os.path.join(cache_dir, cache_filename)
            
            # Check if cached thumbnail exists
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'r') as f:
                        return f.read()
                except:
                    pass  # If cache read fails, regenerate
            
        from PIL import Image
        import io
        import base64
        
        with Image.open(image_path) as img:
            # Calculate new dimensions maintaining aspect ratio
            width, height = img.size
            if width > max_width:
                new_height = int(height * max_width / width)
                img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            # Save to bytes with heavy compression for small size
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=60, optimize=True)
            
            # Encode as base64
            img_data = base64.b64encode(buffer.getvalue()).decode()
            thumbnail_data = f"data:image/jpeg;base64,{img_data}"
            
            # Cache the thumbnail if cache_dir provided
            if cache_dir:
                try:
                    os.makedirs(cache_dir, exist_ok=True)
                    with open(cache_path, 'w') as f:
                        f.write(thumbnail_data)
                except Exception as e:
                    print(f"Warning: Could not cache thumbnail: {e}")
            
            return thumbnail_data
            
    except Exception as e:
        print(f"Warning: Could not create thumbnail for {os.path.basename(image_path)}: {e}")
        return None


def generate_popup_html(water_point: WaterPoint, output_dir: str) -> str:
    """Generate HTML content for marker popup."""
    
    # Create image gallery with small thumbnails
    images_html = ""
    available_photos = water_point.available_photos
    if available_photos:
        photo_items = []
        for i, photo_path in enumerate(available_photos):
            photo_filename = os.path.basename(photo_path)
            # Copy image to output directory and use relative path
            relative_path = f"images/{photo_filename}"
            
            # Ensure images directory exists
            images_dir = os.path.join(output_dir, "images")
            os.makedirs(images_dir, exist_ok=True)
            
            # Copy image file only if it doesn't exist
            import shutil
            dest_path = os.path.join(images_dir, photo_filename)
            if not os.path.exists(dest_path):
                try:
                    shutil.copy2(photo_path, dest_path)
                except Exception as e:
                    print(f"Warning: Could not copy {photo_path}: {e}")
                    continue
            # If file exists, we're reusing it from previous run
            
            # Create small thumbnail for popup with caching
            cache_dir = os.path.join(output_dir, "thumbnails")
            thumbnail_data = create_small_thumbnail(photo_path, max_width=60, cache_dir=cache_dir)
            
            if thumbnail_data:
                photo_items.append(f'''
                    <div class="photo-item" onclick="openLightbox('{relative_path}', '{photo_filename}')">
                        <img src="{thumbnail_data}" class="photo-thumbnail" alt="Photo {i+1}">
                        <div class="photo-name">Photo {i+1}</div>
                    </div>
                ''')
            else:
                # Fallback to placeholder if thumbnail creation fails
                photo_items.append(f'''
                    <div class="photo-item" onclick="openLightbox('{relative_path}', '{photo_filename}')">
                        <div class="photo-placeholder">
                            📷 Photo {i+1}
                        </div>
                        <div class="photo-name">Photo {i+1}</div>
                    </div>
                ''')
        
        if photo_items:
            images_html = f'''
                <div class="image-gallery">
                    <strong>Photos ({len(photo_items)}):</strong><br>
                    <div class="photo-grid">
                        {"".join(photo_items)}
                    </div>
                </div>
            '''
    
    # Create characteristics list
    characteristics = []
    characteristics.append(f"<strong>Type:</strong> {water_point.water_point_type_display}")
    characteristics.append(f"<strong>Usage:</strong> {water_point.usage_level_display}")
    
    if water_point.is_piped:
        characteristics.append("<strong>Piped:</strong> Yes")
    
    if water_point.has_dispenser:
        characteristics.append("<strong>Has Dispenser:</strong> Yes")
        if water_point.chlorine_dispenser_functional is not None:
            functional = "Yes" if water_point.chlorine_dispenser_functional else "No"
            characteristics.append(f"<strong>Dispenser Functional:</strong> {functional}")
    
    if water_point.other_treatment:
        characteristics.append(f"<strong>Other Treatment:</strong> {water_point.other_treatment}")
    
    characteristics_html = "<br>".join(characteristics)
    
    # Create notes section
    notes_html = ""
    if water_point.notes:
        notes_html = f'''
            <div class="notes">
                <strong>Notes:</strong><br>
                <em>{water_point.notes}</em>
            </div>
        '''
    
    popup_html = f'''
        <div class="water-point-popup">
            <h3>{water_point.community}</h3>
            <div class="location-breadcrumb">
                📍 {water_point.location_breadcrumb}
            </div>
            <hr>
            <div class="characteristics">
                {characteristics_html}
            </div>
            {images_html}
            {notes_html}
            <div class="metadata">
                <small>
                    <strong>Survey:</strong> {water_point.time_of_visit.strftime('%Y-%m-%d %H:%M')}<br>
                    <strong>Collector:</strong> {water_point.username} ({water_point.project_name})
                </small>
            </div>
        </div>
    '''
    
    return popup_html


def get_marker_color(water_point: WaterPoint) -> str:
    """Get marker color based on water point type."""
    color_map = {
        'piped_water': '#2E86AB',               # Blue
        'borehole_hand_pump': '#A23B72',       # Purple
        'borehole_motorized_pump': '#8E44AD',   # Dark Purple
        'protected_wells': '#F18F01',           # Orange
        'well': '#E67E22',                      # Light Orange
        'surface_water': '#8B4513',             # Brown
        'storage_tank_tap_stand': '#27AE60',    # Green
        'other': '#95A5A6'                      # Gray
    }
    return color_map.get(water_point.water_point_type, '#666666')


def generate_html_map(water_points: List[WaterPoint], output_dir: str, title: str = "Water Points Map") -> str:
    """Generate complete HTML map with embedded data and styling."""
    
    # Prepare data for JavaScript
    print("Processing water points and generating popups...")
    markers_data = []
    for i, wp in enumerate(water_points):
        if i % 50 == 0:  # Progress update every 50 points
            print(f"  Processing point {i+1}/{len(water_points)}: {wp.community}")
        marker_data = wp.to_dict()
        marker_data['popup_html'] = generate_popup_html(wp, output_dir)
        marker_data['marker_color'] = get_marker_color(wp)
        markers_data.append(marker_data)
    print("Popup generation complete.")
    
    # Calculate map center
    if water_points:
        center_lat = sum(wp.latitude for wp in water_points) / len(water_points)
        center_lon = sum(wp.longitude for wp in water_points) / len(water_points)
    else:
        center_lat, center_lon = 9.0765, 7.3986  # Nigeria center
    
    # Generate statistics
    total_points = len(water_points)
    projects = list(set(wp.project_name for wp in water_points))
    
    # Generate dynamic legend based on actual data
    unique_types = list(set(wp.water_point_type for wp in water_points))
    legend_items = []
    for water_type in sorted(unique_types):  # Sort for consistent ordering
        # Get a sample water point of this type to get display name and color
        sample_wp = next(wp for wp in water_points if wp.water_point_type == water_type)
        color = get_marker_color(sample_wp)
        display_name = sample_wp.water_point_type_display
        legend_items.append(f'''
        <div class="legend-item">
            <div class="legend-color" style="background-color: {color};"></div>
            <span>{display_name}</span>
        </div>''')
    
    legend_html = "".join(legend_items)
    
    html_template = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    
    <!-- Leaflet CSS -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    
    <style>
        body {{
            margin: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f5f5;
        }}
        
        #map {{
            height: 100vh;
            width: 100%;
        }}
        
        .map-header {{
            position: absolute;
            top: 10px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 1000;
            background: rgba(255, 255, 255, 0.95);
            padding: 10px 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }}
        
        .map-header h1 {{
            margin: 0 0 5px 0;
            font-size: 1.5em;
            color: #333;
        }}
        
        .map-header .stats {{
            font-size: 0.9em;
            color: #666;
        }}
        
        .legend {{
            position: absolute;
            bottom: 30px;
            right: 10px;
            z-index: 1000;
            background: rgba(255, 255, 255, 0.95);
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            font-size: 0.9em;
        }}
        
        .legend h4 {{
            margin: 0 0 10px 0;
            color: #333;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 5px 0;
        }}
        
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 50%;
            margin-right: 8px;
            border: 2px solid rgba(255,255,255,0.8);
        }}
        
        /* Popup styling */
        .water-point-popup {{
            max-width: 300px;
            font-family: inherit;
        }}
        
        .water-point-popup h3 {{
            margin: 0 0 8px 0;
            color: #2c3e50;
            font-size: 1.2em;
        }}
        
        .location-breadcrumb {{
            color: #7f8c8d;
            font-size: 0.85em;
            margin-bottom: 10px;
        }}
        
        .characteristics {{
            margin: 10px 0;
            line-height: 1.4;
        }}
        
        .image-gallery {{
            margin: 10px 0;
        }}
        
        .photo-grid {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 5px;
        }}
        
        .photo-item {{
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 6px;
            border: 1px solid #ddd;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
            background: #f9f9f9;
            min-width: 80px;
            max-width: 90px;
        }}
        
        .photo-item:hover {{
            background: #e9e9e9;
            transform: translateY(-2px);
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .photo-thumbnail {{
            width: 60px;
            height: 45px;
            object-fit: cover;
            border-radius: 4px;
            margin-bottom: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.2);
        }}
        
        .photo-placeholder {{
            font-size: 18px;
            margin-bottom: 4px;
            width: 60px;
            height: 45px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #f0f0f0;
            border-radius: 4px;
            transition: background-color 0.2s;
        }}
        
        .photo-placeholder:hover {{
            background: #e0e0e0;
        }}
        
        .photo-name {{
            font-size: 0.7em;
            color: #666;
            text-align: center;
            word-break: break-word;
            margin-top: 2px;
        }}
        
        .notes {{
            margin: 10px 0;
            padding: 8px;
            background: #f8f9fa;
            border-left: 3px solid #3498db;
            border-radius: 4px;
        }}
        
        .metadata {{
            margin-top: 10px;
            padding-top: 8px;
            border-top: 1px solid #eee;
        }}
        
        /* Lightbox */
        .lightbox {{
            display: none;
            position: fixed;
            z-index: 2000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.9);
        }}
        
        .lightbox-content {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            max-width: 90%;
            max-height: 90%;
        }}
        
        .lightbox img {{
            max-width: 100%;
            max-height: 100%;
            border-radius: 8px;
        }}
        
        .loading-indicator {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 15px;
            color: white;
            z-index: 10;
        }}
        
        .spinner {{
            width: 40px;
            height: 40px;
            border: 4px solid rgba(255, 255, 255, 0.3);
            border-top: 4px solid #ffffff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }}
        
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        
        .loading-text {{
            font-size: 16px;
            font-weight: 500;
            text-align: center;
        }}
        
        .close-lightbox {{
            position: absolute;
            top: 15px;
            right: 35px;
            color: #f1f1f1;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
        }}
        
        .close-lightbox:hover {{
            color: #ccc;
        }}
        
        .lightbox-controls {{
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            align-items: center;
            gap: 15px;
            background: rgba(0,0,0,0.7);
            padding: 10px 20px;
            border-radius: 25px;
        }}
        
        .nav-button {{
            background: rgba(255,255,255,0.2);
            border: 1px solid rgba(255,255,255,0.3);
            color: white;
            font-size: 18px;
            padding: 8px 12px;
            border-radius: 50%;
            cursor: pointer;
            transition: all 0.2s;
            min-width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        
        .nav-button:hover {{
            background: rgba(255,255,255,0.3);
            transform: scale(1.1);
        }}
        
        .nav-button:disabled {{
            opacity: 0.3;
            cursor: not-allowed;
            transform: none;
        }}
        
        .image-counter {{
            color: white;
            font-size: 14px;
            font-weight: bold;
            min-width: 60px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="map-header">
        <h1>{title}</h1>
        <div class="stats">
            {total_points} water points across {len(projects)} projects
        </div>
    </div>
    
    <div id="map"></div>
    
    <div class="legend">
        <h4>Water Point Types</h4>
        {legend_html}
    </div>
    
    <!-- Lightbox -->
    <div id="lightbox" class="lightbox" onclick="closeLightbox()">
        <span class="close-lightbox" onclick="closeLightbox()">&times;</span>
        <div class="lightbox-content">
            <div id="loading-indicator" class="loading-indicator">
                <div class="spinner"></div>
                <div class="loading-text">Loading image...</div>
            </div>
            <img id="lightbox-img" src="" alt="">
            <div class="lightbox-controls">
                <button class="nav-button prev-button" onclick="event.stopPropagation(); prevImage();" title="Previous image (Left arrow)">❮</button>
                <div id="image-counter" class="image-counter"></div>
                <button class="nav-button next-button" onclick="event.stopPropagation(); nextImage();" title="Next image (Right arrow)">❯</button>
            </div>
        </div>
    </div>
    
    <!-- Leaflet JS -->
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    
    <script>
        // Water points data
        const waterPoints = {json.dumps(markers_data, indent=2)};
        
        // Initialize map
        const map = L.map('map').setView([{center_lat}, {center_lon}], 8);
        
        // Add tile layer
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap contributors',
            maxZoom: 18
        }}).addTo(map);
        
        // Add markers and store references
        const markers = {{}};
        waterPoints.forEach(function(point) {{
            const marker = L.circleMarker([point.latitude, point.longitude], {{
                radius: 8,
                fillColor: point.marker_color,
                color: 'white',
                weight: 2,
                opacity: 1,
                fillOpacity: 0.8
            }});
            
            marker.bindPopup(point.popup_html, {{
                maxWidth: 350,
                className: 'custom-popup'
            }});
            
            // Store marker reference by formid for URL navigation
            markers[point.formid] = marker;
            
            // Add click handler to update URL
            marker.on('click', function() {{
                updateUrlForPoint(point.formid);
            }});
            
            marker.addTo(map);
        }});
        
        // Add scale control
        L.control.scale({{
            position: 'bottomleft'
        }}).addTo(map);
        
        // URL parameter handling for direct navigation to specific water points
        function checkUrlParams() {{
            const urlParams = new URLSearchParams(window.location.search);
            const pointId = urlParams.get('point') || urlParams.get('guid') || urlParams.get('formid');
            
            if (pointId && markers[pointId]) {{
                const marker = markers[pointId];
                const point = waterPoints.find(p => p.formid === pointId);
                
                if (point) {{
                    // Center map on the point with higher zoom
                    map.setView([point.latitude, point.longitude], 15);
                    
                    // Open the popup after a short delay to ensure map is ready
                    setTimeout(function() {{
                        marker.openPopup();
                    }}, 500);
                    
                    // Optional: Add a temporary highlight effect
                    marker.setStyle({{
                        radius: 12,
                        weight: 4
                    }});
                    
                    // Reset highlight after 3 seconds
                    setTimeout(function() {{
                        marker.setStyle({{
                            radius: 8,
                            weight: 2
                        }});
                    }}, 3000);
                }}
            }}
        }}
        
        // Check URL params when map is ready
        map.whenReady(function() {{
            checkUrlParams();
        }});
        
        // Update URL when a water point is selected
        function updateUrlForPoint(formid) {{
            const newUrl = window.location.pathname + '?point=' + encodeURIComponent(formid);
            
            // Update the URL without reloading the page
            if (window.history && window.history.pushState) {{
                window.history.pushState({{pointId: formid}}, '', newUrl);
            }}
        }}
        
        // Handle browser back/forward buttons
        window.addEventListener('popstate', function(event) {{
            if (event.state && event.state.pointId) {{
                // Navigate to the point from history
                const pointId = event.state.pointId;
                if (markers[pointId]) {{
                    const marker = markers[pointId];
                    const point = waterPoints.find(p => p.formid === pointId);
                    
                    if (point) {{
                        map.setView([point.latitude, point.longitude], 15);
                        setTimeout(function() {{
                            marker.openPopup();
                        }}, 300);
                    }}
                }}
            }} else {{
                // No specific point in URL, close any open popups and reset view
                map.closePopup();
                // Optionally reset to default view
                const center_lat = {center_lat};
                const center_lon = {center_lon};
                map.setView([center_lat, center_lon], 8);
            }}
        }});
        
        // Lightbox state
        let currentImageIndex = 0;
        let currentImageSet = [];
        
        // Enhanced lightbox functions with navigation
        function openLightbox(imageSrc, imageAlt) {{
            // Find all images in the current popup
            const currentPopup = document.querySelector('.leaflet-popup-content');
            if (currentPopup) {{
                const photoItems = currentPopup.querySelectorAll('.photo-item');
                currentImageSet = [];
                
                photoItems.forEach(function(item, index) {{
                    const img = item.querySelector('.photo-thumbnail');
                    if (img) {{
                        const fullImageSrc = item.getAttribute('onclick').match(/'([^']+)'/)[1];
                        const fullImageAlt = item.getAttribute('onclick').match(/'[^']+',\\s*'([^']+)'/)[1];
                        currentImageSet.push({{
                            src: fullImageSrc,
                            alt: fullImageAlt
                        }});
                        
                        // Set current index if this is the clicked image
                        if (fullImageSrc === imageSrc) {{
                            currentImageIndex = index;
                        }}
                    }}
                }});
            }}
            
            document.getElementById('lightbox').style.display = 'block';
            updateLightboxImage();
        }}
        
        function updateLightboxImage() {{
            if (currentImageSet.length > 0) {{
                const currentImage = currentImageSet[currentImageIndex];
                const lightboxImg = document.getElementById('lightbox-img');
                const loadingIndicator = document.getElementById('loading-indicator');
                
                // Show loading indicator
                loadingIndicator.style.display = 'flex';
                lightboxImg.style.display = 'none';
                
                // Create new image to preload
                const newImg = new Image();
                newImg.onload = function() {{
                    // Image loaded successfully
                    lightboxImg.src = currentImage.src;
                    lightboxImg.alt = currentImage.alt;
                    lightboxImg.style.display = 'block';
                    loadingIndicator.style.display = 'none';
                }};
                newImg.onerror = function() {{
                    // Image failed to load
                    lightboxImg.alt = 'Failed to load image';
                    lightboxImg.style.display = 'block';
                    loadingIndicator.style.display = 'none';
                }};
                newImg.src = currentImage.src;
                
                // Update counter
                const counter = document.getElementById('image-counter');
                if (counter) {{
                    if (currentImageSet.length > 1) {{
                        counter.textContent = `${{currentImageIndex + 1}} of ${{currentImageSet.length}}`;
                        counter.style.display = 'block';
                    }} else {{
                        counter.style.display = 'none';
                    }}
                }}
                
                // Update button states
                const prevButton = document.querySelector('.prev-button');
                const nextButton = document.querySelector('.next-button');
                
                if (prevButton && nextButton) {{
                    if (currentImageSet.length <= 1) {{
                        prevButton.style.display = 'none';
                        nextButton.style.display = 'none';
                    }} else {{
                        prevButton.style.display = 'flex';
                        nextButton.style.display = 'flex';
                        
                        // Optional: disable buttons at ends (or keep cycling)
                        // prevButton.disabled = currentImageIndex === 0;
                        // nextButton.disabled = currentImageIndex === currentImageSet.length - 1;
                    }}
                }}
            }}
        }}
        
        function nextImage() {{
            if (currentImageSet.length > 1) {{
                currentImageIndex = (currentImageIndex + 1) % currentImageSet.length;
                updateLightboxImage();
            }}
        }}
        
        function prevImage() {{
            if (currentImageSet.length > 1) {{
                currentImageIndex = (currentImageIndex - 1 + currentImageSet.length) % currentImageSet.length;
                updateLightboxImage();
            }}
        }}
        
        function closeLightbox() {{
            document.getElementById('lightbox').style.display = 'none';
            document.getElementById('loading-indicator').style.display = 'none';
            document.getElementById('lightbox-img').style.display = 'block';
            currentImageSet = [];
            currentImageIndex = 0;
        }}
        
        // Enhanced keyboard navigation
        document.addEventListener('keydown', function(event) {{
            if (document.getElementById('lightbox').style.display === 'block') {{
                switch(event.key) {{
                    case 'Escape':
                        closeLightbox();
                        break;
                    case 'ArrowLeft':
                        event.preventDefault();
                        prevImage();
                        break;
                    case 'ArrowRight':
                        event.preventDefault();
                        nextImage();
                        break;
                }}
            }}
        }});
        
        console.log('Water Points Map loaded with', waterPoints.length, 'points');
    </script>
</body>
</html>'''
    
    return html_template


def create_water_points_map(
    project_name: Optional[str] = None, 
    output_dir: Optional[str] = None,
    water_data_path: str = "data"
) -> str:
    """
    Create interactive water points map.
    
    Args:
        project_name: Specific project to map (None for all projects)
        output_dir: Output directory (None for auto-generated)
        water_data_path: Path to water_data directory
    
    Returns:
        Path to generated HTML file
    """
    
    # Set up output directory with smart image reuse
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create output directory following project pattern: src/water_visualization/output/timestamp/
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "output", timestamp)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Smart image and thumbnail directory handling - reuse existing assets if available
    images_dir = os.path.join(output_dir, "images")
    thumbnails_dir = os.path.join(output_dir, "thumbnails")
    existing_images_dir = None
    existing_thumbnails_dir = None
    
    # Look for existing output directories with images and thumbnails
    output_base = os.path.join(script_dir, "output")
    if os.path.exists(output_base):
        for existing_dir in os.listdir(output_base):
            existing_path = os.path.join(output_base, existing_dir)
            if os.path.isdir(existing_path) and existing_dir != timestamp:
                potential_images = os.path.join(existing_path, "images")
                potential_thumbnails = os.path.join(existing_path, "thumbnails")
                if os.path.exists(potential_images) and os.listdir(potential_images):
                    existing_images_dir = potential_images
                    if os.path.exists(potential_thumbnails):
                        existing_thumbnails_dir = potential_thumbnails
                    break
    
    # Move existing directories if found, otherwise create new ones
    if existing_images_dir:
        print(f"Reusing images from previous run: {existing_images_dir}")
        import shutil
        shutil.move(existing_images_dir, images_dir)
        print(f"Images moved to: {images_dir}")
        
        if existing_thumbnails_dir:
            print(f"Reusing thumbnails from previous run: {existing_thumbnails_dir}")
            shutil.move(existing_thumbnails_dir, thumbnails_dir)
            print(f"Thumbnails moved to: {thumbnails_dir}")
        else:
            print("No existing thumbnails found, will generate fresh thumbnails")
    else:
        print("No existing images found, will copy fresh images and generate thumbnails")
    
    # Load water points
    all_water_points = []
    # Convert relative path to absolute path from script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_water_data_path = os.path.join(script_dir, water_data_path)
    projects = find_projects(absolute_water_data_path)
    
    if project_name:
        if project_name not in projects:
            raise ValueError(f"Project '{project_name}' not found. Available: {list(projects.keys())}")
        water_points = load_water_points(project_name, absolute_water_data_path)
        all_water_points.extend(water_points)
        title = f"CommCare Connect Water Source Research - {project_name}"
    else:
        # Load all projects
        for proj_name in projects:
            water_points = load_water_points(proj_name, absolute_water_data_path)
            all_water_points.extend(water_points)
        title = "CommCare Connect Water Source Research"
    
    print(f"Loaded {len(all_water_points)} water points from {len(projects)} projects")
    
    # Generate HTML
    print(f"Generating HTML for {len(all_water_points)} water points...")
    html_content = generate_html_map(all_water_points, output_dir, title)
    
    # Write to file
    filename = "index.html"
    output_path = os.path.join(output_dir, filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Map generated: {output_path}")
    if existing_images_dir:
        print(f"Images reused from previous run (moved to): {os.path.join(output_dir, 'images')}")
        if existing_thumbnails_dir:
            print(f"Thumbnails reused from previous run (moved to): {os.path.join(output_dir, 'thumbnails')}")
        else:
            print(f"Thumbnails generated and cached to: {os.path.join(output_dir, 'thumbnails')}")
    else:
        print(f"Images copied to: {os.path.join(output_dir, 'images')}")
        print(f"Thumbnails generated and cached to: {os.path.join(output_dir, 'thumbnails')}")
    
    # Open the file in the default browser
    try:
        webbrowser.open(f'file://{os.path.abspath(output_path)}')
        print(f"Opening map in default browser...")
    except Exception as e:
        print(f"Could not auto-open browser: {e}")
        print(f"Please manually open: {output_path}")
    
    return output_path


if __name__ == "__main__":
    # Generate map for all projects
    try:
        output_file = create_water_points_map()
        print(f"Success! Map should open automatically in your browser.")
        print(f"If it doesn't open, manually open: {output_file}")
    except Exception as e:
        print(f"Error generating map: {e}")
        import traceback
        traceback.print_exc()
