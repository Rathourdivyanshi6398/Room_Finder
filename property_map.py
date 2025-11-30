import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster
import re
import numpy as np
from typing import Dict, List, Tuple
import os
import glob
import time
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import random

class PropertyMapVisualizer:
    def __init__(self, data_dir: str):
        """
        Initialize the PropertyMapVisualizer with data directory.
        
        Args:
            data_dir (str): Path to the directory containing CSV files
        """
        self.data_dir = data_dir
        self.df = None
        self.city_averages = {}
        # Initialize Nominatim with a custom user agent
        self.geocoder = Nominatim(
            user_agent="property_map_visualizer_v1",
            timeout=30  # Increased timeout
        )
        
    def load_and_clean_data(self):
        """Load and clean the property data from all CSV files in the data directory."""
        # Find all CSV files in the data directory
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")
        
        # Define required columns
        required_columns = {
            'title', 'link', 'subtitle', 'price', 'area',
            'property_details', 'city', 'location', 'first_image_url'
        }
        
        # Load and combine all CSV files
        dfs = []
        for csv_file in csv_files:
            print(f"Loading {csv_file}...")
            try:
                df = pd.read_csv(csv_file)
                # Convert column names to lowercase
                df.columns = df.columns.str.strip().str.lower()
                # Check for required columns
                missing_columns = required_columns - set(df.columns)
                if missing_columns:
                    print(f"Skipping {csv_file}: Missing columns {', '.join(missing_columns)}")
                    continue
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {csv_file}: {str(e)}")
                if 'df' in locals():
                    print("Columns in the file:", df.columns.tolist())
                continue
        
        if not dfs:
            raise ValueError("No valid data files could be loaded")
        
        self.df = pd.concat(dfs, ignore_index=True)
        
        # Initialize latitude and longitude columns
        self.df['latitude'] = np.nan
        self.df['longitude'] = np.nan  # Fixed typo
        
        # Clean price column - remove non-numeric characters and convert to float
        try:
            self.df['price'] = self.df['price'].astype(str)
            self.df['price'] = self.df['price'].apply(lambda x: re.sub(r'[^\d.]', '', x))
            self.df['price'] = pd.to_numeric(self.df['price'], errors='coerce')
        except Exception as e:
            print(f"Error cleaning price column: {str(e)}")
            print("Sample of price values:", self.df['price'].head().to_dict())
            raise
        
        # Clean area column - extract numeric value and convert to float
        try:
            self.df['area'] = self.df['area'].astype(str)
            self.df['area'] = self.df['area'].apply(lambda x: re.sub(r'[^\d.]', '', x))
            self.df['area'] = pd.to_numeric(self.df['area'], errors='coerce')
        except Exception as e:
            print(f"Error cleaning area column: {str(e)}")
            print("Sample of area values:", self.df['area'].head().to_dict())
            raise
        
        # Remove rows with invalid price or area
        self.df = self.df.dropna(subset=['price', 'area'])
        
        # Clean location and city columns
        self.df['location'] = self.df['location'].fillna('').str.strip()
        self.df['city'] = self.df['city'].fillna('').str.strip()
        
        # Remove rows with empty location or city
        self.df = self.df[(self.df['location'] != '') & (self.df['city'] != '')]
        
        # Calculate price per sqft
        self.df['price_per_sqft'] = self.df['price'] / self.df['area']
        
        # Remove outliers (optional)
        # Remove properties with price_per_sqft beyond 3 standard deviations
        mean_ppsf = self.df['price_per_sqft'].mean()
        std_ppsf = self.df['price_per_sqft'].std()
        self.df = self.df[
            (self.df['price_per_sqft'] >= mean_ppsf - 3 * std_ppsf) &
            (self.df['price_per_sqft'] <= mean_ppsf + 3 * std_ppsf)
        ]
        
        # Calculate city averages
        self.city_averages = self.df.groupby('city')['price'].mean().to_dict()
        
        print(f"Loaded data for cities: {', '.join(self.df['city'].unique())}")
        print(f"Total properties: {len(self.df)}")
        print("\nSummary statistics:")
        print(self.df[['price', 'area', 'price_per_sqft']].describe())
        
    def geocode_with_retry(self, address: str, max_retries: int = 5) -> Tuple[float, float]:
        """
        Geocode an address with retry logic and rate limiting.
        
        Args:
            address (str): Address to geocode
            max_retries (int): Maximum number of retry attempts
            
        Returns:
            Tuple[float, float]: (latitude, longitude) or (None, None) if failed
        """
        for attempt in range(max_retries):
            try:
                # Add random delay between 1-3 seconds to respect rate limits
                time.sleep(random.uniform(1, 3))
                
                # Try to geocode the address
                location = self.geocoder.geocode(address)
                if location:
                    return location.latitude, location.longitude
                else:
                    return None, None
                    
            except (GeocoderTimedOut, GeocoderServiceError) as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed for {address}, retrying...")
                    print(f"Error: {str(e)}")
                    # Exponential backoff with jitter
                    time.sleep(2 ** (attempt + 1) + random.uniform(0, 1))
                else:
                    print(f"Failed to geocode {address} after {max_retries} attempts")
                    return None, None
            except Exception as e:
                print(f"Unexpected error geocoding {address}: {str(e)}")
                return None, None
                    
        return None, None
        
    def geocode_locations(self):
        """Add latitude and longitude columns using geocoding."""
        print("Geocoding locations...")
        
        # Create a dictionary to cache geocoding results
        geocode_cache = {}
        
        def geocode_address(row):
            # Create a cache key from location and city
            cache_key = f"{row['location']}, {row['city']}"
            
            # Check if we already have this location cached
            if cache_key in geocode_cache:
                return pd.Series(geocode_cache[cache_key])
            
            # Try to geocode
            lat, lon = self.geocode_with_retry(f"{cache_key}, India")
            
            # Cache the result
            geocode_cache[cache_key] = (lat, lon)
            
            return pd.Series([lat, lon])
        
        # Process in smaller batches to avoid overwhelming the geocoding service
        batch_size = 25  # Reduced batch size
        total_rows = len(self.df)
        
        for start_idx in range(0, total_rows, batch_size):
            end_idx = min(start_idx + batch_size, total_rows)
            print(f"Processing batch {start_idx + 1} to {end_idx} of {total_rows}...")
            
            batch_df = self.df.iloc[start_idx:end_idx].copy()
            geocoded_coords = batch_df.apply(geocode_address, axis=1)
            
            # Update the coordinates in the main dataframe
            self.df.loc[start_idx:end_idx-1, 'latitude'] = geocoded_coords[0]
            self.df.loc[start_idx:end_idx-1, 'longitude'] = geocoded_coords[1]  # Fixed column name
            
            # Save progress after each batch
            self.df.to_csv('geocoding_progress.csv', index=False)
            
            # Add a longer delay between batches
            time.sleep(5)
        
        # Remove rows where geocoding failed
        self.df = self.df.dropna(subset=['latitude', 'longitude'])
        print(f"Successfully geocoded {len(self.df)} properties")
        
    def create_map(self, city_filter: str = None) -> folium.Map:
        """
        Create an interactive map with property markers.
        
        Args:
            city_filter (str, optional): Filter properties by city
            
        Returns:
            folium.Map: The generated map object
        """
        # Filter data if city is specified
        df_filtered = self.df[self.df['city'] == city_filter] if city_filter else self.df
        
        if len(df_filtered) == 0:
            raise ValueError(f"No data available for city: {city_filter}")
        
        # Create base map centered on the mean coordinates
        center_lat = df_filtered['latitude'].mean()
        center_lon = df_filtered['longitude'].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        
        # Create marker clusters for each city
        marker_clusters = {}
        for city in df_filtered['city'].unique():
            marker_clusters[city] = MarkerCluster(name=f"{city} Properties")
        
        # Add markers for each property
        for idx, row in df_filtered.iterrows():
            # Determine marker color based on price comparison
            price_color = 'green' if row['price'] < self.city_averages[row['city']] else 'red'
            
            # Create popup content
            popup_content = f"""
                <div style='width: 200px'>
                    <h4>{row['title']}</h4>
                    <p><b>Price:</b> ₹{row['price']:,.0f}</p>
                    <p><b>Area:</b> {row['area']} sq.ft</p>
                    <p><b>Price/sq.ft:</b> ₹{row['price_per_sqft']:,.0f}</p>
                    <p><b>Details:</b> {row['property_details']}</p>
                    <img src='{row['first_image_url']}' style='width: 100%; height: auto;'>
                </div>
            """
            
            # Create circle marker
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=min(row['area'] / 100, 30),  # Scale marker size with area, max size 30
                popup=folium.Popup(popup_content, max_width=300),
                color=price_color,
                fill=True,
                fill_color=price_color,
                fill_opacity=0.7
            ).add_to(marker_clusters[row['city']])
        
        # Add marker clusters to map
        for cluster in marker_clusters.values():
            cluster.add_to(m)
        
        # Add heatmap layers
        # Price heatmap
        price_heat_data = [[row['latitude'], row['longitude'], row['price']] 
                          for _, row in df_filtered.iterrows()]
        HeatMap(price_heat_data, name='Price Heatmap').add_to(m)
        
        # Area heatmap
        area_heat_data = [[row['latitude'], row['longitude'], row['area']] 
                         for _, row in df_filtered.iterrows()]
        HeatMap(area_heat_data, name='Area Heatmap').add_to(m)
        
        # Price per sqft heatmap
        ppa_heat_data = [[row['latitude'], row['longitude'], row['price_per_sqft']] 
                        for _, row in df_filtered.iterrows()]
        HeatMap(ppa_heat_data, name='Price per sqft Heatmap').add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        return m

def main():
    # Initialize visualizer with data directory
    visualizer = PropertyMapVisualizer('data')
    
    try:
        # Process data
        visualizer.load_and_clean_data()
        visualizer.geocode_locations()
        
        # Create a single HTML file with city selection
        cities = visualizer.df['city'].unique()
        
        # Create the HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Property Map Viewer</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f0f0f0;
                }}
                .container {{
                    max-width: 800px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #333;
                    text-align: center;
                }}
                .city-select {{
                    width: 100%;
                    padding: 10px;
                    margin: 20px 0;
                    font-size: 16px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                }}
                #map {{
                    height: 600px;
                    width: 100%;
                    margin-top: 20px;
                    border-radius: 4px;
                }}
                .button {{
                    background-color: #4CAF50;
                    color: white;
                    padding: 10px 20px;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 16px;
                    width: 100%;
                }}
                .button:hover {{
                    background-color: #45a049;
                }}
            </style>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.css"/>
            <script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet.markercluster/1.1.0/leaflet.markercluster.js"></script>
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet.markercluster/1.1.0/MarkerCluster.css"/>
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet.markercluster/1.1.0/MarkerCluster.Default.css"/>
            <script src="https://cdn.jsdelivr.net/gh/python-visualization/folium@main/folium/templates/leaflet_heat.min.js"></script>
        </head>
        <body>
            <div class="container">
                <h1>Property Map Viewer</h1>
                <select id="citySelect" class="city-select">
                    <option value="">Select a city</option>
                    {''.join(f'<option value="{city}">{city}</option>' for city in cities)}
                </select>
                <button onclick="showMap()" class="button">Show Map</button>
                <div id="map"></div>
            </div>

            <script>
                let map = null;
                let markerCluster = null;
                let heatmapLayer = null;

                function showMap() {{
                    const city = document.getElementById('citySelect').value;
                    if (!city) {{
                        alert('Please select a city');
                        return;
                    }}

                    // Clear existing map if any
                    if (map) {{
                        map.remove();
                    }}

                    // Create new map
                    map = L.map('map');
                    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                        attribution: '© OpenStreetMap contributors'
                    }}).addTo(map);

                    // Create marker cluster
                    markerCluster = L.markerClusterGroup();
                    map.addLayer(markerCluster);

                    // Fetch and display data for selected city
                    fetch(`property_data_{city.toLowerCase().replace(" ", "_")}.json`)
                        .then(response => response.json())
                        .then(data => {{
                            // Set map center to the first property
                            if (data.length > 0) {{
                                map.setView([data[0].latitude, data[0].longitude], 12);
                            }}

                            // Add markers for each property
                            data.forEach(property => {{
                                const marker = L.circleMarker(
                                    [property.latitude, property.longitude],
                                    {{
                                        radius: Math.min(property.area / 100, 30),
                                        color: property.price < property.city_average ? 'green' : 'red',
                                        fillColor: property.price < property.city_average ? 'green' : 'red',
                                        fillOpacity: 0.7,
                                        weight: 3
                                    }}
                                );

                                const popupContent = `
                                    <div style='width: 200px'>
                                        <h4>${{property.title}}</h4>
                                        <p><b>Price:</b> ₹${{property.price.toLocaleString()}}</p>
                                        <p><b>Area:</b> ${{property.area}} sq.ft</p>
                                        <p><b>Price/sq.ft:</b> ₹${{property.price_per_sqft.toLocaleString()}}</p>
                                        <p><b>Details:</b> ${{property.property_details}}</p>
                                        <img src='${{property.first_image_url}}' style='width: 100%; height: auto;'>
                                    </div>
                                `;

                                marker.bindPopup(popupContent);
                                markerCluster.addLayer(marker);
                            }});

                            // Add heatmap
                            const heatData = data.map(p => [p.latitude, p.longitude, p.price]);
                            if (heatmapLayer) {{
                                map.removeLayer(heatmapLayer);
                            }}
                            heatmapLayer = L.heatLayer(heatData, {{radius: 25, blur: 15}}).addTo(map);
                        }})
                        .catch(error => {{
                            console.error('Error loading property data:', error);
                            alert('Error loading property data. Please try again.');
                        }});
                }}
            </script>
        </body>
        </html>
        """
        
        # Save the HTML file
        with open('property_map.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Save property data for each city as JSON
        for city in cities:
            city_data = visualizer.df[visualizer.df['city'] == city].copy()
            city_data['city_average'] = visualizer.city_averages[city]
            city_data.to_json(f'property_data_{city.lower().replace(" ", "_")}.json', orient='records')
        
        print("Map viewer has been generated successfully!")
        print("Open property_map.html in your web browser to view the interactive map.")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please check your CSV files and make sure they have the required columns:")
        print("- title\n- link\n- subtitle\n- price\n- area\n- property_details\n- city\n- location\n- first_image_url")
        if hasattr(e, 'args'):
            print("Error details:", e.args)

if __name__ == "__main__":
    main()