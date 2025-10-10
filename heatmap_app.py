import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import geopandas as gpd
import json
from datetime import datetime, timedelta
import requests
import streamlit.components.v1 as components

# Page configuration
st.set_page_config(
    page_title="Campus Occupancy Heatmap",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Section headers */
    .section-header {
        text-align: center;
        margin: 2rem 0;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f2f6, #ffffff);
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 3px solid #1f77b4;
    }
    
    /* Chart containers */
    .chart-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and process the WiFi data and building categories - OPTIMIZED VERSION"""
    try:
        # Load building categories first (smaller dataset)
        with open('notebooks/data/campus_buildings_categories.geojson', 'r') as f:
            building_data = json.load(f)
        
        # Convert to GeoDataFrame
        campus = gpd.GeoDataFrame.from_features(building_data['features'], crs="EPSG:4326")
        
        # Create building type classification mapping (do this once)
        def classify_building_type(bldg_type):
            if pd.isna(bldg_type):
                return 'Unknown'
            bldg_type_lower = str(bldg_type).lower()
            if any(keyword in bldg_type_lower for keyword in ['residence', 'dormitory', 'housing', 'greek']):
                return 'Residential'
            else:
                return 'Non-Residential'
        
        # Create building mapping dictionary for fast lookup
        building_mapping = {}
        for _, row in campus.iterrows():
            bldg_code = row['BLDG_CODE']
            building_mapping[bldg_code] = {
                'BLDG_NAME': row['BLDG_NAME'],
                'BLDG_TYPE': row['BLDG_TYPE'],
                'building_category': classify_building_type(row['BLDG_TYPE']),
                'geometry': row['geometry']
            }
        
        # Load WiFi data
        try:
            wifi = pd.read_parquet('notebooks/data/wifi_data_2days.parquet', engine='fastparquet')
        except:
            wifi = pd.read_parquet('notebooks/data/wifi_data_2days.parquet', engine='pyarrow')
        
        # Process time columns efficiently
        wifi['time'] = pd.to_datetime(wifi['time'])
        
        # Sort by time to ensure chronological order
        wifi = wifi.sort_values('time').reset_index(drop=True)
        
        # Get the first timestamp and calculate 24-hour window
        first_timestamp = wifi['time'].min()
        end_timestamp = first_timestamp + pd.Timedelta(hours=24)
        
        # FILTER TO FIRST 24 HOURS ONLY (Kathleen handles second day)
        wifi = wifi[(wifi['time'] >= first_timestamp) & (wifi['time'] < end_timestamp)].copy()
        
        # Add derived time columns after filtering
        wifi['hour'] = wifi['time'].dt.hour
        wifi['date'] = wifi['time'].dt.date
        wifi['day_of_week'] = wifi['time'].dt.day_name()
        
        if len(wifi) == 0:
            st.error("No data found in the first 24 hours!")
            return None, None
        
        # Extract building ID numbers efficiently
        wifi['bid_no_letters'] = wifi['building_id'].str.extract(r'(\d+)')
        
        # Use vectorized operations instead of merge for better performance
        # Map building information using the dictionary
        wifi['BLDG_NAME'] = wifi['bid_no_letters'].map(lambda x: building_mapping.get(x, {}).get('BLDG_NAME'))
        wifi['BLDG_TYPE'] = wifi['bid_no_letters'].map(lambda x: building_mapping.get(x, {}).get('BLDG_TYPE'))
        wifi['building_category'] = wifi['bid_no_letters'].map(lambda x: building_mapping.get(x, {}).get('building_category', 'Unknown'))
        
        # Create occupancy column if it doesn't exist (for real WiFi data)
        if 'occupancy' not in wifi.columns:
            # For WiFi data, we'll count unique devices per building per hour as occupancy
            wifi['occupancy'] = 1  # Each row represents one device connection
        
        # Add geometry information for mapping
        wifi['geometry'] = wifi['bid_no_letters'].map(lambda x: building_mapping.get(x, {}).get('geometry'))
        
        # Sample for performance if dataset is very large
        if len(wifi) > 100000:
            sample_rate = max(1, len(wifi) // 50000)  # Target ~50k rows
            wifi = wifi.iloc[::sample_rate].copy()
        
        return wifi, campus
        
    except Exception as e:
        st.error(f"Error loading real data: {e}")
        st.error("Please ensure the data files are available:")
        st.error("- notebooks/data/wifi_data_2days.parquet")
        st.error("- notebooks/data/campus_buildings_categories.geojson")
        return None, None


def create_heatmap_data(data, selected_date, selected_hour):
    """Create heatmap data for the selected time"""
    # Filter data for the selected hour (don't filter by date since we have 24-hour window)
    filtered_data = data[
        (data['hour'] == selected_hour)
    ].copy()
    
    if filtered_data.empty:
        return None
    
    # Check what columns are available and use appropriate grouping
    available_columns = filtered_data.columns.tolist()
    
    # Determine grouping columns based on what's available
    group_columns = ['building_id', 'building_category']
    
    # Add building_name if it exists, otherwise use building_id as name
    if 'building_name' in available_columns:
        group_columns.append('building_name')
    elif 'BLDG_NAME' in available_columns:
        group_columns.append('BLDG_NAME')
    
    # Aggregate occupancy by building
    # If occupancy is 1 (device count), we want to count unique devices
    if filtered_data['occupancy'].nunique() == 1 and filtered_data['occupancy'].iloc[0] == 1:
        # Count unique devices per building (assuming MAC or IP represents unique devices)
        if 'MAC' in filtered_data.columns:
            building_occupancy = filtered_data.groupby(group_columns)['MAC'].nunique().reset_index()
            building_occupancy.rename(columns={'MAC': 'occupancy'}, inplace=True)
        elif 'IP' in filtered_data.columns:
            building_occupancy = filtered_data.groupby(group_columns)['IP'].nunique().reset_index()
            building_occupancy.rename(columns={'IP': 'occupancy'}, inplace=True)
        else:
            # Fallback to counting rows
            building_occupancy = filtered_data.groupby(group_columns).size().reset_index()
            building_occupancy.rename(columns={0: 'occupancy'}, inplace=True)
    else:
        # Sum occupancy values
        building_occupancy = filtered_data.groupby(group_columns).agg({
            'occupancy': 'sum'
        }).reset_index()
    
    # If no building_name column, create one from building_id
    if 'building_name' not in building_occupancy.columns and 'BLDG_NAME' not in building_occupancy.columns:
        building_occupancy['building_name'] = building_occupancy['building_id']
    elif 'BLDG_NAME' in building_occupancy.columns and 'building_name' not in building_occupancy.columns:
        building_occupancy['building_name'] = building_occupancy['BLDG_NAME']
    
    return building_occupancy

def create_occupancy_timeline(data, selected_date):
    """Create timeline showing occupancy patterns for the selected date"""
    # Use all data since we have 24-hour window
    daily_data = data.copy()
    
    if daily_data.empty:
        return None
    
    # Aggregate by hour and building category
    hourly_data = daily_data.groupby(['hour', 'building_category']).agg({
        'occupancy': 'sum'
    }).reset_index()
    
    return hourly_data


def main():
    st.title("Campus Occupancy Heatmap - First Day Analysis")
    st.markdown("**Interactive heatmap with scroll-over time feature and residential/non-residential color coding**")
    st.markdown("**Scope**: First day data only (Day 2 handled by Kathleen)")
    
    # Load data
    data, campus = load_data()
    
    if data is None or data.empty:
        st.error("Failed to load data or no data available")
        st.error("Please check that the data files exist and contain valid data")
        return
    
    # Sidebar controls
    st.sidebar.header("Controls")
    
    # Date selection (24-hour window may span two dates)
    available_dates = sorted(data['date'].unique())
    if len(available_dates) == 1:
        selected_date = available_dates[0]
    else:
        selected_date = st.sidebar.selectbox(
            "Select Date (First 24 Hours)",
            available_dates,
            index=0
        )
    
    # Hour selection with 12-hour format slider
    available_hours = sorted(data['hour'].unique())
    
    if len(available_hours) == 0:
        st.sidebar.error("No hourly data available")
        return
    elif len(available_hours) == 1:
        selected_hour = available_hours[0]
    else:
        # Create 12-hour format labels for the slider
        hour_labels = []
        for hour in available_hours:
            if hour == 0:
                hour_labels.append("12:00 AM")
            elif hour < 12:
                hour_labels.append(f"{hour}:00 AM")
            elif hour == 12:
                hour_labels.append("12:00 PM")
            else:
                hour_labels.append(f"{hour - 12}:00 PM")
        
        # Create mapping between labels and hours
        hour_mapping = dict(zip(hour_labels, available_hours))
        
        selected_time_label = st.sidebar.select_slider(
            "Select Time (Scroll Over Day)",
            options=hour_labels,
            value=hour_labels[0],
            help="Scroll through the day to see occupancy patterns"
        )
        
        # Convert back to 24-hour format for processing
        selected_hour = hour_mapping[selected_time_label]
    
    # Building category filter
    building_categories = data['building_category'].unique()
    selected_categories = st.sidebar.multiselect(
        "Building Categories",
        building_categories,
        default=building_categories
    )
    
    # Filter data based on selections (don't filter by date since we have 24-hour window)
    filtered_data = data[
        (data['hour'] == selected_hour) &
        (data['building_category'].isin(selected_categories))
    ]
    
    # Main content with better organization
    st.markdown("---")
    
    # Header section with centered title
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2>Campus Occupancy Visualization</h2>
        <p style="font-size: 1.1em; color: #666;">Interactive heatmap with real-time occupancy data</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Convert to 12-hour format for display
    if selected_hour == 0:
        time_display = "12:00 AM"
    elif selected_hour < 12:
        time_display = f"{selected_hour}:00 AM"
    elif selected_hour == 12:
        time_display = "12:00 PM"
    else:
        time_display = f"{selected_hour - 12}:00 PM"
    
    # Time and date display
    col_time, col_date = st.columns([1, 1])
    with col_time:
        st.metric("Selected Time", time_display)
    with col_date:
        st.metric("Selected Date", str(selected_date))
    
    st.markdown("---")
    
    if not filtered_data.empty:
        # Create heatmap data
        heatmap_data = create_heatmap_data(data, selected_date, selected_hour)
        
        if heatmap_data is not None:
            # Create geographic map visualization
            st.markdown("### Interactive Campus Map")
            st.markdown("*Hover over buildings to see occupancy details*")
            
            # Prepare data for mapping
            map_data = data[
                (data['hour'] == selected_hour) &
                (data['building_category'].isin(selected_categories))
            ].copy()
            
            if not map_data.empty:
                # Aggregate by building for map
                building_map_data = map_data.groupby(['building_id', 'building_category']).agg({
                    'occupancy': 'sum'
                }).reset_index()
                
                # Add building names
                if 'BLDG_NAME' in data.columns:
                    building_map_data = building_map_data.merge(
                        data[['building_id', 'BLDG_NAME']].drop_duplicates(),
                        on='building_id',
                        how='left'
                    )
                    name_col = 'BLDG_NAME'
                else:
                    name_col = 'building_id'
                
                # Create building shape heatmap using Folium
                try:
                    import folium
                    from streamlit_folium import st_folium
                    import geopandas as gpd
                    
                    # Get unique buildings with geometry and occupancy
                    building_geo = data[['building_id', 'geometry', 'building_category', 'BLDG_NAME']].drop_duplicates()
                    building_geo = building_geo[building_geo['geometry'].notna()]
                    
                    if not building_geo.empty:
                        # Merge with occupancy data
                        building_geo = building_geo.merge(
                            building_map_data[['building_id', 'occupancy']],
                            on='building_id',
                            how='left'
                        )
                        
                        # Convert to GeoDataFrame and ensure proper coordinate system
                        gdf = gpd.GeoDataFrame(building_geo, geometry='geometry', crs="EPSG:4326")
                        
                        # Ensure geometries are valid and properly aligned
                        gdf = gdf.to_crs("EPSG:4326")  # Ensure WGS84 coordinate system
                        gdf = gdf[gdf.geometry.is_valid]  # Remove invalid geometries
                        
                        # Check if we have valid geometry
                        valid_geometry = gdf[gdf['geometry'].notna()]
                        if len(valid_geometry) == 0:
                            st.error("No valid geometry data found!")
                            raise Exception("No valid geometry data")
                        
                        # Create Folium map with simple, clean tiles
                        m = folium.Map(
                            location=[33.7756, -84.3963], 
                            zoom_start=16, 
                            tiles=None  # Start with no base tiles
                        )
                        
                        # Add simple, minimal tile layer
                        folium.TileLayer(
                            tiles='https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png',
                            attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
                            name='Simple Map',
                            overlay=False,
                            control=False
                        ).add_to(m)
                        
                        # NEW COLOR CALCULATION: Higher occupancy = Darker colors
                        def get_building_color(feature):
                            """Get color based on building type and occupancy intensity - HIGHER OCCUPANCY = DARKER COLOR"""
                            try:
                                props = feature['properties']
                                building_id = props.get('building_id', 'unknown')
                                building_category = props.get('building_category', 'Unknown')
                                occupancy = props.get('occupancy', 0)
                                
                                # Handle None or invalid occupancy values
                                if occupancy is None or pd.isna(occupancy):
                                    occupancy = 0
                                
                                # Get occupancy statistics for this time period
                                min_occupancy = gdf['occupancy'].min()
                                max_occupancy = gdf['occupancy'].max()
                                
                                # Calculate normalized occupancy (0 to 1)
                                if max_occupancy > min_occupancy:
                                    normalized_occupancy = (occupancy - min_occupancy) / (max_occupancy - min_occupancy)
                                else:
                                    normalized_occupancy = 0
                                
                                # Define base colors (light versions for low occupancy)
                                light_colors = {
                                    'Residential': '#FFB3B3',    # Light red
                                    'Non-Residential': '#B3E5E5', # Light teal  
                                    'Unknown': '#D3D3D3'         # Light gray
                                }
                                
                                # Define dark colors (dark versions for high occupancy)
                                dark_colors = {
                                    'Residential': '#CC0000',    # Dark red
                                    'Non-Residential': '#006666', # Dark teal
                                    'Unknown': '#666666'         # Dark gray
                                }
                                
                                # Get light and dark colors for this building type
                                light_color = light_colors.get(building_category, '#D3D3D3')
                                dark_color = dark_colors.get(building_category, '#666666')
                                
                                # Interpolate between light and dark based on occupancy
                                # normalized_occupancy = 0 → light color
                                # normalized_occupancy = 1 → dark color
                                
                                # Convert hex to RGB
                                def hex_to_rgb(hex_color):
                                    hex_color = hex_color.lstrip('#')
                                    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                                
                                def rgb_to_hex(rgb):
                                    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
                                
                                light_rgb = hex_to_rgb(light_color)
                                dark_rgb = hex_to_rgb(dark_color)
                                
                                # Interpolate between light and dark
                                interpolated_rgb = tuple(
                                    int(light_rgb[i] + (dark_rgb[i] - light_rgb[i]) * normalized_occupancy)
                                    for i in range(3)
                                )
                                
                                final_color = rgb_to_hex(interpolated_rgb)
                                
                                return final_color
                                
                            except Exception as e:
                                st.error(f"Color calculation error: {e}")
                                return '#D3D3D3'  # Light gray default
                        
                        # Create style function - eliminate shadows from building geometries
                        def style_function(feature):
                            return {
                                'fillColor': get_building_color(feature),
                                'color': 'transparent',
                                'weight': 0,
                                'fillOpacity': 0.9,
                                'stroke': False,  # No stroke to prevent shadows
                                'shadow': False,  # Explicitly disable shadows
                            }
                        
                        # Add GeoJson to map - completely flat, no popups
                        folium.GeoJson(
                            gdf,
                            style_function=style_function
                        ).add_to(m)
                        
                        # Add individual markers with proper tooltips for hover functionality
                        for _, row in gdf.iterrows():
                            if row.geometry is not None:
                                try:
                                    # Get building info
                                    building_name = row.get('BLDG_NAME', row.get('building_id', 'Unknown'))
                                    category = row.get('building_category', 'Unknown')
                                    occupancy = row.get('occupancy', 0)
                                    
                                    # Convert to 12-hour format with AM/PM
                                    if selected_hour == 0:
                                        time_display = "12:00 AM"
                                    elif selected_hour < 12:
                                        time_display = f"{selected_hour}:00 AM"
                                    elif selected_hour == 12:
                                        time_display = "12:00 PM"
                                    else:
                                        time_display = f"{selected_hour - 12}:00 PM"
                                    
                                    # Create tooltip content
                                    tooltip_text = f"""
                                    <b>{building_name}</b><br>
                                    Category: {category}<br>
                                    Occupancy: {occupancy}<br>
                                    Time: {time_display}
                                    """
                                    
                                    # Add marker at building centroid
                                    folium.Marker(
                                        location=[row.geometry.centroid.y, row.geometry.centroid.x],
                                        tooltip=folium.Tooltip(
                                            tooltip_text,
                                            style="""
                                            font-family: Arial, sans-serif;
                                            font-size: 12px;
                                            background-color: white;
                                            border: 1px solid #ccc;
                                            border-radius: 5px;
                                            padding: 5px;
                                            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                                            """
                                        ),
                                        icon=folium.Icon(
                                            color='red' if category == 'Residential' else 'blue',
                                            icon='building',
                                            icon_size=(10, 10)
                                        )
                                    ).add_to(m)
                                except Exception as e:
                                    pass  # Skip buildings with errors
                        
                        # Display the map using HTML embedding to avoid component conflicts
                        map_html = m._repr_html_()
                        st.components.v1.html(map_html, width=1000, height=700)
                        
                        # Add detailed legend with color intensity examples
                        st.markdown("**Color Intensity Legend:**")
                        
                        # Show actual occupancy range
                        max_occ = gdf['occupancy'].max()
                        min_occ = gdf['occupancy'].min()
                        
                        st.markdown(f"**Occupancy Range:** {min_occ} to {max_occ} devices")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown("**Residential Buildings**")
                            st.markdown("- Light Red = Low Occupancy")
                            st.markdown("- Dark Red = High Occupancy")
                        with col2:
                            st.markdown("**Non-Residential Buildings**")
                            st.markdown("- Light Teal = Low Occupancy") 
                            st.markdown("- Dark Teal = High Occupancy")
                        with col3:
                            st.markdown("**Unknown Buildings**")
                            st.markdown("- Light Gray = Low Occupancy")
                            st.markdown("- Dark Gray = High Occupancy")
                        
                        # Show intensity calculation
                        st.markdown("**Intensity Formula:**")
                        st.markdown(f"- Lightest: 30% intensity (occupancy = {min_occ})")
                        st.markdown(f"- Darkest: 100% intensity (occupancy = {max_occ})")
                        st.markdown("- **Darker colors = More occupants**")
                        
                        # Add occupancy intensity explanation
                        st.markdown("**Color Intensity System:**")
                        st.markdown("- **Light colors** = Low occupancy (30% intensity)")
                        st.markdown("- **Dark colors** = High occupancy (100% intensity)")
                        st.markdown("- **Building type** = Color (Red=Residential, Teal=Non-Residential)")
                        st.markdown("- **Occupancy level** = Intensity (Light=Low, Dark=High)")
                        
                    else:
                        raise Exception("No geometry data available")
                        
                except Exception as e:
                    # Fallback to regular scatter plot if geographic mapping fails
                    st.warning(f"Geographic mapping failed: {e}")
                    st.info("Showing data in chart format instead")
                    
                    # Create color mapping
                    color_map = {
                        'Residential': '#FF6B6B',  # Red for residential
                        'Non-Residential': '#4ECDC4',  # Teal for non-residential
                        'Unknown': '#95A5A6'  # Gray for unknown
                    }
                    
                    # Create scatter plot
                    fig_map = px.scatter(
                        building_map_data,
                        x='building_id',
                        y='occupancy',
                        color='building_category',
                        size='occupancy',
                        hover_data=[name_col] if name_col != 'building_id' else [],
                        color_discrete_map=color_map,
                        title=f"Building Occupancy - {time_display}",
                        height=500
                    )
                    
                    fig_map.update_layout(
                        xaxis_title="Building ID",
                        yaxis_title="Occupancy Count",
                        legend_title="Building Category",
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_map, use_container_width=True)
                
                # Add building details table
                st.subheader("Building Details")
                building_details = building_map_data.sort_values('occupancy', ascending=False)
                display_cols = ['building_id', name_col, 'building_category', 'occupancy']
                if 'lat' in building_details.columns:
                    display_cols.extend(['lat', 'lon'])
                
                st.dataframe(
                    building_details[display_cols].rename(
                        columns={name_col: 'Building Name', 'occupancy': 'Occupancy'}
                    ),
                    use_container_width=True
                )
                
                # Building occupancy chart section
                st.markdown("---")
                st.markdown("### Building Occupancy Analysis")
                
                # Create color mapping for building categories
                color_map = {
                    'Residential': '#FF6B6B',  # Red for residential
                    'Non-Residential': '#4ECDC4',  # Teal for non-residential
                    'Unknown': '#95A5A6'  # Gray for unknown
                }
                
                # Create the bar chart
                fig = px.bar(
                    heatmap_data,
                    x='building_name' if 'building_name' in heatmap_data.columns else 'building_id',
                    y='occupancy',
                    color='building_category',
                    color_discrete_map=color_map,
                    title=f"Building Occupancy at {time_display}",
                    labels={'occupancy': 'Occupancy Count', 'building_name': 'Building'},
                    height=400
                )
                
                fig.update_layout(
                    xaxis_tickangle=-45,
                    showlegend=True,
                    legend_title="Building Category",
                    title_x=0.5,  # Center the title
                    font=dict(size=12)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data available for the selected time and filters")
        else:
            st.warning("No data available for the selected time")
    else:
        st.warning("No data available for the selected filters")
    
    # Timeline Analysis Section - Full width and centered
    st.markdown("---")
    st.markdown("### Timeline Analysis")
    st.markdown("*Track occupancy patterns throughout the day*")
    
    # Create timeline for the selected date
    timeline_data = create_occupancy_timeline(data, selected_date)
    
    if timeline_data is not None:
        # Create two columns for timeline and stats
        col_timeline, col_stats = st.columns([2, 1])
        
        with col_timeline:
            # Create timeline chart
            fig_timeline = px.line(
                timeline_data,
                x='hour',
                y='occupancy',
                color='building_category',
                title="Hourly Occupancy Pattern",
                labels={'hour': 'Hour of Day', 'occupancy': 'Total Occupancy'},
                height=400
            )
            
            # Add vertical line for selected hour
            fig_timeline.add_vline(
                x=selected_hour,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Selected: {time_display}"
            )
            
            # Center the title and improve styling
            fig_timeline.update_layout(
                title_x=0.5,
                font=dict(size=12),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        with col_stats:
            st.markdown("#### Current Hour Statistics")
            
            # Summary statistics
            total_occupancy = timeline_data[timeline_data['hour'] == selected_hour]['occupancy'].sum()
            st.metric("Total Occupancy", f"{total_occupancy:,}")
            
            # Category breakdown
            category_breakdown = timeline_data[timeline_data['hour'] == selected_hour].groupby('building_category')['occupancy'].sum()
            for category, occupancy in category_breakdown.items():
                st.metric(f"{category} Occupancy", f"{occupancy:,}")
            
            # Add trend indicator
            if selected_hour > 0:
                prev_hour_data = timeline_data[timeline_data['hour'] == selected_hour - 1]
                if not prev_hour_data.empty:
                    prev_total = prev_hour_data['occupancy'].sum()
                    change = total_occupancy - prev_total
                    change_pct = (change / prev_total * 100) if prev_total > 0 else 0
                    
                    st.metric("Change from Previous Hour", f"{change:+,}", f"{change_pct:+.1f}%")
    
    # Detailed Analysis Section
    st.markdown("---")
    st.markdown("### Detailed Analysis")
    st.markdown("*Comprehensive data breakdown and trends*")
    
    # Show raw data for selected time
    if not filtered_data.empty:
        # Create tabs for different analysis views
        tab1, tab2, tab3 = st.tabs(["Summary Table", "Trends", "Raw Data"])
        
        with tab1:
            st.markdown("#### Building Occupancy Summary")
            
            # Display summary table - use available columns
            available_cols = filtered_data.columns.tolist()
            group_cols = ['building_category']
            
            # Add building name if available
            if 'building_name' in available_cols:
                group_cols.append('building_name')
            elif 'BLDG_NAME' in available_cols:
                group_cols.append('BLDG_NAME')
            else:
                group_cols.append('building_id')
            
            summary_data = filtered_data.groupby(group_cols).agg({
                'occupancy': ['sum', 'mean', 'count']
            }).round(2)
            
            summary_data.columns = ['Total Occupancy', 'Average Occupancy', 'Data Points']
            st.dataframe(summary_data, use_container_width=True)
        
        with tab2:
            st.markdown("#### Occupancy Trends")
            
            # Compare with previous hour
            if selected_hour > 0:
                prev_hour_data = data[
                    (data['date'] == selected_date) & 
                    (data['hour'] == selected_hour - 1) &
                    (data['building_category'].isin(selected_categories))
                ]
                
                if not prev_hour_data.empty:
                    current_total = filtered_data['occupancy'].sum()
                    prev_total = prev_hour_data['occupancy'].sum()
                    change = current_total - prev_total
                    change_pct = (change / prev_total * 100) if prev_total > 0 else 0
                    
                    # Create metrics in a centered layout
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Hour", f"{current_total:,}")
                    with col2:
                        st.metric("Previous Hour", f"{prev_total:,}")
                    with col3:
                        st.metric("Change", f"{change:+,}", f"{change_pct:+.1f}%")
                else:
                    st.info("No previous hour data available for comparison")
            else:
                st.info("This is the first hour of data - no previous hour to compare")
        
        with tab3:
            st.markdown("#### Raw Data Sample")
            st.markdown("*Showing first 100 rows of filtered data*")
            
            # Show sample of raw data
            sample_data = filtered_data.head(100)
            display_cols = ['building_id', 'building_category', 'occupancy', 'hour']
            if 'BLDG_NAME' in sample_data.columns:
                display_cols.insert(1, 'BLDG_NAME')
            
            st.dataframe(
                sample_data[display_cols],
                use_container_width=True,
                height=400
            )
    
    # Footer
    st.markdown("---")
    st.markdown("**Note:** This heatmap focuses exclusively on the **first day** of occupancy data with scroll-over time functionality. The color coding distinguishes between residential (red) and non-residential (teal) buildings. **Day 2 data is handled by Kathleen.**")

if __name__ == "__main__":
    main()
