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
import folium
from streamlit_folium import st_folium

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
    
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the PRE-PROCESSED 10-minute summary data."""
    try:
        # Load the clean, aggregated 10-minute data
        data = pd.read_parquet('notebooks/data/ten_min_occupancy_summary.parquet')
        
        # Convert the WKT text column back into a real geometry column
        data['geometry'] = gpd.GeoSeries.from_wkt(data['geometry'])
        
        # Convert the pandas DataFrame back into a GeoDataFrame
        data = gpd.GeoDataFrame(data, geometry='geometry', crs="EPSG:4326")

        # Extract date/time components for filtering
        data['time_bin'] = pd.to_datetime(data['time_bin'])
        data['date'] = data['time_bin'].dt.date
        data['hour'] = data['time_bin'].dt.hour
        data['minute'] = data['time_bin'].dt.minute
        data['day_of_week'] = data['time_bin'].dt.day_name()

        # remove incomplete day
        date_to_remove = pd.to_datetime("2025-04-13").date()
        data = data[data['date'] != date_to_remove].copy()
        
        # Load the campus boundaries for the base map
        with open('notebooks/data/campus_buildings_categories.geojson', 'r') as f:
            building_data = json.load(f)
        campus = gpd.GeoDataFrame.from_features(building_data['features'], crs="EPSG:4326")

        return data, campus
        
    except Exception as e:
        st.error(f"Error loading summary data: {e}")
        st.error("Please ensure 'ten_min_occupancy_summary.parquet' exists and you have run the `process_data.py` script.")
        return None, None


def create_heatmap_data(data, selected_date, selected_hour, selected_minute):
    """
    Filter the pre-aggregated data for the selected time.
    The aggregation is already done in the processing script.
    """
    selected_date_obj = pd.to_datetime(selected_date).date()
    
    filtered_data = data[
        (data['date'] == selected_date_obj) &
        (data['hour'] == selected_hour) &
        (data['minute'] == selected_minute)
    ].copy()
    
    if filtered_data.empty:
        return None
    
    # Data is already aggregated, just return it
    return filtered_data


def create_occupancy_timeline(data, selected_date):
    """Create timeline showing occupancy patterns for the selected date."""
    selected_date_obj = pd.to_datetime(selected_date).date()
    
    daily_data = data[data['date'] == selected_date_obj].copy()
    
    if daily_data.empty:
        return None
    
    # Aggregate by hour and building category for the line chart
    # We sum the 'occupancy' which is the pre-calculated unique user count
    hourly_data = daily_data.groupby(['hour', 'building_category']).agg({
        'occupancy': 'sum'
    }).reset_index()
    
    return hourly_data


def create_campus_timeseries(data, interval_minutes: int = 10):
    """Create a campus-wide time series of unique users.

    This helper supports two input shapes:
    - Pre-aggregated 10-minute summary (has 'time_bin' and 'occupancy') -> simply sums occupancy across buildings per time_bin.
    - Raw connection-level data with a 'time' column and device ids ('MAC' or 'IP') -> resamples into interval_minutes and counts unique devices.

    Returns a DataFrame with columns: ['time_bin', 'unique_users'].
    """
    if data is None or data.empty:
        return None

    df = data.copy()

    # Case 1: pre-aggregated 10-minute bins already exist
    if 'time_bin' in df.columns and 'occupancy' in df.columns:
        # Ensure datetime
        df['time_bin'] = pd.to_datetime(df['time_bin'])
        ts = df.groupby('time_bin', as_index=False)['occupancy'].sum()
        ts = ts.rename(columns={'occupancy': 'unique_users'})
        return ts[['time_bin', 'unique_users']]

    # Case 2: raw connection-level data
    if 'time' not in df.columns:
        st.error("Data missing both 'time_bin'/'occupancy' and raw 'time' columns; cannot build time series.")
        return None

    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time')

    # Choose identifier column for uniqueness
    id_col = None
    if 'MAC' in df.columns:
        id_col = 'MAC'
    elif 'IP' in df.columns:
        id_col = 'IP'

    # Set time as index for resampling
    df = df.set_index('time')

    # If we have an identifier, compute unique counts per resample bin
    if id_col is not None:
        ts = df[id_col].resample(f"{interval_minutes}T").nunique()
    else:
        # Fallback: count rows per bin
        ts = df.resample(f"{interval_minutes}T").size()

    ts = ts.rename('unique_users').to_frame().reset_index()
    ts['time_bin'] = ts['time']

    return ts[['time_bin', 'unique_users']]


def main():
    st.title("Campus Occupancy Heatmap")
    st.markdown("**Interactive heatmap showing unique device counts in 10-minute intervals.**")
    
    # Load data
    data, campus = load_data()
    
    if data is None or data.empty:
        st.error("Failed to load data.")
        return
    
    # Sidebar controls
    st.sidebar.header("Controls")

    # Date selection
    available_dates = sorted(data['date'].unique())
    selected_date = st.sidebar.selectbox(
        "Select Date",
        available_dates,
        index=0
    )
    
    # Hour selection
    selected_hour = st.sidebar.slider(
        "Select Hour (24-hr)",
        min_value=0, 
        max_value=23, 
        value=14, # Default to 2 PM
        format="%d:00"
    )
    
    # Minute selection
    selected_minute = st.sidebar.select_slider(
        "Select Minute",
        options=[0, 10, 20, 30, 40, 50],
        value=0
    )
    
    # Building category filter
    building_categories = data['building_category'].unique()
    selected_categories = st.sidebar.multiselect(
        "Building Categories",
        building_categories,
        default=building_categories
    )

    # Appearance controls: color palette, normalization, smoothing
    st.sidebar.markdown("---")
    st.sidebar.subheader("Appearance")
    palette_option = st.sidebar.selectbox(
        "Color Palette",
        options=[
            'Residential/Non-Residential (red/teal)',
            'Colorblind-friendly (Set1)',
            'Viridis',
            'High Contrast'
        ],
        index=0
    )

    normalization = st.sidebar.selectbox(
        "Color normalization",
        options=['min-max', '95th-percentile'],
        index=0,
        help="Choose how occupancy values are scaled into color intensity"
    )

    smoothing_bins = st.sidebar.slider(
        "Timeseries smoothing (number of 10-min bins)",
        min_value=1,
        max_value=12,
        value=6,
        help="Rolling window for smoothing the 10-minute timeseries (default ~60 minutes)"
    )

    # Helper to get category color mapping
    def get_palette_mapping(name):
        if name == 'Colorblind-friendly (Set1)':
            return {
                'Residential': '#e41a1c',
                'Non-Residential': '#377eb8',
                'Unknown': '#4daf4a'
            }
        if name == 'Viridis':
            return {
                'Residential': '#440154',
                'Non-Residential': '#21918c',
                'Unknown': '#fde725'
            }
        if name == 'High Contrast':
            return {
                'Residential': '#d62728',
                'Non-Residential': '#1f77b4',
                'Unknown': '#ff7f0e'
            }
        # default
        return {
            'Residential': 'red',
            'Non-Residential': 'teal',
            'Unknown': 'gray'
        }

    category_colors = get_palette_mapping(palette_option)
    # small helper to convert hex to "r, g, b" string for CSS rgba usage
    def hex_to_rgb_str(h):
        h = h.lstrip('#')
        try:
            r = int(h[0:2], 16)
            g = int(h[2:4], 16)
            b = int(h[4:6], 16)
            return f"{r}, {g}, {b}"
        except:
            return "128, 128, 128"

    # --- Main content ---
    
    # Convert selected time to 12-hour format for display
    hour_12 = selected_hour % 12
    if hour_12 == 0: hour_12 = 12 # 0 and 12 should be 12
    am_pm = "AM" if selected_hour < 12 else "PM"
    time_display = f"{hour_12}:{selected_minute:02d} {am_pm}"
    
    # Time and date display
    col_time, col_date = st.columns([1, 1])
    with col_time:
        st.metric("Selected Time", time_display)
    with col_date:
        st.metric("Selected Date", str(selected_date))
    
    st.markdown("---")
    
    # Get the filtered data for the selected time
    heatmap_data = create_heatmap_data(data, selected_date, selected_hour, selected_minute)
    
    # Filter by category
    if heatmap_data is not None:
        heatmap_data = heatmap_data[heatmap_data['building_category'].isin(selected_categories)]
    
    if heatmap_data is not None and not heatmap_data.empty:
        # --- Create geographic map visualization ---
        st.markdown("### Interactive Campus Map")
        st.markdown("*Hover over buildings to see occupancy details*")
        
        try:
            # Get valid geometry
            gdf = heatmap_data[heatmap_data.geometry.is_valid & heatmap_data.geometry.notna()]
            
            if gdf.empty:
                st.error("No valid building geometry found for this time.")
                return

            # Create the base map
            m = folium.Map(
                location=[33.7756, -84.3963], 
                zoom_start=15, 
                tiles="CartoDB positron"
            )

            # --- 1. Get occupancy range for a STABLE color scale ---
            # We use the full day's data so the legend is consistent
            full_day_data = data[data['date'] == pd.to_datetime(selected_date).date()]
            min_occ = full_day_data['occupancy'].min()
            if normalization == '95th-percentile':
                max_occ = float(np.percentile(full_day_data['occupancy'].dropna(), 95))
            else:
                max_occ = full_day_data['occupancy'].max()

            # --- 2. Define an advanced style_function ---
            def style_function(feature):
                category = feature['properties']['building_category']
                occupancy = feature['properties']['occupancy']
                
                # Set color by category using palette selection
                color = category_colors.get(category, 'gray')

                # Set opacity (shade) by occupancy
                if max_occ > min_occ:
                    # Normalize occupancy from 0 (min) to 1 (max)
                    norm_occupancy = (occupancy - min_occ) / (max_occ - min_occ)
                else:
                    norm_occupancy = 0
                
                # Map occupancy to opacity (e.g., 0.3 for low, 0.9 for high)
                opacity = 0.3 + (norm_occupancy * 0.6)
                
                return {
                    'fillColor': color,
                    'color': 'black',      # A thin black border
                    'weight': 0.5,
                    'fillOpacity': opacity, # Opacity is now based on occupancy
                }

            # --- 3. Create a clean tooltip GeoDataFrame ---
            tooltip_cols = ['BLDG_NAME', 'building_category', 'occupancy']
            gdf_for_map = gdf[tooltip_cols + ['geometry']].copy()
            gdf_for_map['time_display'] = time_display
            tooltip_cols.append('time_display')
            tooltip_aliases = ['Building', 'Category', 'Occupancy', 'Time']

            # --- 4. Create the single GeoJson layer ---
            folium.GeoJson(
                gdf_for_map,
                style_function=style_function,
                tooltip=folium.GeoJsonTooltip(
                    fields=tooltip_cols,
                    aliases=tooltip_aliases,
                    style="""
                        font-family: Arial, sans-serif;
                        font-size: 12px;
                        background-color: white;
                        border: 1px solid #ccc;
                        border-radius: 5px;
                        padding: 5px;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                        """
                )
            ).add_to(m)

            # --- 5. Display the map ---
            st_folium(m, width=1000, height=700)
            
            # --- 6. ADD A CUSTOM HTML LEGEND (Placed below the map) ---
            st.markdown("#### Map Legend")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Residential**")
                rbg = hex_to_rgb_str(category_colors.get('Residential', '#ff0000'))
                st.markdown(f"""
                <div style="background: linear-gradient(to right, rgba({rbg}, 0.3), rgba({rbg}, 0.95)); border-radius: 5px; height: 20px; width: 100%;"></div>
                <div style="display: flex; justify-content: space-between; font-size: 12px;">
                    <span>Low ({min_occ})</span>
                    <span>High ({max_occ})</span>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("**Non-Residential**")
                rgb = hex_to_rgb_str(category_colors.get('Non-Residential', '#008080'))
                st.markdown(f"""
                <div style="background: linear-gradient(to right, rgba({rgb}, 0.3), rgba({rgb}, 0.95)); border-radius: 5px; height: 20px; width: 100%;"></div>
                <div style="display: flex; justify-content: space-between; font-size: 12px;">
                    <span>Low ({min_occ})</span>
                    <span>High ({max_occ})</span>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown("**Unknown**")
                rgb_u = hex_to_rgb_str(category_colors.get('Unknown', '#808080'))
                st.markdown(f"""
                <div style="background: linear-gradient(to right, rgba({rgb_u}, 0.3), rgba({rgb_u}, 0.95)); border-radius: 5px; height: 20px; width: 100%;"></div>
                <div style="display: flex; justify-content: space-between; font-size: 12px;">
                    <span>Low ({min_occ})</span>
                    <span>High ({max_occ})</span>
                </div>
                """, unsafe_allow_html=True)
            
            # --- END OF NEW CODE ---
            
            # --- Bar Chart ---
            st.markdown("### Building Occupancy Analysis")
            fig = px.bar(
                heatmap_data.sort_values('occupancy', ascending=False),
                x='BLDG_NAME',
                y='occupancy',
                color='building_category',
                color_discrete_map=category_colors,
                title=f"Building Occupancy at {time_display}",
                labels={'occupancy': 'Unique User Count', 'BLDG_NAME': 'Building'},
                height=400
            )
            st.plotly_chart(fig, width='stretch')

        except Exception as e:
            st.error(f"An error occurred while creating the map: {e}")

    else:
        st.warning(f"No data available for {time_display} on {selected_date} with the selected filters.")
    
    # --- Timeline Analysis Section ---
    st.markdown("---")
    st.markdown("### Hourly Timeline Analysis")
    st.markdown("*Track total occupancy patterns throughout the day*")
    
    timeline_data = create_occupancy_timeline(data, selected_date)
    
    if timeline_data is not None:
        fig_timeline = px.line(
            timeline_data,
            x='hour',
            y='occupancy',
            color='building_category',
            title=f"Total Hourly Occupancy Pattern on {selected_date}",
            labels={'hour': 'Hour of Day', 'occupancy': 'Total Unique Users'},
            height=400
        )
        
        # Add vertical line for selected hour
        fig_timeline.add_vline(
            x=selected_hour,
            line_dash="dash",
            line_color=category_colors.get('Residential', '#d62728'),
            annotation_text=f"Selected Hour: {selected_hour}:00"
        )
        st.plotly_chart(fig_timeline, width='stretch')
        # --- Campus-wide 10-minute unique-user time series and simple smoothing ---
        ts_10min = create_campus_timeseries(data, interval_minutes=10)
        if ts_10min is not None and not ts_10min.empty:
            ts = ts_10min.copy()
            ts = ts.sort_values('time_bin')
            # Add a rolling mean for smoothing (smoothing_bins * 10 minutes)
            ts['smoothed'] = ts['unique_users'].rolling(window=smoothing_bins, min_periods=1, center=True).mean()
            smoothing_minutes = smoothing_bins * 10

            fig_ts = go.Figure()
            raw_color = '#666666'
            smoothed_color = category_colors.get('Residential', '#d62728')
            fig_ts.add_trace(go.Scatter(
                x=ts['time_bin'],
                y=ts['unique_users'],
                mode='lines',
                name='Unique users (10-min bins)',
                line=dict(color=raw_color, width=1),
                hovertemplate='%{x}<br>Unique users: %{y}<extra></extra>'
            ))
            fig_ts.add_trace(go.Scatter(
                x=ts['time_bin'],
                y=ts['smoothed'],
                mode='lines',
                name=f'Smoothed (rolling {smoothing_minutes} min)',
                line=dict(color=smoothed_color, width=2)
            ))

            fig_ts.update_layout(
                title=f'Campus-wide Unique Users — 10-minute bins on {selected_date}',
                xaxis_title='Time',
                yaxis_title='Unique Users',
                height=350,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )

            st.markdown("#### Campus-wide 10-minute Unique Users")
            st.plotly_chart(fig_ts, use_container_width=True)

    # --- Raw Data View ---
    if st.checkbox("Show Raw Aggregated Data (for this time)"):
        st.subheader(f"Data for {time_display} on {selected_date}")
        st.dataframe(heatmap_data)


if __name__ == "__main__":
    main()