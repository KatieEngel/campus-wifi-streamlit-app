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
    timeline_data = daily_data.groupby(['time_bin', 'building_category']).agg({
        'occupancy': 'sum'
    }).reset_index()
    
    return timeline_data


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
            max_occ = full_day_data['occupancy'].max()

            # --- 2. Define an advanced style_function ---
            def style_function(feature):
                category = feature['properties']['building_category']
                occupancy = feature['properties']['occupancy']
                
                # Set color by category
                colors = {
                    'Residential': 'red',
                    'Non-Residential': 'teal',
                    'Unknown': 'gray'
                }
                color = colors.get(category, 'gray')

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
                st.markdown(f"""
                <div style="background: linear-gradient(to right, rgba(255, 0, 0, 0.3), rgba(255, 0, 0, 0.9)); border-radius: 5px; height: 20px; width: 100%;"></div>
                <div style="display: flex; justify-content: space-between; font-size: 12px;">
                    <span>Low ({min_occ})</span>
                    <span>High ({max_occ})</span>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("**Non-Residential**")
                st.markdown(f"""
                <div style="background: linear-gradient(to right, rgba(0, 128, 128, 0.3), rgba(0, 128, 128, 0.9)); border-radius: 5px; height: 20px; width: 100%;"></div>
                <div style="display: flex; justify-content: space-between; font-size: 12px;">
                    <span>Low ({min_occ})</span>
                    <span>High ({max_occ})</span>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown("**Unknown**")
                st.markdown(f"""
                <div style="background: linear-gradient(to right, rgba(128, 128, 128, 0.3), rgba(128, 128, 128, 0.9)); border-radius: 5px; height: 20px; width: 100%;"></div>
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
                color_discrete_map={
                    'Residential': 'red',
                    'Non-Residential': 'teal',
                    'Unknown': 'gray'
                },
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
            x='time_bin',
            y='occupancy',
            color='building_category',
            title=f"Total Hourly Occupancy Pattern on {selected_date}",
            labels={'hour': 'Hour of Day', 'occupancy': 'Total Unique Users'},
            height=400
        )

        selected_timestamp = pd.to_datetime(f"{selected_date} {selected_hour}:{selected_minute:02d}")
        
        fig_timeline.add_shape(
            type='line',
            x0=selected_timestamp,
            x1=selected_timestamp,
            y0=0,
            y1=1,
            yref='paper',  # 'paper' means span the full height of the plot
            line=dict(color='red', dash='dash')
        )

        fig_timeline.add_annotation(
            x=selected_timestamp,
            y=1.05,       # Position it just above the top of the plot
            yref='paper', # Use 'paper' to reference the plot area
            text=f"Selected: {time_display}",
            showarrow=False,
            font=dict(color="red")
        )
        st.plotly_chart(fig_timeline, width='stretch')

    # --- Raw Data View ---
    if st.checkbox("Show Raw Aggregated Data (for this time)"):
        st.subheader(f"Data for {time_display} on {selected_date}")
        st.dataframe(heatmap_data)


if __name__ == "__main__":
    main()