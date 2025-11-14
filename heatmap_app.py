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
import branca.colormap as cm

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

        # --- THIS IS THE FIX ---
        # We must create the 'building_category' column for the base 'campus' map,
        # just like we did in process_data.py.
        
        def classify_building_type(bldg_type):
            if pd.isna(bldg_type): return 'Unknown'
            bldg_type_lower = str(bldg_type).lower()
            if any(keyword in bldg_type_lower for keyword in ['residence', 'dormitory', 'housing', 'greek']):
                return 'Residential'
            else:
                return 'Non-Residential'
        
        # Your geojson has 'BLDG_TYPE', which we use to create the new column
        if 'BLDG_TYPE' in campus.columns:
            campus['building_category'] = campus['BLDG_TYPE'].apply(classify_building_type)
        else:
            st.error("Error: The GeoJSON file is missing the 'BLDG_TYPE' column, so categories cannot be created.")
            return None, None
        # --- END OF FIX ---

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

    if data is None or campus is None or data.empty or campus.empty:
        st.error("Failed to load data. Please check file paths and run process_data.py.")
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
    
    # === NEW TIME SLIDER ===
    
    # 1. Generate all 10-minute intervals for the selected day
    start_time = pd.to_datetime(selected_date)
    end_time = start_time + pd.Timedelta(days=1)
    
    # Creates a list of timestamps: [00:00, 00:10, ..., 23:50]
    time_intervals = pd.date_range(start=start_time, end=end_time, freq='10T', inclusive='left')
    
    # 2. Set a default value
    default_timestamp = pd.to_datetime(f"{selected_date} 12:00")
    
    # 3. Create the single select_slider
    selected_timestamp = st.sidebar.select_slider(
        "Select Time",
        options=time_intervals,
        value=default_timestamp,
        format_func=lambda ts: ts.strftime('%H:%M')  # Show as 24-hr time
    )
    
    # 4. Extract the hour and minute for the rest of the app
    selected_hour = selected_timestamp.hour
    selected_minute = selected_timestamp.minute
    
    # === END OF NEW TIME SLIDER ===
    
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
    time_display = selected_timestamp.strftime('%I:%M %p')
    
    # Time and date display
    col_time, col_date = st.columns([1, 1])
    with col_time:
        st.metric("Selected Time", time_display)
    with col_date:
        st.metric("Selected Date", str(selected_date))
    
    st.markdown("---")
    
    # Get the filtered data for the selected time
    # A) Filter the base campus map by selected categories
    #    This is the complete list of buildings we want to show.
    campus_filtered = campus[campus['building_category'].isin(selected_categories)].copy()

    # B) --- NEW LOGIC: GET SEPARATE MAX VALUES ---
    # Get all data for the day, NOT filtered by selected categories
    full_day_data = data[(data['date'] == pd.to_datetime(selected_date).date())]
    min_occ = 0 # Min is always 0

    # Calculate max for Residential
    max_res = full_day_data[full_day_data['building_category'] == 'Residential']['occupancy'].max()
    if pd.isna(max_res) or max_res == 0: max_res = 1

    # Calculate max for Non-Residential
    max_non_res = full_day_data[full_day_data['building_category'] == 'Non-Residential']['occupancy'].max()
    if pd.isna(max_non_res) or max_non_res == 0: max_non_res = 1
    
    # Calculate max for Unknown
    max_unk = full_day_data[full_day_data['building_category'] == 'Unknown']['occupancy'].max()
    if pd.isna(max_unk) or max_unk == 0: max_unk = 1
    # --- END NEW LOGIC ---

    # C) Get the (sparse) occupancy data for the specific 10-min interval
    heatmap_data = create_heatmap_data(data, selected_date, selected_hour, selected_minute)

    # D) Merge the (sparse) heatmap data onto our (complete) filtered campus
    #    This brings the occupancy data (or lack of it) to every building.
    if heatmap_data is not None:
        gdf = campus_filtered.merge(
            heatmap_data[['BLDG_CODE', 'occupancy']], 
            on='BLDG_CODE', 
            how='left'
        )
    else:
        # If no data for this time, create gdf from campus_filtered directly
        gdf = campus_filtered.copy()
        gdf['occupancy'] = 0 # Set occupancy to 0 for all
    
    # E) Fill missing occupancy with 0
    #    Any building not in heatmap_data had 0 occupancy at this time.
    gdf['occupancy'] = gdf['occupancy'].fillna(0).astype(int)

    # === 4. MAP VISUALIZATION (NEW LOGIC) ===
    
   # === 4. MAP VISUALIZATION (NEW LOGIC) ===
    
    # --- A) Define the 6-step Colors ---
    RES_COLORS = ['#FFEDF0', '#FFC9D4', '#FF9FAD', '#FF6384', '#DC143C', '#8B0000']
    NON_RES_COLORS = ['#F1F7FF', '#D6E6FF', '#B0D0FF', '#7FB5FF', '#4292C6', '#08519C']
    UNK_COLORS = ['#FAFAFA', '#E0E0E0', '#BDBDBD', '#9E9E9E', '#616161', '#212121']
    
    # --- B) Helper function to create 6 steps for *any* max_val ---
    def get_6_steps(max_val):
        max_val = max(min_occ + 5, max_val) # Ensure max is at least 6 (for 6 bins)
        # Create 7 break points (to get 6 bins)
        breaks = np.linspace(min_occ, max_val, 7).astype(int)
        step_index = breaks[:-1].tolist() # [b0, b1, b2, b3, b4, b5]
        
        legend_ranges = []
        for i in range(6):
            start = breaks[i]
            end = breaks[i+1]
            # For the last bin, make it inclusive. For others, [start - (end-1)]
            if i == 5:
                range_str = f"{start}+"
            else:
                range_str = f"{start} - {end - 1}"
            legend_ranges.append(range_str)
            
        # Handle edge case where breaks are not unique (if max_val is small)
        # e.g., [0, 0, 1, 2, 3, 4] -> [0, 1, 2, 3, 4, 5]
        if len(set(step_index)) < 6:
            step_index = [0, 1, 2, 3, 4, 5]
            legend_ranges = ["0", "1", "2", "3", "4", "5+"]

        return step_index, legend_ranges

    # --- C) Create the 3 SEPARATE colormaps and legends ---
    step_index_res, ranges_res = get_6_steps(max_res)
    res_map = cm.StepColormap(
        colors=RES_COLORS, index=step_index_res, vmin=min_occ, vmax=max_res
    )
    
    step_index_non_res, ranges_non_res = get_6_steps(max_non_res)
    non_res_map = cm.StepColormap(
        colors=NON_RES_COLORS, index=step_index_non_res, vmin=min_occ, vmax=max_non_res
    )
    
    step_index_unk, ranges_unk = get_6_steps(max_unk)
    unk_map = cm.StepColormap(
        colors=UNK_COLORS, index=step_index_unk, vmin=min_occ, vmax=max_unk
    )
    
    if not gdf.empty:
        st.markdown("### Interactive Campus Map")
        st.markdown("*Hover over buildings to see occupancy details*")
        
        try:
            gdf_valid = gdf[gdf.geometry.is_valid & gdf.geometry.notna()]
            
            if gdf_valid.empty:
                st.error("No valid building geometry found for this time.")
                return

            m = folium.Map(location=[33.7756, -84.3963], zoom_start=15, tiles="CartoDB positron")

            # --- D) Define the style_function (This is now using the StepColormap) ---
            def style_function(feature):
                try:
                    occupancy = float(feature['properties']['occupancy'])
                    category = feature['properties']['building_category']
                    
                    if category == 'Residential':
                        fill_color = res_map(occupancy)
                    elif category == 'Non-Residential':
                        fill_color = non_res_map(occupancy)
                    else:
                        fill_color = unk_map(occupancy)
                        
                    return {
                        'fillColor': fill_color,
                        'color': 'black',
                        'weight': 0.5,
                        'fillOpacity': 0.85, # Constant opacity
                    }
                except Exception:
                    return {'fillColor': 'purple', 'fillOpacity': 0.5} # Error color

            # --- E) Create tooltip and GeoJson layer (no change) ---
            tooltip_cols = ['BLDG_NAME', 'building_category', 'occupancy']
            gdf_for_map = gdf_valid[tooltip_cols + ['geometry']].copy() 
            gdf_for_map['time_display'] = time_display
            tooltip_cols.append('time_display')
            tooltip_aliases = ['Building', 'Category', 'Occupancy', 'Time']

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

            st_folium(m, width=1000, height=700)
            
            # --- F) NEW LEGEND (Uses the 3 separate range lists) ---
            st.markdown("#### Map Legend")
            
            # This helper function is the same as before
            def create_step_legend(colors, value_ranges):
                html = "<div style='font-size: 12px; line-height: 1.6;'>"
                for color, text in zip(colors, value_ranges):
                    line = f'<div style="display: flex; align-items: center; margin-bottom: 3px;"><div style="width: 20px; height: 20px; background-color: {color}; border: 1px solid #444; margin-right: 8px; flex-shrink: 0;"></div><span>{text}</span></div>'
                    html += line
                html += "</div>"
                return html

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Residential**")
                # Use the residential colors and ranges
                st.markdown(create_step_legend(RES_COLORS, ranges_res), unsafe_allow_html=True)
            
            with col2:
                st.markdown("**Non-Residential**")
                # Use the non-residential colors and ranges
                st.markdown(create_step_legend(NON_RES_COLORS, ranges_non_res), unsafe_allow_html=True)

            with col3:
                st.markdown("**Unknown**")
                # Use the unknown colors and ranges
                st.markdown(create_step_legend(UNK_COLORS, ranges_unk), unsafe_allow_html=True)
                
            # --- Bar Chart ---
            st.markdown("### Building Occupancy Analysis")
            bar_data = gdf[gdf['occupancy'] > 0].sort_values('occupancy', ascending=False)
            
            if not bar_data.empty:
                fig = px.bar(
                    bar_data,
                    x='BLDG_NAME',
                    y='occupancy',
                    color='building_category',
                    color_discrete_map={ 
                        'Residential': '#DC143C', # Crimson (from new ramp)
                        'Non-Residential': '#4292C6', # Mid-Blue (from new ramp)
                        'Unknown': '#737373'      # Mid-Gray (from new ramp)
                    },
                    title=f"Building Occupancy at {time_display}",
                    labels={'occupancy': 'Unique User Count', 'BLDG_NAME': 'Building'},
                    height=400
                )
                st.plotly_chart(fig, width='stretch')
            else:
                st.info("No building occupancy (greater than 0) to display in bar chart.")

        except Exception as e:
            st.error(f"An error occurred while creating the map: {e}")
            st.exception(e)

    else:
        st.warning(f"No building categories selected. Please select one or more categories from the sidebar.")
    
    # --- Timeline Analysis Section ---
    st.markdown("---")
    st.markdown("### Hourly Timeline Analysis")
    st.markdown("*Track total occupancy patterns throughout the day*")
    
    # We must filter the timeline data by selected categories too
    timeline_data_full = create_occupancy_timeline(data, selected_date)
    
    if timeline_data_full is not None:
        timeline_data_filtered = timeline_data_full[
            timeline_data_full['building_category'].isin(selected_categories)
        ]

        if not timeline_data_filtered.empty:
            # --- FIX 4: Re-added color='building_category' ---
            fig_timeline = px.line(
                timeline_data_filtered, # Use the filtered data directly
                x='time_bin',
                y='occupancy',
                color='building_category', # This brings back the separate lines
                color_discrete_map={ # Match the bar chart/map
                    'Residential': '#e34a33',
                    'Non-Residential': '#2b8cbe',
                    'Unknown': '#969696'
                },
                title=f"Hourly Occupancy Pattern on {selected_date} (Filtered)",
                labels={'time_bin': 'Time', 'occupancy': 'Total Unique Users'},
                height=400
            )
            
            fig_timeline.add_shape(
                type='line',
                x0=selected_timestamp,
                x1=selected_timestamp,
                y0=0,
                y1=1,
                yref='paper',
                line=dict(color='red', dash='dash')
            )

            fig_timeline.add_annotation(
                x=selected_timestamp, y=1.05, yref='paper',
                text=f"Selected: {time_display}", showarrow=False, font=dict(color="red")
            )
            st.plotly_chart(fig_timeline, width='stretch')
        else:
            st.info("No timeline data for the selected categories.")
    else:
        st.warning(f"No timeline data available for {selected_date}.")

    # --- Raw Data View ---
    if st.checkbox("Show Raw Aggregated Data (for this time)"):
        st.subheader(f"Data for {time_display} on {selected_date}")
        st.dataframe(gdf)


if __name__ == "__main__":
    main()