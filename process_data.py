import pandas as pd
import geopandas as gpd
import json

def create_summary_file_10min():
    print("Starting data processing for 10-minute intervals...")
    
    # --- 1. Load Building Geometry ---
    print("Loading building geometry...")
    with open('notebooks/data/campus_buildings_categories.geojson', 'r') as f:
        building_data = json.load(f)
    campus = gpd.GeoDataFrame.from_features(building_data['features'], crs="EPSG:4326")

    # Create a simple mapping for geometry
    building_geometry_map = {
        row['BLDG_CODE']: row['geometry'] 
        for _, row in campus.iterrows()
    }

    # --- 2. Load Raw 2-Day WiFi Data ---
    print("Loading 2-day raw WiFi data...")
    try:
        wifi = pd.read_parquet('notebooks/data/wifi_data_2days.parquet', engine='fastparquet')
    except:
        wifi = pd.read_parquet('notebooks/data/wifi_data_2days.parquet', engine='pyarrow')
    
    # --- 3. Process Time and Building IDs ---
    print("Processing time and building IDs...")
    wifi['time'] = pd.to_datetime(wifi['time'])
    
    # --- THIS IS THE KEY CHANGE ---
    # Create a 'time_bin' column by rounding down to the nearest 10 minutes
    # '10T' is the pandas code for a 10-minute frequency
    wifi['time_bin'] = wifi['time'].dt.floor('10T')
    
    wifi['bid_no_letters'] = wifi['building_id'].str.extract(r'(\d+)')

    # --- 4. Aggregate Unique Users ---
    print("Aggregating unique users (this is the slow part)...")
    
    # -- Change to Username instead of MAC address
    # Group by the new 'time_bin' instead of date and hour
    occupancy_data = wifi.groupby(['time_bin', 'bid_no_letters', 'building_id'])['Username'].nunique().reset_index()
    
    # Rename for clarity
    occupancy_data.rename(columns={'Username': 'occupancy', 'bid_no_letters': 'BLDG_CODE'}, inplace=True)

    # --- 5. Merge Building & Geometry Info ---
    print("Merging building info...")
    campus_info = campus.drop(columns='geometry')
    occupancy_data = occupancy_data.merge(campus_info, on='BLDG_CODE', how='left')

    # Add geometry using the map
    occupancy_data['geometry'] = occupancy_data['BLDG_CODE'].map(building_geometry_map)
    occupancy_data = gpd.GeoDataFrame(occupancy_data, geometry='geometry', crs="EPSG:4326")
    
    # Define building category
    def classify_building_type(bldg_type):
        if pd.isna(bldg_type): return 'Unknown'
        bldg_type_lower = str(bldg_type).lower()
        if any(keyword in bldg_type_lower for keyword in ['residence', 'dormitory', 'housing', 'greek']):
            return 'Residential'
        else:
            return 'Non-Residential'
            
    occupancy_data['building_category'] = occupancy_data['BLDG_TYPE'].apply(classify_building_type)

    # --- 6. Save the Final Files ---
    
    # A) Save the clean summary file for your app
    output_parquet = 'notebooks/data/ten_min_occupancy_summary.parquet'
    
    # Convert geometry to WKT for saving
    occupancy_data['geometry'] = occupancy_data['geometry'].to_wkt()
    occupancy_data.to_parquet(output_parquet)
    print(f"Success! App summary file saved to: {output_parquet}")

    # B) Save the CSV file for your other task
    output_csv = 'notebooks/data/ten_min_occupancy_summary.csv'
    occupancy_data.to_csv(output_csv, index=False)
    print(f"Success! CSV summary file saved to: {output_csv}")


if __name__ == "__main__":
    create_summary_file_10min()