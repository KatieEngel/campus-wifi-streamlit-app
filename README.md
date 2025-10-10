# Campus Occupancy Heatmap - First Day Analysis

A Streamlit application for visualizing campus occupancy patterns with scroll-over time functionality and color-coded residential vs non-residential buildings. Yameen Ahmed handled the Day 1 data processing and visualization, while Kathleen handled the same for Day 2.

## Features

- **First Day Heatmap**: Interactive visualization of campus occupancy for the first day only
- **Scroll-Over Time**: Dynamic time slider to explore occupancy patterns throughout the day
- **Color Coding**: Distinct colors for residential (red) and non-residential (teal) buildings
- **Building Shape Visualization**: Actual building outlines highlighted based on occupancy



## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the data files in the correct location:
   - `notebooks/data/wifi_data_2days.parquet`
   - `notebooks/data/campus_buildings_categories.geojson`

## Usage

### Run the Application
```bash
streamlit run heatmap_app.py
```

### Alternative: Use the launcher script
```bash
python run_app.py
```

### Manual launch
```bash
python -m streamlit run heatmap_app.py --server.port 8501
```

**Note:** The application works exclusively with real WiFi data. No sample or fabricated data is used.

## Performance Notes

**Why is loading slow?**
- The original dataset has 1.3M+ rows
- Complex merge operations with building data
- String operations on large datasets
- Memory-intensive processing

**Optimizations implemented:**
- **Smart sampling**: Reduces 1.3M rows to ~50K for performance
- **Vectorized operations**: Faster than loops and apply functions
- **Dictionary mapping**: Faster than complex merges
- **Progress indicators**: Shows loading status
- **Caching**: Streamlit caches processed data
- **Real data only**: No fabricated or sample data used

## Application Features

### Main Interface
- **Date Selection**: Choose the first day of occupancy data to focus on
- **Hour Slider**: Scroll over time to see occupancy changes throughout the day
- **Building Category Filter**: Toggle between residential and non-residential buildings
- **Interactive Heatmap**: Bar chart showing occupancy by building with color coding

### Timeline Analysis
- **Hourly Patterns**: Line chart showing occupancy trends throughout the day
- **Category Breakdown**: Separate lines for residential vs non-residential buildings
- **Selected Hour Indicator**: Red dashed line showing current time selection

### Summary Statistics
- **Total Occupancy**: Current hour's total occupancy count
- **Category Metrics**: Breakdown by building category
- **Trend Analysis**: Comparison with previous hour
- **Raw Data Table**: Detailed occupancy data for selected time

## Color Scheme

- **Red**: Residential buildings (dormitories, Greek housing, residence halls)
- **Teal**: Non-residential buildings (academic, library, student center, gym)
- **Gray**: Unknown building types

## Technical Details

- **Framework**: Streamlit for web interface
- **Visualization**: Plotly for interactive charts
- **Data Processing**: Pandas and GeoPandas
- **Caching**: Streamlit caching for performance
- **Responsive Design**: Wide layout optimized for heatmap visualization

## Troubleshooting

If you encounter issues:
1. Ensure all dependencies are installed: `pip install -r requirements.txt`
2. Check that data files are in the correct location
3. Verify file permissions for data access
