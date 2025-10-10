#!/usr/bin/env python3
"""
Launcher script for the Campus Occupancy Heatmap Streamlit app
"""
import subprocess
import sys
import os

def main():
    """Launch the Streamlit app"""
    try:
        # Check if streamlit is installed
        import streamlit
        print("Streamlit is available. Launching app...")
        
        # Run the streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "heatmap_app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
        
    except ImportError:
        print("Streamlit not found. Installing requirements...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Requirements installed. Please run the app again.")
        
    except Exception as e:
        print(f"Error launching app: {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    main()

