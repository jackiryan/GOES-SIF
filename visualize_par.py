#!/usr/bin/env python3
import argparse
from datetime import datetime, timedelta
import h5py
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import numpy.typing as npt
from pathlib import Path
import re
import sys


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for the PAR visualization script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Visualize NASA GeoNEX PAR (Photosynthetically Active Radiation) data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python visualize_par.py data/G016_DSR_2021067_h18v10.h5 -o output/plots/
        """
    )
    
    parser.add_argument(
        "input",
        type=str,
        help="Path to input PAR HDF-5 file"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=".",
        help="Output directory for generated plot images"
    )
    
    return parser.parse_args()


def validate_inputs(input_file: str, output_dir: str) -> tuple[Path, Path]:
    """
    Validate input file and output directory.
    
    Args:
        input_file (str): Path to input HDF-5 file
        output_dir (str): Path to output directory
        
    Returns:
        tuple: (Path object for input file, Path object for output directory)
        
    Raises:
        SystemExit: If validation fails
    """
    # Validate input file
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: Input file does not exist: {input_file}")
        sys.exit(1)
    
    # Validate and create output directory
    output_path = Path(output_dir)
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error: Cannot create output directory {output_dir}: {e}")
        sys.exit(1)
    
    return input_path, output_path


def parse_filename(filename: str) -> dict[str, str | int]:
    """
    Parse GeoNEX filename to extract metadata.
    
    Expected format: SATELLITE_PRODUCT_DATEYYYYDDD_hXXvYY.h5
    Example: G016_DSR_2021067_h18v10.h5
    
    Args:
        filename (str): Name of the input file
        
    Returns:
        dict: Dictionary containing parsed metadata
        
    Raises:
        SystemExit: If filename doesn't match expected format
    """
    # Remove .h5 extension and parse components
    basename = Path(filename).stem
    print(len(basename))
    if len(basename) > 23:
        # Sometimes these files have non-standard filenames, but the end part
        # matches the convention
        basename = basename[-23:]
    print(basename)
    
    # Pattern to match GeoNEX filename format
    pattern = r"([A-Z0-9]+)_([A-Z]+)_(\d{7})_h(\d+)v(\d+)"
    match = re.match(pattern, basename)
    
    if not match:
        print(f"Error: Filename '{filename}' doesn't match expected GeoNEX format")
        print("Expected format: SATELLITE_PRODUCT_YYYYDDD_hXXvYY.h5")
        sys.exit(1)
    
    satellite, product, date_code, h_idx, v_idx = match.groups()
    
    # Parse date code (YYYYDDD format)
    year = int(date_code[:4])
    day_of_year = int(date_code[4:])
    
    date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
    
    # Convert grid indices to integers
    h_idx = int(h_idx)
    v_idx = int(v_idx)
    
    # Calculate geographic bounds
    # Upper left longitude = -180 + (h × 6)
    # Upper left latitude = 60 - (v × 6)
    ul_lon = -180 + (h_idx * 6)
    ul_lat = 60 - (v_idx * 6)
    
    # Calculate full extent (6x6 degree square)
    lon_min = ul_lon
    lon_max = ul_lon + 6
    lat_min = ul_lat - 6  
    lat_max = ul_lat

    # Get the sunrise and sunset hour using the center coord
    lat_mid = lat_min + 3
    lon_mid = lon_min + 3
    print(f"Center point: ({lat_mid}, {lon_mid})")
    rise_hr, set_hr = calculate_sunrise_sunset(lat_mid, lon_mid, year, day_of_year)
    
    metadata = {
        "satellite": satellite,
        "product": product,
        "year": year,
        "day_of_year": day_of_year,
        "date": date,
        "h_index": h_idx,
        "v_index": v_idx,
        "lon_min": lon_min,
        "lon_max": lon_max,
        "lat_min": lat_min,
        "lat_max": lat_max,
        "ul_lon": ul_lon,
        "ul_lat": ul_lat,
        "sunrise_hr": rise_hr - 1, # add 1 hour buffer to account for eastern edge
        "sunset_hr": set_hr
    }
    
    return metadata


def load_par_data(input_file: Path) -> npt.NDArray[np.float32]:
    """
    Load PAR data from HDF-5 file.
    
    Args:
        input_file (Path): Path to input HDF-5 file
        
    Returns:
        numpy.ndarray: PAR data array with shape (24, 600, 600)
        
    Raises:
        SystemExit: If file cannot be read or PAR_hourly variable is missing
    """
    try:
        print(f"Opening HDF-5 file: {input_file}")
        
        with h5py.File(input_file, "r") as hdf_file:
            if "PAR_hourly" not in hdf_file:
                print("Error: 'PAR_hourly' variable not found in HDF-5 file")
                print("Available variables:", list(hdf_file.keys()))
                sys.exit(1)
            
            par_data = np.array(hdf_file["PAR_hourly"][:], dtype=np.int16) # type: ignore
            par_masked = np.where(par_data == -9999, np.nan, par_data)
            # Apply a scaling, ideally this would come from Dongdong Wang's User Guide, but I couldn't find it,
            # so I used dead reckoning from reading through https://doi.org/10.5194/essd-15-1419-2023
            par_data_scaled = np.array(par_masked * 0.1, dtype=np.float32)
            print(f"Successfully loaded PAR data with shape: {par_data.shape}")
            print(f"PAR range: {np.nanmin(par_data_scaled):.2f} to {np.nanmax(par_data_scaled):.2f} W/m²")
            
            return par_data_scaled
            
    except Exception as e:
        print(f"Error reading HDF-5 file: {e}")
        sys.exit(1)


def create_coordinate_arrays(metadata: dict[str, str | int]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Create latitude and longitude coordinate arrays based on metadata.
    
    Args:
        metadata (dict): Parsed filename metadata
        
    Returns:
        tuple: (longitude array, latitude array) both with shape (600,)
    """
    # Create coordinate arrays (600 pixels for 6 degrees = 0.01 degree resolution)
    lon = np.linspace(int(metadata["lon_min"]), int(metadata["lon_max"]), 600, endpoint=False)
    lat = np.linspace(int(metadata["lat_max"]), int(metadata["lat_min"]), 600, endpoint=False)
    
    return lon, lat


def calculate_sunrise_sunset(lat: float, lon: float, year: int, day_of_year: int) -> tuple[int, int]:
    """
    Calculate sunrise and sunset times in UTC for a given location and day.
    
    Args:
        lat (float): Latitude in decimal degrees
        lon (float): Longitude in decimal degrees
        year (int): Year associated with this date
        day_of_year (int): Day of year (1-366)
        
    Returns:
        tuple: (sunrise_hour_utc, sunset_hour_utc) rounded to nearest hour
               Returns None for polar day/night conditions
    """
    lat_rad = math.radians(lat)
    
    # Calculate solar declination angle
    declination = math.radians(23.45) * math.sin(math.radians(360 * (284 + day_of_year) / 365))

    # Calculate equation of time (in minutes)
    # source: https://en.wikipedia.org/wiki/Equation_of_time
    D = 6.24004077 + 0.01720197 * (365.25 * (year - 2000) + day_of_year)
    eot_mins = -7.659 * math.sin(D) + 9.863 * math.sin(2 * D + 3.5932)
    
    # Calculate hour angle at sunrise/sunset
    try:
        cos_hour_angle = -math.tan(lat_rad) * math.tan(declination)
        # If |cos(hour_angle)| > 1, the sun never rises or sets but this should
        # never happen with this data since it is all subarctic/antarctic

        # Convert hour angle to degrees for later
        hour_angle = math.degrees(math.acos(cos_hour_angle))
    except ValueError:
        return 0, 23
    
    # Calculate Solar Noon in UTC
    solar_noon_utc = 12.0 - (lon / 15.0) - (eot_mins / 60.0)
    sunrise_utc = solar_noon_utc - (hour_angle / 15.0)
    sunset_utc = solar_noon_utc + (hour_angle / 15.0)
    
    # Round to nearest hour
    sunrise_hour = round(sunrise_utc)
    sunset_hour = round(sunset_utc)
    
    return sunrise_hour, sunset_hour


def create_animated_plot(par_data: npt.NDArray[np.float32], metadata: dict[str, str | int], output_path: Path) -> Path:
    """
    Create animated plot showing PAR data for each hour.
    
    Args:
        par_data (np.ndarray): PAR data array with shape (24, 600, 600)
        metadata (dict): Parsed filename metadata
        output_path (Path): Output directory path
    
    Returns:
        Path: Path to output animation file
    """
    print("Creating animated PAR visualization...")
    
    # Create coordinate arrays
    lon, lat = create_coordinate_arrays(metadata)
    
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Calculate global min/max for consistent colorbar
    vmin = np.nanmin(par_data)
    vmax = np.nanmax(par_data)
    
    # Create meshgrid for plotting
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    # Initialize the plot
    im = ax.pcolormesh(lon_grid, lat_grid, par_data[int(metadata["sunrise_hr"])], 
                       vmin=float(vmin), vmax=float(vmax), cmap="plasma", shading="auto")
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label="PAR (W/m²)")
    
    # Set labels and title
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_aspect("equal")
    
    # Format the title with metadata
    title_base = (f"{metadata["satellite"]} {metadata["product"]} - "
                 f"{metadata["date"].strftime("%Y-%m-%d")} (Day {metadata["day_of_year"]})") # type: ignore
    title = ax.set_title(f"{title_base}\nHour: 00 UTC")
    
    def animate(frame):
        """Animation function for each frame (hour)."""
        real_frame = frame + int(metadata["sunrise_hr"]) + 1
        im.set_array(par_data[real_frame % 24].ravel())
        
        # Update title with current hour
        title.set_text(f"{title_base}\nHour: {(real_frame % 24):02d} UTC")
        
        return [im, title]
    
    # Create animation
    day_hours = int(metadata["sunset_hr"]) - int(metadata["sunrise_hr"])
    anim = animation.FuncAnimation(fig, animate, frames=day_hours, 
                                 interval=500, blit=False, repeat=True)
    
    # Save animation
    output_file = output_path / f"par_animation_{metadata['satellite']}_{metadata['date'].strftime('%Y%j')}_h{metadata['h_index']:02d}v{metadata['v_index']:02d}.gif" # type: ignore
    
    print(f"Saving animation to: {output_file}")
    anim.save(output_file, writer="pillow", fps=2)
    
    plt.tight_layout()
    plt.show()
    
    return output_file


def main() -> int:
    print("NASA GeoNEX PAR Data Visualization Script")
    print("=" * 50)

    args = parse_arguments()    
    input_path, output_path = validate_inputs(args.input, args.output)
    par_data = load_par_data(input_path)
    metadata = parse_filename(input_path.name)
    create_animated_plot(par_data, metadata, output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())