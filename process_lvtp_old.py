from datetime import datetime
import glob
import numpy as np
import numpy.typing as npt
import os
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from scipy.interpolate import griddata, interp1d
from tqdm import tqdm
import xarray as xr


# Constants for standard atmosphere
P0_SEA_LEVEL = 1013.25  # hPa
T0_SEA_LEVEL = 288.15   # K (15°C)
LAPSE_RATE = -0.0065    # K/m
g = 9.80665             # m/s²
R = 287.05              # J/(kg·K)


def calculate_latlon_grid(goes_ds: xr.Dataset) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Convert GOES-ABI fixed grid projection x/y coordinates to lat/lon coordinates.
    This method is derived from the code example here created by the NOAA/NESDIS/STAR Science Team: 
    https://www.star.nesdis.noaa.gov/atmospheric-composition-training/python_abi_lat_lon.php

    See also this figure: https://www.star.nesdis.noaa.gov/atmospheric-composition-training/images/satellite_data_graphics/ABI_Fixed_Grid_Coordinate_Frames.png
    """
    # Read GOES ABI fixed grid projection variables and constants
    x_coords = goes_ds.x.data
    y_coords = goes_ds.y.data

    proj_info = goes_ds["goes_imager_projection"]
    lon_origin = proj_info.longitude_of_projection_origin
    H = proj_info.perspective_point_height + proj_info.semi_major_axis
    r_eq = proj_info.semi_major_axis
    r_pol = proj_info.semi_minor_axis

    # Create 2D coordinate matrices based off vectors from granule
    x_coords_2d, y_coords_2d = np.meshgrid(x_coords, y_coords)

    # Equations to calculate latitude and longitude
    lambda_0 = (lon_origin * np.pi) / 180.0
    # a = sin²(x) + cos²(x)cos²(y) + (r_eq² / r_pol²)sin²(y)
    a_var = \
        np.power(np.sin(x_coords_2d), 2.0) \
        + (np.power(np.cos(x_coords_2d), 2.0) * (np.power(np.cos(y_coords_2d), 2.0) \
        + (((r_eq * r_eq) / (r_pol * r_pol)) * np.power(np.sin(y_coords_2d), 2.0))))
    # b = -2Hcos(x)cos(y)
    b_var = -2.0 * H * np.cos(x_coords_2d) * np.cos(y_coords_2d)
    # c = H² - r_eq²
    c_var = (H ** 2.0) - (r_eq ** 2.0)
    # r_s = (-b - √(b² - 4ac)) / 2a, negative solution to quadratic equation
    r_s = (-b_var - np.sqrt((b_var ** 2) - (4.0 * a_var * c_var))) / (2.0 * a_var)
    s_x = r_s * np.cos(x_coords_2d) * np.cos(y_coords_2d)
    s_y = -r_s * np.sin(x_coords_2d)
    s_z = r_s * np.cos(x_coords_2d) * np.sin(y_coords_2d)

    # Ignore numpy errors for sqrt of negative number; occurs for GOES-16 ABI CONUS sector data
    np.seterr(all="ignore")

    abi_lat = (180.0/np.pi)*(np.arctan(((r_eq*r_eq)/(r_pol*r_pol))*((s_z/np.sqrt(((H-s_x)*(H-s_x))+(s_y*s_y))))))
    abi_lon = (lambda_0 - np.arctan(s_y/(H-s_x)))*(180.0/np.pi)

    return abi_lat, abi_lon


def calculate_surface_pressure(elevation: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """
    Calculate surface pressure from elevation using barometric formula.
    
    Args:
        elevation (np.ndarray): Elevation in meters
        
    Returns:
        pressure (np.ndarray): Surface pressure in hPa
    """
    # Barometric formula: P = P0 * (1 + (L*h)/T0)^(-g*M/(R*L))
    # Simplified for troposphere with constant lapse rate
    pressure = P0_SEA_LEVEL * (1 + (LAPSE_RATE * elevation) / T0_SEA_LEVEL) ** (-g / (R * LAPSE_RATE))
    return pressure


def interpolate_temperature_to_surface(
    temperature_profile: npt.NDArray[np.float32],
    pressure_levels: npt.NDArray[np.float32],
    surface_pressure: float,
    extrapolate: bool = True
) -> float:
    """
    Interpolate temperature profile to surface pressure.
    
    Args:
        temperature_profile (np.ndarray): Temperature values at each pressure level (K or °C)
        pressure_levels (np.ndarray): Pressure levels (hPa)
        surface_pressure (float): Target surface pressure (hPa)
        extrapolate (bool): Whether to extrapolate if surface pressure is outside range
        
    Returns:
        surface_temp (float): Interpolated surface temperature
    """
    # Remove NaN values
    valid_mask = ~np.isnan(temperature_profile)
    if not np.any(valid_mask):
        return np.nan
    
    valid_temps = temperature_profile[valid_mask]
    valid_pressures = pressure_levels[valid_mask]
    
    # Sort by pressure (descending order for stability)
    sort_idx = np.argsort(valid_pressures)[::-1]
    valid_temps = valid_temps[sort_idx]
    valid_pressures = valid_pressures[sort_idx]
    
    # Use log-pressure interpolation for better accuracy
    log_pressures = np.log(valid_pressures)
    log_surface_pressure = np.log(surface_pressure)
    
    if extrapolate:
        # Allow extrapolation with "extrapolate" mode
        interp_func = interp1d(
            log_pressures, valid_temps,
            kind="linear", fill_value="extrapolate" # type: ignore
        )
    else:
        # Clip to bounds
        interp_func = interp1d(
            log_pressures, valid_temps,
            kind="linear", bounds_error=False,
            fill_value=(valid_temps[0], valid_temps[-1]) # type: ignore
        )
    
    surface_temp = interp_func(log_surface_pressure)
    return surface_temp


def load_dem(
        dem_path: str,
) -> np.ndarray:
    """
    Load DEM and extract elevation. Since we know the bounds, resolution and CRS of this
    file a priori, we don't need to faff about with subsetting and metadata.
    
    Args:
        dem_path (str): Path to DEM GeoTIFF file
        
    Returns:
        elevation (np.ndarray): Elevation grid
    """
    with rasterio.open(dem_path) as src:
        elevation = src.read(1)
        
    return elevation


def process_lvtp_files(
    file_paths: list[str],
    dem_path: str,
    output_path: str,
    quality_threshold: float = 0.0
) -> None:
    """
    Process multiple GOES LVTP files to create surface temperature GeoTIFF.
    
    Args:
        file_paths (list[str]): List of GOES LVTP NetCDF file paths
        dem_path (str): Path to DEM GeoTIFF
        output_path (str): Output GeoTIFF path
        quality_threshold (float): DQF threshold (0=best quality)
    """
    print(f"Processing {len(file_paths)} GOES LVTP files...")
    
    # Storage for all surface temperature grids
    all_surface_temps = []
    all_lons = []
    all_lats = []

    # Load DEM and keep it in memory
    print("Loading DEM...")
    dem_elev = load_dem(dem_path)
    dem_lon_vals = -135 + np.arange(8500) * 0.01
    dem_lat_vals = 50 - np.arange(3000) * 0.01
    dem_lon, dem_lat = np.meshgrid(dem_lon_vals, dem_lat_vals)

    file_paths = [file_paths[0]]

    for i, file_path in enumerate(file_paths):
        print(f"Processing file {i+1}/{len(file_paths)}: {os.path.basename(file_path)}")
        
        # Load GOES data
        ds = xr.open_dataset(file_path)
        
        # Get temperature data and pressure levels
        lvt_data = ds["LVT"].values  # Shape: (y, x, pressure)
        pressure_levels = ds["pressure"].values  # Shape: (pressure,)
        
        # Get quality flags
        dqf_overall = ds["DQF_Overall"].values
        
        # Calculate lat/lon for GOES grid
        goes_lat, goes_lon = calculate_latlon_grid(ds)
        
        # Get valid data bounds
        valid_mask = ~np.isnan(goes_lat) & ~np.isnan(goes_lon)
        
        # Apply quality flag filter
        if quality_threshold is not None:
            quality_mask = dqf_overall <= quality_threshold
            valid_mask = valid_mask & quality_mask
        
        # Get extent for DEM loading
        valid_lats = goes_lat[valid_mask]
        valid_lons = goes_lon[valid_mask]
        
        if len(valid_lats) == 0:
            print(f"  No valid data in file, skipping...")
            continue
        
        # Interpolate DEM to GOES grid points
        dem_points = np.column_stack([dem_lon.ravel(), dem_lat.ravel()])
        goes_points = np.column_stack([goes_lon.ravel(), goes_lat.ravel()])
        
        print("griddata")
        elevation_at_goes = griddata(
            dem_points,
            dem_elev.ravel(),
            goes_points,
            method="linear"
        ).reshape(goes_lat.shape)
        
        # Calculate surface pressure at each GOES pixel
        print("Calculating surface pressure...")
        surface_pressure = calculate_surface_pressure(elevation_at_goes)
        
        # Initialize surface temperature array
        surface_temp = np.full_like(goes_lat, np.nan)
        
        # Interpolate temperature to surface for each valid pixel
        ny, nx = goes_lat.shape
        print("interpolating temperature to surface...")
        for j in tqdm(range(ny)):
            for i in range(nx):
                if valid_mask[j, i] and not np.isnan(surface_pressure[j, i]):
                    temp_profile = lvt_data[j, i, :]
                    
                    # Check if we have valid temperature data
                    if not np.all(np.isnan(temp_profile)):
                        surface_temp[j, i] = interpolate_temperature_to_surface(
                            temp_profile,
                            pressure_levels,
                            surface_pressure[j, i],
                            extrapolate=True
                        )
        
        # Store results
        all_surface_temps.append(surface_temp)
        all_lons.append(goes_lon)
        all_lats.append(goes_lat)
        
        ds.close()
    
    print("Combining and regridding all data...")
    
    # Combine all data
    combined_temps = []
    combined_lons = []
    combined_lats = []
    
    for temps, lons, lats in zip(all_surface_temps, all_lons, all_lats):
        valid = ~np.isnan(temps)
        combined_temps.extend(temps[valid])
        combined_lons.extend(lons[valid])
        combined_lats.extend(lats[valid])
    
    combined_temps = np.array(combined_temps)
    combined_lons = np.array(combined_lons)
    combined_lats = np.array(combined_lats)
    
    # Define output grid (0.01 degree resolution)
    output_res = 0.01
    min_lon, max_lon = np.min(combined_lons), np.max(combined_lons)
    min_lat, max_lat = np.min(combined_lats), np.max(combined_lats)
    
    output_lons = np.arange(min_lon, max_lon + output_res, output_res)
    output_lats = np.arange(min_lat, max_lat + output_res, output_res)
    output_lon_grid, output_lat_grid = np.meshgrid(output_lons, output_lats)
    
    # Interpolate to regular grid
    points = np.column_stack([combined_lons, combined_lats])
    output_points = np.column_stack([output_lon_grid.ravel(), output_lat_grid.ravel()])
    
    surface_temp_grid = griddata(
        points,
        combined_temps,
        output_points,
        method="linear"
    ).reshape(output_lon_grid.shape)
    
    # Write to GeoTIFF
    write_geotiff(
        surface_temp_grid,
        output_lons,
        output_lats,
        output_path
    )
    
    print(f"Surface temperature GeoTIFF saved to: {output_path}")


def write_geotiff(
    data: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    output_path: str,
    crs: str = "EPSG:4326"
) -> None:
    """
    Write data to GeoTIFF file.
    
    Args:
        data (np.ndarray): 2D data array
        lons (np.ndarray): 1D longitude coordinates
        lats (np.ndarray): 1D latitude coordinates  
        output_path (str): Output file path
        crs (str): Coordinate reference system
    """
    # Calculate transform
    lon_res = lons[1] - lons[0]
    lat_res = lats[1] - lats[0]
    
    transform = from_bounds(
        lons.min() - lon_res/2,
        lats.min() - lat_res/2,
        lons.max() + lon_res/2,
        lats.max() + lat_res/2,
        len(lons),
        len(lats)
    )
    
    # Write GeoTIFF
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=CRS.from_string(crs),
        transform=transform,
        compress="deflate"
    ) as dst:
        dst.write(data, 1)
        
        # Set NoData value
        dst.nodata = np.nan


def main():
    """
    Main processing function.
    """
    # Configuration
    input_dir = "LVTPC/2020/163/17/"
    dem_path = "conus_gmted_01deg.tif"
    output_dir = "Tair/"
    
    # Example: process one hour of data
    target_datetime = datetime(2020, 6, 11, 17)  # Example from your data
    
    # Find all LVTP files for this hour
    pattern = f"*s{target_datetime.strftime('%Y%j%H')}*.nc"
    file_paths = sorted(glob.glob(f"{input_dir}/{pattern}"))
    
    if len(file_paths) == 0:
        print(f"No files found matching pattern: {pattern}")
        return
    
    print(f"Found {len(file_paths)} files for {target_datetime}")
    
    # Output filename
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"surface_temp_{target_datetime.strftime('%Y%m%d_%H')}00.tif")
    
    # Process files
    process_lvtp_files(
        file_paths,
        dem_path,
        output_path,
        quality_threshold=0.0  # Use only best quality data
    )


if __name__ == "__main__":
    main()

