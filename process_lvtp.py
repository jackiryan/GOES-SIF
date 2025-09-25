from datetime import datetime
import glob
from numba import jit, prange
import numpy as np
import numpy.typing as npt
import os
from pyresample import create_area_def
from pyresample.kd_tree import resample_nearest
from pyresample.geometry import AreaDefinition
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import warnings
import xarray as xr

warnings.filterwarnings('ignore', category=RuntimeWarning)

# Constants for standard atmosphere
P0_SEA_LEVEL = 1013.25  # hPa
T0_SEA_LEVEL = 288.15   # K (15°C)
LAPSE_RATE = -0.0065    # K/m
g = 9.80665             # m/s²
R = 287.05              # J/(kg·K)


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


def load_dem(
        dem_path: str,
) -> tuple[npt.NDArray[np.float32], list[int]]:
    """
    Load DEM and extract elevation. Precompute a regular grid interpolator
    to speed up association of elevation with GOES lat/lons.
    
    Args:
        dem_path (str): Path to DEM GeoTIFF file
        
    Returns:
        elevation (np.ndarray): Gridded elevation values
        extent (list[int]): extent of DEM, also will be target extent of GOES.
    """
    with rasterio.open(dem_path) as src:
        elevation = src.read(1)
        bounds = src.bounds
        dem_extent = [int(bounds.left), int(bounds.bottom), int(bounds.right), int(bounds.top)]

    return (elevation, dem_extent)


def combine_grids(
        all_surface_temps: list[npt.NDArray[np.float32]],
        all_lons: list[npt.NDArray[np.float32]],
        all_lats: list[npt.NDArray[np.float32]]
) -> tuple[npt.NDArray[np.float32], np.ndarray, np.ndarray]:
    """
    Combine grids without regridding since they're already on the same grid.
    
    Args:
        all_surface_temps: List of 2D temperature arrays
        all_lons: List of 2D longitude arrays (should all be identical)
        all_lats: List of 2D latitude arrays (should all be identical)
        method: How to combine overlapping values ("mean", "first", "last")
    
    Returns:
        combined_grid: 2D temperature array
        lons_1d: 1D longitude array  
        lats_1d: 1D latitude array
    """
    if not all_surface_temps:
        raise ValueError("No data to combine")
    
    # All grids should have the same shape and coordinates
    ref_shape = all_surface_temps[0].shape
    
    # Initialize output arrays
    combined_grid = np.zeros(ref_shape, dtype=np.float32)
    count_grid = np.zeros(ref_shape, dtype=np.int32)
    
    # Sum all valid values and count them
    for temp_grid in all_surface_temps:
        valid_mask = ~np.isnan(temp_grid)
        combined_grid[valid_mask] += temp_grid[valid_mask]
        count_grid[valid_mask] += 1
    
    # Calculate mean where we have data
    valid_count = count_grid > 0
    combined_grid[valid_count] /= count_grid[valid_count]
    combined_grid[~valid_count] = np.nan
    
    # Extract 1D coordinate arrays from 2D grids
    # Assuming regular grid, take first row for lats and first column for lons
    lons_1d = all_lons[0][0, :]  # First row (constant latitude)
    lats_1d = all_lats[0][:, 0]  # First column (constant longitude)
    
    return np.flipud(combined_grid), lons_1d, lats_1d


def resample_goes(
        ds: xr.Dataset,
        target_extent: list[int],
        radius: int = 10000
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    lvt_data = ds["LVT"].values  # Shape: (y, x, pressure)

    proj_info = ds["goes_imager_projection"]
    h = proj_info.perspective_point_height
    lon_origin = proj_info.longitude_of_projection_origin
    sweep = proj_info.sweep_angle_axis

    x = ds.x.data * h
    y = ds.y.data * h

    goes_area = AreaDefinition(
        "goes_conus",
        "GOES East CONUS",
        "goes_conus",
        {
            "proj": "geos",
            "lon_0": lon_origin,
            "h": h,
            "sweep": sweep,
            "ellps": "GRS80"
        },
        len(x),
        len(y),
        [x.min(), y.min(), x.max(), y.max()]
    )

    target_area = create_area_def(
        "equirectangular",
        {
            "proj": "eqc",
            "ellps": "WGS84"
        },
        area_extent=target_extent,
        resolution=0.01,
        units="degrees"
    )

    resampled_nn = resample_nearest(
        goes_area,
        lvt_data,
        target_area,
        radius_of_influence=radius,
        fill_value=np.nan, # type: ignore
        epsilon=0.5 # type: ignore
    )
    resampled_data = np.flipud(np.array(resampled_nn, dtype=np.float32))

    lons, lats = target_area.get_lonlats() # type: ignore
    out_lons = np.array(lons, dtype=np.float32)
    out_lats = np.array(lats, dtype=np.float32)
    return (resampled_data, out_lons, out_lats)


@jit(nopython=True, parallel=True)
def vectorized_log_interpolation(
    temp_profiles: npt.NDArray[np.float32],  # Shape: (n_points, n_levels)
    log_pressure_levels: npt.NDArray[np.float32],  # Shape: (n_levels,)
    log_surface_pressures: npt.NDArray[np.float32],  # Shape: (n_points,)
    valid_mask: npt.NDArray[np.bool_],  # Shape: (n_points, n_levels)
) -> npt.NDArray[np.float32]:
    """
    Vectorized interpolation using Numba for massive speedup.
    Performs linear interpolation in log-pressure space.
    """
    n_points = temp_profiles.shape[0]
    surface_temps = np.full(n_points, np.nan, dtype=np.float32)
    
    for idx in prange(n_points):
        if not np.any(valid_mask[idx]):
            continue
            
        # Get valid data for this point
        valid_temps = temp_profiles[idx, valid_mask[idx]]
        valid_log_p = log_pressure_levels[valid_mask[idx]]
            
        # Sort by pressure (descending)
        sort_indices = np.argsort(valid_log_p)[::-1]
        valid_temps = valid_temps[sort_indices]
        valid_log_p = valid_log_p[sort_indices]
        
        log_surf_p = log_surface_pressures[idx]
        
        # Linear interpolation/extrapolation
        if log_surf_p >= valid_log_p[0]:  # Above highest pressure
            # Extrapolate from first two points
            if len(valid_temps) >= 2:
                slope = (valid_temps[1] - valid_temps[0]) / (valid_log_p[1] - valid_log_p[0])
                surface_temps[idx] = valid_temps[0] + slope * (log_surf_p - valid_log_p[0])
            else:
                surface_temps[idx] = valid_temps[0]
        elif log_surf_p <= valid_log_p[-1]:  # Below lowest pressure
            # Extrapolate from last two points
            if len(valid_temps) >= 2:
                slope = (valid_temps[-1] - valid_temps[-2]) / (valid_log_p[-1] - valid_log_p[-2])
                surface_temps[idx] = valid_temps[-1] + slope * (log_surf_p - valid_log_p[-1])
            else:
                surface_temps[idx] = valid_temps[-1]
        else:
            # Interpolate between points
            for i in range(len(valid_log_p) - 1):
                if valid_log_p[i] >= log_surf_p >= valid_log_p[i + 1]:
                    # Linear interpolation
                    t = (log_surf_p - valid_log_p[i]) / (valid_log_p[i + 1] - valid_log_p[i])
                    surface_temps[idx] = valid_temps[i] + t * (valid_temps[i + 1] - valid_temps[i])
                    break
    
    return surface_temps


def interpolate_temperature_batch(
    lvt_data: npt.NDArray[np.float32],  # Shape: (y, x, pressure)
    pressure_levels: npt.NDArray[np.float32],
    surface_pressure: npt.NDArray[np.float32],  # Shape: (y, x)
    valid_indices: tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]
) -> npt.NDArray[np.float32]:
    """
    Optimized batch interpolation of temperature profiles to surface.
    
    Returns:
        surface_temp: 2D array of surface temperatures
    """
    surface_temp = np.full_like(surface_pressure, np.nan)
    
    if len(valid_indices[0]) == 0:
        return surface_temp
    
    # Extract data for valid points only
    temp_profiles = lvt_data[valid_indices[0], valid_indices[1], :]  # Shape: (n_points, n_levels)
    j_indices, i_indices = valid_indices
    surf_pressure_j = 3000 - j_indices - 1
    surf_pressures = surface_pressure[surf_pressure_j, i_indices] # Shape: (n_points,)
    
    # Precompute log pressures
    log_pressure_levels = np.log(pressure_levels.astype(np.float32))
    log_surf_pressures = np.log(surf_pressures.astype(np.float32))
    
    # Create mask for valid temperature data
    valid_mask = ~np.isnan(temp_profiles)
    
    # Use vectorized Numba function
    surface_temps_1d = vectorized_log_interpolation(
        temp_profiles, log_pressure_levels, log_surf_pressures, valid_mask
    )
    
    # Put results back into 2D array
    surface_temp[valid_indices[0], valid_indices[1]] = surface_temps_1d
    
    return surface_temp


def process_lvtp_files(
    file_paths: list[str],
    dem_path: str,
    output_path: str
) -> None:
    """
    Process multiple GOES LVTP files to create surface temperature GeoTIFF.
    
    Args:
        file_paths (list[str]): List of GOES LVTP NetCDF file paths
        dem_path (str): Path to DEM GeoTIFF
        output_path (str): Output GeoTIFF path
    """
    print(f"Processing {len(file_paths)} GOES LVTP files...")
    
    # Storage for all surface temperature grids
    all_surface_temps = []
    all_lons = []
    all_lats = []

    # Load DEM and keep it in memory
    elevation, extent = load_dem(dem_path)
    surface_pressure = calculate_surface_pressure(elevation)

    for i, file_path in enumerate(file_paths):
        print(f"Processing file {i+1}/{len(file_paths)}: {os.path.basename(file_path)}")
        
        # Load GOES data
        ds = xr.open_dataset(file_path)
        # Get temperature data and pressure levels
        pressure_levels = ds["pressure"].values  # Shape: (pressure,)
        lvt_resamp, goes_lon, goes_lat = resample_goes(ds, extent)
        ds.close()

        valid_mask = ~np.isnan(lvt_resamp[:, :, 0]) & ~np.isnan(surface_pressure)
        valid_indices = np.where(valid_mask)

        surface_temp = interpolate_temperature_batch(
            lvt_resamp,
            pressure_levels,
            surface_pressure,
            valid_indices # type: ignore
        )

        # Store results
        all_surface_temps.append(surface_temp)
        all_lons.append(goes_lon)
        all_lats.append(goes_lat)
    

    print("Combining all data...")
    surface_temp_grid, output_lons, output_lats = combine_grids(
        all_surface_temps, all_lons, all_lats
    )
    
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
        output_path
    )


if __name__ == "__main__":
    main()






