from bs4 import BeautifulSoup
import calendar
from datetime import datetime, timedelta, timezone
from glob import glob
import numpy as np
import numpy.typing as npt
import os.path
import pandas as pd
from pathlib import Path
import re
import requests
import sys
from tqdm import tqdm
from urllib.parse import urljoin
import xarray as xr
"""
OCO3_VARIABLES: dict[str, str] = {
    "latitude": "Geolocation/latitude",
    "longitude": "Geolocation/longitude",
    "hour": "Geolocation/hour_of_day",
    "sif740": "Science/SIF_740nm",
    "sif757": "Science/SIF_757nm",
    "tair": "Science/temperature_two_meter",
    "vpd": "Science/vapor_pressure_deficit",
    "saz": "Science/SAz",
    "sza": "Science/SZA",
    "vaz": "Science/VAz",
    "vza": "Science/VZA"
}
"""
OCO3_VARIABLES: dict[str, str] = {
    "latitude": "Geolocation/latitude",
    "longitude": "Geolocation/longitude",
    "hour": "Geolocation/hour_of_day",
    "sif740": "Science/sif740",
    "sif757": "Science/sif757",
    "tair": "Science/tskin",
    "vpd": "Science/vpd",
    "saz": "Science/saa",
    "sza": "Science/sza",
    "vaz": "Science/vaa",
    "vza": "Science/vza"
}


def solar_zenith_angle(
        date_array: npt.NDArray[np.datetime64],
        latitude: npt.NDArray[np.float32],
        longitude:npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """
    Calculate solar zenith angle empirically for BRDF inversion.
    """
    
    # Convert datetime64 to pandas datetime for easier manipulation, then extract components
    # Convert to pandas datetime to easily extract components
    dt_pandas = pd.to_datetime(date_array)
    
    # Calculate day of year (1-365/366)
    day_of_year = dt_pandas.dayofyear.values
    
    # Calculate fractional hour
    fractional_hour = dt_pandas.hour.values + dt_pandas.minute.values/60.0 + dt_pandas.second.values/3600.0
    
    # Convert latitude and longitude to radians
    lat_rad = np.radians(latitude.astype(np.float64))
    lon_rad = np.radians(longitude.astype(np.float64))
    
    # Calculate solar declination angle (in radians)
    # Using Cooper's equation: δ = 23.45° * sin(360° * (284 + n) / 365)
    declination_deg = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365.25))
    declination_rad = np.radians(declination_deg)
    
    # Calculate equation of time (simplified approximation in minutes)
    B = np.radians(360 * (day_of_year - 81) / 365.25)
    equation_of_time = 9.87 * np.sin(2*B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)
    
    # Calculate solar time
    time_correction = equation_of_time + 4 * np.degrees(lon_rad)  # in minutes
    solar_time = fractional_hour + time_correction / 60.0
    
    # Calculate hour angle (in radians)
    # Hour angle = 15° * (solar_time - 12)
    hour_angle_deg = 15 * (solar_time - 12)
    hour_angle_rad = np.radians(hour_angle_deg)
    
    # Calculate solar zenith angle using the fundamental formula:
    # cos(θz) = sin(φ)sin(δ) + cos(φ)cos(δ)cos(h)
    cos_zenith = (np.sin(lat_rad) * np.sin(declination_rad) + 
                  np.cos(lat_rad) * np.cos(declination_rad) * np.cos(hour_angle_rad))
    
    # Ensure cos_zenith is within valid range [-1, 1] to avoid math domain errors
    cos_zenith = np.clip(cos_zenith, -1, 1)
    
    # Calculate zenith angle in radians
    zenith_angle = np.arccos(cos_zenith)
    
    return np.rad2deg(zenith_angle)


def generate_dates(start_date: datetime, end_date: datetime) -> list[datetime]:
    """
    Generate a list of dates from start_date to end_date (inclusive).

    Arguments:
        start_date (datetime): The beginning date.
        end_date (datetime): The ending date.

    Returns:
        List[datetime]: List of dates between start_date and end_date.
    """
    dates: list[datetime] = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date += timedelta(days=1)
    return dates


def get_times(hours: npt.NDArray[np.float32], data_day: datetime) -> npt.NDArray[np.datetime64]:
    base_date = np.datetime64(data_day.strftime("%Y-%m-%d"))
    datetime_array = (base_date + hours.astype("timedelta64[h]") + np.timedelta64(30, "m"))
    return datetime_array


def open_oco3(oco3_granule: str, engine: str = "netcdf4") -> xr.DataTree:
    """
    Open an OCO-3 granule, usually one day, as a datatree since variables
    are only represented in groups.
    """
    dt = xr.open_datatree(
        oco3_granule,
        decode_timedelta=True,
        engine=engine
    )
    return dt


def vectorize_dt(dt: xr.DataTree, data_day: datetime, compute_sza: bool = False) -> npt.NDArray[np.float32]:
    hr = np.asarray(dt[OCO3_VARIABLES["hour"]].data)
    lat = np.asarray(dt[OCO3_VARIABLES["latitude"]].data)
    lon = np.asarray(dt[OCO3_VARIABLES["longitude"]].data)
    sif740 = np.asarray(dt[OCO3_VARIABLES["sif740"]].data)
    sza = np.asarray(dt[OCO3_VARIABLES["sza"]].data)

    expected_shape = (len(hr), len(lon), len(lat))
    if sif740.shape != expected_shape:
        raise ValueError(f"SIF array shape {sif740.shape} doesn't match expected {expected_shape}")
    
    valid_sif = np.where(~np.isnan(sif740))
    hr_ndx, lon_ndx, lat_ndx = valid_sif

    if compute_sza:
        data_times = get_times(hr[hr_ndx], data_day)
        computed_sza = solar_zenith_angle(data_times, lat[lat_ndx], lon[lon_ndx])
        vectorized_data = np.vstack([
            len(hr_ndx) * [float(data_day.strftime("%Y%m%d"))],
            hr[hr_ndx],
            lon[lon_ndx],
            lat[lat_ndx],
            sif740[hr_ndx, lon_ndx, lat_ndx],
            sza[hr_ndx, lon_ndx, lat_ndx],
            computed_sza
        ])
    else:
        vectorized_data = np.vstack([
            len(hr_ndx) * [float(data_day.strftime("%Y%m%d"))],
            hr[hr_ndx],
            lon[lon_ndx],
            lat[lat_ndx],
            sif740[hr_ndx, lon_ndx, lat_ndx],
            sza[hr_ndx, lon_ndx, lat_ndx]
        ])
    return vectorized_data


def process_1day_matrix(
        data_day: datetime,
        data_dir: str,
        engine: str,
        is_1deg: bool = True,
        compute_sza: bool = False
) -> npt.NDArray[np.float32]:
    if compute_sza:
        ndim = 7
    else:
        ndim = 6
    year = data_day.year
    month = f"{int(data_day.month):02d}"
    day = f"{int(data_day.day):02d}"
    if is_1deg:
        granule_wildcard = os.path.join(data_dir, f"{year}/{month}/oco3_goessif_{year}{month}{day}_ndgl_*.nc4")
    else:
        granule_wildcard = os.path.join(data_dir, f"oco3_geossif_0p01d_{year}{month}{day}_ndgl_*.nc4")
    results = glob(granule_wildcard)
    try:
        # Use the first result, otherwise throw IndexError if file not found
        oco3_granule = results[0]
        dt = open_oco3(oco3_granule, engine)
        vectorized_day = vectorize_dt(dt, data_day, compute_sza)
        return vectorized_day
    except IndexError:
        print(f"Warning: no granule found for {data_day.strftime('%Y-%m-%d')}. Skipping...")
        return np.empty((ndim, 0), dtype=np.float32)
    except Exception as e:
        if "dt" in locals().keys():
            dt.close() # type: ignore
        print(f"Unexpected {type(e)} when processing granule for {data_day.strftime('%Y-%m-%d')}: {e}. Skipping...")
        return np.empty((ndim, 0), dtype=np.float32)


def latlon_to_geonex_grid(
        lats: npt.NDArray[np.float32],
        lons: npt.NDArray[np.float32]
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    h_ndx = np.floor((lons + 180) / 6).astype(int)
    v_ndx = np.floor((60 - lats) / 6).astype(int)
    # In practice this isn't necessary since data is only over the CONUS
    h_ndx = np.clip(h_ndx, 0, 59)
    v_ndx = np.clip(v_ndx, 0, 19)
    return (h_ndx, v_ndx)


def find_geonex_tiles(vectorized_data: npt.NDArray[np.float32], date: datetime) -> list[tuple[str, datetime]]:
    lons = vectorized_data[2, :]
    lats = vectorized_data[3, :]
    h_ndx, v_ndx = latlon_to_geonex_grid(lats, lons)
    tiles = [f"h{h_val:02d}v{v_val:02d}" for h_val, v_val in zip(h_ndx, v_ndx)]
    uniq_tiles = list(set(tiles))
    return [(tile, date) for tile in uniq_tiles]


def sample_igbp(df: pd.DataFrame) -> None:
    df["IGBP_Primary"] = np.nan
    df["IGBP_Secondary"] = np.nan

    igbp_file = "conus/modis/MCD12Q1.061_500m_aid0001_2020_CONUS_0p01deg.nc4"
    # igbp_file = "conus/viirs/VIIRS-AST-IGBP17-GEO_v1r0_2020_CONUS_0p01deg.nc4"
    dt = open_oco3(igbp_file)
    surface_type = np.asarray(dt["IGBP17/surface_type"].data)
    ul_lat = 20
    ul_lon = -135
    lon_coords = np.floor((df["longitude"].values - ul_lon) / 0.01).astype(int)
    lat_coords = np.floor((df["latitude"].values - ul_lat) / 0.01).astype(int)
    valid_mask = ((lon_coords >= 0) & (lon_coords < surface_type.shape[0]) & 
                  (lat_coords >= 0) & (lat_coords < surface_type.shape[1]))
    valid_indices = np.where(valid_mask)[0]
    primary_class = surface_type[lon_coords, lat_coords, 0]
    secondary_class = surface_type[lon_coords, lat_coords, 1]

    df.loc[valid_indices, "IGBP_Primary"] = primary_class
    secondary_mask = secondary_class != 0
    df.loc[valid_indices[secondary_mask], "IGBP_Secondary"] = secondary_class[secondary_mask]
    df.loc[valid_indices[~secondary_mask], "IGBP_Secondary"] = primary_class[~secondary_mask]



def _setup_session_and_output(output_dir: str, session: requests.Session | None = None) -> tuple[requests.Session, Path]:
    """Set up the requests session and output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if session is None:
        session = requests.Session()
        # Maybe this is needed to prevent bot detection
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
    
    return session, output_path


def _group_tiles_by_index(tile_list: list[tuple[str, datetime]]) -> dict[str, list[datetime]]:
    """Group dates by tile_index for efficient processing."""
    tile_date_map = {}
    for tile_index, date in tile_list:
        if tile_index not in tile_date_map:
            tile_date_map[tile_index] = []
        tile_date_map[tile_index].append(date)
    return tile_date_map


def _fetch_tile_directory_listing(
        year_dir_url: str,
        session: requests.Session
) -> list[tuple[str, str]]:
    """Fetch and parse directory listing for a GeoNEX URL, returning (href, filename) pairs."""
    response = session.get(year_dir_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Find all links that could be HDF files
    links = soup.find_all("a", href=True)
    
    # Extract href and filename pairs for HDF files
    hdf_links = []
    for link in links:
        href = link["href"]
        if href.endswith(".hdf") or href.endswith(".h5"):
            # Extract just the filename from the href path
            filename = href.split("/")[-1]
            hdf_links.append((href, filename))
    
    return hdf_links


def _find_matching_file(tile_index: str, date: datetime, hdf_links: list[tuple[str, str]]) -> tuple[str, str] | None:
    """Find the matching file for a given tile and date from cached directory listing."""
    year = date.year
    day_of_year = date.timetuple().tm_yday
    
    # Create regex pattern to match the expected filename format
    # GO16_ABI12C_{year}{doy:03d}\d{4}_GLBG_{tile_index}_\d{2}.hdf
    pattern = rf"GO16_ABI12C_{year}{day_of_year:03d}\d{{4}}_GLBG_{re.escape(tile_index)}_\d{{2}}\.hdf"
    
    matching_links = [(href, filename) for href, filename in hdf_links 
                     if re.match(pattern, filename)]
    
    if not matching_links:
        return None
    
    if len(matching_links) > 1:
        print(f"INFO: Multiple matching files found for tile {tile_index}, using first match: {matching_links[0][1]}")
    
    return matching_links[0]


def _construct_file_url(href: str, year_dir_url: str) -> str:
    """Construct the full file URL from href and base directory URL."""
    if href.startswith("/"):
        # Absolute path from root of the server
        return urljoin("https://data.nas.nasa.gov", href)
    else:
        # Relative path
        return urljoin(year_dir_url, href)


def _download_file(file_url: str, output_file_path: Path, session: requests.Session) -> bool:
    """Download a single file with progress tracking. Returns True if successful."""
    try:
        print(f"Downloading: {output_file_path.name}")
        print(f"URL: {file_url}")
        
        with session.get(file_url, stream=True) as file_response:
            file_response.raise_for_status()
            
            total_size = int(file_response.headers.get("content-length", 0))
            
            with open(output_file_path, "wb") as f:
                downloaded_size = 0
                for chunk in file_response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            print(f"\rProgress: {progress:.1f}%", end="", flush=True)
            
            print(f"\nSuccessfully downloaded: {output_file_path}")
            return True
            
    except Exception as e:
        print(f"\nERROR: Failed to download {output_file_path.name}: {e}")
        # Clean up partial download
        if output_file_path.exists():
            output_file_path.unlink()
        return False


def download_goes_brdfs(
        tile_list: list[tuple[str, datetime]], 
        output_dir: str = "./goes_data",
        session: requests.Session | None = None
) -> list[str]:
    """
    Download GOES-16 BRDF HDF files for specified tiles and dates.
    """
    BASE_URL = "https://data.nas.nasa.gov/geonex/GOES16/GEONEX-L2/MAIAC/"
    
    session, output_path = _setup_session_and_output(output_dir, session)
    tile_date_map = _group_tiles_by_index(tile_list)
    
    print(f"Processing {len(tile_date_map)} unique tiles with {len(tile_list)} total requests")
    
    downloaded_files = []
    
    # Process each unique tile_index
    for tile_index, dates in tile_date_map.items():
        try:
            print(f"\nProcessing tile {tile_index} with {len(dates)} dates")
            
            # Get the year from first date (assuming all dates are same year)
            year = dates[0].year
            year_dir_url = urljoin(BASE_URL, f"{tile_index}/{year}/")
            
            # Fetch directory listing once per tile
            hdf_links = _fetch_tile_directory_listing(year_dir_url, session)
            print(f"Found {len(hdf_links)} HDF files in directory")
            
            # Create tile-specific subdirectory
            tile_output_dir = output_path / tile_index
            tile_output_dir.mkdir(exist_ok=True)
            
            # Process each date for this tile using cached directory listing
            for date in dates:
                try:
                    day_of_year = date.timetuple().tm_yday
                    print(f"  Processing date {date.strftime('%Y-%m-%d')} (DOY {day_of_year})")
                    
                    # Find matching file
                    match_result = _find_matching_file(tile_index, date, hdf_links)
                    if not match_result:
                        print(f"    WARNING: No matching file found for {date.strftime('%Y-%m-%d')} (DOY {day_of_year:03d})")
                        continue
                    
                    href, filename = match_result
                    file_url = _construct_file_url(href, year_dir_url)
                    output_file_path = tile_output_dir / filename
                    
                    # Skip if file already exists
                    if output_file_path.exists():
                        print(f"    File already exists: {filename}")
                        downloaded_files.append(str(output_file_path))
                        continue
                    
                    if _download_file(file_url, output_file_path, session):
                        downloaded_files.append(str(output_file_path))
                        
                except Exception as e:
                    print(f"    ERROR: Failed to process date {date.strftime('%Y-%m-%d')}: {e}")
                    continue
                    
        except requests.RequestException as e:
            print(f"ERROR: Failed to fetch directory listing for tile {tile_index}: {e}")
            continue
        except Exception as e:
            print(f"ERROR: Unexpected error processing tile {tile_index}: {e}")
            continue
    
    print(f"\nDownload complete. Successfully downloaded {len(downloaded_files)} files.")
    return downloaded_files

def download_goes_par(
        tile_list: list[tuple[str, datetime]], 
        output_dir: str = "./goes_data",
        session: requests.Session | None = None
) -> list[str]:
    """
    Download GOES-16 PAR HDF files for specified tiles and dates.
    """
    BASE_URL = "https://data.nas.nasa.gov/geonex/GOES16/GEONEX-L2/DSR-PAR/"
    
    session, output_path = _setup_session_and_output(output_dir, session)
    tile_date_map = _group_tiles_by_index(tile_list)
    
    print(f"Processing {len(tile_date_map)} unique tiles with {len(tile_list)} total requests")
    
    downloaded_files = []
    
    # Process each unique tile_index
    for tile_index, dates in tile_date_map.items():
        try:
            print(f"\nProcessing tile {tile_index} with {len(dates)} dates")
            
            # Get the year from first date (assuming all dates are same year)
            year = dates[0].year
            year_dir_url = urljoin(BASE_URL, f"{year}/hourly/{tile_index}/")
            
            # Fetch directory listing once per tile
            hdf_links = _fetch_tile_directory_listing(year_dir_url, session)
            print(f"Found {len(hdf_links)} H5 files in directory")
            
            # Create tile-specific subdirectory
            tile_output_dir = output_path / tile_index
            tile_output_dir.mkdir(exist_ok=True)
            
            # Process each date for this tile using cached directory listing
            for date in dates:
                try:
                    day_of_year = date.timetuple().tm_yday
                    print(f"  Processing date {date.strftime('%Y-%m-%d')} (DOY {day_of_year})")
                    
                    # Find matching file
                    year = date.year
                    day_of_year = date.timetuple().tm_yday
                    
                    # Create regex pattern to match the expected filename format
                    # G016_DSR_{year}{doy:03d}_{tile_index}.h5 (note the 0!!!! Arghhhhh)
                    pattern = rf"G016_DSR_{year}{day_of_year:03d}_{re.escape(tile_index)}\.h5"
                    
                    matching_links = [(href, filename) for href, filename in hdf_links 
                                    if re.match(pattern, filename)]
                    
                    if not matching_links:
                        print(f"    WARNING: No matching file found for {date.strftime('%Y-%m-%d')} (DOY {day_of_year:03d})")
                        continue
                    
                    if len(matching_links) > 1:
                        print(f"INFO: Multiple matching files found for tile {tile_index}, using first match: {matching_links[0][1]}")
                    
                    href, filename = matching_links[0]
                    file_url = _construct_file_url(href, year_dir_url)
                    output_file_path = tile_output_dir / filename
                    
                    # Skip if file already exists
                    if output_file_path.exists():
                        print(f"    File already exists: {filename}")
                        downloaded_files.append(str(output_file_path))
                        continue
                    
                    if _download_file(file_url, output_file_path, session):
                        downloaded_files.append(str(output_file_path))
                        
                except Exception as e:
                    print(f"    ERROR: Failed to process date {date.strftime('%Y-%m-%d')}: {e}")
                    continue
                    
        except requests.RequestException as e:
            print(f"ERROR: Failed to fetch directory listing for tile {tile_index}: {e}")
            continue
        except Exception as e:
            print(f"ERROR: Unexpected error processing tile {tile_index}: {e}")
            continue
    
    print(f"\nDownload complete. Successfully downloaded {len(downloaded_files)} files.")
    return downloaded_files
        
        
def main() -> int:
    is_1deg = False
    compute_sza = False
    if is_1deg:
        data_dir = "oco3_1p00d/ndgl/1p00d/"
    else:
        data_dir = "oco3_0p01d/ndgl/0p01d/"
    engine = "netcdf4"
    year = 2020
    month = 6
    if is_1deg:
        output_csv = f"oco3_1p00d_{year}{month:02d}_sif_lc.csv"
    else:
        output_csv = f"oco3_0p01d_{year}{month:02d}_sif_lc.csv"
    download_brdfs = False
    download_par = False
    start_date = datetime(year, month, 1)
    _, num_days = calendar.monthrange(year, month)
    end_date = datetime(year, month, num_days)
    dates = generate_dates(start_date, end_date)
    daily_data: list[npt.NDArray] = []
    all_tiles: list[tuple[str, datetime]] = []
    for date in tqdm(dates, desc="Processing dates"):
        daily_data.append(process_1day_matrix(date, data_dir, engine, is_1deg, compute_sza))
        if len(daily_data[-1]) > 0:
            all_tiles.extend(find_geonex_tiles(daily_data[-1], date))
    combined_data = np.hstack(daily_data)
    data_transposed = combined_data.T
    columns = ["date", "hour", "longitude", "latitude", "sif_740nm", "SZA"]
    if compute_sza:
        columns.append("comp_SZA")
    df = pd.DataFrame(data_transposed, columns=columns)
    sample_igbp(df)
    df.to_csv(output_csv, index=False)

    # print("All tiles to download:")
    # for tile in all_tiles:
    #     print(f"{tile[1].strftime('%Y-%m-%d')}: {tile[0]}")

    if download_brdfs:
        downloaded_files = download_goes_brdfs(all_tiles)

        print(f"Downloaded {len(downloaded_files)} files.")
    
    if download_par:
        downloaded_par = download_goes_par(all_tiles)

        print(f"Downloaded {len(downloaded_par)} files.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
