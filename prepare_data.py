import calendar
from datetime import datetime, timedelta
from glob import glob
import numpy as np
import numpy.typing as npt
import os.path
import pandas as pd
import xarray as xr

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


def vectorize_dt(dt: xr.DataTree) -> npt.NDArray[np.float32]:
    hr = np.asarray(dt[OCO3_VARIABLES["hour"]].data)
    lat = np.asarray(dt[OCO3_VARIABLES["latitude"]].data)
    lon = np.asarray(dt[OCO3_VARIABLES["longitude"]].data)
    sif740 = np.asarray(dt[OCO3_VARIABLES["sif740"]].data)

    expected_shape = (len(hr), len(lon), len(lat))
    if sif740.shape != expected_shape:
        raise ValueError(f"SIF array shape {sif740.shape} doesn't match expected {expected_shape}")
    
    valid_sif = np.where(~np.isnan(sif740))
    hr_ndx, lon_ndx, lat_ndx = valid_sif

    vectorized_data = np.vstack([
        hr[hr_ndx],
        lon[lon_ndx],
        lat[lat_ndx],
        sif740[hr_ndx, lon_ndx, lat_ndx]
    ])
    return vectorized_data


def process_1day_matrix(data_day: datetime, data_dir: str, engine: str) -> npt.NDArray[np.float32]:
    year = data_day.year
    month = f"{int(data_day.month):02d}"
    day = f"{int(data_day.day):02d}"
    granule_wildcard = os.path.join(data_dir, f"{year}/{month}/oco3_goessif_{year}{month}{day}_ndgl_*.nc4")
    results = glob(granule_wildcard)
    try:
        # Use the first result, otherwise throw IndexError if file not found
        oco3_granule = results[0]
        dt = open_oco3(oco3_granule, engine)
        vectorized_day = vectorize_dt(dt)
        return vectorized_day
    except IndexError:
        print(f"Warning: no granule found for {data_day.strftime('%Y-%m-%d')}. Skipping...")
        return np.empty((4, 0), dtype=np.float32)
    except Exception as e:
        if "dt" in locals().keys():
            dt.close() # type: ignore
        print(f"Unexpected {type(e)} when processing granule for {data_day.strftime('%Y-%m-%d')}: {e}. Skipping...")
        return np.empty((4, 0), dtype=np.float32)
        
        
def main() -> int:
    data_dir = "oco3_1p00d/ndgl/1p00d/"
    engine = "netcdf4"
    year = 2020
    month = 6
    output_csv = f"oco3_1p00d_{year}{month:02d}_vector.csv"
    start_date = datetime(year, month, 1)
    _, num_days = calendar.monthrange(year, month)
    end_date = datetime(year, month, num_days)
    dates = generate_dates(start_date, end_date)
    daily_data: list[npt.NDArray] = []
    for date in dates:
        daily_data.append(process_1day_matrix(date, data_dir, engine))
    combined_data = np.hstack(daily_data)
    data_transposed = combined_data.T
    columns = ["hour", "longitude", "latitude", "sif_740nm"]
    df = pd.DataFrame(data_transposed, columns=columns)
    df.to_csv(output_csv, index=False)
    return 0

if __name__ == "__main__":
    main()
