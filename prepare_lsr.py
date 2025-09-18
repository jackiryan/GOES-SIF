import glob
from datetime import datetime
import numpy as np
import numpy.typing as npt
import os
import pandas as pd
from pyhdf.SD import SD, SDC
import h5py
import sys
from tqdm import tqdm
import traceback


def parse_coeffs(input_brdf: str) -> npt.NDArray[np.float32]:
    """
    Extract the Ross-Thick Li-Sparse Kernel parameters from  the BRDF granule
    Specifically we want:
    - Kiso, the isotropic kernel parameter
    - Kvol, the volumetric kernel (ross-thick) parameter
    - Kgeo, the geometric kernel (li-sparse) parameter

    All parameters need to have the nodata value of -32767 filtered and should be scaled
    by 1e-4. We can further filter values outside the valid range of (-30000, 30000)

    Args:
        input_brdf (str): Name of the input BRDF granule

    Returns:
        A 4-dimensional float32 array where the first element is Kiso in band 0, then Kvol,
        and lastly Kgeo. The first dimension of each array contains the spectral bands from ABI:
        0.47 micron (blue), 0.51 micron (synth green, reference only), 0.64 micron (red), and 0.86
        micron (NIR). The third and fourth dimensions represent lat and lon respectively.

    Raises:
        SystemExit: On failure to parse coefficients from HDF file
    """
    try:
        f = SD(input_brdf, SDC.READ)
        
        var_stack: list[npt.NDArray[np.float32]] = []
        for var in ["Kiso", "Kvol", "Kgeo"]:
            sds = f.select(var)
            var_data = sds.get().astype("int16")
            sds.endaccess()
            filt_data = np.where(var_data < -30000, np.nan, var_data)
            filt_data = np.where(filt_data > 30000, np.nan, filt_data)
            scaled_data = (filt_data * 1e-4).astype("f4")
            var_stack.append(scaled_data)

        coeffs = np.stack(var_stack, axis=0)
        
        return coeffs
    except Exception as e:
        print(f"Error processing BRDF HDF file: {e}")
        traceback.print_exc()
        sys.exit(1)


def parse_par(input_par: str) -> npt.NDArray[np.float32]:
    """
    Load PAR data from HDF-5 file.
    
    Args:
        input_file (str): Path to input HDF-5 file
        
    Returns:
        numpy.ndarray: PAR data array with shape (24, 600, 600)
        
    Raises:
        SystemExit: If file cannot be read or PAR_hourly variable is missing
    """
    try:
        with h5py.File(input_par, "r") as hdf_file:
            if "PAR_hourly" not in hdf_file:
                print("Error: 'PAR_hourly' variable not found in HDF-5 file")
                print("Available variables:", list(hdf_file.keys()))
                sys.exit(1)
            
            par_data = np.array(hdf_file["PAR_hourly"][:], dtype=np.int16) # type: ignore
            par_masked = np.where(par_data == -9999, np.nan, par_data)
            # Apply a scaling, ideally this would come from Dongdong Wang's User Guide, but I couldn't find it,
            # so I guessed based on looking at values in https://doi.org/10.5194/essd-15-1419-2023
            par_data_scaled = np.array(par_masked * 0.1, dtype=np.float32)
            # print(f"Successfully loaded PAR data with shape: {par_data.shape}")
            # print(f"PAR range: {np.nanmin(par_data_scaled):.2f} to {np.nanmax(par_data_scaled):.2f} W/m²")
            
            return par_data_scaled
            
    except Exception as e:
        print(f"Error reading HDF-5 file: {e}")
        sys.exit(1)


def ross_thick_kernel(
        theta_i: npt.NDArray[np.float32],
        theta_v: npt.NDArray[np.float32],
        phi: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
    """
    Compute the Ross-Thick kernel.
    
    Args:
        theta_i (npt.NDArray[np.float32]): Solar zenith angle
        theta_v (npt.NDArray[np.float32]): View zenith angle
        phi (npt.NDArray[np.float32]): View-sun relative azimuth angle (radians)
    
    Returns:
        npt.NDArray[np.float32]: Ross-Thick kernel value
    """
    cos_theta_i = np.cos(theta_i)
    cos_theta_v = np.cos(theta_v)
    # since 0° ≤ SZA ≤ 90° and 0° ≤ VZA ≤ 90°
    sin_theta_i = np.sqrt(1 - (cos_theta_i ** 2))
    sin_theta_v = np.sqrt(1 - (cos_theta_v ** 2))
    
    # Cosine of scattering angle
    cos_xi = cos_theta_i * cos_theta_v + sin_theta_i * sin_theta_v * np.cos(phi)
    
    # Ross-Thick kernel formula
    xi = np.arccos(cos_xi)
    return np.array(((np.pi/2 - xi) * cos_xi + np.sin(xi)) / (cos_theta_i + cos_theta_v) - np.pi/4, dtype=np.float32)


def li_sparse_kernel(
        theta_i: npt.NDArray[np.float32],
        theta_v: npt.NDArray[np.float32],
        phi: npt.NDArray[np.float32],
        h_b: float = 2.0,
        b_r: float = 1.0
    ) -> npt.NDArray[np.float32]:
    """
    Compute the Li-Sparse kernel.
    
    Args:
        theta_i (npt.NDArray[np.float32]): Solar zenith angle
        theta_v (npt.NDArray[np.float32]): View zenith angle
        phi (npt.NDArray[np.float32]): View-sun relative azimuth angle (radians)
        h_b (float): h/b ratio (height to center of crown divided by vertical crown radius)
        b_r (float): b/r ratio (vertical to horizontal crown radius)
    
    Returns:
        npt.NDArray[np.float32]: Li-Sparse kernel value
    """
    cos_theta_i = np.cos(theta_i)
    cos_theta_v = np.cos(theta_v)
    # As with the sine in ross-thick, this is allowed because
    # 0° ≤ SZA ≤ 90° and 0° ≤ VZA ≤ 90°
    tan_theta_i = np.sqrt(1 / cos_theta_i ** 2 - 1)
    tan_theta_v = np.sqrt(1 / cos_theta_v ** 2 - 1)
    
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    
    # Adjusted angles for crown shape
    theta_i_prime = np.arctan(b_r * tan_theta_i)
    theta_v_prime = np.arctan(b_r * tan_theta_v)
    cos_theta_ip = np.cos(theta_i_prime)
    sin_theta_ip = np.sin(theta_i_prime)
    cos_theta_vp = np.cos(theta_v_prime)
    sin_theta_vp = np.sin(theta_v_prime)
    tan_theta_ip = b_r * tan_theta_i
    tan_theta_vp = b_r * tan_theta_v
    
    # Secant terms
    sec_theta_ip = 1.0 / np.cos(theta_i_prime)
    sec_theta_vp = 1.0 / np.cos(theta_v_prime)
    
    # Scattering angle
    cos_gamma = cos_theta_ip * cos_theta_vp + sin_theta_ip * sin_theta_vp * cos_phi
    
    # Distance parameter
    D_square = tan_theta_ip**2 + tan_theta_vp**2 - 2 * tan_theta_ip * tan_theta_vp * cos_phi
    #D = np.sqrt(D_square)
    
    # Cos_t parameter for overlap calculation
    cos_t = (h_b * np.sqrt(D_square + (tan_theta_ip * tan_theta_vp * sin_phi)**2)) / (sec_theta_ip + sec_theta_vp)
    
    # Make sure temp is in valid range for arccos
    cos_t = np.clip(cos_t, -1.0, 1.0)
    t = np.arccos(cos_t)
    
    # Overlap function
    O = (1 / np.pi) * (t - np.sin(t) * cos_t) * (sec_theta_ip + sec_theta_vp)
    
    # Li-Sparse kernel formula
    return np.array(O - sec_theta_ip - sec_theta_vp + 0.5 * (1 + cos_gamma) * sec_theta_ip * sec_theta_vp, dtype=np.float32)


def compute_reflectance(
        Kiso: npt.NDArray[np.float32], 
        Kvol: npt.NDArray[np.float32],
        Kgeo: npt.NDArray[np.float32],
        thetaI: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
    """
    Compute reflectance using Ross-Thick Li-Sparse model.
    
    Args:
        Kiso (npt.NDArray[np.float32]): Isotropic parameter
        Kvol (npt.NDArray[np.float32]): Volumetric parameter
        Kgeo (npt.NDArray[np.float32]): Geometric parameter
        thetaI (npt.NDArray[np.float32]): Solar zenith angle
    
    Returns:
        npt.NDArray[np.float32]: Surface reflectance
    """
    nadirs = np.zeros(len(Kiso), dtype=np.float32)
    Fvol = ross_thick_kernel(thetaI, nadirs, nadirs)
    Fgeo = li_sparse_kernel(thetaI, nadirs, nadirs)
    
    reflectance = Kiso + Kvol * Fvol + Kgeo * Fgeo
    return np.array(reflectance, dtype=np.float32)


def invert_brdf(filename, lats, lons, szas):
    tile_index = filename.split("/")[1]
    h_ndx = int(tile_index[1:3])
    v_ndx = int(tile_index[4:6])
    ul_lat = 60 - (v_ndx * 6)
    ul_lon = -180 + (h_ndx * 6)
    x_coords = np.floor((lons - ul_lon) / 0.01).astype(int)
    y_coords = np.floor((lats - ul_lat) / 0.01).astype(int)
    reflectance = np.zeros((4, len(x_coords)), dtype=np.float32)
    kernel_parms = parse_coeffs(filename)
    for i in range(4):
        Kiso = kernel_parms[0, i, y_coords, x_coords].flatten()
        Kvol = kernel_parms[1, i, y_coords, x_coords].flatten()
        Kgeo = kernel_parms[2, i, y_coords, x_coords].flatten()
        reflectance[i] = compute_reflectance(Kiso, Kvol, Kgeo, szas)
    return reflectance


def sample_par(filename, lats, lons, hours):
    tile_index = filename.split("/")[1]
    h_ndx = int(tile_index[1:3])
    v_ndx = int(tile_index[4:6])
    ul_lat = 60 - (v_ndx * 6)
    ul_lon = -180 + (h_ndx * 6)
    x_coords = np.floor((lons - ul_lon) / 0.01).astype(int)
    y_coords = np.floor((lats - ul_lat) / 0.01).astype(int)
    par = np.zeros(len(x_coords), dtype=np.float32)
    all_par = parse_par(filename)
    par = all_par[hours.astype(int), y_coords, x_coords]
    return par

def find_matching_brdf(row, root_dir="goes_data/"):
    """
    Find the matching BRDF file for a given row of data.
    """
    date_str = str(int(row["date"]))
    year = int(date_str[:4])
    month = int(date_str[4:6]) 
    day = int(date_str[6:8])
    date_obj = datetime(year, month, day)
    day_of_year = date_obj.timetuple().tm_yday
    date = f"{year}{day_of_year:03d}"
    lon = row["longitude"]
    lat = row["latitude"]
    
    # Calculate tile indices
    h_ndx = int(np.floor((lon + 180) / 6))
    v_ndx = int(np.floor((60 - lat) / 6))
    
    # Create tile directory path
    tile_dir = f"h{h_ndx:02d}v{v_ndx:02d}"
    full_tile_path = os.path.join(root_dir, tile_dir)
    
    # Check if tile directory exists
    if not os.path.exists(full_tile_path):
        return None
    
    # Look for files matching the date pattern
    # Pattern: GO16_ABI12C_{date}HHMM_GLBG_h{h_ndx}v{v_ndx}_*.hdf
    pattern = f"GO16_ABI12C_{date}*_GLBG_h{h_ndx:02d}v{v_ndx:02d}_*.hdf"
    search_pattern = os.path.join(full_tile_path, pattern)
    
    # Find matching files
    matching_files = glob.glob(search_pattern)
    
    if matching_files:
        # Return the first match (there should typically be only one)
        return matching_files[0]
    else:
        return None


def find_matching_par(row, root_dir="goes_data/"):
    """
    Find the matching DSR-PAR file for a given row of data.
    """
    date_str = str(int(row["date"]))
    year = int(date_str[:4])
    month = int(date_str[4:6]) 
    day = int(date_str[6:8])
    date_obj = datetime(year, month, day)
    day_of_year = date_obj.timetuple().tm_yday
    date = f"{year}{day_of_year:03d}"
    lon = row["longitude"]
    lat = row["latitude"]
    
    # Calculate tile indices
    h_ndx = int(np.floor((lon + 180) / 6))
    v_ndx = int(np.floor((60 - lat) / 6))
    
    # Create tile directory path
    tile_dir = f"h{h_ndx:02d}v{v_ndx:02d}"
    full_tile_path = os.path.join(root_dir, tile_dir)
    
    # Check if tile directory exists
    if not os.path.exists(full_tile_path):
        return None
    
    # Look for files matching the date pattern
    # Pattern: G016_DSR_{date}_GLBG_h{h_ndx}v{v_ndx}.h5
    pattern = f"G016_DSR_{date}_h{h_ndx:02d}v{v_ndx:02d}.h5"
    search_pattern = os.path.join(full_tile_path, pattern)
    
    # Find matching files
    matching_files = glob.glob(search_pattern)
    
    if matching_files:
        # Return the first match (there should typically be only one)
        return matching_files[0]
    else:
        return None
    

def process_brdf_data(df):
    """
    Batch process rows one BRDF file at a time. Result adds LSR columns to DF.
    """
    # Initialize new columns with NaN
    df["LSR_Blue"] = np.nan
    df["LSR_Red"] = np.nan 
    df["LSR_NIR"] = np.nan
    
    # Only process rows that have valid file paths
    valid_rows = df[df["hdf_file_path"].notna()].copy()
    
    if len(valid_rows) == 0:
        print("No valid HDF file paths found. Skipping BRDF processing.")
        return df
    
    # Group by HDF file path
    grouped = valid_rows.groupby("hdf_file_path")
    
    print(f"\nProcessing BRDF data for {len(grouped)} unique HDF files...")
    
    processed_files = 0
    total_samples = 0
    
    for hdf_file, group in tqdm(grouped, desc="Computing LSR from BRDFs"):
        try:
            # Extract coordinates and SZA values for this group
            lats = group["latitude"].values
            lons = group["longitude"].values 
            #szas = group["comp_SZA"].values
            szas = group["SZA"].values
            
            # print(f"Processing file: {os.path.basename(hdf_file)} ({len(group)} samples)")
            
            # Call the BRDF inversion function
            brdf_results = invert_brdf(hdf_file, lats, lons, np.deg2rad(szas))
            
            # Verify the results have the expected shape
            if brdf_results.shape[0] != 4 or brdf_results.shape[1] != len(group):
                print(f"Warning: Unexpected BRDF result shape {brdf_results.shape}, expected (4, {len(group)})")
                continue
            
            # Assign results back to the original DataFrame using the group indices
            df.loc[group.index, "LSR_Blue"] = brdf_results[0, :]
            df.loc[group.index, "LSR_Red"] = brdf_results[2, :]
            df.loc[group.index, "LSR_NIR"] = brdf_results[3, :]
            processed_files += 1
            total_samples += len(group)
            
        except Exception as e:
            print(f"Error processing file {hdf_file}: {str(e)}")
            continue
    
    print(f"\nBRDF processing complete:")
    print(f"Files processed successfully: {processed_files}")
    print(f"Total samples processed: {total_samples}")
    print(f"Rows with LSR data: {df[['LSR_Blue', 'LSR_Red', 'LSR_NIR']].notna().all(axis=1).sum()}")
    
    return df


def process_par_data(df):
    """
    Batch process rows one BRDF file at a time. Result adds LSR columns to DF.
    """
    # Initialize new columns with NaN
    df["PAR"] = np.nan
    
    # Only process rows that have valid file paths
    valid_rows = df[df["par_file_path"].notna()].copy()
    
    if len(valid_rows) == 0:
        print("No valid H5 file paths found. Skipping PAR processing.")
        return df
    
    # Group by PAR file path
    grouped = valid_rows.groupby("par_file_path")
    
    print(f"\nProcessing PAR data for {len(grouped)} unique H5 files...")
    
    processed_files = 0
    total_samples = 0
    
    for hdf_file, group in tqdm(grouped, desc="Sampling PAR"):
        try:
            # Extract coordinates and hour values for this group
            lats = group["latitude"].values
            lons = group["longitude"].values  
            hours = group["hour"].values
            
            # print(f"Processing file: {os.path.basename(hdf_file)} ({len(group)} samples)")
            
            par_results = sample_par(hdf_file, lats, lons, hours)
            
            # Verify the results have the expected shape
            if len(par_results) != len(group):
                print(f"Warning: Unexpected PAR result shape {par_results.shape}, expected ({len(group)},)")
                continue
            
            # Assign results back to the original DataFrame using the group indices
            df.loc[group.index, "PAR"] = par_results
            processed_files += 1
            total_samples += len(group)
            
        except Exception as e:
            print(f"Error processing file {hdf_file}: {str(e)}")
            continue
    
    print(f"\nPAR processing complete:")
    print(f"Files processed successfully: {processed_files}")
    print(f"Total samples processed: {total_samples}")
    print(f"Rows with LSR and PAR data: {df[['LSR_Blue', 'LSR_Red', 'LSR_NIR', 'PAR']].notna().all(axis=1).sum()}")
    return df


def main() -> int:
    year = 2021
    month = 6
    #input_csv = f"oco3_1p00d_{year}{month:02d}_sif_lc.csv"
    input_csv = f"new_oco3_0p01d_{year}{month:02d}_sif_lc.csv"
    #output_csv = f"oco3_1p00d_{year}{month:02d}_lsr_par.csv"
    output_csv = f"new_oco3_0p01d_{year}{month:02d}_lsr_par_clean.csv"
    df = pd.read_csv(input_csv)
    df["hdf_file_path"] = df.apply(find_matching_brdf, axis=1)
    df["par_file_path"] = df.apply(find_matching_par, axis=1)
    df = process_brdf_data(df)
    df = process_par_data(df)
    #df_nofname = df.drop(["hdf_file_path", "par_file_path", "SZA", "comp_SZA"], axis=1)
    df_nofname = df.drop(["hdf_file_path", "par_file_path", "SZA"], axis=1)
    df_clean = df_nofname.dropna()
    df_clean.to_csv(output_csv, index=False)

    return 0

if __name__ == "__main__":
    sys.exit(main())