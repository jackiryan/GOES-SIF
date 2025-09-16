import h5py
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from pyhdf.SD import SD, SDC
import sys
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


def parse_angles(input_refl: str) -> npt.NDArray[np.float32]:
    """
    Extract the required metadata from a surface reflectance granule to use for BRDF inversion.
    Specifically we want:
    - cosSZA or the cosine of the solar zenith angle, it needs to be scaled by 1e-4
    - cosVZA or the cosine of the viewing zenith angle, it also needs to be scaled by 1e-4
    - RelAZ or the relative azimuth which is VAZ - SAZ, also requires 1e-4 scaling

    These angles are given on a 5km grid so we use nearest neighbor to scale them to the
    1km grid used in the BRDF. This may become more sophisticated later if needed.

    Args:
        input_refl (str): Name of the input surface reflectance granule

    Returns:
        A 3-dimensional array where the first band is cosSZA, second is cosVZA, and third is RelAZ. The
        other two dimensions are lat and lon respectively.

    Raises:
        SystemExit: On failure to parse angles from HDF file
    """
    try:
        f = SD(input_refl, SDC.READ)
        
        var_stack: list[npt.NDArray] = []
        for var in ["cosSZA", "cosVZA", "RelAZ"]:
            sds = f.select(var)
            var_data = sds.get().astype("int16")
            sds.endaccess()
            scaled_data = np.array(var_data * 1e-4, dtype=np.float32)
            # Upscale from 5km to 1km (x5)
            upscaled_data = np.repeat(np.repeat(scaled_data, 5, axis=0), 
                                      5, axis=1)
            var_stack.append(upscaled_data)
        
        angles = np.stack(var_stack, axis=0)

        print(f"Successfully extracted angles from {input_refl}")
        print(f"  Final angles array shape: {angles.shape}")
        print(f"  cosSZA range: [{np.min(angles[0, :, :]):.4f}, {np.max(angles[0, :, :]):.4f}]")
        print(f"  cosVZA range: [{np.min(angles[1, :, :]):.4f}, {np.max(angles[1, :, :]):.4f}]")
        print(f"  RelAZ range: [{np.min(angles[2, :, :]):.2f}, {np.max(angles[2, :, :]):.2f}]")
        
        return angles
    except Exception as e:
        print(f"Error processing Surface Reflectance HDF file: {e}")
        traceback.print_exc()
        sys.exit(1)


def ross_thick_kernel(
        cos_theta_i: npt.NDArray[np.float32],
        cos_theta_v: npt.NDArray[np.float32],
        phi: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """
    Compute the Ross-Thick kernel.
    
    Args:
        cos_theta_i (npt.NDArray[np.float32]): Cosine of solar zenith angle
        cos_theta_v (npt.NDArray[np.float32]): Cosine of view zenith angle
        phi (npt.NDArray[np.float32]): View-sun relative azimuth angle (radians)
    
    Returns:
        npt.NDArray[np.float32]: Ross-Thick kernel value
    """
    # since 0° ≤ SZA ≤ 90° and 0° ≤ VZA ≤ 90°
    sin_theta_i = np.sqrt(1 - (cos_theta_i ** 2))
    sin_theta_v = np.sqrt(1 - (cos_theta_v ** 2))
    
    # Cosine of scattering angle
    cos_xi = cos_theta_i * cos_theta_v + sin_theta_i * sin_theta_v * np.cos(phi)
    
    # Ross-Thick kernel formula
    xi = np.arccos(cos_xi)
    return np.array(((np.pi/2 - xi) * cos_xi + np.sin(xi)) / (cos_theta_i + cos_theta_v) - np.pi/4, dtype=np.float32)


def li_sparse_kernel(
        cos_theta_i: npt.NDArray[np.float32],
        cos_theta_v: npt.NDArray[np.float32],
        phi: npt.NDArray[np.float32],
        h_b: float = 2.0,
        b_r: float = 1.0
) -> npt.NDArray[np.float32]:
    """
    Compute the Li-Sparse kernel.
    
    Args:
        cos_theta_i (npt.NDArray[np.float32]): Cosine of solar zenith angle
        cos_theta_v (npt.NDArray[np.float32]): Cosine of view zenith angle
        phi (npt.NDArray[np.float32]): View-sun relative azimuth angle (radians)
        h_b (float): h/b ratio (height to center of crown divided by vertical crown radius)
        b_r (float): b/r ratio (vertical to horizontal crown radius)
    
    Returns:
        npt.NDArray[np.float32]: Li-Sparse kernel value
    """
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
        cosThetaI: npt.NDArray[np.float32],
        cosThetaV: npt.NDArray[np.float32],
        rAz: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """
    Compute reflectance using Ross-Thick Li-Sparse model.
    
    Args:
        Kiso (npt.NDArray[np.float32]): Isotropic parameter
        Kvol (npt.NDArray[np.float32]): Volumetric parameter
        Kgeo (npt.NDArray[np.float32]): Geometric parameter
        cosThetaI (npt.NDArray[np.float32]): Cosine of solar zenith angle
        cosThetaV (npt.NDArray[np.float32]): Cosine of view zenith angle
        rAz (npt.NDArray[np.float32]): View-sun relative azimuth angle (radians)
    
    Returns:
        npt.NDArray[np.float32]: Surface reflectance
    """
    nadirs = np.zeros((600, 600), dtype=np.float32)
    cosThetaV = np.ones_like(nadirs)
    Fvol = ross_thick_kernel(cosThetaI, cosThetaV, nadirs) #cosThetaV, rAz)
    Fgeo = li_sparse_kernel(cosThetaI, cosThetaV, nadirs) #cosThetaV, rAz)

    """
    print("kernel data ranges:")
    print(f"Kiso: {np.nanmin(Kiso)} to {np.nanmax(Kiso)}")
    print(f"Kvol: {np.nanmin(Kvol)} to {np.nanmax(Kvol)}")
    print(f"Fvol: {np.nanmin(Fvol)} to {np.nanmax(Fvol)}")
    print(f"Kgeo: {np.nanmin(Kgeo)} to {np.nanmax(Kgeo)}")
    print(f"Fgeo: {np.nanmin(Fgeo)} to {np.nanmax(Fgeo)}")
    """
    
    reflectance = Kiso + Kvol * Fvol + Kgeo * Fgeo
    return np.array(reflectance, dtype=np.float32)


def create_coordinate_arrays(
        lon_min: int,
        lon_max: int,
        lat_min: int,
        lat_max: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Create latitude and longitude coordinate arrays based on metadata.
    
    Args:
        lon_min, lon_max, lat_min, lat_max (int): longitude and latitude bounds for grid square
        
    Returns:
        tuple: (longitude array, latitude array) both with shape (600,)
    """
    # Create coordinate arrays (600 pixels for 6 degrees = 0.01 degree resolution)
    lon = np.linspace(lon_min, lon_max, 600, endpoint=False)
    lat = np.linspace(lat_max, lat_min, 600, endpoint=False)
    
    return lon, lat


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


if __name__ == "__main__":
    # Southern California (Los Angeles)
    #tile_index = "h10v04"
    #doy = "2020164"
    #hr_set = ["14", "19", "01"]
    #min_set = ["00", "00", "00"]
    # Lake Michigan (Chicago)
    #tile_index = "h15v02"
    #doy = "2020163"
    #hr_set = ["12", "17", "23"]
    #min_set = ["00", "10", "00"]
    # Southeast US (Atlanta)
    tile_index = "h16v04"
    doy = "2020163"
    hr_set = ["12", "17", "23"]
    min_set = ["00", "00", "00"]
    # Montana
    #tile_index = "h12v02"
    #doy = "2020163"
    #hr_set = ["22", "17", "23"]
    #min_set = ["00", "00", "00"]

    par_vmax = 500

    nbar_band = 3
    nbar_title = ["Blue (0.47µm) NBAR", "Green (0.51µm) NBAR", "Red (0.64µm) NBAR", "NIR (0.86µm) NBAR"]

    h_ndx = int(tile_index[1:3])
    v_ndx = int(tile_index[4:6])
    titles = ["Morning (8:00)", "Midday (13:00)", "Evening (19:00)"]
    #titles = ["Afternoon (22:00 UTC)", "Midday (13:00)", "Evening (19:00)"]
    input_brdf = f"goes_data/{tile_index}/GO16_ABI12C_{doy}2350_GLBG_{tile_index}_02.hdf"
    input_par  = f"goes_data/{tile_index}/G016_DSR_{doy}_{tile_index}.h5"
    kernel_parms = parse_coeffs(input_brdf)
    par_data = parse_par(input_par)

    ul_lon = -180 + (h_ndx * 6)
    ul_lat = 60 - (v_ndx * 6)
    
    # Calculate full extent (6x6 degree square)
    lon_min = ul_lon
    lon_max = ul_lon + 6
    lat_min = ul_lat - 6
    lat_max = ul_lat

    lon, lat = create_coordinate_arrays(lon_min, lon_max, lat_min, lat_max)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    #fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    for i, ax in enumerate(axes.flat):
        row = i // 3 # 0 for rtls plots, 1 for par plots
        col = i % 3 # 0, 1, 2 for morning, noon, evening columns
        #row = i
        #col = 0
        hr = hr_set[col]
        mins = min_set[col]

        if row == 0:
            input_refl = f"GO16_ABI12B_{doy}{hr}{mins}_GLBG_{tile_index}_02.hdf"
            # Get the 5km grid values for solar zenith angle, viewing zenith angle,
            # and relative azimuth. Ideally the reflectance granule is on the hour.
            angles = parse_angles(input_refl)
            
            # Compute reflectance for all bands, can do a composite or single band
            # based on what communicates the idea better
            reflectance = np.zeros((4, 600, 600), dtype=np.float32)
            for j in range(4):
                Kiso = kernel_parms[0, i, :, :]
                Kvol = kernel_parms[1, i, :, :]
                Kgeo = kernel_parms[2, i, :, :]
                cosSZA = angles[0, :, :]
                cosVZA = angles[1, :, :]
                RelAZ  = angles[2, :, :]
                reflectance[j] = compute_reflectance(Kiso, Kvol, Kgeo, cosSZA, cosVZA, RelAZ)

            ax.set_title(titles[col], fontsize=12)
            im = ax.pcolormesh(lon_grid, lat_grid, reflectance[nbar_band],
                               vmin=0.0, vmax=0.6, cmap="viridis", shading="auto")
            if col == 0:
                ax.set_ylabel("Latitude (°)", fontsize=11)
            else:
                ax.set_yticklabels([])
            if col == 2:
                pos = ax.get_position()
                cbar_ax = fig.add_axes((pos.x1 + 0.02, pos.y0 + 0.04, 0.02, pos.height * 0.8))
                cbar = plt.colorbar(im, cax=cbar_ax)
                cbar.set_label(nbar_title[nbar_band], fontsize=11, rotation=270, labelpad=15)
            ax.set_xticklabels([])
            ax.set_aspect('equal', adjustable='box')
        elif row == 1:
            im = ax.pcolormesh(lon_grid, lat_grid, par_data[int(hr)],
                               vmin=0.0, vmax=par_vmax, cmap="plasma", shading="auto")
            if col == 0:
                ax.set_ylabel("Latitude (°)", fontsize=11)
            else:
                ax.set_yticklabels([])
            if col == 2:
                pos = ax.get_position()
                cbar_ax = fig.add_axes((pos.x1 + 0.02, pos.y0 + 0.04, 0.02, pos.height * 0.8))
                cbar = plt.colorbar(im, cax=cbar_ax)
                cbar.set_label("PAR (W/m²)", fontsize=11, rotation=270, labelpad=15)
            ax.set_xlabel('Longitude (°)', fontsize=11)
            ax.set_aspect('equal', adjustable='box')
            
    #plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    plt.show()