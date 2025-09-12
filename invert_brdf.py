#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import os
from PIL import Image
from pyhdf.SD import SD, SDC
import sys
import traceback


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for the BRDF inversion script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="BRDF Inversion (Surface Reflectance calculation) for GeoNEX MAIAC granules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python invert_brdf.py data/BRDF/GO16_ABI12C_20200222200_GLBG_h16v04_02.hdf data/SurfRefl/GO16_ABI12B_20200221800_GLBG_h16v04_02.hdf
        """
    )
    
    parser.add_argument(
        "input_brdf",
        help="Path to input BRDF HDF-4 file"
    )
    parser.add_argument(
        "input_refl",
        help="Path to input Surface Reflectance HDF-4 file"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output filename for generated RTLS image"
    )
    
    return parser.parse_args()


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


def scale_rgb(
        data_r: npt.NDArray,
        data_g: npt.NDArray,
        data_b: npt.NDArray,
        width: int,
        height: int,
        max_scale: float = 0.7
    ) -> npt.NDArray[np.uint8]:
    """
    Apply logarithmic stretch to match human visual sensibility
    
    Args:
    data_r, data_g, data_b: Input data for red, green, and blue channels
    width, height: Dimensions of the output image
    max_scale: Maximum scale value for normalization
    
    Returns:
        data_rgb: RGB image array scaled for visualization
    """
    # logarithmic stretch parameters
    max_in = max_scale
    max_out = 255
    ref = max_in * 0.20          # 20% reflectance as the middle gray
    offset = max_out * 0.5       # corresponding to ref
    scale = max_out * 0.20 / np.log(2.0)  # 20% linear increase for 2x in reflectance
  
    data_rgb = np.zeros((height, width, 3), "u1")
  
    # Process red channel
    x = np.clip(data_r, 1e-4, max_in).astype("f4")  # 1 will be zero in logarithm
    x = (np.log(x) - np.log(ref)) * scale + offset  
    x = np.clip(x, 0, max_out)
    data_rgb[:, :, 0] = x.reshape(height, width).astype("u1")
   
    # Process green channel
    x = np.clip(data_g, 1e-4, max_in).astype("f4")
    x = (np.log(x) - np.log(ref)) * scale + offset
    x = np.clip(x, 0, max_out)
    data_rgb[:, :, 1] = x.reshape(height, width).astype("u1")
   
    # Process blue channel
    x = np.clip(data_b, 1e-4, 0.3).astype("f4")
    x = (np.log(x) - np.log(0.3 * 0.2)) * scale + offset
    x = np.clip(x, 0, max_out)
    data_rgb[:, :, 2] = x.reshape(height, width).astype("u1")

    return data_rgb


def numpy_to_rgb_png(array_data: npt.NDArray[np.float32], quantize: bool = False, nodata_value: int = -32767):
    """
    Convert a multi-band numpy array to RGBA PNG with transparency for nodata values.
    For float data, normalization is done based on min and max values across the selected bands.
    
    Args:
        array_data (npt.NDArray[np.float32]): numpy array with shape (bands, height, width)
        quantize (bool): Flag to scale data per band
        nodata_value (int): value to be treated as transparent in any band
    
    Returns:
        PIL Image
    """
    # Extract the bands for RGB
    rawr = array_data[2, :, :]
    rawg = (0.48358168*array_data[2, :, :] + 
            0.45706946*array_data[0, :, :] + 
            0.06038137*array_data[3, :, :])
    rawb = array_data[0, :, :]

    print("reflectance data ranges")
    print(f"red {np.nanmin(rawr)} to {np.nanmax(rawr)}")
    print(f"green {np.nanmin(rawg)} to {np.nanmax(rawg)}")
    print(f"blue {np.nanmin(rawb)} to {np.nanmax(rawb)}")

    # Create alpha channel (fully opaque by default)
    alpha = np.ones_like(rawr) * 255
    
    # Set alpha to 0 (transparent) where any band has nodata_value
    nodata_mask = (np.isnan(rawr)) | (np.isnan(rawg)) | (np.isnan(rawb))
    alpha[nodata_mask] = 0

    width, height = rawr.shape
    rgb = scale_rgb(rawr, rawg, rawb, width, height)
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    if (array_data.dtype == np.float32 or array_data.dtype == np.float64) and quantize:
        # Find global min and max across all three bands, ignoring nodata values
        combined = np.stack([r, g, b])
        data_min = np.nanmin(combined)
        data_max = np.nanmax(combined)
        print("data_min:", data_min)
        print("data_max:", data_max)
        
        # Avoid division by zero if all values are the same
        if data_max == data_min:
            data_range = 1.0
        else:
            data_range = data_max - data_min
        
        # Normalize each band using the same min/max values
        r_norm = (r - data_min) / data_range
        g_norm = (g - data_min) / data_range 
        b_norm = (b - data_min) / data_range
        
        # Replace NaNs (from nodata) with 0
        r_norm = np.nan_to_num(r_norm, nan=0.0)
        g_norm = np.nan_to_num(g_norm, nan=0.0)
        b_norm = np.nan_to_num(b_norm, nan=0.0)
        
        # Convert to 8-bit
        r = np.clip(r_norm * 255, 0, 255).astype(np.uint8)
        g = np.clip(g_norm * 255, 0, 255).astype(np.uint8)
        b = np.clip(b_norm * 255, 0, 255).astype(np.uint8)
    else:
        # For integer types, just replace nodata and convert to uint8
        r = np.where(r == nodata_value, 0, r).astype(np.uint8)
        g = np.where(g == nodata_value, 0, g).astype(np.uint8)
        b = np.where(b == nodata_value, 0, b).astype(np.uint8)
    
    # Convert alpha to uint8
    alpha = alpha.astype(np.uint8)

    # Stack the bands to create RGBA
    rgba = np.dstack((r, g, b, alpha))
    
    # Create PIL Image with alpha channel
    img = Image.fromarray(rgba)
    return img


def main() -> int:
    print("GeoNEX BRDF Inversion Script")
    print("=" * 50)

    args = parse_arguments()
    kernel_parms = parse_coeffs(args.input_brdf)

    # Get the 5km grid values for solar zenith angle, viewing zenith angle,
    # and relative azimuth. Ideally the reflectance granule is on the hour.
    angles = parse_angles(args.input_refl)
    
    reflectance = np.zeros((4, 600, 600), dtype=np.float32)
    for i in range(4):
        Kiso = kernel_parms[0, i, :, :]
        Kvol = kernel_parms[1, i, :, :]
        Kgeo = kernel_parms[2, i, :, :]
        cosSZA = angles[0, :, :]
        cosVZA = angles[1, :, :]
        RelAZ  = angles[2, :, :]
        reflectance[i] = compute_reflectance(Kiso, Kvol, Kgeo, cosSZA, cosVZA, RelAZ)
    rgb_image = numpy_to_rgb_png(reflectance)
    if args.output:
        outfile = args.output
    else:
        data_datetime = os.path.basename(args.input_refl).split("_")[2]
        data_date = data_datetime[0:7]
        data_time = data_datetime[7:11]
        data_cell = os.path.basename(args.input_refl).split("_")[4]
        outfile = f"GO16_ABI_RTLS_{data_date}_{data_time}_{data_cell}.png"
    rgb_image.save(outfile)

    return 0

if __name__ == "__main__":
    sys.exit(main())