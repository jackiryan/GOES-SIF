#!/usr/bin/env python3
from math import cos, sin, tan, pi
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pyhdf.SD import SD, SDC

def get_var(fname, var):
    f = SD(fname, SDC.READ)
    sds = f.select(var)
    var_data = sds.get().astype("float32")
    sds.endaccess()
    f.end()
    return var_data

def ross_thick_kernel(theta_i, theta_v, raz):
    """Compute the Ross-Thick kernel.
    
    Args:
        theta_i: Solar zenith angle (radians)
        theta_v: View zenith angle (radians)
        raz: View-sun relative azimuth angle (radians)
    
    Returns:
        Ross-Thick kernel value
    """
    cos_theta_i = cos(theta_i)
    cos_theta_v = cos(theta_v)
    
    # Relative azimuth
    phi = raz
    
    # Scattering angle cosine
    cos_xi = cos_theta_i * cos_theta_v + sin(theta_i) * sin(theta_v) * cos(phi)
    
    # Ross-Thick kernel formula
    xi = np.arccos(cos_xi)
    return ((pi/2 - xi) * cos_xi + sin(xi)) / (cos_theta_i + cos_theta_v) - pi/4

def li_sparse_kernel(theta_i, theta_v, raz, h_b=2.0, b_r=1.0):
    """Compute the Li-Sparse kernel.
    
    Args:
        theta_i: Solar zenith angle (radians)
        theta_v: View zenith angle (radians)
        raz: View-sun relative azimuth angle (radians)
        h_b: h/b ratio (height to center of crown divided by vertical crown radius)
        b_r: b/r ratio (vertical to horizontal crown radius)
    
    Returns:
        Li-Sparse kernel value
    """
    # Relative azimuth
    phi = raz
    
    # Basic trig functions
    tan_theta_i = np.tan(theta_i)
    tan_theta_v = np.tan(theta_v)
    
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
    
    # Original Li-Sparse kernel
    return O - sec_theta_ip - sec_theta_vp + 0.5 * (1 + cos_gamma) * sec_theta_ip * sec_theta_vp


def compute_reflectance(k_iso, k_vol, k_geo, theta_i, theta_v, raz):
    """Compute reflectance using Ross-Thick Li-Sparse model.
    
    Args:
        k_iso: Isotropic parameter
        k_vol: Volumetric parameter
        k_geo: Geometric parameter
        theta_i: Solar zenith angle (radians)
        theta_v: View zenith angle (radians)
        raz: View-sun relative azimuth angle (radians)
    
    Returns:
        Surface reflectance
    """
    f_vol = ross_thick_kernel(theta_i, theta_v, raz)
    f_geo = li_sparse_kernel(theta_i, theta_v, raz)
    
    reflectance = k_iso + k_vol * f_vol + k_geo * f_geo
    return reflectance

def scale_rgb(data_r, data_g, data_b, width, height, max_scale=6000):
    """
    Apply logarithmic stretch to match human visual sensibility
    
    Parameters:
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
    x = np.clip(data_r, 1, max_in).astype('f4')  # 1 will be zero in logarithm
    x = (np.log(x) - np.log(ref)) * scale + offset  
    x = np.clip(x, 0, max_out)
    data_rgb[:, :, 0] = x.reshape(height, width).astype("u1")
   
    # Process green channel
    x = np.clip(data_g, 1, max_in).astype('f4')
    x = (np.log(x) - np.log(ref)) * scale + offset
    x = np.clip(x, 0, max_out)
    data_rgb[:, :, 1] = x.reshape(height, width).astype("u1")
   
    # Process blue channel
    x = np.clip(data_b, 1, max_in).astype('f4')
    x = (np.log(x) - np.log(ref)) * scale + offset
    x = np.clip(x, 0, max_out)
    data_rgb[:, :, 2] = x.reshape(height, width).astype("u1")

    return data_rgb

def plot_reflectance_rgb(red, green, blue, title="Surface BRF"):
    """Plot RGB reflectance image."""
    # Normalize to 0-1 range
    red_norm = red #np.clip(red / np.percentile(red, 99), 0, 1)
    green_norm = green #np.clip(green / np.percentile(red, 99), 0, 1)
    blue_norm = blue #np.clip(blue / np.percentile(red, 99), 0, 1)
    
    # Stack RGB
    rgb = np.stack([red_norm, green_norm, blue_norm], axis=2)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(rgb)
    plt.title(title)
    plt.show()

def numpy_to_rgb_png(array_data, r_band=2, g_band=1, b_band=0, nodata_value=-32767):
    """
    Convert a multi-band numpy array to RGBA PNG with transparency for nodata values.
    For float data, normalization is done based on min and max values across the selected bands.
    
    Parameters:
    array_data: numpy array with shape (bands, height, width)
    r_band, g_band, b_band: indices of bands to use for RGB channels
    nodata_value: value to be treated as transparent in any band
    """
    # Extract the bands for RGB
    rawr = array_data[r_band, :, :]
    #rawg = array_data[g_band, :, :]
    rawg = (0.48358168*array_data[2, :, :] + 
            0.45706946*array_data[0, :, :] + 
            0.06038137*array_data[3, :, :]).astype('u2')
    rawb = array_data[b_band, :, :]
    width, height = rawr.shape
    rgb = scale_rgb(rawr, rawg, rawb, width, height)
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    
    # Create alpha channel (fully opaque by default)
    alpha = np.ones_like(r) * 255
    
    # Set alpha to 0 (transparent) where any band has nodata_value
    nodata_mask = (np.isnan(r)) | (np.isnan(g)) | (np.isnan(b))
    alpha[nodata_mask] = 0
    
    # Replace nodata values with NaN for min/max calculation
    '''
    r_valid = np.where(r == nodata_value, np.nan, r)
    g_valid = np.where(g == nodata_value, np.nan, g)
    b_valid = np.where(b == nodata_value, np.nan, b)
    '''
    #r_valid = np.where(r < nodata_value, np.nan, r)
    #g_valid = np.where(g < nodata_value, np.nan, g)
    #b_valid = np.where(b < nodata_value, np.nan, b)

    quantize = False
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
    rgba = np.dstack((r, r, r, alpha))
    
    # Create PIL Image with alpha channel
    img = Image.fromarray(rgba, mode='RGBA')
    return img

if __name__ == "__main__":
    #hdf_file = "GO16_ABI12C_20203142140_GLBG_h16v04_02.hdf"
    
    hdf_file = "GO16_2020022_ABI2B/GO16_ABI12B_20200221700_GLBG_h16v04_02.hdf"
    sur_refl_raw = get_var(hdf_file, "sur_refl1km")
    sur_refl_1km = np.where(sur_refl_raw == -28672, np.nan, sur_refl_raw)
    
    # multiplying by the scale factor
    hdf_file = "GO16_ABI12C_20200222200_GLBG_h16v04_02.hdf"
    Kiso_raw = get_var(hdf_file, "Kiso")
    Kiso = np.where(Kiso_raw == -32767, 0, Kiso_raw)# * 1e-4)
    Kgeo_raw = get_var(hdf_file, "Kgeo")
    Kgeo = np.where(Kgeo_raw == -32767, 0, Kgeo_raw)# * 1e-4)
    Kvol_raw = get_var(hdf_file, "Kvol")
    Kvol = np.where(Kvol_raw == -32767, 0, Kvol_raw)# * 1e-4)


    # solar zenith angle
    theta_i = np.radians(50)
    # relative azimuth angle
    raz = np.radians(90)
    # vza
    theta_v = np.radians(0)

    reflectance = np.zeros((4, 600, 600), dtype=np.float32)
    for i in range(4):
        reflectance[i] = compute_reflectance(Kiso[i], Kvol[i], Kgeo[i], theta_i, theta_v, raz)
    #data_g = (0.48358168*reflectance[2, :, :] + 
    #          0.45706946*reflectance[0, :, :] + 
    #          0.06038137*reflectance[3, :, :])
    #reflectance[1] = data_g
    #reflectance[0] = reflectance[0] ** (1.0/1.3)
    #rgb_data = scale_rgb(reflectance[2], reflectance[1], reflectance[0], 600, 600, max_scale=0.2)
    #rgb_image = Image.fromarray(rgb_data)
    #plot_reflectance_rgb(sur_refl_1km[2], sur_refl_1km[1], sur_refl_1km[0])
    #plot_reflectance_rgb(reflectance[2], reflectance[1], reflectance[0])
    #print(reflectance)
    # nd = -6.08251133e4
    # nd = -32767
    # nd = -6.13144570e4
    # nd = -28672
    # nd = -6921.6226
    rgb_image = numpy_to_rgb_png(reflectance)
    rgb_image.save("rtls3_022_sza50_raz90.png")
    #rgb_image = numpy_to_rgb_png(sur_refl_1km)
    #rgb_image.save("surf_022_angled.png")