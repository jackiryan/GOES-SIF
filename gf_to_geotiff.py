#!/usr/bin/env python3
"""
Convert MODIS MCD43GF HDF4 gap-filled albedo data to GeoTIFF format.

This script reads three HDF4 files (Band 1, Band 4, and Band 3) containing
gap-filled black-sky albedo data at 30 arc-second resolution and combines them
into a single GeoTIFF with downsampling to 8k resolution.

Usage:
    python hdf4_mcd43gf_to_geotiff.py <year> <day_of_year> <output.tif> [--input-dir .]
"""

import sys
import os
import argparse
import numpy as np
from pyhdf.SD import SD, SDC # type: ignore [import-untyped]
from scipy import ndimage

# Set GDAL plugin path for conda environment
if 'CONDA_PREFIX' in os.environ:
    gdal_plugins = os.path.join(os.environ['CONDA_PREFIX'], 'Library', 'lib', 'gdalplugins')
    if os.path.exists(gdal_plugins):
        os.environ['GDAL_DRIVER_PATH'] = gdal_plugins

from osgeo import gdal, osr # type: ignore [import-untyped]


def find_hdf_file(input_dir, band_num, day_of_year, year):
    """
    Find the HDF file for a specific band, day, and year.
    
    Args:
        input_dir: Directory containing HDF files
        band_num: Band number (1, 3, or 4)
        day_of_year: Day of year (1-366)
        year: Year (4 digits)
    
    Returns:
        Full path to the HDF file
    """
    filename = f"MCD43GF_wsa_Band{band_num}_{day_of_year:03d}_{year}_V061.hdf"
    filepath = os.path.join(input_dir, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find file: {filepath}")
    
    return filepath


def get_wavelength_for_band(band_num):
    """
    Get the wavelength identifier for the subdataset name.
    
    Args:
        band_num: Band number (1, 3, or 4)
    
    Returns:
        Wavelength string for the subdataset
    """
    wavelengths = {
        1: "0.659",  # Band 1 (Red)
        3: "0.47",   # Band 3 (Blue)
        4: "0.555"   # Band 4 (Green/NIR)
    }
    
    if band_num not in wavelengths:
        raise ValueError(f"Invalid band number: {band_num}. Must be 1, 3, or 4.")
    
    return wavelengths[band_num]


def read_mcd43gf_band(hdf_file, band_num):
    """
    Read albedo data from an MCD43GF HDF4 file using pyhdf.
    
    Args:
        hdf_file: Path to HDF4 file
        band_num: Band number (1, 3, or 4)
    
    Returns:
        numpy array containing the band data
    """
    wavelength = get_wavelength_for_band(band_num)
    dataset_name = f"Albedo_Map_{wavelength}"
    
    # Open the HDF file
    hdf = SD(hdf_file, SDC.READ)
    
    # Get list of datasets to help debug
    datasets = hdf.datasets()
    
    # Try to find the dataset
    if dataset_name not in datasets:
        print(f"Available datasets in {os.path.basename(hdf_file)}:")
        for ds_name in datasets.keys():
            print(f"  - {ds_name}")
        hdf.end()
        raise ValueError(f"Dataset '{dataset_name}' not found in {hdf_file}")
    
    # Select and read the dataset
    sds = hdf.select(dataset_name)
    band_data = sds.get()
    
    # Clean up
    sds.endaccess()
    hdf.end()
    
    return band_data


def downsample_array(array, target_width=8192, nodata_value=32767):
    """
    Downsample a 2D array to target width while preserving aspect ratio.
    Uses averaging for downsampling to reduce aliasing, properly handling NoData values.
    
    Args:
        array: Input numpy array
        target_width: Target width in pixels (default: 8192)
        nodata_value: NoData value to exclude from averaging (default: 32767)
    
    Returns:
        Downsampled numpy array
    """
    input_height, input_width = array.shape
    
    # Calculate target height to preserve aspect ratio (2:1 for global data)
    target_height = target_width // 2
    
    # Calculate downsampling factors
    scale_y = input_height / target_height
    scale_x = input_width / target_width
    
    print(f"  Downsampling from {input_width}x{input_height} to {target_width}x{target_height}")
    print(f"  Scale factors: X={scale_x:.2f}, Y={scale_y:.2f}")
    
    # Count NoData pixels in input
    nodata_count = np.sum(array == nodata_value)
    total_pixels = array.size
    print(f"  NoData pixels in input: {nodata_count:,} ({100*nodata_count/total_pixels:.2f}%)")
    
    # Use GDAL for high-quality downsampling with averaging
    # Create in-memory dataset for input
    mem_driver = gdal.GetDriverByName('MEM')
    src_ds = mem_driver.Create('', input_width, input_height, 1, gdal.GDT_Float32)
    src_band = src_ds.GetRasterBand(1)
    src_band.WriteArray(array)
    
    # Set NoData value so GDAL excludes it from averaging
    src_band.SetNoDataValue(float(nodata_value))
    
    # Set geotransform for source (global extent, 30 arc-second resolution)
    # Input: 43200 x 21600 pixels = 360 degrees x 180 degrees
    src_pixel_width = 360.0 / input_width
    src_pixel_height = -180.0 / input_height
    src_geotransform = (
        -180.0,           # Top-left X
        src_pixel_width,  # Pixel width
        0,                # Rotation
        90.0,             # Top-left Y
        0,                # Rotation
        src_pixel_height  # Pixel height (negative)
    )
    src_ds.SetGeoTransform(src_geotransform)
    
    # Set projection for source (EPSG:4326)
    src_srs = osr.SpatialReference()
    src_srs.ImportFromEPSG(4326)
    src_ds.SetProjection(src_srs.ExportToWkt())
    
    # Create in-memory dataset for output
    dst_ds = mem_driver.Create('', target_width, target_height, 1, gdal.GDT_Float32)
    dst_band = dst_ds.GetRasterBand(1)
    
    # Set NoData value for destination
    dst_band.SetNoDataValue(float(nodata_value))
    
    # Set geotransform for destination (same extent, different resolution)
    dst_pixel_width = 360.0 / target_width
    dst_pixel_height = -180.0 / target_height
    dst_geotransform = (
        -180.0,           # Top-left X
        dst_pixel_width,  # Pixel width
        0,                # Rotation
        90.0,             # Top-left Y
        0,                # Rotation
        dst_pixel_height  # Pixel height (negative)
    )
    dst_ds.SetGeoTransform(dst_geotransform)
    
    # Set projection for destination (EPSG:4326)
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromEPSG(4326)
    dst_ds.SetProjection(dst_srs.ExportToWkt())
    
    # Perform resampling with average method
    # GDAL will automatically exclude NoData values from the averaging
    gdal.ReprojectImage(
        src_ds, dst_ds,
        src_srs.ExportToWkt(), dst_srs.ExportToWkt(),
        gdal.GRA_Average
    )
    
    # Read the downsampled data
    downsampled = dst_band.ReadAsArray()
    
    # Count NoData pixels in output
    nodata_count_out = np.sum(downsampled == nodata_value)
    total_pixels_out = downsampled.size
    print(f"  NoData pixels in output: {nodata_count_out:,} ({100*nodata_count_out/total_pixels_out:.2f}%)")
    
    # Clean up
    src_ds = None
    dst_ds = None
    
    return downsampled


def fill_ocean_with_blend(data_arrays, ocean_color, blend_radius=5, nodata_value=0):
    """
    Fill NoData (ocean) pixels with a specified color, blending smoothly from coastal data.
    
    Args:
        data_arrays: List of numpy arrays (one per band) - will be modified in place
        ocean_color: Tuple of (R, G, B) values for deep ocean (e.g., (1, 11, 20))
        blend_radius: Radius in pixels for blending transition (default: 5)
        nodata_value: NoData value to replace (default: 32767)
    
    Returns:
        List of modified numpy arrays with ocean filled
    """
    from scipy import ndimage
    
    print(f"\nFilling ocean with color {ocean_color} and {blend_radius}-pixel blend...")
    
    # Create a mask of valid (non-NoData) pixels (same for all bands)
    valid_mask = data_arrays[0] != nodata_value
    
    # Calculate distance from each NoData pixel to nearest valid pixel
    # Distance transform gives distance to nearest zero (False) pixel
    distance_from_data = ndimage.distance_transform_edt(~valid_mask)
    
    # Create blend weight: 1.0 at data edge, 0.0 at blend_radius distance
    # Pixels beyond blend_radius get pure ocean color
    blend_weight = np.clip(1.0 - (distance_from_data / blend_radius), 0.0, 1.0)
    
    # Process each band
    filled_arrays = []
    for i, (data_array, target_color) in enumerate(zip(data_arrays, ocean_color)):
        print(f"  Processing band {i+1} (target ocean value: {target_color})...")
        
        # Create output array
        filled = data_array.copy()
        
        # For NoData pixels, blend between nearest valid value and ocean color
        nodata_mask = data_array == nodata_value
        
        if np.any(nodata_mask):
            # Find nearest valid value for each NoData pixel using distance transform
            # We'll use nearest neighbor interpolation for the valid data
            indices = ndimage.distance_transform_edt(
                ~valid_mask, 
                return_distances=False, 
                return_indices=True
            )
            
            # Get the nearest valid values
            nearest_valid_values = data_array[tuple(indices)]
            
            # Blend between nearest valid value and ocean color based on distance
            # At the coast (distance=0): use nearest_valid_value
            # At distance=blend_radius: use ocean_color
            blended_values = (
                blend_weight * nearest_valid_values + 
                (1.0 - blend_weight) * target_color
            )
            
            # Apply blended values only to NoData pixels
            filled[nodata_mask] = blended_values[nodata_mask]
            
            pixels_filled = np.sum(nodata_mask)
            print(f"    Filled {pixels_filled:,} pixels")
        
        filled_arrays.append(filled)
    
    return filled_arrays


def create_geotiff(output_file, data_arrays, nodata_value=None, projection='EPSG:4326'):
    """
    Create a GeoTIFF file from numpy arrays with proper georeferencing.
    
    Args:
        output_file: Path to output GeoTIFF file
        data_arrays: List of numpy arrays (one per band)
        nodata_value: NoData value to set (default: 0)
        projection: Projection string (default: EPSG:4326)
    """
    # Get dimensions from first array
    rows, cols = data_arrays[0].shape
    num_bands = len(data_arrays)
    
    # Create the output driver
    driver = gdal.GetDriverByName('GTiff')
    
    # Create the output dataset
    # Using GDT_Float32 to preserve albedo values
    out_dataset = driver.Create(
        output_file,
        cols,
        rows,
        num_bands,
        gdal.GDT_Float32,
        options=['COMPRESS=LZW', 'TILED=YES', 'PREDICTOR=3']
    )
    
    if out_dataset is None:
        raise RuntimeError(f"Could not create output file: {output_file}")
    
    # Set geotransform for global extent
    # GeoTransform format: (top-left X, pixel width, rotation, top-left Y, rotation, pixel height)
    pixel_width = 360.0 / cols
    pixel_height = -180.0 / rows  # Negative because Y decreases
    
    # Top-left corner: -180 longitude, +90 latitude
    geotransform = (
        -180.0,           # Top-left X (longitude)
        pixel_width,      # Pixel width
        0,                # Rotation (0 for north-up)
        90.0,             # Top-left Y (latitude)
        0,                # Rotation (0 for north-up)
        pixel_height      # Pixel height (negative)
    )
    
    out_dataset.SetGeoTransform(geotransform)
    
    # Set projection to EPSG:4326 (WGS84)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    out_dataset.SetProjection(srs.ExportToWkt())
    
    # Write each band
    band_descriptions = ["Band 1 (Red - 0.659μm)", "Band 4 (NIR - 0.555μm)", "Band 3 (Blue - 0.47μm)"]
    
    for i, data_array in enumerate(data_arrays, start=1):
        band = out_dataset.GetRasterBand(i)
        band.WriteArray(data_array)
        
        # Set band description
        if i <= len(band_descriptions):
            band.SetDescription(band_descriptions[i-1])
        
        # Set NoData value
        if nodata_value is not None:
            band.SetNoDataValue(float(nodata_value))
        
        # Set statistics for better visualization
        band.ComputeStatistics(False)
        
        # Flush the band
        band.FlushCache()
        band = None
    
    # Close the dataset
    out_dataset.FlushCache()
    out_dataset = None
    
    print(f"\nSuccessfully created GeoTIFF: {output_file}")
    print(f"  Dimensions: {cols} x {rows}")
    print(f"  Bands: {num_bands}")
    print(f"  Resolution: {pixel_width:.6f} degrees ({pixel_width*3600:.1f} arc-seconds)")
    print(f"  Projection: EPSG:4326")
    if nodata_value is not None:
        print(f"  NoData value: {nodata_value}")


def main():
    """Main function to process MCD43GF HDF4 files to GeoTIFF."""
    
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Convert MODIS MCD43GF HDF4 gap-filled albedo data to GeoTIFF',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python hdf4_mcd43gf_to_geotiff.py 2021 315 output.tif --input-dir /data/modis
  
This will look for files:
  - MCD43GF_bsa_Band1_315_2021_V061.hdf
  - MCD43GF_bsa_Band3_315_2021_V061.hdf
  - MCD43GF_bsa_Band4_315_2021_V061.hdf
        """
    )
    
    parser.add_argument('year', type=int,
                        help='Year (4 digits, e.g., 2021)')
    parser.add_argument('day_of_year', type=int,
                        help='Day of year (1-366)')
    parser.add_argument('output_tif', metavar='OUTPUT_TIF',
                        help='Output GeoTIFF file path')
    parser.add_argument('--input-dir', default='.',
                        help='Directory containing input HDF files (default: current directory)')
    parser.add_argument('--width', type=int, default=8192,
                        help='Output width in pixels (default: 8192, height will be half)')
    parser.add_argument('--fill-ocean', action='store_true',
                        help='Fill ocean (NoData) areas with color and smooth blending')
    parser.add_argument('--ocean-color', type=str, default='1,11,20',
                        help='RGB color for ocean as comma-separated values (default: 1,11,20)')
    parser.add_argument('--blend-radius', type=int, default=5,
                        help='Radius in pixels for coastal blending (default: 5)')
    
    args = parser.parse_args()
    
    year = args.year
    day_of_year = args.day_of_year
    output_tif = args.output_tif
    input_dir = args.input_dir
    target_width = args.width
    fill_ocean = args.fill_ocean
    blend_radius = args.blend_radius
    
    # Parse ocean color
    try:
        ocean_color = tuple(map(float, args.ocean_color.split(',')))
        if len(ocean_color) != 3:
            raise ValueError("Ocean color must have exactly 3 values (R,G,B)")
    except Exception as e:
        print(f"Error parsing ocean color: {e}")
        print("Use format: --ocean-color R,G,B (e.g., --ocean-color 1,11,20)")
        sys.exit(1)
    
    # Validate inputs
    if day_of_year < 1 or day_of_year > 366:
        print("Error: Day of year must be between 1 and 366")
        sys.exit(1)
    
    if year < 2000 or year > 2100:
        print("Warning: Year seems unusual, but continuing...")
    
    print(f"Processing MCD43GF data for year {year}, day {day_of_year}")
    print(f"Input directory: {input_dir}")
    print(f"Target output resolution: {target_width}x{target_width//2}")
    if fill_ocean:
        print(f"Ocean filling: ENABLED (color: {ocean_color}, blend radius: {blend_radius} pixels)")
    else:
        print(f"Ocean filling: DISABLED (use --fill-ocean to enable)")

    gdal.UseExceptions()
    
    try:
        # Find and read Band 1 (Red)
        print("\n" + "="*60)
        print("Reading Band 1 (Red - 0.659μm)...")
        band1_file = find_hdf_file(input_dir, 1, day_of_year, year)
        print(f"  File: {os.path.basename(band1_file)}")
        band1_data = read_mcd43gf_band(band1_file, 1)
        print(f"  Original dimensions: {band1_data.shape}")
        print(f"  Value range: {band1_data.min():.4f} to {band1_data.max():.4f}")
        
        # Find and read Band 4 (NIR/Green)
        print("\n" + "="*60)
        print("Reading Band 4 (NIR - 0.555μm)...")
        band4_file = find_hdf_file(input_dir, 4, day_of_year, year)
        print(f"  File: {os.path.basename(band4_file)}")
        band4_data = read_mcd43gf_band(band4_file, 4)
        print(f"  Original dimensions: {band4_data.shape}")
        print(f"  Value range: {band4_data.min():.4f} to {band4_data.max():.4f}")
        
        # Find and read Band 3 (Blue)
        print("\n" + "="*60)
        print("Reading Band 3 (Blue - 0.47μm)...")
        band3_file = find_hdf_file(input_dir, 3, day_of_year, year)
        print(f"  File: {os.path.basename(band3_file)}")
        band3_data = read_mcd43gf_band(band3_file, 3)
        print(f"  Original dimensions: {band3_data.shape}")
        print(f"  Value range: {band3_data.min():.4f} to {band3_data.max():.4f}")
        
        # Verify dimensions
        expected_shape = (21600, 43200)
        for band_name, band_data in [("Band 1", band1_data), ("Band 4", band4_data), ("Band 3", band3_data)]:
            if band_data.shape != expected_shape:
                print(f"Warning: {band_name} has unexpected shape {band_data.shape}, expected {expected_shape}")
        
        # Downsample the data
        print("\n" + "="*60)
        print("Downsampling data...")
        print("Band 1 (Red):")
        band1_downsampled = downsample_array(band1_data, target_width)
        
        print("Band 4 (NIR):")
        band4_downsampled = downsample_array(band4_data, target_width)
        
        print("Band 3 (Blue):")
        band3_downsampled = downsample_array(band3_data, target_width)
        
        # Apply ocean filling if requested
        if fill_ocean:
            print("\n" + "="*60)
            downsampled_arrays = [band1_downsampled, band4_downsampled, band3_downsampled]
            downsampled_arrays = fill_ocean_with_blend(
                downsampled_arrays, 
                ocean_color, 
                blend_radius=blend_radius
            )
            band1_downsampled, band4_downsampled, band3_downsampled = downsampled_arrays
        
        # Create the GeoTIFF with bands in RGB order (Band1, Band4, Band3)
        print("\n" + "="*60)
        print("Creating GeoTIFF...")
        # If ocean is filled, don't set NoData value (all pixels have valid data)
        if fill_ocean:
            create_geotiff(output_tif, [band1_downsampled, band4_downsampled, band3_downsampled])
        else:
            create_geotiff(output_tif, [band1_downsampled, band4_downsampled, band3_downsampled], nodata_value=0)
        print("\n" + "="*60)
        print("Conversion completed successfully!")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nMake sure all three required files are present:")
        print(f"  - MCD43GF_bsa_Band1_{day_of_year:03d}_{year}_V061.hdf")
        print(f"  - MCD43GF_bsa_Band3_{day_of_year:03d}_{year}_V061.hdf")
        print(f"  - MCD43GF_bsa_Band4_{day_of_year:03d}_{year}_V061.hdf")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nError during conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()