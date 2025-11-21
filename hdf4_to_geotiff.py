#!/usr/bin/env python3
"""
Convert MODIS MCD43C4 HDF4 BRDF data to GeoTIFF format.

This script reads HDF4 files containing Nadir BRDF-Adjusted Reflectance data
from the MODIS instrument (Terra and Aqua combined) on the Climate Modeling Grid
at 0.05 degree resolution, and exports selected bands to a georeferenced GeoTIFF.

Usage:
    python hdf4_to_geotiff.py input.hdf output.tif
"""
import argparse
import numpy as np
import os
import sys

# Set GDAL plugin path for conda environment
if 'CONDA_PREFIX' in os.environ:
    gdal_plugins = os.path.join(os.environ['CONDA_PREFIX'], 'Library', 'lib', 'gdalplugins')
    if os.path.exists(gdal_plugins):
        os.environ['GDAL_DRIVER_PATH'] = gdal_plugins

from osgeo import gdal, osr # type: ignore [import-untyped]


def read_hdf4_band(hdf_file, band_num):
    """
    Read a specific band from the HDF4 file.
    
    Args:
        hdf_file: GDAL dataset object
        band_num: Band number (1-7)
    
    Returns:
        numpy array containing the band data
    """
    subdataset_name = f'HDF4_EOS:EOS_GRID:"{hdf_file.GetDescription()}":MCD_CMG_BRDF_0.05Deg:Nadir_Reflectance_Band{band_num}'
    
    # Open the subdataset
    subdataset = gdal.Open(subdataset_name, gdal.GA_ReadOnly)
    if subdataset is None:
        raise ValueError(f"Could not open subdataset for Band {band_num}")
    
    # Read the data
    band_data = subdataset.ReadAsArray()
    subdataset = None
    
    return band_data


def read_quality_flag(hdf_file):
    """
    Read the Albedo_Quality flag from the HDF4 file.
    
    Args:
        hdf_file: GDAL dataset object
    
    Returns:
        numpy array containing the quality flag data
    """
    subdataset_name = f'HDF4_EOS:EOS_GRID:"{hdf_file.GetDescription()}":MCD_CMG_BRDF_0.05Deg:Albedo_Quality'
    
    # Open the subdataset
    subdataset = gdal.Open(subdataset_name, gdal.GA_ReadOnly)
    if subdataset is None:
        raise ValueError("Could not open Albedo_Quality subdataset")
    
    # Read the data
    quality_data = subdataset.ReadAsArray()
    subdataset = None
    
    return quality_data


def apply_quality_filter(data_arrays, quality_flag, threshold, nodata_value=32767):
    """
    Apply quality filtering to data arrays based on quality flag threshold.
    
    Args:
        data_arrays: List of numpy arrays (one per band)
        quality_flag: Numpy array containing quality flags
        threshold: Maximum acceptable quality value (inclusive)
        nodata_value: Value to set for filtered pixels
    
    Returns:
        List of filtered numpy arrays
    """
    # Create mask where quality is good (quality <= threshold)
    quality_mask = quality_flag <= threshold
    
    filtered_arrays = []
    for data_array in data_arrays:
        # Create a copy to avoid modifying original
        filtered_data = data_array.copy()
        
        # Set pixels to nodata where quality is poor
        filtered_data[~quality_mask] = nodata_value
        
        filtered_arrays.append(filtered_data)
    
    return filtered_arrays


def create_geotiff(output_file, data_arrays, projection='EPSG:4326'):
    """
    Create a GeoTIFF file from numpy arrays with proper georeferencing.
    
    Args:
        output_file: Path to output GeoTIFF file
        data_arrays: List of numpy arrays (one per band)
        projection: Projection string (default: EPSG:4326)
    """
    # Get dimensions from first array
    rows, cols = data_arrays[0].shape
    num_bands = len(data_arrays)
    
    # Create the output driver
    driver = gdal.GetDriverByName('GTiff')
    
    # Create the output dataset
    # Using GDT_Float32 to preserve reflectance values
    out_dataset = driver.Create(
        output_file,
        cols,
        rows,
        num_bands,
        gdal.GDT_Float32,
        options=['COMPRESS=LZW', 'TILED=YES']
    )
    
    if out_dataset is None:
        raise RuntimeError(f"Could not create output file: {output_file}")
    
    # Set geotransform for global extent at 0.05 degree resolution
    # GeoTransform format: (top-left X, pixel width, rotation, top-left Y, rotation, pixel height)
    # CMG 0.05 degree grid: -180 to 180 longitude, -90 to 90 latitude
    # Pixel size: 360/7200 = 0.05 degrees in X, 180/3600 = 0.05 degrees in Y
    pixel_width = 360.0 / cols  # Should be 0.05
    pixel_height = -180.0 / rows  # Should be -0.05 (negative because Y decreases)
    
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
    for i, data_array in enumerate(data_arrays, start=1):
        band = out_dataset.GetRasterBand(i)
        band.WriteArray(data_array)
        
        # Set band description
        band_names = {1: "Band 1 (Red)", 2: "Band 4 (NIR)", 3: "Band 3 (Blue)"}
        if i in band_names:
            band.SetDescription(band_names[i])
        
        # Set NoData value if needed (common MODIS fill value is 32767)
        band.SetNoDataValue(32767)
        
        # Flush the band
        band.FlushCache()
        band = None
    
    # Close the dataset
    out_dataset.FlushCache()
    out_dataset = None
    
    print(f"Successfully created GeoTIFF: {output_file}")
    print(f"  Dimensions: {cols} x {rows}")
    print(f"  Bands: {num_bands}")
    print(f"  Resolution: {pixel_width} degrees")
    print(f"  Projection: EPSG:4326")


def main():
    """Main function to process HDF4 to GeoTIFF conversion."""
    
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Convert MODIS MCD43C4 HDF4 BRDF data to GeoTIFF with quality filtering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quality threshold values:
  0 = best quality only
  1 = best and good quality
  2 = best, good, and moderate quality (default)
  255 = no filtering (accept all quality levels)
        """
    )
    
    parser.add_argument('input_hdf', metavar='INPUT_HDF',
                        help='Input HDF4 file path')
    parser.add_argument('output_tif', metavar='OUTPUT_TIF',
                        help='Output GeoTIFF file path')
    parser.add_argument('-q', '--quality-threshold', type=int, default=2,
                        metavar='N',
                        help='Maximum quality flag value to accept (0=best, default=2)')
    
    args = parser.parse_args()
    
    input_hdf = args.input_hdf
    output_tif = args.output_tif
    quality_threshold = args.quality_threshold
    
    print(f"Processing: {input_hdf}")
    print(f"Quality threshold: {quality_threshold} (pixels with quality > {quality_threshold} will be masked)")

    gdal.UseExceptions()
    
    # Open the HDF4 file
    hdf_dataset = gdal.Open(input_hdf, gdal.GA_ReadOnly)
    if hdf_dataset is None:
        print(f"Error: Could not open HDF4 file: {input_hdf}")
        sys.exit(1)
    
    # Print available subdatasets for reference
    subdatasets = hdf_dataset.GetSubDatasets()
    print(f"\nFound {len(subdatasets)} subdatasets in the HDF4 file")
    
    try:
        # Read the required bands (1, 4, 3 in that order for RGB mapping)
        print("\nReading bands...")
        band1_data = read_hdf4_band(hdf_dataset, 1)
        print(f"  Band 1 (Red): {band1_data.shape}")
        
        band4_data = read_hdf4_band(hdf_dataset, 4)
        print(f"  Band 4 (NIR): {band4_data.shape}")
        
        band3_data = read_hdf4_band(hdf_dataset, 3)
        print(f"  Band 3 (Blue): {band3_data.shape}")
        
        # Read quality flag
        print("\nReading quality flag...")
        quality_flag = read_quality_flag(hdf_dataset)
        print(f"  Albedo_Quality: {quality_flag.shape}")
        print(f"  Quality values range: {quality_flag.min()} to {quality_flag.max()}")
        
        # Verify dimensions
        expected_shape = (3600, 7200)
        for band_name, band_data in [("Band 1", band1_data), ("Band 4", band4_data), 
                                      ("Band 3", band3_data), ("Quality", quality_flag)]:
            if band_data.shape != expected_shape:
                print(f"Warning: {band_name} has unexpected shape {band_data.shape}, expected {expected_shape}")
        
        # Apply quality filtering
        print(f"\nApplying quality filter (threshold <= {quality_threshold})...")
        data_arrays = [band1_data, band4_data, band3_data]
        filtered_arrays = apply_quality_filter(data_arrays, quality_flag, quality_threshold)
        
        # Calculate filtering statistics
        total_pixels = quality_flag.size
        good_pixels = np.sum(quality_flag <= quality_threshold)
        filtered_pixels = total_pixels - good_pixels
        print(f"  Total pixels: {total_pixels:,}")
        print(f"  Good quality pixels (kept): {good_pixels:,} ({100*good_pixels/total_pixels:.1f}%)")
        print(f"  Poor quality pixels (masked): {filtered_pixels:,} ({100*filtered_pixels/total_pixels:.1f}%)")
        
        # Create the GeoTIFF with filtered bands
        print("\nCreating GeoTIFF...")
        create_geotiff(output_tif, filtered_arrays)
        
        print("\nConversion completed successfully!")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        hdf_dataset = None


if __name__ == "__main__":
    main()
