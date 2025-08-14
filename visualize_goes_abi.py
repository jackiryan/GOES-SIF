#!/usr/bin/env python3
import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pyhdf.SD import SD, SDC
from pathlib import Path

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
  
    data_rgb = np.zeros((height, width, 3), 'u1')
  
    # Process red channel
    x = np.clip(data_r, 1, max_in).astype('f4')  # 1 will be zero in logarithm
    x = (np.log(x) - np.log(ref)) * scale + offset  
    x = np.clip(x, 0, max_out)
    data_rgb[:, :, 0] = x.reshape(height, width).astype('u1')
   
    # Process green channel
    x = np.clip(data_g, 1, max_in).astype('f4')
    x = (np.log(x) - np.log(ref)) * scale + offset
    x = np.clip(x, 0, max_out)
    data_rgb[:, :, 1] = x.reshape(height, width).astype('u1')
   
    # Process blue channel
    x = np.clip(data_b, 1, max_in).astype('f4')
    x = (np.log(x) - np.log(ref)) * scale + offset
    x = np.clip(x, 0, max_out)
    data_rgb[:, :, 2] = x.reshape(height, width).astype('u1')

    return data_rgb

def list_datasets(hdf_file):
    """List all datasets in an HDF file"""
    f = SD(hdf_file, SDC.READ)
    datasets = f.datasets()
    
    print(f"\nDatasets in {hdf_file}:")
    print("-" * 50)
    for name, info in datasets.items():
        print(f"Dataset: {name}")
        print(f"  - Dimensions: {info[1]}")
        print(f"  - Type: {info[0]}")
        
        # List attributes if available
        try:
            sds = f.select(name)
            attrs = sds.attributes()
            if attrs:
                print("  - Attributes:")
                for attr_name, attr_value in attrs.items():
                    print(f"    * {attr_name}: {attr_value}")
            sds.endaccess()
        except Exception as e:
            print(f"  - Error reading attributes: {e}")
        print()
    
    print(f"Global Attributes:")
    print("-" * 50)
    try:
        for attr_name, attr_value in f.attributes().items():
            print(f"  - {attr_name}: {attr_value}")
    except Exception as e:
        print(f"  Error reading global attributes: {e}")
    
    f.end()

def process_hdf_file(hdf_file, output_file, red_ds=None, green_ds=None, blue_ds=None, 
                    qa_ds=None, interactive=False, max_scale=6000, dpi=100):
    """
    Process an HDF file and generate visualization
    
    Parameters:
    hdf_file: Path to the input HDF file
    output_file: Path to save the output image
    red_ds, green_ds, blue_ds: Dataset names for RGB channels
    qa_ds: Dataset name for quality assurance data
    interactive: Whether to display an interactive plot
    max_scale: Maximum scale value for normalization
    dpi: DPI for the output image
    """
    try:
        f = SD(hdf_file, SDC.READ)
        datasets = f.datasets()
        dataset_names = list(datasets.keys())
        
        # Auto-detect datasets if not specified
        if red_ds is None and green_ds is None and blue_ds is None:
            # Try to find reflectance data
            if "sur_refl1km" in dataset_names:
                print("Found standard reflectance dataset 'sur_refl1km'")
                reflectance_ds = "sur_refl1km"
            elif any("refl" in ds.lower() for ds in dataset_names):
                # Try to find any dataset with 'refl' in the name
                reflectance_candidates = [ds for ds in dataset_names if "refl" in ds.lower()]
                reflectance_ds = reflectance_candidates[0]
                print(f"Auto-detected reflectance dataset: {reflectance_ds}")
            else:
                # If no reflectance datasets found, try to use the first 3 datasets
                if len(dataset_names) >= 3:
                    print("No reflectance datasets found. Using the first 3 datasets as RGB channels.")
                    red_ds, green_ds, blue_ds = dataset_names[:3]
                else:
                    raise ValueError("Could not auto-detect suitable datasets for RGB channels")
        
        # Auto-detect QA dataset if not specified
        if qa_ds is None and "Status_QA" in dataset_names:
            qa_ds = "Status_QA"
            print(f"Auto-detected QA dataset: {qa_ds}")
        
        # Read reflectance data if specified
        if red_ds is None and 'reflectance_ds' in locals():
            # Read multi-band reflectance dataset
            sds = f.select(reflectance_ds)
            data = sds.get().astype("float32")
            sds.endaccess()
            
            # Get dimensions
            if len(data.shape) == 3 and data.shape[0] >= 3:
                # Assuming band, height, width format
                height, width = data.shape[1], data.shape[2]
                
                # Extract bands
                data_r = data[1, :, :].astype('u2')  # Band 2 (usually red)
                data_b = data[0, :, :].astype('u2')  # Band 1 (usually blue)
                
                # Create green band using a weighted combination
                if data.shape[0] >= 3:
                    data_g = (0.48358168*data[1, :, :] + 
                              0.45706946*data[0, :, :] + 
                              0.06038137*data[2, :, :]).astype('u2')
                else:
                    # If third band is not available, use average of first two
                    data_g = ((data[0, :, :] + data[1, :, :]) / 2).astype('u2')
            else:
                raise ValueError(f"Unexpected shape for reflectance data: {data.shape}")
        else:
            # Read individual datasets for RGB channels
            if not all([red_ds, green_ds, blue_ds]):
                raise ValueError("If not using a multi-band reflectance dataset, you must specify red_ds, green_ds, and blue_ds")
            
            # Read red channel
            sds = f.select(red_ds)
            data_r = sds.get().astype("float32")
            height, width = data_r.shape
            sds.endaccess()
            
            # Read green channel
            sds = f.select(green_ds)
            data_g = sds.get().astype("float32")
            sds.endaccess()
            
            # Read blue channel
            sds = f.select(blue_ds)
            data_b = sds.get().astype("float32")
            sds.endaccess()
        
        # Read QA data if specified
        if qa_ds:
            sds = f.select(qa_ds)
            QA = sds.get().astype("u2")
            sds.endaccess()
            
            # Process QA mask - this is specific to the original format
            # May need adjustment for different datasets
            mask = np.bitwise_and(QA, 0xF000)
            mask = np.where(mask==0x4000, 3, mask)
            mask = np.where(mask==0x8000, 6, mask)
            mask = np.where(mask==0xC000, 9, mask)
        else:
            mask = None
        
        f.end()
        
        # Scale RGB values for visualization
        data_rgb = scale_rgb(data_r, data_g, data_b, width, height, max_scale)
        
        # Create output directory if it doesn't exist
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create figure for visualization
        plt.switch_backend("Agg")  # Use non-interactive backend
        
        if mask is not None:
            # Create figure with two subplots
            fig = plt.figure(figsize=(10, 8), frameon=False)
            
            # Left subplot for RGB image
            ax = plt.axes([0, 0, 1, 1])
            ax.set_axis_off()
            ax.imshow(data_rgb)
            #ax.text(0.05*width, 0.95*height, Path(hdf_file).stem, 
            #        fontsize=14, color='#335533')
            
            # Right subplot for mask visualization
            '''
            ax2 = plt.axes([0.5, 0, 0.5, 1])
            ax2.set_axis_off()
            im = ax2.imshow(mask, vmin=-0.5, vmax=11.5, cmap='Paired')
            
            # Add colorbar for mask
            cax = plt.axes([0.65, 0.1, 0.2, 0.03])
            plt.colorbar(im, cax=cax, orientation='horizontal')
            '''
        else:
            # Create figure with just the RGB image
            fig = plt.figure(figsize=(10, 8), frameon=False)
            ax = plt.axes([0, 0, 1, 1])
            ax.set_axis_off()
            ax.imshow(data_rgb)
            ax.text(0.05*width, 0.95*height, Path(hdf_file).stem, 
                    fontsize=14, color='#335533')
        
        # Save figure
        fig.savefig(output_file, dpi=dpi)
        print(f"Image saved to: {output_file}")
        
        # Interactive mode
        if interactive:
            def click(event):
                if event.ydata is not None and event.xdata is not None:
                    y, x = int(event.ydata), int(event.xdata)
                    if 0 <= y < height and 0 <= x < width:
                        print(f"Position: y={y}, x={x}")
                        print(f"RGB value: {data_rgb[y, x, :]}")
                        if mask is not None and 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                            print(f"Mask value: {mask[y, x]}")
            
            plt.switch_backend("TkAgg")  # Switch to interactive backend
            plt.connect('button_press_event', click)
            plt.show()
        
        return True
        
    except Exception as e:
        print(f"Error processing HDF file: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='Process and visualize HDF files')
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List datasets in an HDF file')
    list_parser.add_argument('input', help='Input HDF file')
    
    # Visualize command
    vis_parser = subparsers.add_parser('visualize', help='Visualize an HDF file')
    vis_parser.add_argument('input', help='Input HDF file')
    vis_parser.add_argument('-o', '--output', help='Output image file (PNG)', 
                           default=None)
    vis_parser.add_argument('-r', '--red', help='Dataset name for red channel')
    vis_parser.add_argument('-g', '--green', help='Dataset name for green channel')
    vis_parser.add_argument('-b', '--blue', help='Dataset name for blue channel')
    vis_parser.add_argument('-q', '--qa', help='Dataset name for QA data')
    vis_parser.add_argument('-m', '--max-scale', type=int, default=6000,
                           help='Maximum scale value for normalization (default: 6000)')
    vis_parser.add_argument('-d', '--dpi', type=int, default=100,
                           help='DPI for output image (default: 100)')
    vis_parser.add_argument('-i', '--interactive', action='store_true',
                           help='Show interactive plot')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    if args.command == 'list':
        if not os.path.exists(args.input):
            print(f"Error: File not found: {args.input}")
            return 1
        list_datasets(args.input)
        return 0
        
    elif args.command == 'visualize':
        if not os.path.exists(args.input):
            print(f"Error: File not found: {args.input}")
            return 1
            
        # Set default output filename if not specified
        if args.output is None:
            input_path = Path(args.input)
            args.output = f"{input_path.stem}.png"
            
        success = process_hdf_file(
            args.input, 
            args.output,
            red_ds=args.red,
            green_ds=args.green,
            blue_ds=args.blue,
            qa_ds=args.qa,
            interactive=args.interactive,
            max_scale=args.max_scale,
            dpi=args.dpi
        )
        
        return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())