#!/usr/bin/env python3
import argparse
from glob import glob
import shlex
import subprocess as sp
import sys


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Download NASA GEONEX remote sensing data by date and grid cell',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s 2021 067 h15v02
  %(prog)s 2021 067 h15v02 -o ./downloads
  %(prog)s 2021 067 h15v02 --dry-run
        """
    )
    
    parser.add_argument('year', type=int, help='Year (e.g., 2021)')
    parser.add_argument('doy', type=str, help='Day of year (e.g., 067)')
    parser.add_argument('grid_cell', type=str, 
                       help='Grid cell identifier (e.g., h15v02)')
    parser.add_argument('-o', '--output-dir', type=str, default='.',
                       help='Output directory for downloaded files (default: current directory)')
    parser.add_argument('--timeout', type=int, default=30,
                       help='Request timeout in seconds (default: 30)')
    parser.add_argument('--retry', type=int, default=3,
                       help='Number of retry attempts for failed downloads (default: 3)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.year < 2000 or args.year > 2100:
        parser.error(f"Invalid year: {args.year}")
    
    # Ensure day of year is zero-padded
    try:
        doy_int = int(args.doy)
        if doy_int < 1 or doy_int > 366:
            parser.error(f"Day of year must be between 1 and 366: {args.doy}")
        args.doy = f"{doy_int:03d}"
    except ValueError:
        parser.error(f"Invalid day of year: {args.doy}")
    
    # Validate grid cell format
    if not args.grid_cell.startswith('h') or 'v' not in args.grid_cell:
        parser.error(f"Invalid grid cell format: {args.grid_cell}. Expected format: hXXvYY")
    
    return args


def main():
    args = parse_arguments()
    download_cmd = f"python3 download_geonex.py {args.year} {args.doy} {args.grid_cell} --timeout {args.timeout} --retry {args.retry}"
    if args.output_dir:
        download_cmd += f" -o {args.output_dir}"
    
    response = sp.check_output(shlex.split(download_cmd))
    print(response.decode())

    data_dir = f"{args.output_dir}/{args.grid_cell}/{args.year}"
    refl_dir = f"{data_dir}/{args.doy}"
    brdf_glob = glob(f"{data_dir}/GO16_ABI12C_{args.year}{int(args.doy):03d}*_GLBG_{args.grid_cell}_02.hdf")
    if len(brdf_glob) > 0:
        brdf_file = brdf_glob[0]
    else:
        raise FileNotFoundError("could not find BRDF file.")
    refl_files = glob(f"{refl_dir}/*.hdf")
    if len(refl_files) < 1:
        raise FileNotFoundError("could not find surface reflectance files.")
    
    for rf in refl_files:
        invert_cmd = f"python3 invert_brdf.py {brdf_file} {rf}"
        sp.check_output(shlex.split(invert_cmd))

if __name__ == "__main__":
    sys.exit(main())