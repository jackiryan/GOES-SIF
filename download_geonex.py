#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
import time


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
    parser.add_argument('--dry-run', action='store_true',
                       help='Show URLs that would be downloaded without actually downloading')
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


def list_directory_files(url, timeout=30):
    """List files in a directory URL."""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        files = []
        
        for link in soup.find_all('a'):
            href = link.get('href')
            if href and not href.startswith(('?', '/', '../')):
                files.append(href)
        
        return files
    except requests.RequestException as e:
        print(f"Error listing directory {url}: {e}", file=sys.stderr)
        return []


def download_file(url, output_path, timeout=30, retry=3):
    """Download a file with retry logic."""
    attempt = 0
    while attempt < retry:
        try:
            print(f"Downloading: {os.path.basename(url)}")
            response = requests.get(url, timeout=timeout, stream=True)
            response.raise_for_status()
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Write file in chunks
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            print(f"  ✓ Saved to: {output_path}")
            return True
            
        except requests.RequestException as e:
            attempt += 1
            if attempt < retry:
                print(f"  ✗ Download failed (attempt {attempt}/{retry}): {e}")
                time.sleep(2)  # Wait before retry
            else:
                print(f"  ✗ Failed after {retry} attempts: {e}", file=sys.stderr)
                return False


def find_brdf_file(base_url, grid_cell, year, doy, timeout=30):
    """Find BRDF file by trying common time codes directly."""
    brdf_dir = f"{base_url}/MAIAC/{grid_cell}/{year}/"
    print(f"\nSearching for BRDF file in: {brdf_dir}")
    
    date_str = f"{year}{doy}"
    
    # Generate time codes every 10 minutes from 0000 to 2350
    time_codes = []
    for hour in range(24):
        for minute in range(0, 60, 10):
            time_codes.append(f"{hour:02d}{minute:02d}")
    
    common_times = time_codes
    
    for time_str in common_times:
        filename = f"GO16_ABI12C_{date_str}{time_str}_GLBG_{grid_cell}_02.hdf"
        url = f"{brdf_dir}{filename}"
        
        try:
            print(f"  Trying: {filename}")
            response = requests.head(url, timeout=timeout)
            if response.status_code == 200:
                print(f"  ✓ Found BRDF file: {filename}")
                return url
        except requests.RequestException as e:
            # Continue trying other time codes
            continue
    
    # If no common times work, try a broader search with HTML scraping as fallback
    print("  Common time codes failed, trying HTML directory listing...")
    try:
        files = list_directory_files(brdf_dir, timeout)
        if files:
            pattern = f"GO16_ABI12C_{date_str}"
            matching_files = [f for f in files if f.startswith(pattern) and f.endswith('.hdf')]
            
            if matching_files:
                print(f"  ✓ Found via directory listing: {matching_files[0]}")
                return f"{brdf_dir}{matching_files[0]}"
    except Exception as e:
        print(f"  Directory listing also failed: {e}")
    
    print(f"  ✗ No BRDF file found for {date_str}")
    return None


def generate_reflectance_urls(base_url, grid_cell, year, doy):
    """Generate surface reflectance URLs for each hour."""
    urls = []
    date_str = f"{year}{doy}"
    refl_base = f"{base_url}/MAIAC/{grid_cell}/{year}/{doy}/"
    
    # Generate URLs for each hour (00:00 to 23:00)
    for hour in range(24):
        time_str = f"{hour:02d}00"
        filename = f"GO16_ABI12B_{date_str}{time_str}_GLBG_{grid_cell}_02.hdf"
        urls.append(f"{refl_base}{filename}")
    
    return urls


def main():
    args = parse_arguments()
    
    # Base URL
    base_url = "https://data.nas.nasa.gov/geonex/GOES16/GEONEX-L2"
    
    # Create output directory structure
    output_base = Path(args.output_dir) / args.grid_cell / str(args.year) / args.doy
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be downloaded\n")
    
    # Find and download BRDF file
    print("=" * 60)
    print("BRDF Data Product")
    print("=" * 60)
    
    brdf_url = find_brdf_file(base_url, args.grid_cell, args.year, args.doy, args.timeout)
    
    if brdf_url:
        print(f"Found BRDF file: {brdf_url}")
        if not args.dry_run:
            output_path = output_base.parent / os.path.basename(brdf_url)
            download_file(brdf_url, output_path, args.timeout, args.retry)
    else:
        print(f"Warning: No BRDF file found for {args.year} DOY {args.doy}", file=sys.stderr)
    
    # Generate and download surface reflectance files
    print("\n" + "=" * 60)
    print("Surface Reflectance Products (Hourly)")
    print("=" * 60)
    
    reflectance_urls = generate_reflectance_urls(base_url, args.grid_cell, args.year, args.doy)
    
    successful = 0
    failed = 0
    
    for url in reflectance_urls:
        if args.dry_run:
            print(f"Would download: {url}")
        else:
            output_path = output_base / os.path.basename(url)
            if download_file(url, output_path, args.timeout, args.retry):
                successful += 1
            else:
                failed += 1
    
    # Summary
    if not args.dry_run:
        print("\n" + "=" * 60)
        print("Download Summary")
        print("=" * 60)
        print(f"Successfully downloaded: {successful} files")
        if failed > 0:
            print(f"Failed downloads: {failed} files", file=sys.stderr)
        print(f"Output directory: {output_base}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())