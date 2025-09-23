#!/usr/bin/env python3
"""
GOES-R Data Retrieval Script for LVMP and LVTP Products

This script downloads GOES-R Legacy Atmospheric Profile data from NOAA's AWS S3 bucket
for vapor pressure deficit calculations and atmospheric analysis.

Usage:
    python goes_retrieval.py --date 2020-06-15 --products LVMPC LVTPC --output-dir ./data
    python goes_retrieval.py --date 2020-06-15 --hour 18 --products LVMPC --output-dir ./data
    python goes_retrieval.py --year 2020 --doy 167 --products LVMPC LVTPC --output-dir ./data

Requirements:
    pip install boto3 tqdm argparse
"""

import argparse
import boto3
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import re
from typing import List, Optional
import concurrent.futures
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GOESDataRetriever:
    """
    Class for retrieving GOES-R data from NOAA's AWS S3 bucket
    """
    
    def __init__(self, satellite='goes16'):
        """
        Initialize the GOES data retriever
        
        Args:
            satellite (str): Satellite name ('goes16', 'goes17', 'goes18')
        """
        self.satellite = satellite
        self.bucket_name = f'noaa-{satellite}'
        
        # Initialize S3 client with unsigned requests
        self.s3_client = boto3.client(
            's3',
            config=Config(signature_version=UNSIGNED)
        )
        
        # Available CONUS products for atmospheric profiling
        self.available_products = {
            'LVMPC': 'Legacy Vertical Moisture Profile - CONUS',
            'LVTPC': 'Legacy Vertical Temperature Profile - CONUS',
            'DSIC': 'Derived Stability Indices - CONUS',
            'TPWC': 'Total Precipitable Water - CONUS'
        }
    
    def date_to_doy(self, date_str: str) -> tuple:
        """
        Convert date string to year and day of year
        
        Args:
            date_str (str): Date in format 'YYYY-MM-DD'
            
        Returns:
            tuple: (year, day_of_year)
        """
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        year = date_obj.year
        doy = date_obj.timetuple().tm_yday
        return year, doy
    
    def doy_to_date(self, year: int, doy: int) -> str:
        """
        Convert year and day of year to date string
        
        Args:
            year (int): Year
            doy (int): Day of year
            
        Returns:
            str: Date in format 'YYYY-MM-DD'
        """
        date_obj = datetime(year, 1, 1) + timedelta(days=doy - 1)
        return date_obj.strftime('%Y-%m-%d')
    
    def list_files_for_day(self, product: str, year: int, doy: int, hour: Optional[int] = None) -> List[str]:
        """
        List all files for a given product, year, day of year, and optionally hour
        
        Args:
            product (str): Product code (e.g., 'LVMPC', 'LVTPC')
            year (int): Year
            doy (int): Day of year
            hour (int, optional): Hour (0-23). If None, all hours are included
            
        Returns:
            List[str]: List of S3 object keys
        """
        if product not in self.available_products:
            raise ValueError(f"Product {product} not available. Choose from: {list(self.available_products.keys())}")
        
        # Construct base prefix
        base_prefix = f'ABI-L2-{product}/{year:04d}/{doy:03d}/'
        
        if hour is not None:
            # List files for specific hour
            hour_prefix = f'{base_prefix}{hour:02d}/'
            return self._list_s3_objects(hour_prefix)
        else:
            # List files for all hours of the day
            all_files = []
            for h in range(24):
                hour_prefix = f'{base_prefix}{h:02d}/'
                hour_files = self._list_s3_objects(hour_prefix)
                all_files.extend(hour_files)
            return all_files
    
    def _list_s3_objects(self, prefix: str) -> List[str]:
        """
        List S3 objects with given prefix
        
        Args:
            prefix (str): S3 prefix
            
        Returns:
            List[str]: List of object keys
        """
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)
            
            files = []
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        if obj['Key'].endswith('.nc'):
                            files.append(obj['Key'])
            
            return files
        except Exception as e:
            logger.warning(f"Error listing objects with prefix {prefix}: {e}")
            return []
    
    def download_file(self, s3_key: str, local_path: str, overwrite: bool = False) -> bool:
        """
        Download a single file from S3
        
        Args:
            s3_key (str): S3 object key
            local_path (str): Local file path
            overwrite (bool): Whether to overwrite existing files
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Check if file already exists
        if os.path.exists(local_path) and not overwrite:
            file_size = os.path.getsize(local_path)
            if file_size > 0:
                logger.debug(f"File already exists: {local_path}")
                return True
        
        try:
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            logger.debug(f"Downloaded: {s3_key} -> {local_path}")
            return True
        except Exception as e:
            logger.error(f"Error downloading {s3_key}: {e}")
            return False
    
    def sync_day(self, products: List[str], year: int, doy: int, output_dir: str, 
                 hour: Optional[int] = None, overwrite: bool = False, max_workers: int = 4) -> dict:
        """
        Sync all data for given products and day
        
        Args:
            products (List[str]): List of product codes
            year (int): Year
            doy (int): Day of year
            output_dir (str): Output directory
            hour (int, optional): Specific hour to download
            overwrite (bool): Whether to overwrite existing files
            max_workers (int): Number of parallel download threads
            
        Returns:
            dict: Download statistics
        """
        stats = {
            'total_files': 0,
            'downloaded': 0,
            'skipped': 0,
            'failed': 0,
            'products': {}
        }
        
        for product in products:
            logger.info(f"Processing product: {product} ({self.available_products[product]})")
            
            # List files for this product
            files = self.list_files_for_day(product, year, doy, hour)
            
            if not files:
                logger.warning(f"No files found for {product} on {year:04d}-{doy:03d}")
                continue
            
            logger.info(f"Found {len(files)} files for {product}")
            
            # Prepare download tasks
            download_tasks = []
            for s3_key in files:
                # Extract filename from S3 key
                filename = os.path.basename(s3_key)
                
                # Create organized directory structure: output_dir/product/year/doy/hour/
                match = re.search(r'/(\d{2})/', s3_key)  # Extract hour from path
                file_hour = match.group(1) if match else '00'
                
                local_dir = os.path.join(output_dir, product, f'{year:04d}', f'{doy:03d}', file_hour)
                local_path = os.path.join(local_dir, filename)
                
                download_tasks.append((s3_key, local_path))
            
            # Download files in parallel
            product_stats = {'downloaded': 0, 'skipped': 0, 'failed': 0}
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all download tasks
                future_to_task = {
                    executor.submit(self.download_file, s3_key, local_path, overwrite): (s3_key, local_path)
                    for s3_key, local_path in download_tasks
                }
                
                # Process completed downloads with progress bar
                with tqdm(total=len(download_tasks), desc=f"Downloading {product}", unit="files") as pbar:
                    for future in concurrent.futures.as_completed(future_to_task):
                        s3_key, local_path = future_to_task[future]
                        try:
                            success = future.result()
                            if success:
                                if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                                    product_stats['downloaded'] += 1
                                else:
                                    product_stats['skipped'] += 1
                            else:
                                product_stats['failed'] += 1
                        except Exception as e:
                            logger.error(f"Download failed for {s3_key}: {e}")
                            product_stats['failed'] += 1
                        
                        pbar.update(1)
            
            # Update overall statistics
            stats['products'][product] = product_stats
            stats['total_files'] += len(files)
            stats['downloaded'] += product_stats['downloaded']
            stats['skipped'] += product_stats['skipped']
            stats['failed'] += product_stats['failed']
            
            logger.info(f"Product {product} completed: {product_stats['downloaded']} downloaded, "
                       f"{product_stats['skipped']} skipped, {product_stats['failed']} failed")
        
        return stats


def main():
    parser = argparse.ArgumentParser(
        description='Download GOES-R atmospheric profile data from NOAA AWS S3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Download LVMP and LVTP for June 15, 2020:
    python goes_retrieval.py --date 2020-06-15 --products LVMPC LVTPC --output-dir ./data
  
  Download only hour 18 UTC:
    python goes_retrieval.py --date 2020-06-15 --hour 18 --products LVMPC --output-dir ./data
  
  Use day of year directly:
    python goes_retrieval.py --year 2020 --doy 167 --products LVMPC LVTPC --output-dir ./data
        """
    )
    
    # Date specification (mutually exclusive)
    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument('--date', type=str, help='Date in YYYY-MM-DD format')
    date_group.add_argument('--year-doy', nargs=2, type=int, metavar=('YEAR', 'DOY'),
                           help='Year and day of year (1-366)')
    
    # Data selection
    parser.add_argument('--products', nargs='+', default=['LVMPC'], 
                       choices=['LVMPC', 'LVTPC', 'DSIC', 'TPWC'],
                       help='Products to download (default: LVMPC)')
    
    parser.add_argument('--hour', type=int, choices=range(24),
                       help='Specific hour to download (0-23, default: all hours)')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='./goes_data',
                       help='Output directory (default: ./goes_data)')
    
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing files')
    
    # Performance options
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Number of parallel download threads (default: 4)')
    
    parser.add_argument('--satellite', type=str, default='goes16',
                       choices=['goes16', 'goes17', 'goes18'],
                       help='GOES satellite (default: goes16)')
    
    # Logging
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse date
    if args.date:
        try:
            year, doy = GOESDataRetriever().date_to_doy(args.date)
        except ValueError:
            logger.error(f"Invalid date format: {args.date}. Use YYYY-MM-DD")
            sys.exit(1)
    else:
        year, doy = args.year_doy
    
    # Validate day of year
    max_doy = 366 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 365
    if not (1 <= doy <= max_doy):
        logger.error(f"Invalid day of year: {doy}. Must be 1-{max_doy} for year {year}")
        sys.exit(1)
    
    # Initialize retriever
    retriever = GOESDataRetriever(satellite=args.satellite)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Display information
    date_str = retriever.doy_to_date(year, doy)
    hour_str = f" hour {args.hour:02d}" if args.hour is not None else ""
    logger.info(f"Downloading {args.satellite.upper()} data for {date_str} (DOY {doy}){hour_str}")
    logger.info(f"Products: {', '.join(args.products)}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Sync data
    try:
        stats = retriever.sync_day(
            products=args.products,
            year=year,
            doy=doy,
            output_dir=args.output_dir,
            hour=args.hour,
            overwrite=args.overwrite,
            max_workers=args.max_workers
        )
        
        # Print summary
        logger.info("Download completed!")
        logger.info(f"Total files: {stats['total_files']}")
        logger.info(f"Downloaded: {stats['downloaded']}")
        logger.info(f"Skipped: {stats['skipped']}")
        logger.info(f"Failed: {stats['failed']}")
        
        if stats['failed'] > 0:
            logger.warning(f"{stats['failed']} files failed to download")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Download failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()