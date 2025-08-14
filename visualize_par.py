import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def load_par_data(filename):
    """Load PAR data from HDF5 file"""
    with h5py.File(filename, 'r') as f:
        # Load the PAR data - shape (24, 600, 600)
        par_data = f['PAR_hourly'][:]
        
    # Create coordinate arrays for the specified geographic extent
    # Data spans 0° to 6°S latitude and 78°W to 72°W longitude
    lat = np.linspace(0, -6, par_data.shape[1])  # 600 points from 0° to -6°
    lon = np.linspace(-72, -66, par_data.shape[2])  # 600 points from -72° to -66°
    time = np.arange(par_data.shape[0])  # 24 hours (0-23)
    
    # Transpose to get shape (lat, lon, time) for easier plotting
    par_data = par_data.transpose(1, 2, 0)  # Now shape is (600, 600, 24)
    
    return par_data, lat, lon, time

def plot_par_heatmaps(par_data, lat, lon, time, selected_hours=None):
    """Create subplot heatmaps for selected hours"""
    if selected_hours is None:
        selected_hours = [12, 15, 18, 21]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    # Create custom colormap for PAR (blue to yellow to red)
    colors = ['#000033', '#000080', '#0080FF', '#80FF80', '#FFFF00', '#FF8000', '#FF0000']
    cmap = LinearSegmentedColormap.from_list('PAR', colors)
    
    for i, hour in enumerate(selected_hours):
        if hour >= len(time):
            continue
            
        ax = axes[i]
        im = ax.imshow(par_data[:, :, hour], 
                      extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                      cmap=cmap, origin='lower', aspect='auto')
        
        ax.set_title(f'PAR at Hour {hour:02d}:00')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='PAR (μmol/m²/s)')
    
    plt.tight_layout()
    plt.show()

def plot_par_contour_animation(par_data, lat, lon, time, save_gif=False):
    """Create animated contour plot"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create meshgrid for contour plotting
    LON, LAT = np.meshgrid(lon, lat)
    
    # Set up the plot
    ax.set_xlim(lon.min(), lon.max())
    ax.set_ylim(lat.min(), lat.max())
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('PAR Over 24 Hours - Ecuador/Peru')
    
    # Initialize contour plot and colorbar
    levels = np.linspace(par_data.min(), par_data.max(), 20)
    cs = ax.contourf(LON, LAT, par_data[:, :, 0], levels=levels, cmap='plasma')
    cbar = plt.colorbar(cs, ax=ax, label='PAR (μmol/m²/s)')
    
    # Animation function
    def animate(frame):
        # Clear only the contour collections, not the entire axes
        for collection in ax.collections:
            collection.remove()
        
        # Set labels and title
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'PAR at Hour {(frame+11):02d}:00 UTC (Local: {(frame+7)%24:02d}:00)')
        
        # Create new contour plot
        cs = ax.contourf(LON, LAT, par_data[:, :, frame+11], levels=levels, cmap='plasma')
        
        # No need to return anything for blit=False
        return []
    
    # Create animation with blit=False to avoid collection issues
    anim = animation.FuncAnimation(fig, animate, frames=12, 
                                 interval=500, blit=False, repeat=True)
    
    if save_gif:
        anim.save('par_animation.gif', writer='pillow', fps=2)
        print("Animation saved as 'par_animation.gif'")
    
    plt.show()
    return anim

def plot_par_time_series(par_data, lat, lon, time, sample_points=None):
    """Plot time series for selected spatial points"""
    if sample_points is None:
        # Select some sample points across Brazil
        sample_points = [
            (300, 300, "Center (3°S, 75°W)"),
            (100, 100, "North West (1°S, 71°W)"),
            (500, 100, "South West (5°S, 71°W)"),
            (100, 500, "North East (1°S, 67°W)"),
            (500, 500, "South East (5°S, 67°W)")
        ]
    
    plt.figure(figsize=(12, 8))
    
    for lat_idx, lon_idx, label in sample_points:
        if lat_idx < len(lat) and lon_idx < len(lon):
            par_series = par_data[lat_idx, lon_idx, :]
            plt.plot(time, par_series, marker='o', linewidth=2, markersize=4, 
                    label=label)
    
    plt.xlabel('Hour of Day (UTC)')
    plt.ylabel('PAR (μmol/m²/s)')
    plt.title('PAR Time Series at Selected Locations - Brazil')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 23)
    plt.xticks(range(0, 24, 3))
    
    # Add local time reference (approximately UTC-5 for this region)
    ax2 = plt.gca().twiny()
    ax2.set_xlim(0, 23)
    ax2.set_xticks(range(0, 24, 3))
    ax2.set_xticklabels([f'{(h-5)%24:02d}:00' for h in range(0, 24, 3)])
    ax2.set_xlabel('Local Time (UTC-5, approximate)')
    
    plt.tight_layout()
    plt.show()

def plot_par_geographic(par_data, lat, lon, time, hour=12):
    """Plot PAR data on a geographic map using Cartopy"""
    fig = plt.figure(figsize=(12, 8))
    
    # Use PlateCarree projection for lat/lon data
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.OCEAN, alpha=0.3)
    ax.add_feature(cfeature.LAND, alpha=0.3)
    
    # Create meshgrid
    LON, LAT = np.meshgrid(lon, lat)
    
    # Plot PAR data
    im = ax.contourf(LON, LAT, par_data[:, :, hour], 
                    levels=20, cmap='plasma', transform=ccrs.PlateCarree())
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1, shrink=0.7)
    cbar.set_label('PAR (μmol/m²/s)')
    
    # Set extent to your specific region (slightly larger for context)
    ax.set_extent([-73, -65, -7, 1], ccrs.PlateCarree())
    ax.gridlines(draw_labels=True, alpha=0.5)
    
    plt.title(f'PAR Distribution at Hour {hour:02d}:00 - Brazil')
    plt.show()

def analyze_par_statistics(par_data, lat, lon, time):
    """Calculate and display basic statistics"""
    print("PAR Data Statistics:")
    print(f"Data shape: {par_data.shape}")
    print(f"Latitude range: {lat.min():.2f} to {lat.max():.2f}")
    print(f"Longitude range: {lon.min():.2f} to {lon.max():.2f}")
    print(f"Time range: {time.min()} to {time.max()} hours")
    print(f"PAR range: {par_data.min():.2f} to {par_data.max():.2f} μmol/m²/s")
    print(f"Mean PAR: {par_data.mean():.2f} μmol/m²/s")
    
    # Daily cycle statistics
    daily_mean = np.mean(par_data, axis=(0, 1))
    peak_hour = np.argmax(daily_mean)
    print(f"Peak PAR occurs at hour {peak_hour}:00")
    
    # Spatial statistics
    spatial_mean = np.mean(par_data, axis=2)
    max_lat_idx, max_lon_idx = np.unravel_index(np.argmax(spatial_mean), spatial_mean.shape)
    print(f"Highest average PAR at: {lat[max_lat_idx]:.2f}°N, {lon[max_lon_idx]:.2f}°E")

# Main execution example
if __name__ == "__main__":
    # Load your data
    filename = "/Users/jryan/general/GeoNEX/data.nas.nasa.gov/geonex/GOES16/GEONEX-L2/DSR-PAR/2021/hourly/h18v10/G016_DSR_2021067_h18v10.h5"
    
    try:
        par_data, lat, lon, time = load_par_data(filename)
        
        # Display basic statistics
        analyze_par_statistics(par_data, lat, lon, time)
        
        # Create different visualizations
        print("\n1. Creating heatmap subplots...")
        plot_par_heatmaps(par_data, lat, lon, time)
        
        print("\n2. Creating time series plot...")
        plot_par_time_series(par_data, lat, lon, time)
        
        print("\n3. Creating animated contour plot...")
        anim = plot_par_contour_animation(par_data, lat, lon, time, save_gif=True)
        
        # Uncomment if you have Cartopy installed
        print("\n4. Creating geographic plot...")
        plot_par_geographic(par_data, lat, lon, time, hour=12)
        
    except FileNotFoundError:
        print(f"File '{filename}' not found. Please check the file path.")
    except Exception as e:
        print(f"Error loading data: {e}")
        
        # Create sample data for demonstration
        print("Creating sample data for demonstration...")
        
        # Generate sample PAR data for Ecuador/Peru coast region
        lat_sample = np.linspace(0, -6, 600)  # 0° to 6°S
        lon_sample = np.linspace(-78, -72, 600)  # 78°W to 72°W
        time_sample = np.arange(24)
        
        # Create realistic PAR pattern
        LAT, LON, TIME = np.meshgrid(lat_sample, lon_sample, time_sample, indexing='ij')
        
        # Diurnal cycle (peaks at solar noon, accounting for longitude)
        # Solar noon occurs around 12:00 local time for this longitude range
        local_solar_time = TIME + (LON + 75) / 15  # Approximate local solar time
        diurnal = np.maximum(0, np.cos(2 * np.pi * (local_solar_time - 12) / 24))
        
        # Spatial variation (accounting for latitude effect)
        # PAR generally decreases with distance from equator
        lat_factor = np.cos(np.radians(LAT))  # Cosine of latitude
        
        # Ocean vs land effect (simplified - higher over ocean)
        # This region is mostly ocean with some coastal areas
        distance_from_coast = np.minimum(np.abs(LON + 75), 2)  # Simple coastal proximity
        ocean_factor = 1 + 0.1 * (distance_from_coast / 2)
        
        # Add some realistic cloud effects (random but spatially correlated)
        np.random.seed(42)  # For reproducible results
        cloud_effect = 0.7 + 0.3 * np.random.random(LAT.shape)
        
        par_sample = 2200 * diurnal * lat_factor * ocean_factor * cloud_effect
        par_sample = np.maximum(0, par_sample)  # Ensure no negative values
        
        print("Running demonstration with sample data...")
        analyze_par_statistics(par_sample, lat_sample, lon_sample, time_sample)
        plot_par_heatmaps(par_sample, lat_sample, lon_sample, time_sample)
        plot_par_time_series(par_sample, lat_sample, lon_sample, time_sample)