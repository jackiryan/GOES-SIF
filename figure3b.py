import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Assuming your DataFrame is called 'df'
df = pd.read_csv("fig3_oco3_0p01d_20200611_data.csv")  # Load your data

# Define the hours to plot
hours_to_plot = [14, 19, 22]

# Create figure with three subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
#fig.suptitle('SIF 740nm Values by Hour', fontsize=16, y=1.02)

# Define colormap limits
vmin, vmax = -0.2, 2.5
cmap = 'YlGn'

for i, hour in enumerate(hours_to_plot):
    # Filter data for the specific hour
    hour_data = df[df['hour'] == hour]
    
    if len(hour_data) == 0:
        print(f"Warning: No data found for hour {hour}")
        continue
    
    # Extract coordinates and values
    lon = hour_data['longitude'].values
    lat = hour_data['latitude'].values
    sif = hour_data['sif_740nm'].values
    
    # Create the scatter plot
    scatter = axes[i].scatter(
        lon, lat, c=sif,
        cmap=cmap, 
        vmin=vmin, 
        vmax=vmax,
        s=50,  # Point size - adjust as needed
        alpha=0.8,  # Transparency - adjust as needed
        edgecolors='none'  # Remove point borders for cleaner look
    )
    
    # Set labels and title
    axes[i].set_title(f'Hour {hour}', fontsize=14)
    axes[i].set_xlabel('Longitude', fontsize=12)
    if i == 0:  # Only label y-axis for the first subplot
        axes[i].set_ylabel('Latitude', fontsize=12)
    
    # Set aspect ratio to be equal for proper geographic display
    axes[i].set_aspect('equal', adjustable='box')
    
    # Add grid for better readability
    axes[i].grid(True, alpha=0.3)
    
    # Optional: Set axis limits to ensure all subplots have the same extent
    # Uncomment these lines if you want consistent axis ranges across all plots
    # all_lon = df['longitude']
    # all_lat = df['latitude']
    # axes[i].set_xlim(all_lon.min(), all_lon.max())
    # axes[i].set_ylim(all_lat.min(), all_lat.max())

# Add a single colorbar for all subplots
#cbar = fig.colorbar(scatter, ax=axes, orientation='horizontal', 
#                   pad=0.1, shrink=0.8, aspect=30)
#cbar.set_label('SIF 740nm (W/m²/sr/μm)', fontsize=12)

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()

# Optional: Save the figure
# plt.savefig('sif_hourly_scatter.png', dpi=300, bbox_inches='tight')

# Print some basic statistics for each hour
print("\nData summary by hour:")
for hour in hours_to_plot:
    hour_data = df[df['hour'] == hour]
    if len(hour_data) > 0:
        print(f"Hour {hour}: {len(hour_data)} points, "
              f"SIF range: {hour_data['sif_740nm'].min():.3f} to {hour_data['sif_740nm'].max():.3f}")
    else:
        print(f"Hour {hour}: No data available")