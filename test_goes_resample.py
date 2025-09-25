import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from pyresample import create_area_def
from pyresample.kd_tree import resample_nearest
from pyresample.geometry import AreaDefinition
import rasterio
from scipy.interpolate import RegularGridInterpolator
import xarray as xr



with rasterio.open("conus_gmted_01deg.tif") as src:
    elevation = src.read(1)
    bounds = src.bounds
    height, width = elevation.shape

    dem_lons = np.linspace(bounds.left, bounds.right, width)
    dem_lats = np.linspace(bounds.top, bounds.bottom, height)

    if dem_lats[0] > dem_lats[-1]:
        dem_lats = dem_lats[::-1]
        elevation = elevation[::-1, :]

    dem_interpolator = RegularGridInterpolator(
        (dem_lats, dem_lons),
        elevation,
        method="linear",
        bounds_error=False,
        fill_value=np.nan
    )


fname = "LVTPC/2020/163/17/OR_ABI-L2-LVTPC-M6_G16_s20201631701146_e20201631703519_c20201631705350.nc"
ds = xr.open_dataset(fname)

lvtp = ds["LVT"].values
proj_info = ds["goes_imager_projection"]
h = proj_info.perspective_point_height
lon_origin = proj_info.longitude_of_projection_origin
sweep = proj_info.sweep_angle_axis

x = ds.x.data * h
y = ds.y.data * h

ds.close()

goes_area = AreaDefinition(
    'goes_conus',
    'GOES East CONUS',
    'goes_conus',
    {
        'proj': 'geos',
        'lon_0': lon_origin,
        'h': h,
        'sweep': sweep,
        'ellps': 'GRS80'
    },
    len(x),
    len(y),
    [x.min(), y.min(), x.max(), y.max()]
)

target_area = create_area_def(
    'equirectangular',
    {
        'proj': 'eqc',  # Equirectangular projection
        'ellps': 'WGS84'
    },
    area_extent=[-135, 20, -50, 50],
    resolution=0.01,
    units='degrees'
)


lons, lats = target_area.get_lonlats()
print(lons.shape)

resampled_nn = resample_nearest(
    goes_area,
    lvtp,
    target_area,
    radius_of_influence=10000,
    fill_value=np.nan,
    epsilon=0.5
)

resampled_nn = np.flipud(resampled_nn)

# Visualize the results
fig = plt.figure(figsize=(20, 6))

# Original data in GOES projection (left subplot)
ax1 = plt.subplot(1, 2, 1)
ax1.imshow(lvtp[:, :, 0], cmap='viridis', origin='upper')
ax1.set_title('Original GOES Fixed Grid')
ax1.set_xlabel('X pixels')
ax1.set_ylabel('Y pixels')

# Resampled data with Cartopy basemap (right subplot)
ax2 = plt.subplot(1, 2, 2, projection=ccrs.PlateCarree())

# Get the extent for the plot
extent = target_area.area_extent_ll  # [lon_min, lon_max, lat_min, lat_max]
plot_extent = [extent[0], extent[2], extent[1], extent[3]]

# Add the temperature data
im1 = ax2.imshow(resampled_nn[:, :, 0], cmap='viridis', extent=plot_extent, 
                 origin='lower', transform=ccrs.PlateCarree(), alpha=0.8)

# Add geographic features
ax2.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax2.add_feature(cfeature.BORDERS, linewidth=0.5)
ax2.add_feature(cfeature.STATES, linewidth=0.3, alpha=0.7)
ax2.add_feature(cfeature.LAKES, alpha=0.3)
ax2.add_feature(cfeature.RIVERS, alpha=0.3)

# Add gridlines
gl = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                   linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 10}
gl.ylabel_style = {'size': 10}

# Set the extent to match your data
ax2.set_extent(plot_extent, crs=ccrs.PlateCarree())

ax2.set_title('Nearest Neighbor Resampling with Geographic Context')

# Add colorbar
plt.colorbar(im1, ax=ax2, orientation='horizontal', pad=0.1, shrink=0.8)

plt.tight_layout()
plt.show()