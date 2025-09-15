
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import xarray as xr


def plot_map(
    data: npt.NDArray[np.float32],
    lat: npt.NDArray[np.float32],
    lon: npt.NDArray[np.float32],
    cmap: str = "viridis",
    fig_size: tuple[int, int] = (16, 8),
    vmin: float | None = None,
    vmax: float | None = None,
    extents: list[float] = [-180, 180, -90, 90],
    title: str | None = None,
    label: str | None = None,
    outfile: str | None = None,
) -> None:
    if not (len(data) == len(lat) == len(lon)):
        raise ValueError("samples, lat, and lon must all have the same length.")

    _, ax = plt.subplots(
        subplot_kw={"projection": ccrs.PlateCarree()}, figsize=fig_size
    )

    ax.set_global() # type: ignore
    ax.add_feature(cfeature.LAND, facecolor="lightgray") # type: ignore
    ax.add_feature(cfeature.OCEAN, facecolor="white") # type: ignore
    # Renders coastlines and borders underneath data
    ax.coastlines(linewidth=0.5, zorder=-1) # type: ignore
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="black", zorder=-1) # type: ignore
    ax.set_extent(extents, crs=ccrs.PlateCarree()) # type: ignore

    # Matplotlib pcolormesh displays the gridded data as a pixel-like grid
    # Note vmax is lower here than in the previous example, SIF can vary seasonally.
    chart = ax.pcolormesh(
        lon,
        lat,
        data,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        transform=ccrs.PlateCarree(),
    )

    cbar = plt.colorbar(chart, ax=ax, orientation="horizontal", pad=0.05, fraction=0.05)

    if label:
        cbar.set_label(label)
    else:
        cbar.set_label("Sample values")

    if title:
        plt.title(title)

    if outfile:
        plt.savefig(outfile, bbox_inches="tight", dpi=300)
        print(f"Plot saved to {outfile}")

    plt.show()

if __name__ == "__main__":
    sif_raster = "jun_2020_sif.nc4"
    ds = xr.open_dataset(sif_raster)
    sif = ds["SIF_740nm"]
    lats = ds["lat"][:]
    lons = ds["lon"][:]

    daily_avg_data = sif[...]
    ds.close()
    lon2d, lat2d = np.meshgrid(lons, lats)
    # Transpose the meshgrid result to be of shape (lon_res, lat_res) ex. (360, 180)
    lon2d = lon2d.T
    lat2d = lat2d.T

    # Create a masked array where the fill_val is masked
    data_masked = np.ma.masked_where(np.isnan(daily_avg_data), daily_avg_data)
    # Average over axis 0 (the "time" dimension), produces a masked array of shape (lon_res, lat_res)
    mean_data_masked = data_masked.mean(axis=0)

    # Be sure to change the title and label if you change the monthly gridded raster you want to display
    # To avoid overwriting the output image when making changes, you should also change the value of outfile
    plot_map(
        mean_data_masked, lat2d, lon2d,
        vmax=2.0,
        vmin=-0.2,
        cmap="YlGn",
        # Uncomment to window the plot to the CONUS
        extents=[-130, -50, 20, 50],
        title=f"OCO-3 June 2020 Mean Daily SIF$_{{740}}$",
        label=r"SIF (W/$\mathrm{m}^2$/sr/μm)",
        outfile=f"figure_2.png"
    )
