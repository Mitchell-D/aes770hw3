from pprint import pprint as ppt
import json
import netCDF4 as nc
import gc
import numpy as np
from pathlib import Path
import pickle as pkl
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt

from scipy.interpolate import griddata
from scipy.interpolate import NearestNDInterpolator
from scipy.stats import linregress

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from krttdkit.acquire import modis
from krttdkit.operate import enhance as enh
from krttdkit.products import FeatureGrid
from krttdkit.products import HyperGrid
from krttdkit.visualize import guitools as gt
from krttdkit.visualize import geoplot as gp

def search_modis():
    modis_grans = modis.query_modis_l2(
            product_key="MOD09",
            start_time=datetime(2021, 8, 21, 19, 41),
            end_time=datetime(2021, 8, 21, 19, 43),
            debug=True,
            )
    print(modis_grans)

def load_modis_l2(mod09_path:Path, bands:list):
    data, info, geo = modis.get_modis_data(
            datafile=mod09_path,
            bands=bands,
            )
    return FeatureGrid(labels=bands, data=data, info=info)

def get_world_latlon(res=.1):
    return np.meshgrid(np.arange(-90,90,res),
                       np.arange(-180,180,res),
                       indexing="ij")

def contour_plot(hcoords, vcoords, data, bins):
    plt.contourf(hcoords, vcoords, data, bins, cmap="jet")
    plt.colorbar()
    plt.show()

def parse_ceres(ceres_nc:Path):
    """
    Parses fields from a full-featured CERES SSF file
    """
    label_mapping = [
        ## (M,) time and geometry information
        ("Time_of_observation", "jday"),
        ("lat", "lat"),
        ("lon", "lon"),
        ("CERES_viewing_zenith_at_surface", "vza"),
        ("CERES_relative_azimuth_at_surface", "raa"),
        ("CERES_solar_zenith_at_surface", "sza"),

        ## (M,8) Most prominent surface types, in decreasing order
        ("Surface_type_index",
         ("id_s1","id_s2","id_s3","id_s4",
          "id_s5","id_s6","id_s7","id_s8")),
        ("Surface_type_percent_coverage",
         ("pct_s1","pct_s2","pct_s3","pct_s4",
          "pct_s5","pct_s6","pct_s7","pct_s8")),

        ## (M,) ADM-corrected fluxes
        ("CERES_SW_TOA_flux___upwards", "swflux"),
        ("CERES_WN_TOA_flux___upwards", "wnflux"),
        ("CERES_LW_TOA_flux___upwards", "lwflux"),

        ## (M,2) COD for each layer weighted by PSF and cloud fraction
        ("Mean_visible_optical_depth_for_cloud_layer",
         ("l1_cod","l2_cod")),
        ("Stddev_of_visible_optical_depth_for_cloud_layer",
         ("l1_sdcod","l2_sdcod")),

        ## (M,) PSF weighted percentage of pixels in the footprint which
        ## have either land or ocean aerosol values
        ("Percentage_of_CERES_FOV_with_MODIS_land_aerosol", "aer_land_pct"),
        ## (M,) PSF weighted cloud frac from MOD04:
        ## Cloud fraction from Land aerosol cloud mask from retrieved
        ## and overcast pixels not including cirrus mask
        ("PSF_wtd_MOD04_cloud_fraction_land", "aer_land_cfrac"),
        ## (M,) Weighted integer percentage bins of aerosol types
        ("PSF_wtd_MOD04_aerosol_types_land", "aer_land_type"),
        ("PSF_wtd_MOD04_corrected_optical_depth_land__0_550_", "aod_land"),

        ## (M,) Optical depth with the deep blue method (?)
        ("Percentage_of_CERES_FOV_with_MODIS_deep_blue_aerosol", "aer_db_pct"),
        ("PSF_wtd_MOD04_deep_blue_aerosol_optical_depth_land__0_550_", "aod_db"),

        ## (M,) Over-ocean aerosol properties
        ("Percentage_of_CERES_FOV_with_MODIS_ocean_aerosol", "aer_ocean_pct"),
        ("PSF_wtd_MOD04_cloud_fraction_ocean", "aer_ocean_cfrac"),
        ("PSF_wtd_MOD04_effective_optical_depth_average_ocean__0_550_", "aod_ocean"),
        ("PSF_wtd_MOD04_optical_depth_small_average_ocean__0_550_", "aod_ocean_small"),
        ]
    ds = nc.Dataset(ceres_nc, 'r')
    data = []
    labels = []
    for ncl,l in label_mapping:
        X = ds.variables[ncl][:]
        if not type(l) is str:
            assert len(l) == X.shape[1]
            for i in range(len(l)):
                data.append(X[:,i])
                labels.append(l[i])
        else:
            assert len(X.shape)==1
            data.append(X)
            labels.append(l)
    return labels, data

class FG1D:
    def __init__(self, labels, data):
        self._data = data
        self.labels = labels

    def data(self, label=None):
        return self._data[self.labels.index(label)]

    def mask(self, mask:np.ndarray):
        return FG1D(self.labels, [X[mask] for X in self._data])

    def scatter(self, xlabel, ylabel, clabel=None, get_trend=False, show=True,
                fig_path:Path=None, plot_spec:dict={}):
        """
        Make a scatter plot of the 2 provided datasets stored in this FG1D,
        optionally coloring points by a third dataset
        """
        ps = {"xlabel":xlabel, "ylabel":ylabel, "trend_color":"red",
              "trend_width":3, "marker_size":4, "cmap":"nipy_spectral",
              "text_size":12, }
        ps.update(plot_spec)

        plt.rcParams.update({"font.size":ps["text_size"]})

        X = self.data(xlabel)
        Y = self.data(ylabel)
        C = None if clabel is None else ceres.data(clabel)

        fig, ax = plt.subplots()
        scat = ax.scatter(X, Y, c=C,
                          s=ps.get("marker_size"),
                          cmap=ps.get("cmap"),
                          )
        if not clabel is None:
            fig.colorbar(scat)
        if get_trend:
            slope,intc,rval = self.trend(xlabel, ylabel)
            Tx = np.copy(X)
            Ty = Tx*slope + intc
            ax.plot(Tx, Ty,
                    linewidth=ps.get("trend_width"),
                    label=f"y={slope:.3f}x+{intc:.3f}\nR^2 = {rval**2:.3f}",
                    color=ps.get("trend_color"),
                    )
            ax.legend()
        ax.set_title(ps.get("title"))
        ax.set_xlabel(ps.get("xlabel"))
        ax.set_ylabel(ps.get("ylabel"))
        if show:
            plt.show()
        if fig_path:
            plt.savefig(fig_path.as_posix())
        plt.clf()

    def heatmap(self, xlabel, ylabel, xbins=256, ybins=256, get_trend=False,
                show=True, fig_path:Path=None, plot_spec:dict={}):
        """
        Generate a heatmap of the 2 provided datasets in this FG1D
        """
        ps = {"xlabel":xlabel, "ylabel":ylabel, "trend_color":"red",
              "trend_width":3, "cmap":"gist_ncar", "text_size":12, }
        ps.update(plot_spec)
        X = self.data(xlabel)
        Y = self.data(ylabel)
        M, coords = enh.get_nd_hist(
                arrays=(X, Y),
                bin_counts=(xbins, ybins),
                )
        hcoords, vcoords = tuple(coords)
        extent = (min(hcoords), max(hcoords), min(vcoords), max(vcoords))

        plt.rcParams.update({"font.size":ps["text_size"]})
        fig, ax = plt.subplots()
        im = ax.pcolormesh(hcoords, vcoords, M,
                cmap=plot_spec.get("cmap"),
                #vmax=plot_spec.get("vmax"),
                #extent=extent,
                #norm=plot_spec.get("imshow_norm"),
                #origin="lower",
                #aspect=plot_spec.get("imshow_aspect")
                )
        if get_trend:
            slope,intc,rval = self.trend(xlabel, ylabel)
            Tx = np.copy(hcoords)
            Ty = Tx * slope + intc
            ax.plot(Tx, Ty,
                    linewidth=ps.get("trend_width"),
                    label=f"y={slope:.3f}x+{intc:.3f}\nR^2 = {rval**2:.3f}",
                    color=ps.get("trend_color"),
                    )
            ax.legend()
        #ax.set(aspect=1)
        cbar = fig.colorbar(im, ax=ax, orientation="vertical", label="Count")
        ax.set_title(ps.get("title"))
        ax.set_xlabel(ps.get("xlabel"))
        ax.set_ylabel(ps.get("ylabel"))
        #ax.set_xticklabels([f"{c:.2f}" for c in hcoords])
        #ax.set_yticklabels([f"{c:.2f}" for c in vcoords])
        #ax.set_ylim(extent[0], extent[1])
        #ax.set_xlim(extent[2], extent[3])
        if show:
            plt.show()
        if not fig_path is None:
            fig.savefig(fig_path.as_posix(), bbox_inches="tight")

    def geo_scatter(self, clabel, xlabel="lat", ylabel="lon",
                    bounds:tuple=None):
        """ """
        ax = plt.axes(projection=ccrs.PlateCarree())
        fig = plt.gcf()
        if bounds is None:
            bounds = [
                np.amin(self.data(ylabel)),
                np.amax(self.data(ylabel)),
                np.amin(self.data(xlabel)),
                np.amax(self.data(xlabel)),
                ]
        ax.set_extent(bounds)

        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.LAKES)
        ax.add_feature(cfeature.RIVERS)
        ax.coastlines()

        scat = ax.scatter(self.data(ylabel),self.data(xlabel),
                          c=self.data(clabel),
                          transform=ccrs.PlateCarree(), zorder=100)
        fig.colorbar(scat)

        plt.show()
        plt.clf()

    def trend(self, xlabel, ylabel):
        """
        Returns the linear regression slope, intercept, and Pearson coefficient
        of the 2 provided dataset labels.
        """
        result = linregress(self.data(xlabel), self.data(ylabel))
        return (result.slope, result.intercept, result.rvalue)

def get_ocod_mask(ceres):
    """
    Returns the mask for oceanic cloud optical depth
    """
    m_sdcod = ceres.data("l1_sdcod") < 5
    m_aopct_ub = ceres.data("aer_ocean_pct") < 2

if __name__=="__main__":
    #mod09_path = Path("data/MOD09.A2021232.1900.061.2021234021126.hdf")
    #mod09_path = Path("data/MOD09.A2021232.1855.061.2021234021011.hdf")
    #mod09_path = Path("data/MOD09.A2021230.1915.061.2021232021510.hdf")
    #ceres_path = Path(
    #    "data/CERES_SSF_Terra-XTRK_Edition4A_Subset_2021082100-2021082223.nc")

    ceres_path = Path(
        "data/CERES_SSF_Terra-XTRK_Edition4A_Subset_2021081805-2021081820.nc")
    mod09_path = Path("data/MOD09.A2021230.1910.061.2021232021325.hdf")

    labels, data = parse_ceres(ceres_path)
    ## File swath includes 2 passes; only take the first one.
    second_pass = data[labels.index("jday")] > 2459445
    data = list(map(lambda X: X[second_pass], data))
    ceres = FG1D(labels, data)

    #mask = m_sdcod & m_aopct_ub & m_aopct_lb ## for l1 cloud COD
    mask = m_sdcod & m_aopct_lb & m_aopct_lb ## for small oceanic AOD

    ceres = ceres.mask(mask)

    exit(0)

    ceres.scatter(
            #"l1_sdcod", "swflux", "lwflux",
            "l1_cod", "swflux", "lwflux",
            #"lat", "lon", "l1_cod",
            get_trend=True,
            )
    ceres.heatmap(
            #"l1_cod", "swflux",
            "aod_ocean_small", "swflux",
            xbins=64, ybins=64, get_trend=True,
            plot_spec={
                })
    #ceres.geo_scatter(
    #        clabel="swflux",
    #        bounds=[-135, -120, 30, 45],
    #        )
    exit(0)

    plt.scatter(ceres.data("lat"), ceres.data("lon"),
                c=ceres.data("l1_cod"), cmap="jet")
                #c=ceres.data("l1_sdcod"), cmap="jet")
                #c=ceres.data("l2_cod"), cmap="jet")
                #c=ceres.data("l2_sdcod"), cmap="jet")
                #c=ceres.data("aer_ocean_pct"), cmap="jet")
                #c=ceres.data("aer_db_pct"), cmap="jet")
                #c=ceres.data("aod_ocean_small"), cmap="jet")
                #c=ceres.data("id_s1"), cmap="jet")
                #c=ceres.data("swflux"), cmap="jet")
                #c=ceres.data("lwflux"), cmap="jet")
                #c=ceres.data("aer_db_pct"), cmap="jet")
                #c=ceres.data("aer_ocean_pct"), cmap="jet")
    plt.colorbar()
    plt.show()

    exit(0)

    '''
    """ Interpolate onto a regular grid """
    clat = np.arange(30,45,.1)
    clon = np.arange(-135,-120,.1)
    gflux = griddata((lat, lon), swflux, (clat[:,None], clon[None,:]),
                     method="nearest")
    contour_plot(clon,clat,gflux, 50)
    '''

    """ Load a L2 (atmospherically corrected) file with geolocation """
    fg = load_modis_l2(
            mod09_path=mod09_path,
            #bands=["R1_500", "R2_500", "R3_500", "R4_500",
            #       "R5_500", "R6_500", "R7_500", "latitude", "longitude"]
            bands=["R1_1000", "R2_1000", "R3_1000", "R4_1000",
                   "R5_1000", "R6_1000", "R7_1000", "latitude", "longitude"]
            )
    fg = fg.subgrid(vrange=(400,1400), hrange=(200, 1100))
    print(fg.shape, fg.labels)

    '''
    """ Get an RGB """
    norm = "norm1 gaussnorm"
    gt.quick_render(fg.get_rgb(norm+"R1_1000", norm+"R4_1000", norm+"R3_1000"))
    '''

    """ NN-Interpolate shortwave fluxes onto the 1km grid """
    interp = NearestNDInterpolator(list(zip(lon,lat)),swflux)
    gflux = interp(fg.data("longitude"), fg.data("latitude"))
    plt.pcolormesh(fg.data("longitude"), fg.data("latitude"), gflux)
    plt.show()
