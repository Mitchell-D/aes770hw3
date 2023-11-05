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
#from sklearn.linear_model import LinearRegression

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

def print_ceres(ceres_nc:Path):
    ds = nc.Dataset(ceres_nc, 'r')
    data = []
    labels = []
    for k in ds.variables.keys():
        X = ds.variables[k][:]
        stat = enh.array_stat(X)
        print(f"{str(X.shape):<12} {k:<70} {stat['min']:.3f} {stat['max']:.3f}")

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
        ("Clear_layer_overlap_percent_coverages",
         ("pct_clr","pct_l1","pct_l2","pct_ol")),

        ## (M,) ADM-corrected fluxes
        ("CERES_SW_TOA_flux___upwards", "swflux"),
        ("CERES_WN_TOA_flux___upwards", "wnflux"),
        ("CERES_LW_TOA_flux___upwards", "lwflux"),

        ("Cloud_mask_clear_strong_percent_coverage", "nocld"),
        ("Cloud_mask_clear_weak_percent_coverage", "nocld_wk"),

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
    unq_ids = np.ma.unique(ds.variables["Surface_type_index"][:])
    print(f"Valid IDs: {unq_ids}")
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
        self.size = data[0].size
        assert all(X.size==self.size for X in data)
        self._data = data
        self.labels = labels

    def data(self, label=None):
        return self._data[self.labels.index(label)]

    def fill(self, label, value):
        """
        Convenience method to replace masked values with a number for a
        single dataset.
        """
        self._data[self.labels.index(label)] = \
                self.data(label).filled(value)

    def mask(self, mask:np.ndarray):
        return FG1D(self.labels, [X[mask] for X in self._data])

    def scatter(self, xlabel, ylabel, clabel=None, get_trend=False, show=True,
                fig_path:Path=None, plot_spec:dict={}):
        """
        Make a scatter plot of the 2 provided datasets stored in this FG1D,
        optionally coloring points by a third dataset
        """
        ps = {"xlabel":xlabel, "ylabel":ylabel, "clabel":clabel,
              "trend_color":"red", "trend_width":3, "marker_size":4,
              "cmap":"nipy_spectral", "text_size":12, "title":"",
              "norm":"linear", "logx":False,"figsize":(16,12)}
        ps.update(plot_spec)

        plt.clf()
        plt.rcParams.update({"font.size":ps["text_size"]})

        X, Y = self.mutual_valid(self.data(xlabel), self.data(ylabel))
        C = None if clabel is None else self.data(clabel)

        fig, ax = plt.subplots()
        if get_trend:
            slope,intc,rval = self.trend(X, Y)
            Tx = np.copy(X)
            Ty = Tx*slope + intc
            ax.scatter(
            #ax.plot(
                    Tx, Ty,
                    linewidth=ps.get("trend_width"),
                    label=f"y={slope:.3f}x+{intc:.3f}\n$R^2$ = {rval**2:.3f}",
                    color=ps.get("trend_color"),
                    s=ps.get("marker_size")+30,
                    marker="2",
                    zorder=100,
                    )
            ax.legend()
        if ps["logx"]:
            plt.semilogx()
        scat = ax.scatter(
                X, Y, c=C, s=ps.get("marker_size"), cmap=ps.get("cmap"),
                norm=ps.get("norm"))
        if not clabel is None:
            fig.colorbar(scat, label=ps.get("clabel"))
        ax.set_title(ps.get("title"))
        ax.set_xlabel(ps.get("xlabel"))
        ax.set_ylabel(ps.get("ylabel"))
        if show:
            plt.show()
        if not fig_path is None:
            fig.set_size_inches(*ps.get("figsize"))
            fig.savefig(fig_path.as_posix(), bbox_inches="tight",dpi=80)

    def heatmap(self, xlabel, ylabel, xbins=256, ybins=256, get_trend=False,
                show=True, fig_path:Path=None, plot_spec:dict={}):
        """
        Generate a heatmap of the 2 provided datasets in this FG1D
        """
        ps = {"xlabel":xlabel, "ylabel":ylabel, "trend_color":"red",
              "trend_width":3, "cmap":"gist_ncar", "text_size":12,
              "figsize":(12,12)}
        ps.update(plot_spec)
        X, Y = self.mutual_valid(self.data(xlabel), self.data(ylabel))
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
            slope,intc,rval = self.trend(X, Y)
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
        if not fig_path is None:
            fig.set_size_inches(*ps.get("figsize"))
            fig.savefig(fig_path.as_posix(), bbox_inches="tight",dpi=80)
        if show:
            plt.show()

    def geo_scatter(self, clabel, xlabel="lat", ylabel="lon", bounds=None,
                    plot_spec={}, show=True, fig_path=None):
        """ """
        ps = {"xlabel":xlabel, "ylabel":ylabel, "marker_size":4,
              "cmap":"nipy_spectral", "text_size":12, "title":clabel,
              "norm":"linear","figsize":(12,12)}
        plt.clf()
        ps.update(plot_spec)

        plt.rcParams.update({"font.size":ps["text_size"]})

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

        ax.set_title(ps.get("title"))
        ax.set_xlabel(ps.get("xlabel"))
        ax.set_ylabel(ps.get("ylabel"))

        scat = ax.scatter(self.data(ylabel),self.data(xlabel),
                          c=self.data(clabel), s=ps.get("marker_size"),
                          transform=ccrs.PlateCarree(), zorder=100,
                          cmap=ps.get("cmap"), norm=ps.get("norm"))
        fig.colorbar(scat)

        if not fig_path is None:
            fig.set_size_inches(*ps.get("figsize"))
            fig.savefig(fig_path.as_posix(), bbox_inches="tight",dpi=80)
        if show:
            plt.show()

    @staticmethod
    def trend(X, Y):
        """
        Returns the linear regression slope, intercept, and Pearson coefficient
        of the 2 provided dataset labels.
        """
        #'''
        res = linregress(X, Y)
        slope,intc,rval = (res.slope, res.intercept, res.rvalue)
        '''
        res = LinearRegression().fit(X[:,None], Y)
        slope,intc,rval = res.coef_[0],res.intercept_,res.score(X[:,None],Y)
        '''
        return (slope, intc, rval)

    @staticmethod
    def mutual_valid(X, Y):
        valid = np.isfinite(X.filled(np.nan)) & np.isfinite(Y.filled(np.nan))
        print(enh.array_stat(X[valid]))
        print(enh.array_stat(Y[valid]))
        return X[valid], Y[valid]

def get_ocod_mask(ceres):
    """
    Returns the mask for oceanic cloud optical depth
    """
    pass

if __name__=="__main__":
    #mod09_path = Path("data/MOD09.A2021232.1900.061.2021234021126.hdf")
    #mod09_path = Path("data/MOD09.A2021232.1855.061.2021234021011.hdf")
    #mod09_path = Path("data/MOD09.A2021230.1915.061.2021232021510.hdf")
    data_dir = Path("data")
    mod09_path = data_dir.joinpath(Path(
        "data/MOD09.A2021230.1910.061.2021232021325.hdf"
        ))
    aqua_path = data_dir.joinpath(Path(
        "CERES_SSF_Aqua-XTRK_Edition4A_Subset_2021081509-2021082522.nc"))
    terra_path = data_dir.joinpath(Path(
        #"CERES_SSF_Terra-XTRK_Edition4A_Subset_2021081805-2021081820.nc"
        "CERES_SSF_Terra-XTRK_Edition4A_Subset_2021081505-2021082520.nc"
        ))

    labels, adata = parse_ceres(aqua_path)
    _, tdata = parse_ceres(terra_path)

    data = list(map(np.concatenate, zip(tdata, adata)))
    #print_ceres(aqua_path)

    ceres = FG1D(labels, data)
    print(f"Total data points: {ceres.size}")

    ## Time mask
    ceres = ceres.mask(ceres.data("sza")<75)

    ## Cloud property masks
    m_sdcod_ub = ceres.data("l1_sdcod") < 10
    m_nocloud = ceres.data("nocld") > 75
    m_nocloud_wk = ceres.data("nocld_wk") > 75
    m_aocfrac_ub = ceres.data("aer_ocean_cfrac") < 5 ## MOD04 cloud fraction
    m_clr_lb = ceres.data("pct_clr") > 98
    ## Ocean aerosol masks
    m_aopct_ub = ceres.data("aer_ocean_pct") < 2
    m_aopct_lb = ceres.data("aer_ocean_pct") > 80
    ## Land aerosol masks
    m_alpct_lb = ceres.data("aer_land_pct") >= 0
    m_alcfrac_ub = ceres.data("aer_land_cfrac") < 10
    ## Surface type masks
    m_uniform = ceres.data("pct_s1") > 80
    m_water = ceres.data("id_s1") == 17
    m_nowater =  np.logical_not(m_water) & (ceres.data("id_s2") != 17)
    m_evergreen = (ceres.data("id_s1") == 1) | (ceres.data("id_s1") == 2)
    m_deciduous = (ceres.data("id_s1") == 3) | (ceres.data("id_s1") == 4)
    m_forest = m_evergreen | m_deciduous | ceres.data("id_s1") == 5
    m_shrub = (ceres.data("id_s1") == 6) | (ceres.data("id_s1") == 7)
    m_savanna = (ceres.data("id_s1") == 8) | (ceres.data("id_s1") == 9) \
            | (ceres.data("id_s1") == 10)
    m_crop = (ceres.data("id_s1") == 12) | (ceres.data("id_s1") == 14)
    m_bare = ceres.data("id_s1") == 16
    m_urban = ceres.data("id_s1") == 13
    ## Flux masks
    m_swf = ceres.data("swflux") < 10000
    m_laod = ceres.data("aod_land") < 100

    ## OLD
    #mask = m_aopct_lb & m_aocfrac_ub & m_water & m_uniform & m_nocloud

    ## oceanic AOD
    #mask = m_aopct_lb & m_aocfrac_ub
    ## l1 cloud COD
    #mask = m_sdcod_ub & m_aopct_ub &
    ## Ocean AOD
    #mask = m_clr_lb & m_aopct_lb & m_water
    ## Land AOD
    #mask = m_alpct_lb & m_alcfrac_ub & m_nowater & m_uniform & m_swf
    mask = m_clr_lb & m_alpct_lb & m_nowater & m_uniform & m_laod
    #mask = mask & m_evergreen

    #ceres = ceres.mask(m_nocloud & m_evergreen)
    ceres = ceres.mask(mask)

    ceres.scatter(
            #"l1_cod", "swflux", "lwflux",
            #"l1_cod", "lwflux", "l1_sdcod",

            "aod_land", "swflux", "sza",
            #"aod_land", "lwflux", "sza",

            #"aod_ocean_small", "swflux", "sza",
            #"aod_ocean_small", "lwflux", "sza",

            #"pct_s1", "swflux", "id_s1",

            #"lat", "lon", "l1_cod",
            #"lat", "lon", "l1_cod",
            get_trend=True,
            show=False,
            #fig_path=Path("figures/oaod_lwf.png"),
            fig_path=Path("figures/laod_swf.png"),
            plot_spec={
                "title":"SW Smoke Aerosol Effect Over Land",
                "xlabel":"Aerosol Optical Depth",
                "ylabel":"Reflected Shortwave Flux",
                #"ylabel":"Outgoing LW Flux",
                "clabel":"Solar Zenith",
                #"cmap": "tab10",
                "cmap": "nipy_spectral_r",
                "marker_size":80,
                "norm":"linear",
                "logx":True,
                "text_size":26,
                "figsize":(24,12),
                })

    #'''
    ceres.geo_scatter(
            #clabel="swflux",
            clabel="aod_ocean_small",
            #clabel="aer_land_cfrac",
            #clabel="aer_ocean_cfrac",
            #clabel="aod_ocean",
            #clabel="id_s1",
            bounds=[-135, -120, 30, 45],
            #fig_path=Path("figures/oaod_geoplot.png"),
            fig_path=Path("figures/aod_geoplot.png"),
            show=False,
            plot_spec={
                "title":"Aerosol Optical Depth ($0.55\mu m$)",
                "cmap":"nipy_spectral_r",
                "norm":"log",
                })
    #'''
    exit(0)


    '''
    ceres.heatmap(
            #"l1_cod", "swflux",
            "aod_ocean_small", "swflux",
            xbins=64, ybins=64, get_trend=True,
            plot_spec={
                })
    '''
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
