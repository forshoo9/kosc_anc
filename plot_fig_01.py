from pathlib import Path
import netCDF4 as nc4
import seaborn as sns
import sys, os
from datetime import date, datetime
from dateutil.rrule import rrule, DAILY, HOURLY
from scipy import interpolate, stats
from numpy import arange, dtype
import numpy as np
import numpy.ma as ma
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib import ticker
import pandas as pd
from netCDF4 import Dataset
import h5py
from matplotlib.lines import Line2D

#import cartopy
#print(cartopy.__version__)
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cf
#import matplotlib.ticker as mticker

from matplotlib.gridspec import GridSpec
from copy import copy
import shapely.geometry as sgeom

from tempfile import TemporaryFile

#pd.options.mode.chained_assignment = None


def find_side(ls, side):
    """
    Given a shapely LineString which is assumed to be rectangular, return the
    line corresponding to a given side of the rectangle.
    
    """
    minx, miny, maxx, maxy = ls.bounds
    points = {'left': [(minx, miny), (minx, maxy)],
              'right': [(maxx, miny), (maxx, maxy)],
              'bottom': [(minx, miny), (maxx, miny)],
              'top': [(minx, maxy), (maxx, maxy)],}
    return sgeom.LineString(points[side])


def lambert_xticks(ax, ticks):
    """Draw ticks on the bottom x-axis of a Lambert Conformal projection."""
    te = lambda xy: xy[0]
    lc = lambda t, n, b: np.vstack((np.zeros(n) + t, np.linspace(b[2], b[3], n))).T
    xticks, xticklabels = _lambert_ticks(ax, ticks, 'bottom', lc, te)
    ax.xaxis.tick_bottom()
    ax.set_xticks(xticks)
    ax.set_xticklabels([ax.xaxis.get_major_formatter()(xtick) for xtick in xticklabels])
    

def lambert_yticks(ax, ticks):
    """Draw ricks on the left y-axis of a Lamber Conformal projection."""
    te = lambda xy: xy[1]
    lc = lambda t, n, b: np.vstack((np.linspace(b[0], b[1], n), np.zeros(n) + t)).T
    yticks, yticklabels = _lambert_ticks(ax, ticks, 'left', lc, te)
    ax.yaxis.tick_left()
    ax.set_yticks(yticks)
    ax.set_yticklabels([ax.yaxis.get_major_formatter()(ytick) for ytick in yticklabels])

def _lambert_ticks(ax, ticks, tick_location, line_constructor, tick_extractor):
    """Get the tick locations and labels for an axis of a Lambert Conformal projection."""
    outline_patch = sgeom.LineString(ax.outline_patch.get_path().vertices.tolist())
    axis = find_side(outline_patch, tick_location)
    n_steps = 30
    extent = ax.get_extent(ccrs.PlateCarree())
    _ticks = []
    for t in ticks:
        xy = line_constructor(t, n_steps, extent)
        proj_xyz = ax.projection.transform_points(ccrs.Geodetic(), xy[:, 0], xy[:, 1])
        xyt = proj_xyz[..., :2]
        ls = sgeom.LineString(xyt.tolist())
        locs = axis.intersection(ls)
        if not locs:
            tick = [None]
        else:
            tick = tick_extractor(locs.xy)
        _ticks.append(tick[0])
    # Remove ticks that aren't visible:    
    ticklabels = copy(ticks)
    while True:
        try:
            index = _ticks.index(None)
        except ValueError:
            break
        _ticks.pop(index)
        ticklabels.pop(index)
    return _ticks, ticklabels


def draw_etopo1(fname, cmap, vminmax=[10,10000]):

    ip = 10 # iskip
    
    minlat, maxlat, minlon, maxlon = [110, 290, 230, 450]
    minlat1, maxlat1, minlon1, maxlon1 = [110, 290, 230, 450]
    minlat2, maxlat2, minlon2, maxlon2 = [300, 470, 90, 280]
    
    bands = [412,443,490,555,660,680,745,865]
    
    nx, ny = [5685, 5567]
    ngrid = nx*ny
    dx, dy = 1., 1.
    minLat, maxLat, minLon, maxLon = 21., 50., 111., 150.
    lat_1deg = np.arange(minLat,maxLat+dy,dy)
    lon_1deg = np.arange(minLon,maxLon+dx,dx)
    lat_25km = np.arange(minLat,maxLat+dy/4.,dy/4.) 
    lon_25km = np.arange(minLon,maxLon+dx/4.,dx/4.)
    
    
    lon = np.memmap('../../extra/LON.img',dtype='<f4').reshape((nx,ny))
    lat = np.memmap('../../extra/LAT.img',dtype='<f4').reshape((nx,ny))
    #dem = np.memmap('../../extra/GOCI_DEM.pxl',dtype='<f4').reshape((nx,ny))
    #mu1 = np.memmap('../../extra/GDPS_MASK.in',dtype='<u1').reshape((nx,ny))
    
#    fname_he5 = "./COMS_GOCI_L2C_GA_20190415031643.he5"
#    with h5py.File(fname_he5, mode='r') as f:
#        dset = f['/HDFEOS/GRIDS/Image Data/Data Fields/FLAG Image Pixel Values'][:,:]
#        bits = np.unpackbits(dset.view(np.uint8),bitorder='little').reshape(*dset.shape, 32)
#        bits1 = bits[:,:,31]*1000 + bits[:,:,30]*100 + bits[:,:,29]*10 + bits[:,:,28]
#        bits1 = bits1.astype(np.str)
#        slot = np.array([ int(bits1[i,j],2) for i in range(nx) for j in range(ny) ]).reshape((nx,ny))
#        slot = np.where(bits[:,:,27] == 1, np.nan, slot)
#        np.save("./Slot_GOCI.npy",slot)
    
    slot = np.load("./Slot_GOCI.npy")

#    import metpy.calc as mpcalc
#    slot0 = mpcalc.smooth_n_point(slot, 9, 1)
#    print(slot0)
#    slot = slot - slot0
#    slot = np.where(slot == 0.0, np.nan, slot)
    
#    slot = slot1 - slot2

    lccproj = ccrs.LambertConformal(central_latitude=36, central_longitude=130)
    orthproj = ccrs.Orthographic(central_latitude=36,central_longitude=130)
    mecproj = ccrs.Mercator(central_longitude=130,min_latitude=21,max_latitude=51)
    geoproj = ccrs.Geostationary(central_longitude=130)
    plateproj = ccrs.PlateCarree()

    proj = plateproj
    #x, y, z = lon, lat, slot
    x, y, z = lon[::ip,::ip], lat[::ip,::ip], slot[::ip,::ip]
    vmin, vmax = vminmax

    fig = plt.figure(figsize=(10,4))

    ax = fig.add_subplot(121, projection = lccproj, aspect = "equal")
            #projection = orthproj,
            #projection = plateproj, 
            #projection = geoproj, 
    ax.set_extent([115,145.5,21,49.5])
#    ax.stock_img()

    n = 16

    cmap = cm.get_cmap(cmap,n)
    bounds = np.arange(n+1)
    vals = bounds[:-1]
    norm = colors.BoundaryNorm(bounds,cmap.N)

    cmap.set_under(color=cf.COLORS['land'])
#    cmap.set_bad(color='k')

    mesh = ax.pcolormesh(x, y, z, transform=proj,
            cmap= cmap, norm=norm )#vmin=-0.5, vmax=0.5)

    cb = plt.colorbar(mesh, ax=ax, fraction=0.03, 
            boundaries=bounds, values=vals)
    cb.ax.tick_params(length=0)
    cb.set_label(label='Slot number of GOCI', size=12, labelpad=10, rotation=-90)

    cb.set_ticks(vals[::2] + 0.5)
    cb.set_ticklabels(['Slot_{:02d}'.format(i+1) for i in vals[::2]])

    ax.set_title('GOCI Slot and Test sites',fontsize=14,)# weight='bold')
    ax.set_xlabel('Longitude',fontsize=12)
    ax.set_ylabel('Latitude',fontsize=12)
    ax.coastlines(resolution='10m', color='black', linewidth=1)
    fig.canvas.draw()

    of = pd.read_csv('./geoinfo_obs_sample.csv') 
    for i, (flat, flon, name) in enumerate(zip(np.array(of["Lat"]),np.array(of["Lon"]),np.array(of["Name"]))):
        ax.scatter(x=flon, y=flat+0.25, s=25, marker='o', c='r', transform=proj)
        ax.text( flon, flat+1.5, name, c='r', fontsize=14, 
            weight='bold', ha='center',va='center', transform=proj)

    xticks=[105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155]
    yticks=[15, 20, 25, 30, 35, 40, 45, 50, 55]

    ax.gridlines(xlocs=xticks, ylocs=yticks, 
            lw=1, color='gray', alpha=0.5, linestyle='--')

    ax.add_feature(cf.LAND)
    ax.add_feature(cf.OCEAN)#, facecolor='blue')
    ax.add_feature(cf.BORDERS) 

    ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER) 
    ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)
    lambert_xticks(ax, xticks)
    lambert_yticks(ax, yticks)

    axL = fig.add_subplot(122, projection = lccproj, aspect = "equal")
    axL.set_extent([122.5,133,32,38.5])
#    ax.stock_img()
    axL.set_title('Meteorological observation',fontsize=14,) # weight='bold')
    #axL.set_title('MET OBS',fontsize=14, weight='bold')
    axL.set_xlabel('Longitude',fontsize=12)
    axL.set_ylabel('Latitude',fontsize=12)
    axL.coastlines(resolution='10m', color='black', linewidth=1)
    fig.canvas.draw()

    of = pd.read_csv('./geoinfo_obs.csv') 
    for i, (flat, flon, name, ha, inst) in enumerate(zip(np.array(of["Lat"]),
        np.array(of["Lon"]),np.array(of["Name"]),np.array(of["ha"]),np.array(of["Instrument"]))):
        if ha=='left': fha = 0.25
        else: fha = -0.25
        if inst=='Buoy': 
            axL.scatter(x=flon, y=flat, s=25, marker='o', label='Buoy', 
                    facecolors='none', edgecolors='k', linewidth=1.25, transform=proj)
            axL.text( flon+fha, flat-0.025, name, c='k', fontsize=12, 
                    ha=ha,va='center', transform=proj)
        else: 
            axL.scatter(x=flon, y=flat, s=25, marker='o', label='IGRA',
                    c='k', transform=proj)
            axL.text( flon+fha, flat-0.025, name, c='k', fontsize=12, 
                    ha=ha,va='center', transform=proj)

    xticks=[120, 122, 124, 126, 128, 130, 132, 134]
    yticks=[30, 32, 34, 36, 38, 40]

    axL.gridlines(xlocs=xticks, ylocs=yticks, 
            lw=1, color='gray', alpha=0.5, linestyle='--')

    axL.add_feature(cf.LAND)
    axL.add_feature(cf.OCEAN)#, facecolor='blue')
    axL.add_feature(cf.BORDERS) 

    axL.xaxis.set_major_formatter(LONGITUDE_FORMATTER) 
    axL.yaxis.set_major_formatter(LATITUDE_FORMATTER)
    lambert_xticks(axL, xticks)
    lambert_yticks(axL, yticks)

    legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Buoy', 
                markeredgecolor='k', markerfacecolor='none', markersize=12),
            Line2D([0], [0], marker='o', color='w', label='IGRA', 
                markerfacecolor='k', markersize=12), ]
    axL.legend(handles=legend_elements, loc='lower right')

    axL.text(x=-0.05,y=1.0, s="b", fontsize=20, weight='bold',    
            ha='center', va='bottom', transform=axL.transAxes) 
    ax.text(x=-0.05,y=1.0, s="a", fontsize=20, weight='bold',    
            ha='center', va='bottom', transform=ax.transAxes) 

    plt.tight_layout()
    plt.savefig(fname,dpi=500)
    plt.show()


if __name__ == '__main__':

    #---
    a = datetime(2011, 1, 1, 0)
    b = datetime(2011, 1, 1, 0)

    fname = '../../figs/fig_01_domain.png'

    #draw_etopo1(fname, cmap='YlGnBu')
    #draw_etopo1(fname, cmap='YlOrBr')
    draw_etopo1(fname, cmap='Greys')
    #draw_etopo1(fname, cmap='jet')
    #draw_etopo1(fname, cmap='ocean_r', vminmax=[10,10000])

    sys.exit()
