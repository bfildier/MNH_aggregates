import xarray as xr
import netCDF4
import numpy as np
import numpy.ma as ma
from datetime import datetime as dt
from datetime import timedelta

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import argparse

import glob, sys, os, re
from pprint import pprint
import subprocess

##-- Directories

sim_dir = "/lustre/fsn1/projects/rech/lsv/ubd61nl/ObedSaba_RECONCILE_MesoNH/production/WPAGG_2km"
fig_dir = '../figures'

#-- Define simulation subfolders
sim_names = [f for f in os.listdir(sim_dir) if re.match("(.*)2016(.*)",f) ]
sim_names.sort(reverse=True)


##-- Functions

#- MNH File identification

def getMNHFileAtTime(target_time,time_ref=dt(2016,9,10),lag_h=0):
    """For a given lag (in hours) at the start of the simulation,
    get the filename of the right simulation at that given time.
    """
    
    # simulation index
    i_lag = np.where(lags_h == lag_h)[0][0]
    # simulation output hourly frequency
    Dt = time_freq_MNH/3600
    
    #-- get corresponding time file
    # time difference
    delta_t = target_time - (time_ref+timedelta(seconds=lag_h*3600))
    # find corresponding subfolder
    i_seg = int((delta_t.total_seconds()/3600/Dt - 1) / (12/Dt) ) +1 # segment number
    subfolder = "seg0%d_12h_tege"%i_seg
    # build corresponding filename
    i_t_seg = int((delta_t.total_seconds()/3600/Dt - 1) % int(12/Dt) + 1)
    filename = "FBC01.1.MNH0%d.OUT.%s.nc"%(i_seg,str(i_t_seg).zfill(3))
    # full path
    path = os.path.join(sim_dir,sim_names[i_lag],subfolder,filename)

    return path

#- Conversions between OLR and Tb

def getOLRFromTb(Tb):
    """Compute OLR estimate in W/m2 from Tb in K, using the conversion formula from Ohring & Gruber (1984),
    with coefficients reported in Yang & Slingo (2001), also used in Feng et al. (2024):

    OLR = sigma * Tf^4

    where

    Tf = Tb * (a + b Tb)

    with a = 1.228 and b = -1.106e-3 K-1.
    """

    sigma = 5.67e-8 # W/m2/K4
    a = 1.228 # no unit
    b = -1.106e-3 # K-1
    
    return sigma * (a * Tb + b * Tb**2)**4

def getTbFromOLR(OLR):
    """Compute Tb estimate in K from OLR in W/m2, by reverting the conversion formula from Ohring & Gruber (1984),
    with coefficients reported in Yang & Slingo (2001), also used in Feng et al. (2024):

    OLR = sigma * Tf^4

    where

    Tf = Tb * (a + b Tb)

    with a = 1.228 and b = -1.106e-3 K-1.
    """

    sigma = 5.67e-8 # W/m2/K4
    a = 1.228 # no unit
    b = -1.106e-3 # K-1
    
    return (-a + np.sqrt(a**2 + 4 * b* (OLR/sigma)**(1/4)))/(2*b)

#- Adjust target time for file selection

def getAdjustedTime(target_time,time_freq=1800):
    """Adjust the target datetime to the current time, according to the frequency of data files.

    Args:
    - target_time (datetime object): target time
    - time_freq (float): frequency of the data files, in s

    Returns:
    - adjusted time (datetime object): adjusted time on the data files time array.
    """

    # compute the number of seconds of IMERG file since start of day
    n_steps = int((target_time-dt(target_time.year,target_time.month,target_time.day)).seconds / (time_freq))
    adjusted_seconds = int(n_steps * time_freq)
    # adjusted time
    adjusted_time = dt(target_time.year,target_time.month,target_time.day)+timedelta(seconds=adjusted_seconds)

    return adjusted_time

#- Show subplot

def showSubplot(lon_array, lat_array, data_var, data_crs, extent, vmin=70, vmax=170, cmap=None):
    
    ax.set_extent(extent, crs=data_crs)
    # data
    pcm = ax.pcolormesh(lon_array, lat_array, 
                        data_var,
                        transform=data_crs,
                        vmin = vmin,
                        vmax = vmax,
                        cmap=cmap)
    # Côtes
    ax.coastlines(resolution='10m', color='black', linewidth=0.8)
    # grille
    gl = ax.gridlines(crs=data_crs, draw_labels=True,
                       color='black', linewidth=0.2, linestyle='-')
    gl.top_labels = False
    gl.right_labels = False
    # colorbar
    cb = plt.colorbar(pcm, ax=ax, orientation='vertical', shrink=0.8)
    cb.ax.set_ylabel('OLR (W/m2)')




##-- Main execution

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Get parameters for figure.')
    parser.add_argument('--target_day', type=str, default="2016-09-10", help='Target time, in YY-mm-dd format.')
    parser.add_argument('--target_time', type=str, default="15:00", help='Target time, in HH:MM format.')
    args = parser.parse_args()
    
    # Input target time
    time_ref = dt(2016,9,10)
    target_time = dt.strptime(args.target_day+"T"+args.target_time,"%Y-%m-%dT%H:%M")
    
    # Input minimum display value
    vmin = 70 # W/m2
    # Input maximum display value
    vmax = 170 # W/m2
    # Input time info
    time_freq_GOESW = 1800 # s
    time_freq_MNH = 1200 # s
    time_step_MNH = 3 # s
    # Simulation lag info
    lags_h = np.array([0,6,12])
    lag_6h = timedelta(seconds=6*3600)
    lag_12h = timedelta(seconds=12*3600)
    
    # Conversion factors
    s_to_h = 1/3600
    rho_w = 1e3 # km/m3
    m_to_mm = 1e3 

    
    ##-- Adjust target time to MNH and IMERG available times
    # Adjusted time on IMERG time array
    target_time_GOESW = getAdjustedTime(target_time,time_freq_GOESW)
    # Adjusted time on MNH time array
    target_time_MNH = getAdjustedTime(target_time,time_freq_MNH)


    ##-- Get Meso-NH data and coords

    #- Datasets
    data_lag0 = xr.open_dataset(getMNHFileAtTime(target_time_MNH,lag_h=0))
    data_lag6 = xr.open_dataset(getMNHFileAtTime(target_time_MNH,lag_h=6))
    data_lag12 = xr.open_dataset(getMNHFileAtTime(target_time_MNH,lag_h=12))
    
    #- Domain coords
    lon_array = data_lag0.longitude.values
    lat_array = data_lag0.latitude.values
    longitude = data_lag0.longitude[0].values
    latitude = data_lag0.latitude[:,0].values
    extent = [longitude[0],longitude[-1],latitude[0],latitude[-1]]
    
    #- OLR data, cropped in target range
    RLUT_lag0 = data_lag0.RLUT_INST[0].values
    RLUT_lag0[RLUT_lag0 > vmax] = np.nan
    RLUT_lag0[RLUT_lag0 < vmin] = np.nan
    RLUT_lag6 = data_lag6.RLUT_INST[0].values
    RLUT_lag6[RLUT_lag6 > vmax] = np.nan
    RLUT_lag6[RLUT_lag6 < vmin] = np.nan
    RLUT_lag12 = data_lag12.RLUT_INST[0].values
    RLUT_lag12[RLUT_lag12 > vmax] = np.nan
    RLUT_lag12[RLUT_lag12 < vmin] = np.nan

    
    ##-- Get GOES-W Tb data, convert it to OLR, and crop to target domain
    
    #- Path to observed GOES-W Tb
    #
    # Data copied to jeanzay from spirit1:/bdd/GEOgrid_coldcloud/
    #
    
    # define observations data path
    parent_dir = '/lustre/fswork/projects/rech/lsv/uae56hc/observations/GEOgrid_coldcloud/GOES-W-1350'
    data_dir = os.path.join(parent_dir,str(target_time.year),target_time.strftime("%Y_%m_%d"))
    # create file name
    file_name = 'GEO_L1C-GOES15_%s-00_G_IR107_004_V1.1.nc'%target_time.strftime("%Y-%m-%dT%H-%M")
    # create file path 
    file_path = os.path.join(data_dir,file_name)
    # Load dataset
    data_obs = xr.open_dataset(file_path)

    #- Format observations to display

    # Select the same region as in Meso-NH
    lon_slice = slice(extent[0]+360,extent[1]+360)
    lat_slice = slice(extent[3],extent[2])
    longitude_obs = data_obs.longitude.sel(longitude=lon_slice)
    latitude_obs = data_obs.latitude.sel(latitude=lat_slice)
    lon_array_obs, lat_array_obs = np.meshgrid(longitude_obs,latitude_obs)
    
    # Convert observed Tb to OLR estimate (W/m2)
    data_OLR = getOLRFromTb(data_obs.Harmonized_irBT.sel(longitude=lon_slice,latitude=lat_slice)).values
    
    # Cap data in range
    data_OLR[data_OLR > vmax] = np.nan
    data_OLR[data_OLR < vmin] = np.nan

    
    ##-- Make figure

    data_crs = ccrs.PlateCarree()  # projection des données lon/lat

    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(10, 10),
                             subplot_kw={'projection': data_crs})
    axs = axs.flatten()
    
    ##-- (0) observations
    ax = axs[0]
    showSubplot(lon_array_obs, lat_array_obs, data_OLR[0], data_crs, extent, vmin=vmin, vmax=vmax)
    ax.set_title('GOES-W (estimated from IR Tb)')
    
    ##-- (1) simulations depuis t=0
    ax = axs[1]
    showSubplot(lon_array, lat_array, RLUT_lag0, data_crs, extent, vmin=vmin, vmax=vmax)
    ax.set_title('$t_0 = $%s'%(dt.strftime(time_ref,"%Y-%m-%dT%H:%M")))
    
    ##-- (2) simulations depuis t=6h
    ax = axs[2]
    showSubplot(lon_array, lat_array, RLUT_lag6, data_crs, extent, vmin=vmin, vmax=vmax)
    ax.set_title('$t_0 = $%s'%(dt.strftime(time_ref+lag_6h,"%Y-%m-%dT%H:%M")))
    
    ##-- (3) simulations depuis t=12h
    ax = axs[3]
    showSubplot(lon_array, lat_array, RLUT_lag12, data_crs, extent, vmin=vmin, vmax=vmax)
    ax.set_title('$t_0 = $%s'%(dt.strftime(time_ref+lag_12h,"%Y-%m-%dT%H:%M")))
    
    save_path = os.path.join(fig_dir,'RC5_GOESW_MNH_%s.png'%(dt.strftime(target_time_MNH,"%Y%m%dT%H%M")))
    plt.savefig(save_path,bbox_inches='tight')


    ##-- Exit
    
    sys.exit(0)