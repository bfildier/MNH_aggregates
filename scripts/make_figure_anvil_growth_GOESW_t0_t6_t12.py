import xarray as xr
import numpy as np
import numpy.ma as ma
from math import floor
from datetime import datetime as dt
from datetime import timedelta

import argparse

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import glob, sys, os, re
from pprint import pprint
import warnings
import pickle

##-- Directories

sim_dir = "/lustre/fsn1/projects/rech/lsv/ubd61nl/ObedSaba_RECONCILE_MesoNH/production/WPAGG_2km"
fig_dir = '../figures'
out_dir = '../results'

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
    if delta_t.total_seconds() >= 0:
        # number of output steps
        n_out = int(delta_t.total_seconds()/3600/Dt - 1)
        # compute corresponding subfolder number
        i_seg = floor(n_out / (12/Dt) ) +1 # segment number
        # compute corresponding file step
        i_t_seg = int(n_out % int(12/Dt) + 1)
        # build corresponding subfolder
        subfolder = "seg0%d_12h_tege"%i_seg
        # build corresponding filename
        filename = "FBC01.1.MNH0%d.OUT.%s.nc"%(i_seg,str(i_t_seg).zfill(3))
        # full path
        path = os.path.join(sim_dir,sim_names[i_lag],subfolder,filename)
        
    else:
        path = None

    return path

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

#- Conversion to string

def dateToStr(dt:datetime,fmt="%Y%m%dT%H%M"):

    return dt.strftime(fmt)

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

#- Loader


def getDataAtTimeGOESW(target_time):
    
    # define observations data path
    parent_dir = '/lustre/fswork/projects/rech/lsv/uae56hc/observations/GEOgrid_coldcloud/GOES-W-1350'
    data_dir = os.path.join(parent_dir,str(target_time.year),target_time.strftime("%Y_%m_%d"))
    # create file name
    file_name = 'GEO_L1C-GOES15_%s-00_G_IR107_004_V1.1.nc'%target_time.strftime("%Y-%m-%dT%H-%M")
    # create file path 
    file_path = os.path.join(data_dir,file_name)
    # Load dataset
    if len(glob.glob(file_path)) > 0 :
        with warnings.catch_warnings(action="ignore"):
            data_obs = xr.open_dataset(file_path)
    else:
        data_obs = None

    return data_obs
    
def getDataset(filepath):

    if filepath is not None and not re.match(r".*000.nc", filepath):
        if len(glob.glob(filepath)) > 0 :
            with warnings.catch_warnings(action="ignore"):
                return xr.open_dataset(filepath)
    else:
        return None

#- Analysis

def computeFractionOLR(dataset,var_thres):

    if dataset is not None:
            
        # get data
        var = dataset.RLUT_INST[0]
        # compute fraction
        frac = (np.sum(var < var_thres)/var.size).values

    else:

        frac = None

    return frac

#- Compute the fraction of the domain below Tb_ref / OLR_ref

def computeDomainFractionGOESW(data_lag0,data_obs,Tb_ref):
    
    #- Domain coords
    
    if data_lag0 is not None:
        
        lon_array = data_lag0.longitude.values
        lat_array = data_lag0.latitude.values
        longitude = data_lag0.longitude[0].values
        latitude = data_lag0.latitude[:,0].values
        extent = [longitude[0],longitude[-1],latitude[0],latitude[-1]]
        
        # Select the same region as in Meso-NH
        lon_slice = slice(extent[0]+360,extent[1]+360)
        lat_slice = slice(extent[3],extent[2])
    
    else :
        lon_slice = lat_slice = slice(None)
    
    if data_obs is not None:
        
        # Get variable
        Tb_obs = data_obs.Harmonized_irBT.sel(longitude=lon_slice,latitude=lat_slice)
        
        # Compute fraction below Tb
        Tb_frac_obs = (np.sum(Tb_obs < Tb_ref)/Tb_obs.size).values
    
    else:
    
        Tb_frac_obs = None

    return Tb_frac_obs

#- String formatting

def toStr(value):
    """Format float or None to string"""

    if value is not None:
        return f"{value:,.4f}"
    else:
        return ''


##-- Main execution

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Get parameters for figure.')
    parser.add_argument('--Tb_ref', type=int, default="235", help='Threshold value of brightness temperature, in K.')
    parser.add_argument('--title', type=str, default="Anvil cloud fraction", help='Figure title.')
    parser.add_argument('--overwrite', type=bool, default=False, help='Overwrite calculation results stored on disk.')
    parser.add_argument('--verbose', type=bool, default=False, help='Show printed information.')
    args = parser.parse_args()

    #- Define threshold OLR
    Tb_ref = args.Tb_ref
    OLR_ref = getOLRFromTb(Tb_ref)

    # Input time info
    time_freq_GOESW = 1800 # s
    time_freq_MNH = 1200 # s
    
    # Simulation lag info
    lags_h = np.array([0,6,12])
    
    ##-- Load data ref MNH for domain information
    
    # reference time with valid MNH data
    valid_time = dt(2016,9,10,12,00)
    # valid data
    data_ref_MNH = getDataset(getMNHFileAtTime(valid_time))
    
    ##-- Compute domain fraction at all times
    
    N_t = int(6*24*1.5) # Number of time slices
    t_delta = timedelta(seconds=600) # 10-minute increments
    time_ref = dt(2016,9,10)

    # saving info
    fileroot = 'RC5_frac_Tb_%d_GOESW_t0_t6_t12'%Tb_ref
    filename = fileroot+'.pickle'
    filepath = os.path.join(out_dir,filename)
    
    if args.overwrite or len(glob.glob(filepath)) == 0:
    
        # Initialize array of anvil fractions
        frac_array = np.full((4,N_t),np.nan)
        
        # Iterate over time
        for i_t in range(N_t):
        
            # time to show
            target_time = time_ref + i_t*t_delta
            # Adjusted time on MNH time array
            target_time_MNH = getAdjustedTime(target_time,time_freq_MNH)
            # Adjusted time on GOES-W time array
            target_time_GOESW = getAdjustedTime(target_time,time_freq_GOESW)
        
            # Get data for GOES-W
            data_obs = getDataAtTimeGOESW(target_time_GOESW)
            # Compute domain fractions for GOES-W
            frac_obs = computeDomainFractionGOESW(data_ref_MNH,data_obs,Tb_ref)
            
            # Get data for MNH
            data_lag0 = getDataset(getMNHFileAtTime(target_time_MNH,lag_h=0))
            data_lag6 = getDataset(getMNHFileAtTime(target_time_MNH,lag_h=6))
            data_lag12 = getDataset(getMNHFileAtTime(target_time_MNH,lag_h=12))
        
            # Compute fraction for MNH
            frac_lag0 = computeFractionOLR(data_lag0,OLR_ref)
            frac_lag6 = computeFractionOLR(data_lag6,OLR_ref)
            frac_lag12 = computeFractionOLR(data_lag12,OLR_ref)
        
            # Save calculation
            frac_array[0,i_t] = frac_obs
            frac_array[1,i_t] = frac_lag0
            frac_array[2,i_t] = frac_lag6
            frac_array[3,i_t] = frac_lag12
        
            if args.verbose:
                print("%s : MNH at %s, GOES-W at %s -- Obs: %s ; MNH t0: %s ; MNH t0+6h: %s ; MNH t0+12h: %s ; "%(dateToStr(target_time),
                                                                                                                  dateToStr(target_time_MNH),
                                                                                                                  dateToStr(target_time_GOESW),
                                                                                                                  toStr(frac_obs),
                                                                                                                  toStr(frac_lag0),
                                                                                                                  toStr(frac_lag6),
                                                                                                                  toStr(frac_lag12)))
        
        ##-- Store calculation results on disk
        pickle.dump(frac_array,open(filepath,'wb'))
    
    else:

        if args.verbose:
            print("Reload %s"%filename)
            
        frac_array = pickle.load(open(filepath,'rb'))


    ##-- Display

    # Time array
    time_array = np.arange(np.datetime64(time_ref),np.datetime64(time_ref+N_t*t_delta),np.timedelta64(t_delta))

    fig,ax = plt.subplots(figsize=(5,5))
    
    # GOES-W
    ax.plot(time_array,frac_array[0]*100,'k',label='GOES-W')
    # t_0
    ax.plot(time_array,frac_array[1]*100,'brown',label=r'MNH $t_0$ = 00:00')
    # t_0 + 6h
    ax.plot(time_array,frac_array[2]*100,'red',label=r'MNH $t_0$ = 06:00')
    # t_0 + 12h
    ax.plot(time_array,frac_array[3]*100,'orange',label=r'MNH $t_0$ = 12:00')
    
    # legend
    ax.legend()
    # axes
    ax.set_ylabel('Fraction > %dK (%%)'%Tb_ref)
    plt.xticks(rotation=45)
    # title
    ax.set_title(args.title)
    
    # save
    filename = fileroot+'.pdf'
    plt.savefig(os.path.join(fig_dir,filename),bbox_inches='tight')


    ##-- Exit
    sys.exit(0)