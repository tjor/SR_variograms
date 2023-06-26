#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 11:49:47 2022


@author: tjor
"""

#import sys
#import xarray as xr

# import netCDF4 as nc 
import os
import numpy as np
import numpy.ma as ma
# import csv
# import itertools

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
from scalebar2 import scale_bar

#import scipy
#from scipy.interpolate import interp1d # interpolation and filtering 
#from scipy import signal
#from sklearn.linear_model import LinearRegression

import pandas as pd
import glob

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.geodesic as gd

from datetime import datetime
# import matplotlib.dates as mdates


def plot_shiptracks():

    plt.figure(figsize=(14,8))
    plt.rc('font', size=18)
          
    if deployment == 'Balaton2019':
       extent = [17.890, 17.902, 46.878,  46.890] 
    elif deployment == 'Plymouth2021':
       extent = [-4.40, -4.10, 50.00, 50.40] 
    elif deployment == 'Lisbon2021':
       min_lat = min(np.hstack(np.asarray(LAT)))
       max_lat = max(np.hstack(np.asarray(LAT)))
       min_lon = min(np.hstack(np.asarray(LON)))
       max_lon = max(np.hstack(np.asarray(LON)))
       extent = [min_lon, max_lon, min_lat, max_lat] 
    elif deployment == 'Danube2021': 
       min_lat = min(np.hstack(np.asarray(LAT)))
       max_lat = max(np.hstack(np.asarray(LAT)))
       min_lon = min(np.hstack(np.asarray(LON)))
       max_lon = max(np.hstack(np.asarray(LON)))
       extent = [min_lon, max_lon, min_lat, max_lat] 
       
    request=cimgt.Stamen('terrain-background')
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent, ccrs.Geodetic())
   
    if deployment == 'Balaton2019':
      ax.add_image(request,13)
    elif deployment == 'Plymouth2021':
      ax.add_image(request,11)
    elif deployment == 'Lisbon2021':
      ax.add_image(request,13)
    elif deployment == 'Danube2021': 
      ax.add_image(request,11)

    
#    ax.set_extent(extent, ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True)
    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter =  LONGITUDE_FORMATTER
    gl.yformatter =  LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 18,  'rotation': 0}
    gl.ylabel_style = {'size': 18,  'rotation': 0}
    
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.tick_params(labelsize = 10)
      
    
    print(plot_index)
    for i in range(len(LAT)):
        if len(LAT[i]) > 1:
            plt.plot(LON[i],LAT[i],alpha=0.5,color='black')
    plt.title(deployment_string + ': ship transects')
    plt.plot(LON[i],LAT[i],alpha=0.5,color='black', label= 'All transects')
    if deployment == 'Balaton2019':
        plt.plot(LON[plot_index],LAT[plot_index],alpha=1,linewidth=3,color='magenta',label='Example of daily transect')
    elif deployment == 'Plymouth2021':
        plt.plot(LON[plot_index],LAT[plot_index],alpha=1,linewidth=3,color='magenta',label='Example of daily transect')
    elif deployment == 'Lisbon2021':
        plt.plot(LON[plot_index],LAT[plot_index],alpha=1,linewidth=3,color='magenta',label='Example of daily transect')
    elif deployment == 'Danube2021':
        plt.plot(LON[plot_index],LAT[plot_index],alpha=1,linewidth=3,color='magenta',label='Example of daily transect')
     
    plt.legend(fontsize=16)
    if deployment == 'Balaton2019':
        scale_bar(ax, (0.1, 0.1), 500, metres_per_unit=1, unit_name='m')
        scale_bar(ax, (0.1, 0.1), 500, metres_per_unit=1, unit_name='m')
    elif deployment == 'Plymouth2021':
        scale_bar(ax, (0.1, 0.8), 2500, metres_per_unit=1, unit_name='m')
    elif deployment == 'Lisbon2021':
        scale_bar(ax, (0.1, 0.8), 1000, metres_per_unit=1, unit_name='m') 
    elif deployment == 'Danube2021':
        scale_bar(ax, (0.1, 0.8), 5000, metres_per_unit=1, unit_name='m') 
    
    return



def plot_ref():

    plt.figure(figsize=(8,8))
    plt.rc('font', size=19)
    plt.title('Remote-sensing reflectance: $R_{rs}$')
    colors = cm.plasma(np.linspace(0, 1, len(time)))# color mask to match rrs with time series    
    for i in range(len(time)):
        plt.plot(wl,rrs[i,:],color=colors[i,:],linewidth=1,alpha=0.8)
    plt.xlim(350,900)
    plt.gca().set_ylim(bottom =-0.001)
    plt.grid()
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('$R_{rs}$  [sr$^{-1}$]')
    
    
    return


def plot_daily_coverage():
    
    plt.figure(figsize=(14,8))
    plt.rc('font', size=18)
    
    if deployment == 'Balaton2019':
       extent = [17.890, 17.902, 46.878,  46.890] 
    elif deployment == 'Plymouth2021':
       extent = [-4.40, -4.10, 50.00, 50.40] 
    elif deployment == 'Lisbon2021':
       min_lat = min(np.hstack(np.asarray(LAT)))
       max_lat = max(np.hstack(np.asarray(LAT)))
       min_lon = min(np.hstack(np.asarray(LON)))
       max_lon = max(np.hstack(np.asarray(LON)))
       extent = [min_lon, max_lon, min_lat, max_lat] 
    elif deployment == 'Danube2021': 
       min_lat = min(np.hstack(np.asarray(LAT)))
       max_lat = max(np.hstack(np.asarray(LAT)))
       min_lon = min(np.hstack(np.asarray(LON)))
       max_lon = max(np.hstack(np.asarray(LON)))
       extent = [min_lon, max_lon, min_lat, max_lat] 
    
    request=cimgt.Stamen('terrain-background')
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent, ccrs.Geodetic()) 
       
    if deployment == 'Balaton2019':
      ax.add_image(request,13)
    elif deployment == 'Plymouth2021':
      ax.add_image(request,11)
    elif deployment == 'Lisbon2021':
      ax.add_image(request,13)
    elif deployment == 'Danube2021': 
      ax.add_image(request,11)
       
    gl = ax.gridlines(draw_labels=True)
    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter =  LONGITUDE_FORMATTER
    gl.xlabel_style = {'size': 18,  'rotation': 0}
    gl.ylabel_style = {'size': 18,  'rotation': 0}
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.tick_params(labelsize = 10)
               
    timenum = [mdates.date2num(time[j]) for j in range(len(time))]
    loc = mdates.AutoDateLocator()
    sc = plt.scatter(lon,lat,s=15, c = timenum, cmap ='plasma', vmin = timenum[0], vmax =timenum[-1])
    cbar = plt.colorbar(sc,format=mdates.DateFormatter('%H'))
    cbar.ax.set_yticklabels(['8','9','10','11','12','13','14','15'])
    cbar.set_label('Hour of day (UTC)')   

    plt.legend(fontsize=16)
    if deployment == 'Balaton2019':
        scale_bar(ax, (0.1, 0.1), 500, metres_per_unit=1, unit_name='m')
    elif deployment == 'Plymouth2021':
        scale_bar(ax, (0.1, 0.8), 5000, metres_per_unit=1, unit_name='m')
    elif deployment == 'Lisbon2021':
        scale_bar(ax, (0.1, 0.8), 1000, metres_per_unit=1, unit_name='m') 
    elif deployment == 'Danube2021':
        scale_bar(ax, (0.1, 0.8), 5000, metres_per_unit=1, unit_name='m') 

    plt.title(deployment_string + ': '+ str(time[0].date()) + str())
    plt.legend()
  
    
    return


def plot_shiptracks_daily_cov():
    
      
    fig = plt.figure(figsize=(14,5))    
    plt.rc('font', size=14)
    gs=GridSpec(4,10) # 2 rows, 3 columns
    
      
    if deployment == 'Balaton2019':
        extent = [17.890, 17.902, 46.878,  46.890] 
    elif deployment == 'Plymouth2021':
         extent = [-4.40, -4.10, 50.00, 50.40] 
    elif deployment == 'Lisbon2021':
         extent = [-9.23, -9.11, 38.66, 38.73] 
            
    request=cimgt.Stamen('terrain-background')
    ax = fig.add_subplot(gs[:,5:10], projection=ccrs.PlateCarree())
    ax.set_extent(extent, ccrs.Geodetic())

    
    if deployment == 'Balaton2019':
       ax.add_image(request,13)
    elif deployment == 'Plymouth2021':
       ax.add_image(request,11)
    elif deployment == 'Lisbon2021':
       ax.add_image(request,13)
    elif deployment == 'Danube2021': 
       ax.add_image(request,11)
     
     # gridlines
    gl = ax.gridlines(draw_labels=True)
    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter =  LONGITUDE_FORMATTER
    gl.xlabel_style = {'size': 12,  'rotation': 0}
    gl.ylabel_style = {'size': 12,  'rotation': 0}
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.tick_params(labelsize = 10)
 
    # black transect lines
    plt.suptitle(deployment_string)
    for i in range(len(LAT)):
        if len(LAT[i]) > 1:
            plt.plot(LON[i],LAT[i],alpha=0.3,color='gray',zorder=1)
    plt.plot(LON[i],LAT[i],alpha=0.33,color='gray', label= 'All transects',zorder=1)

    
    # colormap
    time = TIME[plot_index]
    lat = LAT[plot_index]
    lon = LON[plot_index]
    timenum = [mdates.date2num(time[j]) for j in range(len(time))]
    
    Vmin = np.floor(timenum[0]) + (window_center-3)/24;
    Vmax = np.floor(timenum[0]) + (window_center+3)/24;
    hourticks = np.floor(timenum[0])  + [(window_center-3)/24, (window_center-2)/24,  (window_center-1)/24,  window_center/24, (window_center+1)/24, (window_center+2)/24, (window_center+3)/24]
    loc = mdates.AutoDateLocator()
    if deployment == 'Lisbon2021':
        sc = plt.scatter(LON[plot_index], LAT[plot_index],s=15, c = timenum, cmap ='rainbow', label= 'Example day', vmin=Vmin, vmax=Vmax, zorder=2)
        cbar = plt.colorbar(sc,format=mdates.DateFormatter('%H %M'),ticks=hourticks,location="bottom")
    else:
        sc = plt.scatter(LON[plot_index], LAT[plot_index],s=15, c = timenum, cmap ='rainbow', label= 'Example day', vmin=Vmin, vmax=Vmax, zorder=2)
        cbar = plt.colorbar(sc,format=mdates.DateFormatter('%H %M'),ticks=hourticks)
        #cbar.ax.set_yticklabels(['8','9','10','11','12','13'])
    cbar.set_label('Hour of day (UTC)')   

    plt.legend(fontsize=12, loc=plot_loc)
    if deployment == 'Balaton2019':
        scale_bar(ax, (0.1, 0.1), 500, metres_per_unit=1, unit_name='m')
    elif deployment == 'Plymouth2021':
        scale_bar(ax, (0.1, 0.8), 5000, metres_per_unit=1, unit_name='m')
    elif deployment == 'Lisbon2021':
        scale_bar(ax, (0.1, 0.8), 2000, metres_per_unit=1, unit_name='m') 
    elif deployment == 'Danube2021':
        scale_bar(ax, (0.1, 0.8), 5000, metres_per_unit=1, unit_name='m') 
    
      
    #plt.figure(figsize=(13,10))
    #plt.subplot(1,2,2)  
    rrs = RRS[plot_index]
    ax = fig.add_subplot(gs[:,0:4])
    #plt.title('Remote-sensing reflectance: $R_{rs}$')
    timenum_int = [(round(10000*(timenum[j] - timenum[0]))) for j in range(len(time))]
  
    colors = cm.rainbow(np.linspace(0 ,1,round(10000*(Vmax-Vmin))))# color mask to match rrs with time series    
    for j in range(len(time)-1):
        plt.plot(wl,rrs[j,:],color=colors[timenum_int[j]],linewidth=1.5,alpha=0.6)
    plt.xlim(350,900)
    plt.gca().set_ylim(bottom =-0.001)
    plt.grid()
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('$R_{rs}$  [sr$^{-1}$]')
    
    filename  =  fig_dir +  '/' + deployment + '_coverage.png'
    plt.savefig(filename,dpi=900)
    
    return


if __name__ == '__main__':
    
    hours = 3 # window +/- of satelite pass
       
    deployment = 'Balaton2019'
    path_MSI = '/users/rsg/nse/scratch_network/MONOCLE/ac_comparison/msi/data/L2/v1.4.0/UTM_33TYN/' # Balaton car ferry
    path_sr = '/users/rsg/tjor/monocle_network/sorad_rrs_alldeployments/hyperspectral/Balaton2019_3C/'
    deployment_string = '' #'Lake Balaton' #' 28-05-2019 - 05-07-2019'
    plot_index = 26
    plot_loc = 1
    window_center = 11
    
    #deployment = 'Plymouth2021'
    #path_MSI = '/users/rsg/nse/scratch_network/MONOCLE/ac_comparison/msi/data/L2/v1.4.0/UTM_30UVA/'   # Plymouth
    #path_sr = '/users/rsg/tjor/monocle_network/sorad_rrs_alldeployments/hyperspectral/Plymouth2021_3C/'
    #deployment_string = ''#'Plymouth Sound' #' 25-04-2021 - 03-10-2021'
    #plot_index = 81 # 32 also looks good
    #plot_loc = 4
    #window_center = 12
    
   # deployment = 'Lisbon2021'
    #path_MSI = '/data/datasets/Projects/lakes_MSI/data/L2/v1.5.0/UTM_29SMC'
    #path_sr = '/users/rsg/tjor/monocle_network/sorad_rrs_alldeployments/hyperspectral/Lisbon2021_3C/'
    #deployment_string = '' #'Tagus River: 09-06-2021 - 27-11-2021'
    #plot_index = 17
    #plot_loc = 1
    #window_center = 12
    
    # deployment = 'Danube2021'
    # path_MSI = '/data/datasets/Projects/lakes_MSI/data/L2/v1.5.0/UTM_35TPK' # Danube
    # path_sr = '/users/rsg/tjor/monocle_network/sorad_rrs_alldeployments/hyperspectral/Danube2021_3C/'
    # deployment_string = 'Danube Delta, 2021'
    # plot_index = 5
        
    meta_files = sorted(glob.glob(os.path.join(path_sr, '*data*')))     # 
    rrs_files = sorted(glob.glob(os.path.join(path_sr, '*rrs*')))  #    
    # files_MSI = glob.glob(os.path.join(path_MSI, '*UTM*'))  # all S2A and S2
    wl = np.arange(340, 900, 1) # all 3C and FP(pre 2019) 
    
    fig_dir = '/users/rsg/tjor/monocle/sat_val/Paperplots/'
    
    # 1. unpack lat-lon for deployment for 6 hr windows for ship track plot
    LAT = []
    LON = []
    RRS = []
    TIME = []
    for i in range(len(meta_files)): # use to output best plot for paper
        print(i)
        #  UTMdata = xr.open_dataset(files_MSI[0]) # picks first UTM tile (doesnt really matter - this is just to resample insitu data to MSI grid)
        #  meta_MSI = UTMdata.attrs
            
        meta = pd.read_csv(os.path.join(meta_files[i]))  #
        time = [datetime.strptime(meta['timestamp'][j][0:19],'%Y-%m-%d %H:%M:%S') for j in range(len(meta))] # convert
        time = np.array(time)  # convert to np array for masking
        lat =  np.array(meta['lat'])
        lon =  np.array(meta['lon'])
        q =  np.array(meta['q_3'])
        rrs = np.loadtxt(open(rrs_files[i],"rb"), delimiter=",")
        
        lat = lat[q==1] # filter in-situ rrs subject to qc mask
        lon = lon[q==1] 
        rrs = rrs[q==1, :]
        time = time[q==1]
        
        if len(time) > 0: 
            time_sat = datetime(year = time[0].year, month = time[0].month, day = time[0].day, hour = window_center) # this is not the real time (it is centre point of window)   
            tol = int(hours*3600) # time window tolerance in seconds
            deltat = np.abs(np.array([(time_sat - time[i]).total_seconds() for i in range(len(time))])) #   
    
            lat = lat[deltat < tol] #  filter in-situ rrs subject to time window mask
            lon = lon[deltat < tol]
            rrs = rrs[deltat < tol, :]
            time = time[deltat < tol]
            
        LAT.append(lat)
        LON.append(lon)
        TIME.append(time)
        RRS.append(rrs)

    #  plot_index = plot_index 
    #  plot_shiptracks()  
    plot_shiptracks_daily_cov()

    #  2. Rrs and time series plots
    #  for i in range(plot_index, plot_index + 1): # use to output plot for paper
       # for i in range(len(meta_files)): # use to output plot for 
          #  UTMdata = xr.open_dataset(files_MSI[0]) # picks first UTM tile (doesnt really matter - this is just to resample insitu data to MSI grid)
           # meta_MSI = UTMdata.attrs
        
           # #meta = pd.read_csv(os.path.join(meta_files[i]))  #
           # time = [datetime.strptime(meta['timestamp'][j][0:19],'%Y-%m-%d %H:%M:%S') for j in range(len(meta))] # convert
           # time = np.array(time)  # convert to np array for masking
          #  lat = np.array(meta['lat'])
          #  lon = np.array(meta['lon'])
          #  q = np.array(meta['q_3'])
          #  rrs = np.loadtxt(open(rrs_files[i],"rb"), delimiter=",")
            
          #  lat = lat[q==1] # filter in-situ rrs subject to qc mask
         #   lon = lon[q==1] 
         #   rrs = rrs[q==1,:]
         #   time = time[q==1]
            
         #   if len(time) > 0: 
         #            if abs(np.nanmin(lat)-np.nanmax(lat)) > 0.05:
         #         time_sat = datetime(year = time[0].year, month = time[0].month, day = time[0].day, hour = 11) # this is not the real time (it is centre point of window)   
         #        tol = int(hours*3600) # time window tolerance in seconds
         #        deltat = np.abs(np.array([(time_sat - time[i]).total_seconds() for i in range(len(time))])) #   
            
              #      lat = lat[deltat < tol] #  filter in-situ rrs subject to time window mask
             #       lon = lon[deltat < tol]
             #       rrs = rrs[deltat < tol, :]
             #       time = time[deltat < tol]
                    
                  #  plot_ref()
                    #plot_daily_coverage()
#
    
           
    
         
         
    
