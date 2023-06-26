#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 14:14:43 2021
author: tj9717 - 

Code for Rrs variogram paper submitted to Fronteirs in RS. When running on PML
machine use: /users/rsg/tjor/.conda/envs/geospatial/bin/ as conda evn.

"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 12:35:48 2021

@author: tjor
"""

import sys
#sys.path.append('/users/rsg/anla/anaconda3/envs/geospatial/*')
#import xarray as xr

#import netCDF4 as nc 
import os
import numpy as np
import numpy.ma as ma
import csv
import itertools

from scalebar2 import scale_bar
import sklearn
import skgstat as skg

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec


import scipy
# from scipy.interpolate import interp1d # interpolation and filtering 
# from scipy import signal
# from sklearn.linear_model import LinearRegression

import pandas as pd
import glob


import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.geodesic as gd

from datetime import datetime
import fnmatch
import utm


def plot_data_map_V2(data_is, bands, title_string, map_resolution=12):  
    '''plots coverage map for each deploymnet - not used in MS'''
    
    lat = data_is['lat']
    lon = data_is['lon']
    
    lat_min = (1/1000)*np.floor(np.min(lat*1000)) - 0.0010
    lat_max = (1/1000)*np.ceil(np.max(lat*1000)) +  0.0010
         
    lon_min = (1/1000)*np.floor(np.min(lon*1000)) - 0.0010
    lon_max = (1/1000)*np.ceil(np.max(lon*1000)) +  0.0010
           
    extent = [lon_min, lon_max, lat_min,  lat_max]
    # extent = [17.888, 17.904, 46.878, 46.890]   Balaton deafualt

    if deployment == 'Balaton2019':
        #fig = plt.figure(figsize=(19,13),dpi=300)    
        fig = plt.figure(figsize=(19,13))    
    elif deployment == 'Plymouth2021':
        fig = plt.figure(figsize=(16,13))        
      
    plt.suptitle(title_string)
            
    plt.rc('font',size=20)
    for i in range(len(bands)):
        rrs = data_is['rrs_' +  bands[i]]    
        if i == 0:
            plot_no = 221
        elif i == 1:
            plot_no = 222
        elif i == 2:
            plot_no = 223
        elif i ==3:
            plot_no = 224
            
        ax = fig.add_subplot(plot_no, projection=ccrs.PlateCarree(central_longitude=17.901))
        # request = cimgt.StamenTerrain()
        request=cimgt.Stamen('terrain-background')
        ax.add_image(request, map_resolution)
        ax.set_extent(extent, ccrs.Geodetic())

        if deployment == 'Plymouth2021':
            scale_bar(ax, (0.6, 0.1), 1500, metres_per_unit=1, unit_name='m')
        elif deployment == 'Balaton2019':
            scale_bar(ax, (0.1, 0.1), 500, metres_per_unit=1, unit_name='m')
      
        gl = ax.gridlines(draw_labels=True)
        # gl.xlabels_top =
        
        gl.xlabels_top = gl.ylabels_right = False
        gl.xformatter =  LONGITUDE_FORMATTER
        gl.yformatter =  LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 18,  'rotation': 0}
        gl.ylabel_style = {'size': 18,  'rotation': 0}
        
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.tick_params(labelsize = 10)
        plt.scatter(lon,lat, c = rrs, cmap ='viridis', vmin = np.percentile(rrs,5), vmax = np.percentile(rrs,95),  transform=ccrs.PlateCarree())
        plt.colorbar(fraction=0.035, pad=0.04)
        plt.title ('$R_{rs}$('+ bands[i] + ') [sr$^{-1}$]')
        
    #    plt.savefig('testmap.png', dpi=400)
     #   plt.close()
       
    return


def plot_data_map_zoom(data_is, bands, title_string, map_resolution=12):  
    '''rrs plot function for match-up - not used in MS'''
    
    lat = data_is['lat']
    lon = data_is['lon']
    
    lat_min = (1/1000)*np.floor(np.min(lat*1000)) - 0.0010
    lat_max = (1/1000)*np.ceil(np.max(lat*1000)) +  0.0010
         
    lon_min = (1/1000)*np.floor(np.min(lon*1000)) - 0.0010
    lon_max = (1/1000)*np.ceil(np.max(lon*1000)) +  0.0010
           
    extent = [lon_min, lon_max, lat_min,  lat_max]
    # extent = [17.888, 17.904, 46.878, 46.890]   Balaton deafualt

    if deployment == 'Balaton2019':
        #fig = plt.figure(figsize=(19,13),dpi=300)    
        fig = plt.figure(figsize=(19,13),dpi=300)    
    elif deployment == 'Plymouth2021':
        fig = plt.figure(figsize=(16,13))        
      
    plt.suptitle(title_string)
            
    plt.rc('font',size=20)
  
    rrs = data_is['rrs_' +  bands[1]]    

    ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=17.901))
    # request = cimgt.StamenTerrain()
    request=cimgt.Stamen('terrain-background')
    ax.add_image(request, map_resolution)
    ax.set_extent(extent, ccrs.Geodetic())

    if deployment == 'Plymouth2021':
        scale_bar(ax, (0.6, 0.1), 1500, metres_per_unit=1, unit_name='m')
    elif deployment == 'Balaton2019':
        scale_bar(ax, (0.1, 0.1), 500, metres_per_unit=1, unit_name='m')
  
    gl = ax.gridlines(draw_labels=True)
    # gl.xlabels_top =
    
    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter =  LONGITUDE_FORMATTER
    gl.yformatter =  LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 18,  'rotation': 0}
    gl.ylabel_style = {'size': 18,  'rotation': 0}
    
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.tick_params(labelsize = 10)
    plt.scatter(lon,lat, c = rrs, cmap ='viridis', vmin = np.percentile(rrs,5), vmax = np.percentile(rrs,95),  transform=ccrs.PlateCarree())
    plt.colorbar(fraction=0.035, pad=0.04)
  #  plt.title ('$R_{rs}$('+ bands[i] + ') [sr$^{-1}$]')
        
    #    plt.savefig('testmap.png', dpi=400)
     #   plt.close()
       
    return




    
def variogram_rrs_bands(matchdata, lag_bound, n_lags, bands):
    
    ''' Function to compute experimental and theoretical variograms, along with VG fit data.
    Uses Gaussian VG model and nMAE as filter parameter for fit.'''
    
    X = np.asarray(utm.from_latlon(np.array(matchdata['lat']),np.array(matchdata['lon']))[0:2]).T # coordinates
    
    gamma_e = [] # empty lists to append semivariance vectors: experimental VG notated by _e subscript
    gamma = [] #
    x_e = [] #
    x = [] 
    Q = []
    VG_data = {}

    for i in range(len(bands)):   
        V = skg.Variogram(X, matchdata['rrs_' + str(bands[i])], fit_method ='lm',model = 'gaussian', n_lags = 12, maxlag = lag_bound, fit_sigma ='linear', normalize=True, use_nugget=True, Force=False) #  VG function
        # V = skg.Variogram(X, matchdata['rrs_' + str(bands[i])], fit_method ='lm',model = 'exponential', n_lags = 12, maxlag = lag_bound,normalize=False, use_nugget=True,Force=False) #  VG function
        

        b = V.describe()['nugget'] # gaussian fit parameters to theoretical VG
        c0 = V.describe()['sill'] #  model
        a = V.describe()['effective_range']/2
      
        x_e.append(V.__dict__['_bins']) # experimental distance (lag)        
        gamma_e.append(V.experimental) # experimental VG (aka. emperical semi-variance)

        x_i = np.arange(0,lag_bound + 100, 1)
        gamma_i = b + c0*(1 - np.exp(-(x_i*x_i)/(a*a))) # theoretical VG (aka.fitted semi-variance). gaussiam
        gamma_i_e = b + c0*(1 - np.exp(-(V.__dict__['_bins']*V.__dict__['_bins'])/(a*a))) # theoretical VG down-sampled to experimental bins - not currently used in output
       
 
        nMAE = 100*np.mean(abs(np.sqrt(gamma_i_e) - np.sqrt(V.experimental))/np.sqrt(V.experimental)) # mean % error    - not nMAE               
                                            
        x.append(x_i)
        gamma.append(gamma_i)
  
        # Quality control flag:
        Q_i = 1
        if b < 0 or c0 < 0 or a < 0 or nMAE > 12:   
            Q_i = 0
        Q.append(Q_i)
        
        N_pairs = np.fromiter(
            (g.size for g in V.lag_classes()), dtype=int
        )
       
     
        VG_data.update({'Q_' + str(bands[i]): Q, 'b_' + str(bands[i]): b, 'c0_'+ str(bands[i]): c0, 'a_' + str(bands[i]):  a, 'nMAE_' + str(bands[i]):  nMAE}) # fit data that depends on band
     
    VG_data.update({'N_points': len(X),'max_dist': np.max(scipy.spatial.distance.pdist(X)),'mean_dist': np.mean(scipy.spatial.distance.pdist(X))}) # fit data that does not depend on band
                                  

    return X, x_e, gamma_e, x, gamma, Q, VG_data, N_pairs


def rrs_band_mean(matchdata,bands):
 
    'Survey-mean Rrs'
    
    mean_rrs = []
    for i in range(len(bands_MSI)):
        mean_rrs_i = np.mean(matchdata['rrs_' + str(bands_MSI[i])])
        mean_rrs.append(mean_rrs_i)
    
    return mean_rrs



def variogram_bandplot_V2(x_e, gamma_e, x, gamma, mean_rrs, VG_data, date):
    
    ' not used in MS - superseeded by  plot_rrs_VG()'
    
    plt.figure(figsize=(34,8),dpi=300) 
    plt.suptitle(str(date) + '  ' + str(j))
    colors=['b','g','r','k']
    
    plt.subplot(1,3, 1)     
    for i in range(len(bands_MSI)):           
        
        L = abs(2*VG_data['a_' + bands_MSI[i]]) # really L
        C_0 = VG_data['b_' + bands_MSI[i]] # really c_0
        C_infty = VG_data['c0_' + bands_MSI[i]] + VG_data['b_' + bands_MSI[i]]
        
        # really c_0-c_infty
       # plt.title(bands_MSI[i] + ' : nMAE = '  + str(round(VG_data['nMAE_' + bands_MSI[i]],2)) 
               #     + ' 2a = ' + f'{2*a:.3}' +   '\n' +', \gamma_{0} = ' + f'{b:.3}' +  ', \gamma_{\infty}  = ' + f'{c0:.3}')
        plt.plot(x[i],gamma[i], color=colors[i],label=str(bands_MSI[i]) + ': fit error = '  + str(round(VG_data['nMAE_' + bands_MSI[i]],2)) 
                  + '$\%$, L = ' + f'{L:.3}' + '\n' +  '$C_{0}$ = ' + f'{C_0:.3}' +  ', $C_{\infty}$  = ' + f'{C_infty:.3}')
        plt.legend(loc='lower left', bbox_to_anchor=(-1, 0.5))
 
    
        plt.scatter(x_e[i],gamma_e[i], color=colors[i])
        #plt.gca().set_ylim(bottom = 0)
      
        plt.xlim(0,500)
        #plt.xlim(0,1000)
        
        plt.rc('font', size=20)
        if i == 2 or i == 3:
            plt.xlabel('Lag [m]')
        if i == 0 or i == 2:
            plt.ylabel('Semivariance: $\gamma$   [sr$^{-2}$]')
    plt.gca().set_ylim(bottom = 0)    
       
    plt.subplot(1,3, 2)    
    for i in range(len(bands_MSI)):           
        
         a = VG_data['a_' + bands_MSI[i]]
         b = VG_data['b_' + bands_MSI[i]]
         c0 = VG_data['c0_' + bands_MSI[i]]
         #plt.title(bands_MSI[i] + ' : nMAE = '  + str(round(VG_data['nMAE_' + bands_MSI[i]],2)) 
        #     + ' a = ' + f'{a:.3}' +   '\n' +', b = ' + f'{b:.3}' +  ', c0 = ' + f'{c0:.3}')
         plt.plot(x[i],np.sqrt(gamma[i]), color=colors[i],label=str(bands_MSI[i]))
         plt.scatter(x_e[i],np.sqrt(gamma_e[i]), color=colors[i])

         if deployment == 'Balaton2019':
            plt.xlim(0,500)
         elif deployment == 'Plymouth2021':
            plt.ylim(0,0.001)  
   
    
        
         plt.rc('font', size=18)
         if i == 2 or i == 3:
            plt.xlabel('Lag [m]')
         if i == 0 or i == 2:
            plt.ylabel('$\sqrt{\gamma}$  [sr$^{-1}$]')
    #plt.gca().set_ylim(bottom = 0)  
            
    plt.subplot(1,3, 3)    
    for i in range(len(bands_MSI)):           
        
        # a = VG_data['a_' + bands_MSI[i]]
        # b = VG_data['b_' + bands_MSI[i]]
        # c0 = VG_data['c0_' + bands_MSI[i]]
        # plt.title(bands_MSI[i] + ' : nMAE = '  + str(round(VG_data['nMAE_' + bands_MSI[i]],2)) 
               #     + ' a = ' + f'{a:.3}' +   '\n' +', b = ' + f'{b:.3}' +  ', c0 = ' + f'{c0:.3}')
         plt.plot(x[i],np.sqrt(gamma[i])/mean_rrs[i], color=colors[i],label=str(bands_MSI[i]))
         plt.scatter(x_e[i],np.sqrt(gamma_e[i])/mean_rrs[i], color=colors[i])
  
         plt.legend()
         if deployment == 'Balaton2019':
            plt.xlim(0,500)
         elif deployment == 'Plymouth2021':
            plt.ylim(0,0.001)  
        
         plt.rc('font', size=18)
         if i == 2 or i == 3:
            plt.xlabel('Lag [m]')
         if i == 0 or i == 2:
            plt.ylabel(r'$\tilde{CV}[R_{rs}] = \sqrt{\gamma}/\mu[R_{rs}]$')
    plt.ylim(0,0.25)
    plt.tight_layout()
            
    return



def var_plots():
    ' Spatial variance percentage at 300 m length scale'

    plt.figure(figsize=(15,4),dpi=300);
   # plt.suptitle('Percentage of $R_{rs}$ variability due to spatial structure at OLCI pixel scale: ' + deployment_string)      
    rootgamma = np.sqrt(gamma)  
    colors = ['royalblue','limegreen','red','lightgray']        
    for i in range(len(bands_MSI)):   
        plt.subplot(1,4, i +1)    
        #if deployment == 'Balaton2019':
        plt.title(str(bands_MSI[i]) + ' nm')
        R_300 = (rootgamma[:,i, 300]-rootgamma[:,i, 0])/(rootgamma[:,i,300]) # fraction of variance resolved at 300 m 
        R_300 = R_300[Q[:,i]==1]
        plt.xlabel('$f_{300}$ [$\%$]') 
        if i == 0:
            plt.ylabel('Frequency')   
        plt.hist(100*R_300,bins = 10, color=colors[i], edgecolor='k',range=(0,100),label= 'Median = \n' + str(round(np.median(100*R_300))) + ' $\%$')                               
        plt.legend(fontsize=14)
        plt.xlim(0,100)
        if deployment == 'Lisbon2021':
            plt.ylim(0,45)
        elif deployment == 'Plymouth2021':
            plt.ylim(0,12)
        elif deployment == 'Balaton2019':
            plt.ylim(0,5)
        ax=plt.gca()
        plt.text(.05, .95,  subplotlab[i], ha='left', va='top', transform=ax.transAxes,fontsize=18) 
        
    plt.tight_layout()  
    
    filename  =  fig_dir +  '/' + deployment + '_f300_hist.png'
    plt.savefig(filename,dpi=900)
    
    return


def Nbins_plots():
    
    ' number of samples in each VG bin: not used in paper'
   
    plt.figure(figsize=(10,5),dpi=300) 
    # Nbins = Nbins[Q[:,0]==1] + Nbins[Q[:,1]==1] + Nbins[Q[:,2]==1] + Nbins[Q[:,3]==1]
    bp = plt.boxplot(Nbins[Q[:,1]==1], showfliers=True,patch_artist=True, medianprops=dict(color='black'), whis=[10,90]) 
    plt.ylim(0,800)
    plt.xlim(0.5, 13)
    plt.xticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5], ['0','50','100','150', '200', '300', '300','350', '400', '450',  '500',  '550', '600'])
    plt.xlabel('GSD: h')
    plt.ylabel('Number of samples')

    return

def CV_plots():
    ' intrinsic CV plot'
   
    plt.figure(figsize=(15,4))
    # plt.suptitle('Intrinsic coefficient of variation: ' + deployment_string)      
    root_gamma = np.sqrt(gamma)             
    colors = ['royalblue','limegreen','red','lightgray']            
    
    for i in range(len(bands_MSI)):   
        plt.subplot(1,4, i +1)    
        #if deployment == 'Balaton2019':
        plt.title(str(bands_MSI[i]) + ' nm')
        CV_0 = 100*root_gamma[:,i, 0]/mean_rrs[:,i]
        CV_0 = CV_0[Q[:,i]==1]
        plt.xlabel(r'$\tilde{CV}(0)$'  + '[%]')    
        print(CV_0)
        
    
        if i == 0:
            plt.ylabel('Frequency')   
         
        plt.hist(CV_0, bins = 9, edgecolor='k', color=colors[i],range=(0,45), label = 'Median =' + '\n' + str(int(np.round(np.median(CV_0)))) + '%')
        plt.legend(fontsize=14)
        plt.tight_layout()  
        plt.xlim(0,45)
        ax = plt.gca()
        plt.text(.05, .95,  subplotlab[i], ha='left', va='top', transform=ax.transAxes,fontsize=18) 
        if deployment == 'Lisbon2021':
            plt.ylim(0,28)
        elif deployment == 'Plymouth2021':
            plt.ylim(0,10)
        elif deployment == 'Balaton2019':
            plt.ylim(0,8)
       
        # plt.ylim(0,8.5)
        #  plt.figure(figsize=(14,10));
        #  plt.suptitle('Coefficient of variation at 300 m')               
        #  for i in range(len(bands_MSI)):   
        #  plt.subplot(2,2, i +1)    
        #  plt.title(str(bands_MSI[i]))
        #  CV_0 = root_gamma[:,i, 300]/mean_rrs[:,i]
        #  plt.xlabel('$\sqrt{\gamma(300)}/\mu[R_{rs}]$')    
        #  plt.ylabel('Frequency')   
        #  plt.hist(CV_0,bins = 10,color=colors[i],edgecolor='k', range=(0,0.2))
        #   plt.tight_layout()  
        
    filename  =  fig_dir +  '/' + deployment + '_CV_0_hist.png'
    plt.savefig(filename,dpi=900)
    
    return


def range_plots():
    ' auto-correlation length plot'
    
    plt.figure(figsize=(15,4),dpi=900);
    # plt.suptitle('$R_{rs}$ correlation length: ' + deployment_string)     
    colors = ['royalblue','limegreen','red','lightgray'] 
    
    if deployment == 'Balaton2019':
        hist_range = (0,1000)
        n_bins = 8
    else:
        hist_range = (0,2000)
        n_bins = 8
    
    
    for i in range(len(bands_MSI)): 
        plt.subplot(1,4, i +1)    
       # if deployment == 'Balaton2019':
        plt.title(str(bands_MSI[i]) + ' nm')
        a_i = np.array([VG_data[j]['a_' + str(bands_MSI[i])] for j in range(len(VG_data))])  #
        med_L = np.median(np.sqrt(3)*a_i)
        #plt.ylim(0,6.5)
        plt.hist(np.sqrt(3)*a_i,bins = n_bins,color= colors[i],edgecolor='k', range=hist_range, label = 'Median =' + '\n' + str(round(med_L)) + ' m')  # NOTE THIS DEFINES AC-length using sqrt(3 def)
        if i == 0:
            plt.ylabel('Frequency')   
        plt.legend(fontsize=14)   
        plt.xlabel('L [m]')
       
        if deployment == 'Lisbon2021':
            plt.ylim(0,13)
            plt.xlim(0,2000)
        elif deployment == 'Plymouth2021':
            plt.ylim(0,6)
            plt.xlim(0,2000)
        elif deployment == 'Balaton2019':
            plt.ylim(0,8)
            plt.xlim(0,1000)
        ax=plt.gca()
        plt.text(.05, .95,  subplotlab[i], ha='left', va='top', transform=ax.transAxes,fontsize=18) 
    plt.tight_layout()  
    
    filename  =  fig_dir +  '/' + deployment + '_rangehist.png'
    plt.savefig(filename,dpi=900)
    
    
    return


def pairwise_xt():
    'pairwise space and time plots'   
    
    plt.figure(figsize=(6,10),dpi=300)
    plt.suptitle(deployment_string)
    plt.subplot(2,1,1)
    plt.hist(pairwise_time, weights=np.ones_like(pairwise_time)/len(pairwise_time),bins = 12,range=(0,6),color='royalblue',edgecolor='k')
    plt.xlabel('Pairwise time difference [hours]')
    plt.ylabel('Normalized frequency')
    plt.xlim(0,6)
    
    plt.subplot(2,1,2)
    if deployment == 'Balaton2019': # force axes within each deployment for consitency
         histrange = (0,1200)
         xlimit = 1200
    elif deployment == 'Plymouth2021':
         histrange = (0,36000)
         xlimit = 36000
    elif deployment == 'Lisbon2021':
         histrange = (0,8000)
         xlimit = 8000
         
    plt.hist(pairwise_distance, weights=np.ones_like(pairwise_distance)/len(pairwise_distance),bins = 12, range =histrange,color='gray',edgecolor='k')
    plt.xlabel('Pairwise distance [m]')
    plt.ylabel('Normalized frequency')
    plt.tight_layout()  
    plt.xlim(0, xlimit)
    
    filename  =  fig_dir +  '/' + deployment + '_xt.png'
    plt.savefig(filename,dpi=900)
        
    return


def VGsummary_plots_V2():  
    ' summary plots for CV- and root-variograms: as used in paper sub'
    
   
    plt.figure(figsize = (14,4),dpi=300)
    for k in range(len(bands_MSI)):    
        # calculate median fit parameters
        a = []
        b = []
        c0 = []
        for i in range(len(VG_data)):
           VG_data_i = VG_data[i]
           a.append(VG_data_i['a_' + bands_MSI[k]])
           b.append(VG_data_i['b_' + bands_MSI[k]])
           c0.append(VG_data_i['c0_' + bands_MSI[k]])
        a = np.stack(a)[Q[:,k]==1]
        b = np.stack(b)[Q[:,k]==1]
        c0 = np.stack(c0)[Q[:,k]==1]
       
        a_med = np.median(a) # median paramters for med curve
        b_med = np.median(b)
        c0_med = np.median(c0)
        
        # median curves
        x = x_j[0]
        CV_med = np.sqrt(b_med + c0_med*(1 - np.exp(-(x*x)/(a_med*a_med))))/np.median(mean_rrs,0)[k] 
        CV_k = np.sqrt(gamma[:,k,:][Q[:,k]==1].T)/mean_rrs[:,k][Q[:,k]==1].T
       
        plt.subplot(1,4,k + 1)  
        colors_2 = [cm.cool(np.linspace(0, 1, len(CV_k.T))),cm.summer(np.linspace(0, 1, len(CV_k.T))), cm.autumn(np.linspace(0, 1, len(CV_k.T))), cm.bone(np.linspace(0, 1, len(CV_k.T)))]
        for j in range(len(CV_k.T)):
            plt.plot(x,100*CV_k[:,j],alpha=0.8,linewidth=1.5,color =colors_2[k][j])
        plt.plot(x,100*CV_med,color ='black',linewidth=4, label = 'Median curve')
        plt.xlim(0,600)
        if k == 0:
             plt.ylabel(r'$\tilde{CV}(h) = 100\sqrt{\gamma(h)}/\bar{R}_{rs}$   [%]')
        if deployment == 'Balaton2019':
         plt.title(str(bands_MSI[k]) + ' nm')
        plt.xlabel('$h$ [m]')
        
        if deployment == 'Balaton2019': # force axes within each deployment for consitency
            plt.ylim(0,40)
            plt.xlim(0,600)
        elif deployment == 'Plymouth2021':
            plt.ylim(0,40)
            plt.xlim(0,1500)
        elif deployment == 'Lisbon2021':
            plt.ylim(0,40)
            plt.xlim(0,1500)

      #  plt.legend()
        ax=plt.gca()
        plt.text(.05, .95,  subplotlab[k], ha='left', va='top', transform=ax.transAxes,fontsize=18)  
        plt.text(.60, .95,  'N = '  + str(len(CV_k.T)), ha='left', va='top', transform=ax.transAxes,fontsize=18)
        if k ==0:
           plt.legend(loc=4)
    plt.tight_layout(pad=1.3)

    filename  =  fig_dir +  '/' + deployment + '_CVgram.png'
    plt.savefig(filename,dpi=900)
    
    
    #  root-variogram summary plot #
    plt.figure(figsize = (16,4),dpi=300)
  #  plt.suptitle('$R_{rs}$ root-variograms: ' + deployment_string)
    for k in range(len(bands_MSI)):
        # calculate median fit parameters
        a = []
        b = []
        c0 = []
        for i in range(len(VG_data)):
           VG_data_i = VG_data[i]
           a.append(VG_data_i['a_' + bands_MSI[k]])
           b.append(VG_data_i['b_' + bands_MSI[k]])
           c0.append(VG_data_i['c0_' + bands_MSI[k]])
        a = np.stack(a)[Q[:,k]==1]
        b = np.stack(b)[Q[:,k]==1]
        c0 = np.stack(c0)[Q[:,k]==1]
        a_med = np.median(a)
        b_med = np.median(b)
        c0_med = np.median(c0)
       # b_med_root = np.sqrt(np.median(b))
        #c0_med_root = np.sqrt(np.median(c0))

        # median fit curves
        x = x_j[0]
        gamma_med = np.sqrt(b_med + c0_med*(1 - np.exp(-(x*x)/(a_med*a_med))))
        gamma_k = np.sqrt(gamma[:,k,:][Q[:,k]==1].T)
    
        # plots
        plt.subplot(1,4,k + 1)
        colors_2 = [cm.cool(np.linspace(0, 1, len(gamma_k.T))),cm.summer(np.linspace(0, 1, len(gamma_k.T))), cm.autumn(np.linspace(0, 1, len(gamma_k.T))), cm.bone(np.linspace(0, 1, len(gamma_k.T)))]
        for j in range(len(gamma_k.T)):
            plt.plot(x,gamma_k[:,j],alpha=0.8,linewidth=1.5,color =colors_2[k][j])
     #   plt.plot(x,gamma_med,color = 'black',linewidth=4, label = 'Median fit parameters:' + '\n' + 'L = ' + f'{2*a_med:.3}' + '  m, '  + '\n' +  '$\sqrt{C_{0}}$ = ' + f'{b_med_root:.3}' + ' sr$^{-1}$'  + '\n' + '$\sqrt{C_{\infty}}$  = ' + f'{c0_med_root:.3}' + ' sr$^{-1}$')
        plt.plot(x,gamma_med,color = 'black',linewidth=4, label = 'Median curve')
        if deployment == 'Balaton2019':
            plt.title(str(bands_MSI[k]) + ' nm')
      
      
        if k == 0:
             plt.ylabel(r'$\sqrt{\gamma(h)}$   [sr$^{-1}$]')
   
        plt.xlabel('$h$ [m]')
        
        if deployment == 'Balaton2019': # force axes within each deployment for consitency
            plt.ylim(0,0.01)
            plt.xlim(0,600)
            plt.yticks(rotation = 0)
        elif deployment == 'Plymouth2021':
            plt.ylim(0,0.001)  
            plt.xlim(0,1500)
            plt.yticks(rotation = 0)
        elif deployment == 'Lisbon2021':
            plt.ylim(0,0.003)  
            plt.xlim(0,1500)
            plt.xticks([0, 500, 1000, 1500])
            plt.yticks(rotation = 0)
        ax=plt.gca()
        plt.text(.05, .95,  subplotlab[k], ha='left', va='top', transform=ax.transAxes,fontsize=18) 
        plt.text(.60, .95,  'N = '  + str(len(gamma_k.T)), ha='left', va='top', transform=ax.transAxes,fontsize=18)    
        if k ==0:
           plt.legend(loc=4)
   #     plt.legend()
    plt.tight_layout(pad=1.1)
    
    filename  =  fig_dir +  '/' + deployment + '_rootVgram.png'
    plt.savefig(filename,dpi=900)
    
    return



def plot_rrs_VG():
    ' Plots Root-var, CV-var and maps: as used in paper sub'
    
    fig = plt.figure(figsize=(14,5))    
    plt.suptitle(deployment_string + ': ' +  date)
    plt.rc('font', size=16)
    gs=GridSpec(6,20) # 2 rows, 3 columns
   
    lat = matchdata['lat']
    lon = matchdata['lon']    
    rrs = matchdata['rrs_' +  bands_MSI[1]]    
   
    if deployment == 'Balaton2019':
        extent = [17.890, 17.902, 46.878,  46.890] 
    elif deployment == 'Plymouth2021':
         extent = [-4.30, -4.10, 50.20, 50.40] 
    elif deployment == 'Lisbon2021':
         extent = [-9.23, -9.11, 38.66, 38.73] 
            
    request=cimgt.Stamen('terrain-background')
    ax = fig.add_subplot(gs[:,14:20], projection=ccrs.PlateCarree())
    ax.set_extent(extent, ccrs.Geodetic())

    if deployment == 'Balaton2019':
       ax.add_image(request,13)
    elif deployment == 'Plymouth2021':
       ax.add_image(request,11)
    elif deployment == 'Lisbon2021':
       ax.add_image(request,13)
     
    # gridlines
    gl = ax.gridlines(draw_labels=True)
    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter =  LONGITUDE_FORMATTER
    gl.xlabel_style = {'size': 12,  'rotation': 0}
    gl.ylabel_style = {'size': 12,  'rotation': 0}
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.tick_params(labelsize = 11)
    
    plt.scatter(lon,lat, c = rrs, cmap ='viridis', vmin = np.percentile(rrs,5), vmax = np.percentile(rrs,95),  transform=ccrs.PlateCarree())
    if deployment != 'Plymouth2021':
        cbar = plt.colorbar(location='bottom')
    else:
        cbar = plt.colorbar(location='bottom')
    cbar.set_label('$R_{rs}$(560) [sr$^{-1}$]')  
    
    
    plt.legend(fontsize=12) 
    if deployment == 'Balaton2019':
        scale_bar(ax, (0.1, 0.1), 500, metres_per_unit=1, unit_name='m')
    elif deployment == 'Plymouth2021':
        scale_bar(ax, (0.1, 0.8), 5000, metres_per_unit=1, unit_name='m')
    elif deployment == 'Lisbon2021':
        scale_bar(ax, (0.1, 0.8), 2000, metres_per_unit=1, unit_name='m') 
 
    # ax = fig.add_subplot(gs[0:3,0:4])
    # plt.plot(x_j[0], gamma_j[1], color='green')
    # plt.legend(loc='lower left', bbox_to_anchor=(-1, 0.5))
    # plt.scatter(x_e_j[1],gamma_e_j[1], color='green')
    
   
    L = abs(np.sqrt(3)*VG_data_j['a_' + bands_MSI[1]]) # L
    root_C_0 = np.sqrt(VG_data_j['b_' + bands_MSI[1]]) # sqrt c_0
    root_C_infty= np.sqrt(VG_data_j['c0_' + bands_MSI[1]] + VG_data_j['b_' + bands_MSI[1]])    

    ax = fig.add_subplot(gs[:,0:5])
    
    plt.plot(x_j[0], np.sqrt(gamma_j[1]), color='green', label = 'Fit error = '  + str(round(VG_data_j['nMAE_' + bands_MSI[1]],1)) + '$\%$' + '\n' 
             +  '$\sqrt{C_{0}}$ = ' + f'{root_C_0:.2}' + ' sr$^{-1}$' + '\n' 
             +  '$\sqrt{C_{\infty}}$  = ' + f'{root_C_infty:.2}' + ' sr$^{-1}$' + '\n'                                                                       
             +  'L = ' + str(round(L)) + ' m' )

    
  #  plt.legend(loc='lower left', bbox_to_anchor=(-1, 0.5))
    plt.legend(loc='lower right')
    plt.scatter(x_e_j[1],np.sqrt(gamma_e_j[1]), color='green')
    plt.gca().set_ylim(bottom = 0)
    plt.ylabel(r'$\sqrt{\gamma(h)}$   [sr$^{-1}$]')
    if deployment == 'Balaton2019':
        plt.xlim(0,600)
        plt.ylim(0,0.009)
    else:
        plt.xlim(0,1600)
        plt.xticks([0,500,1000,1500])
        
    if deployment == 'Plymouth2021':
        plt.ylim(0,0.0008)
    elif deployment == 'Lisbon2021':
        plt.ylim(0,0.0030)

    plt.xlabel('$h$ [m]')
    
    ax = fig.add_subplot(gs[:,7:12])
    

    plt.plot(x_j[0], 100*np.sqrt(gamma_j[1])/(mean_rrs_j[1]), color='green', label =    r'$\tilde{CV}(0)$' +' = '  +  str(round(100*(np.sqrt(gamma_j[1][0])/mean_rrs_j[1]),1)) +   ' %'  + '\n'  +  r'$\tilde{CV}(300)$' +' = '  +  str(round(100*(np.sqrt(gamma_j[1][301])/mean_rrs_j[1]),1)) +   ' %'  + '\n'  
             '$f_{300}$ = ' + str(round(100*(np.sqrt(gamma_j[1][301])-np.sqrt(gamma_j[1][0]))/np.sqrt(gamma_j[1][0]),1)) +  ' %'
             )
    
    
    #  plt.legend(loc='lower left', bbox_to_anchor=(-1, 0.5))
    plt.legend(loc='lower right')
   # plt.scatter(x_e_j[1],100*np.sqrt(gamma_e_j[1])/(mean_rrs_j), color='green')
    plt.gca().set_ylim(bottom = 0)
    plt.ylabel(r'$\tilde{CV}(h) = 100\sqrt{\gamma(h)}/\bar{R}_{rs}$   [%]')   
    if deployment == 'Balaton2019':
        plt.xlim(0,600)
        plt.ylim(0,16)
    else:
        plt.xlim(0,1600)
        plt.xticks([0,500,1000,1500])

    plt.xlabel('$h$ [m]')

    if deployment == 'Plymouth2021':
        plt.ylim(0,16)
    elif deployment == 'Lisbon2021':
        plt.ylim(0,25)
    # plt.title ('$R_{rs}$('+ bands[i] + ') [sr$^{-1}$]')
    # calculate median fit parameters
    plt.tight_layout(pad=1.6)

    filename  =  fig_dir +  '/' + deployment + '_VG_annotated.png'
    plt.savefig(filename,dpi=900)

    return


def plot_rrs_VG_PS_red():
   
    fig = plt.figure(figsize=(14,5))    
    plt.suptitle(deployment_string + ': ' +  date)
    plt.rc('font', size=16)
    gs=GridSpec(6,20) # 2 rows, 3 columns
   
    lat = matchdata['lat']
    lon = matchdata['lon']    
    rrs = matchdata['rrs_' +  bands_MSI[2]]    
   
    if deployment == 'Balaton2019':
        extent = [17.890, 17.902, 46.878,  46.890] 
    elif deployment == 'Plymouth2021':
         extent = [-4.40, -4.30, 50.20, 50.40] 
    elif deployment == 'Lisbon2021':
         extent = [-9.23, -9.11, 38.66, 38.73] 
            
    request=cimgt.Stamen('terrain-background')
    ax = fig.add_subplot(gs[:,14:20], projection=ccrs.PlateCarree())
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
    ax.tick_params(labelsize = 11)
    
    plt.scatter(lon,lat, c = rrs, cmap ='viridis', vmin = np.percentile(rrs,5), vmax = np.percentile(rrs,95),  transform=ccrs.PlateCarree())
    if deployment != 'Plymouth2021':
        cbar = plt.colorbar(location='bottom')
    else:
        cbar = plt.colorbar(location='bottom')
    cbar.set_label('$R_{rs}$(665) [sr$^{-1}$]')  
    
    
    plt.legend(fontsize=12) 
    if deployment == 'Balaton2019':
        scale_bar(ax, (0.1, 0.1), 500, metres_per_unit=1, unit_name='m')
    elif deployment == 'Plymouth2021':
        scale_bar(ax, (0.1, 0.8), 5000, metres_per_unit=1, unit_name='m')
    elif deployment == 'Lisbon2021':
        scale_bar(ax, (0.1, 0.8), 2000, metres_per_unit=1, unit_name='m') 
    elif deployment == 'Danube2021':
        scale_bar(ax, (0.1, 0.8), 5000, metres_per_unit=1, unit_name='m') 
    
    # ax = fig.add_subplot(gs[0:3,0:4])
    # plt.plot(x_j[0], gamma_j[1], color='green')
    # plt.legend(loc='lower left', bbox_to_anchor=(-1, 0.5))
    # plt.scatter(x_e_j[1],gamma_e_j[1], color='green')
    
   
    L = abs(np.sqrt(3)*VG_data_j['a_' + bands_MSI[2]]) # L
    root_C_0 = np.sqrt(VG_data_j['b_' + bands_MSI[2]]) # sqrt c_0
    root_C_infty= np.sqrt(VG_data_j['c0_' + bands_MSI[2]] + VG_data_j['b_' + bands_MSI[2]])    

    ax = fig.add_subplot(gs[:,0:5])
    
    plt.plot(x_j[0], np.sqrt(gamma_j[2]), color='red', label = 'Fit error = '  + str(round(VG_data_j['nMAE_' + bands_MSI[2]],1)) + '$\%$' + '\n' 
             +  '$\sqrt{C_{0}}$ = ' + f'{root_C_0:.2}' + ' sr$^{-1}$' + '\n' 
             +  '$\sqrt{C_{\infty}}$  = ' + f'{root_C_infty:.2}' + ' sr$^{-1}$' + '\n'                                                                       
             +  'L = ' + str(round(L)) + ' m' )

    
  #  plt.legend(loc='lower left', bbox_to_anchor=(-1, 0.5))
    plt.legend(loc='lower right')
    plt.scatter(x_e_j[2],np.sqrt(gamma_e_j[2]), color='red')
    plt.gca().set_ylim(bottom = 0)
    plt.ylabel(r'$\sqrt{\gamma(h)}$  [sr$^{-1}$]')
    if deployment == 'Balaton2019':
        plt.xlim(0,600)
        plt.ylim(0,0.009)
    else:
        plt.xlim(0,1600)
        plt.xticks([0,500,1000,1500])

    plt.xlabel('$h$ [m]')
    
    ax = fig.add_subplot(gs[:,7:12])
    

    plt.plot(x_j[0], 100*np.sqrt(gamma_j[2])/(mean_rrs_j[2]), color='red', label =    r'$\tilde{CV}(0)$' +' = '  +  str(round(100*(np.sqrt(gamma_j[2][0])/mean_rrs_j[2]),1)) +   ' [%]'  + '\n'  +  r'$\tilde{CV}(300)$' +' = '  +  str(round(100*(np.sqrt(gamma_j[2][301])/mean_rrs_j[2]),1)) +   ' [%]'  + '\n'  
             '$f_{300}$ = ' + str(round(100*(np.sqrt(gamma_j[2][301])-np.sqrt(gamma_j[2][0]))/np.sqrt(gamma_j[2][0]),1)) +  ' [%]'
             )
    
    
    #  plt.legend(loc='lower left', bbox_to_anchor=(-1, 0.5))
    plt.legend(loc='lower right')
   # plt.scatter(x_e_j[1],100*np.sqrt(gamma_e_j[1])/(mean_rrs_j), color='green')
    plt.gca().set_ylim(bottom = 0)
    plt.ylabel(r'$\tilde{CV}(h) = 100\sqrt{\gamma(h)}/\bar{R}_{rs}$   [%]')   
    if deployment == 'Balaton2019':
        plt.xlim(0,600)
        plt.ylim(0,16)
    else:
        plt.xlim(0,1600)
        plt.xticks([0,500,1000,1500])
        
    if deployment == 'Plymouth2021':
        plt.ylim(0,0.0008)

    plt.xlabel('$h$ [m]')

    # plt.title ('$R_{rs}$('+ bands[i] + ') [sr$^{-1}$]')
    # calculate median fit parameters
    plt.tight_layout(pad=1.6)

    filename  =  fig_dir +  '/' + deployment + '_VG_annotated.png'
    plt.savefig(filename,dpi=900)

    return


def plot_rrs_VG_PS_blue():
       
    fig = plt.figure(figsize=(14,5))    
    plt.suptitle(deployment_string + ': ' +  date)
    plt.rc('font', size=16)
    gs=GridSpec(6,20) # 2 rows, 3 columns
   
    lat = matchdata['lat']
    lon = matchdata['lon']    
    rrs = matchdata['rrs_' +  bands_MSI[0]]    
   
    if deployment == 'Balaton2019':
        extent = [17.890, 17.902, 46.878,  46.890] 
    elif deployment == 'Plymouth2021':
         extent = [-4.30, -4.10, 50.20, 50.40] 
    elif deployment == 'Lisbon2021':
         extent = [-9.23, -9.11, 38.66, 38.73] 
            
    request=cimgt.Stamen('terrain-background')
    ax = fig.add_subplot(gs[:,14:20], projection=ccrs.PlateCarree())
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
    ax.tick_params(labelsize = 11)
    
    plt.scatter(lon,lat, c = rrs, cmap ='viridis', vmin = np.percentile(rrs,5), vmax = np.percentile(rrs,95),  transform=ccrs.PlateCarree())
    if deployment != 'Plymouth2021':
        cbar = plt.colorbar(location='bottom')
    else:
        cbar = plt.colorbar(location='bottom')
    cbar.set_label('$R_{rs}$(443) [sr$^{-1}$]')  
    
    
    plt.legend(fontsize=12) 
    if deployment == 'Balaton2019':
        scale_bar(ax, (0.1, 0.1), 500, metres_per_unit=1, unit_name='m')
    elif deployment == 'Plymouth2021':
        scale_bar(ax, (0.1, 0.8), 5000, metres_per_unit=1, unit_name='m')
    elif deployment == 'Lisbon2021':
        scale_bar(ax, (0.1, 0.8), 2000, metres_per_unit=1, unit_name='m') 
    elif deployment == 'Danube2021':
        scale_bar(ax, (0.1, 0.8), 5000, metres_per_unit=1, unit_name='m') 
    
    # ax = fig.add_subplot(gs[0:3,0:4])
    # plt.plot(x_j[0], gamma_j[1], color='green')
    # plt.legend(loc='lower left', bbox_to_anchor=(-1, 0.5))
    # plt.scatter(x_e_j[1],gamma_e_j[1], color='green')
    
   
    L = abs(np.sqrt(3)*VG_data_j['a_' + bands_MSI[0]]) # L
    root_C_0 = np.sqrt(VG_data_j['b_' + bands_MSI[0]]) # sqrt c_0
    root_C_infty= np.sqrt(VG_data_j['c0_' + bands_MSI[0]] + VG_data_j['b_' + bands_MSI[0]])    

    ax = fig.add_subplot(gs[:,0:5])
    
    plt.plot(x_j[0], np.sqrt(gamma_j[0]), color='blue', label = 'Fit error = '  + str(round(VG_data_j['nMAE_' + bands_MSI[0]],1)) + '$\%$' + '\n' 
             +  '$\sqrt{C_{0}}$ = ' + f'{root_C_0:.2}' + ' sr$^{-1}$' + '\n' 
             +  '$\sqrt{C_{\infty}}$  = ' + f'{root_C_infty:.2}' + ' sr$^{-1}$' + '\n'                                                                       
             +  'L = ' + str(round(L)) + ' m' )

    
  #  plt.legend(loc='lower left', bbox_to_anchor=(-1, 0.5))
    plt.legend(loc='lower right')
    plt.scatter(x_e_j[0],np.sqrt(gamma_e_j[0]), color='blue')
    plt.gca().set_ylim(bottom = 0)
    plt.ylabel(r'$\sqrt{\gamma(h)}$   [sr$^{-1}$]')
    if deployment == 'Balaton2019':
        plt.xlim(0,600)
        plt.ylim(0,0.009)
    else:
        plt.xlim(0,1600)
        plt.xticks([0,500,1000,1500])
    
    if deployment == 'Plymouth2021':
        plt.ylim(0,0.0008)

    plt.xlabel('$h$ [m]')
    
    ax = fig.add_subplot(gs[:,7:12])
    

    plt.plot(x_j[0], 100*np.sqrt(gamma_j[0])/(mean_rrs_j[0]), color='blue', label =    r'$\tilde{CV}(0)$' +' = '  +  str(round(100*(np.sqrt(gamma_j[0][0])/mean_rrs_j[0]),1)) +   ' %'  + '\n'  +  r'$\tilde{CV}(300)$' +' = '  +  str(round(100*(np.sqrt(gamma_j[0][301])/mean_rrs_j[0]),1)) +   ' %'  + '\n'  
             '$f_{300}$ = ' + str(round(100*(np.sqrt(gamma_j[0][301])-np.sqrt(gamma_j[0][0]))/np.sqrt(gamma_j[0][0]),1)) +  ' %'
             )
       
   # plt.legend(loc='lower left', bbox_to_anchor=(-1, 0.5))
    plt.legend(loc='lower right')
   # plt.scatter(x_e_j[1],100*np.sqrt(gamma_e_j[1])/(mean_rrs_j), color='green')
    plt.gca().set_ylim(bottom = 0)
    plt.ylabel(r'$\tilde{CV}(h) = 100\sqrt{\gamma(h)}/\bar{R}_{rs}$   [%]')   
    if deployment == 'Balaton2019':
        plt.xlim(0,600)
        plt.ylim(0,16)
    else:
        plt.xlim(0,1600)
        plt.xticks([0,500,1000,1500])

    if deployment == 'Plymouth2021':
        plt.ylim(0,16)

    plt.xlabel('$h$ [m]')

    # plt.title ('$R_{rs}$('+ bands[i] + ') [sr$^{-1}$]')
    # calculate median fit parameters
    plt.tight_layout(pad=1.6)

    filename  =  fig_dir +  '/' + deployment + '_VG_annotated.png'
    plt.savefig(filename,dpi=900)

    return


if __name__ == '__main__':
    
    # bands_MSI  = ma.array(['443', '490', '560', '665', '705', '740', '783', '842', '865'], mask = [False, True, False, False, True, True, False, True, True])
    bands_MSI  = ['443', '560', '665', '783'] # just consider 4 target bands

    # deployment = 'Balaton2019' # Initalize VG paramters for each deployment
    # deployment = 'Plymouth2021'
    deployment = 'Lisbon2021'
  
    fig_dir = '/users/rsg/tjor/monocle/sat_val/Paperplots/'
    
    if deployment == 'Balaton2019':
        # lag_bound = 480
        # n_lags = 12
        lag_bound = 600 # 
        n_lags = 15
        path_matches = '/users/rsg/tjor/monocle/sat_val/Balaton2019matches/'
        files_matches = sorted(glob.glob(os.path.join(path_matches, 'matches_3hr_v2_*')))  
        path_sr = '/users/rsg/tjor/monocle_network/sorad_rrs_alldeployments/hyperspectral/Balaton2019_3C/'
        meta_files = sorted(glob.glob(os.path.join(path_sr, '*data*')))
        deployment_string = 'Lake Balaton'
        min_points = 60
        subplotlab = ['A', 'B', 'C', 'D']
        plot_loc = 1      
        plot_index = 26
    elif deployment == 'Plymouth2021':
        # lag_bound = 600
        # n_lags = 12
        lag_bound = 1500
        n_lags = 15
        path_matches = '/users/rsg/tjor/monocle/sat_val/Plymouth2021matches/'
        files_matches = sorted(glob.glob(os.path.join(path_matches, 'matches_3hr_v2_*')))  #
        path_sr = '/users/rsg/tjor/monocle_network/sorad_rrs_alldeployments/hyperspectral/Plymouth2021_3C/'
        meta_files = sorted(glob.glob(os.path.join(path_sr, '*data*')))
        deployment_string = 'Western Channel'
        min_points = 150
        subplotlab = ['E', 'F', 'G', 'H']
        plot_loc = 4
        plot_index = 81 # 32 also looks good
    elif deployment == 'Lisbon2021':
        lag_bound = 1500
        n_lags = 15
        path_matches = '/users/rsg/tjor/monocle/sat_val/Lisbon2021matches/'
        files_matches = sorted(glob.glob(os.path.join(path_matches, 'matches_3hr_v2_*')))  #
        path_sr = '/users/rsg/tjor/monocle_network/sorad_rrs_alldeployments/hyperspectral/Lisbon2021_3C/'
        meta_files = sorted(glob.glob(os.path.join(path_sr, '*data*')))
        deployment_string = 'Tagus Estuary'
        subplotlab = ['I', 'J', 'K', 'L']
        min_points = 150
        plot_loc = 1
        plot_index = 48 # 2
        
    pathfigs = '/users/rsg/tjor/monocle/sat_val/outputfigs/'       
    fig_dir = '/users/rsg/tjor/monocle/sat_val/Paperplots_29_03/'

    # intiialize data lists    
    gamma = []    # theoretical semi-variance (fit to model)
    gamma_e = []  # experimental semi-varince
    VG_data = []  # variogram data
    mean_rrs = [] # survey mean RRs
    Q = [] # QC binary variable
    LAT = [] # geographic coordinates
    LON = []
    Nbins =[] # numver of bins
    R = [] 
    pairwise_time = []
    pairwise_distance =[]
    
    # loop over all data files in deplployment
    for j in range(0,len(meta_files),1): # j for day index
    
        matchdata = pd.read_csv(files_matches[j]) 
        date = str(files_matches[j])[-14:-4] 
        print(len(matchdata))
        
        if len(matchdata) > min_points: # and j != 18 and  j != 89: # and  j != 49 and  j != 89:)
            r_j, x_e_j, gamma_e_j, x_j, gamma_j, Q_j, VG_data_j, Nbins_j = variogram_rrs_bands(matchdata,lag_bound,n_lags,bands_MSI) # calculate VGS                
            mean_rrs_j = rrs_band_mean(matchdata, bands_MSI)  # calculated survey-mean Rrs
      
            # append VG output to lists
            gamma_e.append(gamma_e_j)
            gamma.append(gamma_j)
            VG_data.append(VG_data_j)
            Q.append(Q_j)
            mean_rrs.append(mean_rrs_j)
            R.append(r_j)
            Nbins.append(Nbins_j)
            
            # plot rrs variogram
            plot_rrs_VG()
            # variogram_bandplot_V2(x_e_j, gamma_e_j, x_j, gamma_j, mean_rrs_j, VG_data_j, date)   - superseeded
            # plot_data_map_V2(matchdata, bands_MSI, date, map_resolution=12) - siperseeded
            
            # pairwise time differences
            time_cell_hours = matchdata['time_cell']/(60*60)
            pairwise_time_j = sklearn.metrics.pairwise_distances(np.array(time_cell_hours).reshape(-1,1)).flatten() 
            pairwise_time.append(pairwise_time_j)
        
            # pariwise distances 
            X = np.asarray(utm.from_latlon(np.array(matchdata['lat']),np.array(matchdata['lon']))[0:2]).T # 
            pairwise_distance_j = scipy.spatial.distance.pdist(X)
            pairwise_distance.append(pairwise_distance_j)
        
    gamma_e = np.stack(gamma_e)
    gamma = np.stack(gamma)
    mean_rrs = np.stack(mean_rrs)
    Q = np.stack(Q)
    Nbins = np.stack(Nbins,axis=0)
    pairwise_time = np.concatenate(pairwise_time)
    pairwise_distance = np.concatenate(pairwise_distance)

    # output plots in paper submission
    plt.rc('font', size=18)
    plt.rcParams['legend.fontsize'] = 13
    var_plots() 
    range_plots()
    CV_plots() 
    
    # output VG summary plots in paper submission
    plt.rc('font', size=16)
    plt.rcParams['legend.fontsize'] = 13
    VGsummary_plots_V2()
    
    # output pariwise distance plot in paper submission
    pairwise_xt()
        
    

    