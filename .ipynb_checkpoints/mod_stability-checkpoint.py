# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 08:48:10 2023

@author: DELL
"""

import numpy as np
import sys
sys.path.append('../../module')
from mod_statistic import*
from MJD_BY import*




# Conducting overlapping allan standard deviation
# see Gattano, C., S. B. Lambert, and K. Le Bail. "Extragalactic radio source 
# stability and VLBI celestial reference frame: insights from the Allan standard 
# deviation." Astronomy & Astrophysics 618 (2018): A80.
def allanover(y,yerr,t,types):
    #calculate some parameters
    n = np.arange(3,1+round(len(y)//3),1)
    tau0 = np.mean(np.diff(t))
    step = tau0

    allanvar = []
    tau = []
    for jj in range(len(n)):
         nconstraint = 2 #define constraint of number of data point in window
         #sliding windows
         left = min(t)#0
         right = min(t) + n[jj]*tau0#n[jj]
         ytemp = []
         temp = []
         #while right <= len(y):
             #ytemp.append(np.mean(y[left:right]))
         while right <= max(t):
             temp.append(len(y[(t>=left) & (t<=right)]))
             if len(y[(t>=left) & (t<=right)]) == 0:
                 ytemp.append(np.average(y[(t>=left) & (t<=right)]))
             else:
                 if types == 1: #types = 1 for weighted average
                     weight = 1/yerr[(t>=left) & (t<=right)]**2
                     ytemp.append(np.average(y[(t>=left) & (t<=right)],weights=weight))
                 else:
                     ytemp.append(np.average(y[(t>=left) & (t<=right)]))
             left = left + step
             right = right + step
         temp = np.array(temp)
         ytemp = np.array(ytemp)
         M = len(ytemp)
         #D = np.count_nonzero(np.isnan(ytemp))
         D = len(ytemp[temp<=nconstraint])
         print(jj,len(n), M, D)
         
         #discard the window with too few data
         #ytemp = np.array(ytemp)[~np.isnan(ytemp)].tolist()
         ytemp = ytemp[temp>nconstraint]
         
         #calculate Allan variance
         ysum2 = []
         i = 0
         while i < M-2*n[jj]+1-D:
             j = i
             ysum = []
             while j <= i+n[jj]-1:
                 ysum.append(ytemp[j+n[jj]]-ytemp[j])
                 j = j+1
             ysum2.append(0.5*(np.sum(ysum)/n[jj])**2)
             i = i+1
         allanvar.append(np.sum(ysum2)/(M-2*n[jj]+1-D))
         tau.append(tau0*n[jj])

    allanstd = np.sqrt(allanvar)

    return tau,allanvar,allanstd




def allanoverold(y,t):
    #calculate some parameters
    n = np.arange(3,1+round(len(y)//3),1)
    step = 1
    tau0 = np.mean(np.diff(t))

    allanvar = []
    tau = []
    for jj in range(len(n)):
        print(jj,len(n))
        #sliding windows
        left = 0
        right = n[jj]
        ytemp = []
        while right <= len(y):
            ytemp.append(np.mean(y[left:right]))
        
            left = left + step
            right = right + step
        M = len(ytemp)
        
        #calculate Allan variance
        ysum2 = []
        i = 0
        while i < M-2*n[jj]+1:
            j = i
            ysum = []
            while j <= i+n[jj]-1:
                ysum.append(ytemp[j+n[jj]]-ytemp[j])
                j = j+step
            ysum2.append(0.5*(np.sum(ysum)/n[jj])**2)
            i = i+step
        allanvar.append(np.sum(ysum2)/(M-2*n[jj]+1))
        tau.append(tau0*n[jj])

    allanstd = np.sqrt(allanvar)

    return tau,allanvar,allanstd


#read data from IVSOPAR website or VieVs result
def readdata(dataname_dec,dataname_ra,datatype):
    
    if datatype == 1:
        data_dec = np.genfromtxt(dataname_dec)
        data_ra = np.genfromtxt(dataname_ra)

        mjd = data_dec[:,0]
        by  = mjdtoby(mjd)
        dec = data_dec[:,1]
        ra  = data_ra[:,1]
        errdec = data_dec[:,2]
        errra  = data_ra[:,2]

        #convert ra
        ra = ra*15 #convert R.A. from milisecond to mas
        errra = errra*15  #convert R.A. from milisecond to mas

        #calculate R.A.*cos(dec) and its error propagation
        decr = np.deg2rad(dec/1000/3600); #declination in radian
        ra_n = ra*np.cos(decr);
        dra_ndra = np.cos(decr);
        dra_nddec = -ra*np.sin(decr);
        errra_n = np.sqrt(((dra_ndra)**2)*(errra**2) + ((dra_nddec)**2)*(errdec**2));

    elif datatype == 2:
        data_dec = np.genfromtxt(dataname_dec,comments='%')
        data_ra = np.genfromtxt(dataname_ra,comments='%')

        mjd = data_dec[:,0]
        by  = mjdtoby(mjd)
        dec = data_dec[:,2]
        ra  = data_ra[:,1]
        errdec = data_dec[:,4]
        errra  = data_ra[:,3]


        #calculate R.A.*cos(dec) and its error propagation
        decr = np.deg2rad(dec/1000/3600); #declination in radian
        ra_n = ra*np.cos(decr);
        errra_n = errra
        
    return mjd,by,dec,ra_n,errdec,errra_n