import numpy as np
import matplotlib.pyplot as plt 
import allantools as at
from allantools import noise
import sys
sys.path.append('../../module')
from mod_statistic import*
from MJD_BY import*
from mod_stability import*


#Input
####################
datatype = 2  #1: VieVS; 2: ivsopar

#dataname_dec = 'data/2201+315_dec.txt'
#dataname_ra = 'data/2201+315_ra.txt'
#dataname_dec = 'data/2201+315_ivsopar.txt'
#dataname_ra = 'data/2201+315_ivsopar.txt'

dataname_dec = 'data/1803+784_ivsopar.txt'
dataname_ra = 'data/1803+784_ivsopar.txt'

#dataname_dec = 'data/ICRF3_Dem/1611+343_dem.txt'
#dataname_ra = 'data/ICRF3_Dem/1611+343_dem.txt'


mjd,by,dec,ra_n,errdec,errra_n = readdata(dataname_dec,dataname_ra,datatype)


#convert from mas to microas
dec     = 1000*dec
errdec  = 1000*errdec
ra_n    = 1000*ra_n
errra_n = 1000*errra_n

#calculate overlapping allan standard deviation
tau_dec,allanvar_dec,allanstd_dec = allanover(dec,errdec,by,1)
tau_ra,allanvar_ra,allanstd_ra = allanover(ra_n,errra_n,by,1)


# values = np.array([10, 15, 12, 17])
# std = np.array([2, 3, 1.5, 2.5])
# ww= 1/std**2
# values_ww = np.sum(values*ww)
# sum_ww = np.sum(ww)
# values_mean = values_ww/sum_ww
# values_mean2 = np.average(values,weights=ww)

#Plot Allan standard deviation
#---------------------------------
fs = 12
color = 'black'
ymin = 1
ymax = 10000

plt.figure(1)
ax = plt.gca()
plt.subplot(211)
axes = plt.gca()
plt.errorbar(by,ra_n/1000,yerr=abs(errra_n)/1000,fmt='o',color='red',ecolor='grey', capthick=0,ms=1)
plt.ylim(-5,5)
axes.xaxis.set_tick_params(labelsize=fs)
axes.yaxis.set_tick_params(labelsize=fs)
plt.ylabel(r'$\alpha\cos(\delta) [mas]$', fontsize=fs)
plt.tight_layout()
plt.subplot(212)
axes = plt.gca()
plt.errorbar(by,dec/1000,yerr=errdec/1000,fmt='o',color='red',ecolor='grey', capthick=0,ms=1)
plt.ylim(-5,5)
axes.xaxis.set_tick_params(labelsize=fs)
axes.yaxis.set_tick_params(labelsize=fs)
plt.ylabel(r'$\delta$ [mas]', fontsize=fs)
plt.tight_layout()

plt.figure(2)
ax = plt.gca()
slopes_dec = np.diff(np.log10(allanstd_dec)) / np.diff(np.log10(tau_dec))
plt.plot(tau_dec,allanstd_dec,color=color)
plt.fill_between(tau_dec[1:], ymin,ymax, where= slopes_dec  < -0.25,color='lightgray',linewidth=1.0,zorder=1)
plt.fill_between(tau_dec[1:], ymin,ymax, where= (-0.25 <= slopes_dec ) & (slopes_dec  <= 0.25),color='pink',linewidth=1.0)
plt.fill_between(tau_dec[1:], ymin,ymax, where= slopes_dec  > 0.25,color='red',step='mid',linewidth=1.0)
plt.yscale('log')
#plt.xscale('log')
ax.xaxis.set_tick_params(labelsize=fs)
ax.yaxis.set_tick_params(labelsize=fs)
plt.xlabel('Time scale [year]', fontsize=fs)
plt.ylabel(r'slope of ADEV in dec [$\mu$as]', fontsize=fs)
plt.legend(fontsize=fs)
plt.ylim(ymin,ymax)
plt.xlim(0.5,11)
plt.tight_layout()

plt.figure(3)
ax = plt.gca()
slopes_ra = np.diff(np.log10(allanstd_ra)) / np.diff(np.log10(tau_ra))
plt.plot(tau_ra,allanstd_ra,color=color)
plt.fill_between(tau_ra[1:], ymin,ymax, where= slopes_ra  < -0.25,color='lightgray',linewidth=1.0,zorder=1)
plt.fill_between(tau_ra[1:], ymin,ymax, where= (-0.25 <= slopes_ra ) & (slopes_ra  <= 0.25),color='pink',linewidth=1.0)
plt.fill_between(tau_ra[1:], ymin,ymax, where= slopes_ra  > 0.25,color='red',step='mid',linewidth=1.0)
plt.yscale('log')
#plt.xscale('log')
ax.xaxis.set_tick_params(labelsize=fs)
ax.yaxis.set_tick_params(labelsize=fs)
plt.xlabel('Time scale [year]', fontsize=fs)
plt.ylabel(r'slope of ADEV in RA*cos(dec) [$\mu$as]', fontsize=fs)
plt.legend(fontsize=fs)
plt.ylim(ymin,ymax)
plt.xlim(0.5,11)
plt.tight_layout()





plt.show()


