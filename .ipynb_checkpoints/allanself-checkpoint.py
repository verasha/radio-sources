import numpy as np
import sys
sys.path.append('../../module')
import pandas as pd
from tweezepy import allanvar as tpy
import scipy

def process_data(data_txt):
    # Read data
    cols = ['MJD','R.A. (mas)', 'Dec. (mas)', 'Err. R.A. (mas)', 'Err. Dec. (mas)', 'Corr.', 'Del.']
    cols_widths = [(0,9),(9,26),(27,46),(47,56),(57,66),(67,74),(75,79)]
    df = pd.read_fwf(data_txt, colspecs=cols_widths, skiprows=4, header=None, names=cols) 
    
    # Convert to float
    df_new = df.apply(pd.to_numeric, errors='coerce').dropna()    
    
    # Convert from mas to microas
    df_new['R.A. (micas)'] = df_new['R.A. (mas)']*1000
    df_new['Dec. (micas)'] = df_new['Dec. (mas)']*1000
    df_new['Err. R.A. (micas)'] = df_new['Err. R.A. (mas)']*1000
    df_new['Err. Dec. (micas)'] = df_new['Err. Dec. (mas)']*1000
    
    # Convert from mas to degrees to radian
    df_new['Dec. (rad)'] = (df_new['Dec. (mas)']/(3600*1000)).apply(np.deg2rad)
    
    # Calculate RA*cos(dec) column
    
    df_new['RAcosDEC'] = df_new['R.A. (micas)']*np.cos(df_new['Dec. (rad)'])
    
    # Calculate Besselian Year column
    df_new['BY'] = 2000-(51544.333981-df_new['MJD'])/365.242198781
    
    return df_new

def remove_outliers(df,datatype):
    if datatype == 'RA':
        df_sel = df[['MJD','RAcosDEC','Err. R.A. (micas)']].copy()
        datas = 'RAcosDEC'
    elif datatype == 'DEC':
        df_sel = df[['MJD','Dec. (micas)','Err. Dec. (micas)']].copy()
        datas = 'Dec. (micas)'
    else:
        print('Please choose RA/DEC data')

    for index, row in df_sel.iterrows():
        data_values = df_sel[(df_sel['MJD'] >= (row['MJD'] - 365)) & (df_sel['MJD'] <= (row['MJD'] + 365))]
        mean = data_values[datas].mean()
        std = data_values[datas].std()
        if abs(row[datas] - mean) > 6*std:
            df_sel.drop(index, inplace = True)
            
    return df_sel

def allanoadev(y,t,error):
        
    # === Parameters ===
    
    # Length of observation
    N = len(y)
        
    # m array where 3 < m < N/3 
    m = np.arange(3,1+round(N//3),1)
    # initial sampling time
    tau0 =  np.mean(np.diff(t))
    # steps
    step = tau0
    
    # Empty variance array
    allanvar = []
    # Empty tau values array 
    tau = []
    
    edfs = np.empty(len(m))

    # Iterate over values of m
    for k in range(len(m)): 
        # Sliding windows 
        left = min(t)
        right = min(t) + m[k]*tau0

        # Array for averages
        ybar = []

        # Array for amount of data points
        Nrray = []

        # Define no of windows with <= 2 data points
        nconstraint = 2
                
        # Check for missing windows 
        while right <= max(t):
            data_values = y[(t>=left) & (t<=right)]
            Ndata = len(data_values)
            Nrray.append(Ndata)

            if Ndata == 0:
                ybar.append(np.nan)
            else:
                if error is None: 
                    avg = np.average(data_values)
                    ybar.append(avg)
                else:
                    weights_std = 1/error[(t>=left) & (t<=right)]**2
                    avg = np.average(data_values, weights=weights_std)
                    ybar.append(avg)
            left += step 
            right += step    

        # Define empty windows 
        Nrray = np.array(Nrray)
        ybar = np.array(ybar)
        
        D = len(ybar[Nrray<=nconstraint])
        M = len(ybar)
#         print(k,len(m),M,D)

        # Discard windows with too few data points
        ybar = ybar[Nrray>nconstraint]

        # Calculate allan var 
        outersum = []
        j = 0
        while j < M-2*m[k]+1-D:
            i = j
            innersum = []
            while i <= j+m[k]-1:
                innersum.append(ybar[i+m[k]]-ybar[i])
                i += 1
            outersum.append(0.5*(np.sum(innersum)/m[k])**2)
            j += 1


        # Calculate allan variance
        allanvar.append(np.sum(outersum)/(M-2*m[k]+1-D))
        
        # Calculate tau
        tau.append(tau0*m[k])
        
        # Calculate edf
        if N // m[k] > 32:
            alpha_int = tpy.noise_id(y,m[k])[0]
#             print(alpha_int)
        if (alpha_int<3) and (alpha_int>-3):
            edfs[k] = tpy.edf_greenhall(alpha_int,2,m[k],N,overlapping=True)
        else:
            edfs[k] = tpy.edf_approx(N,m[k])

    # Calculate allan std dev
    allanstd = np.sqrt(allanvar)
        
    return tau, allanstd, edfs

def process_allan(y,tau,edfs):
    tau_year = [x/365.242198781 for x in tau]
    samplevar = scipy.stats.variation(y)
    bottom = edfs*samplevar/scipy.stats.chi2.ppf(1-0.1/2,edfs)
    top = scipy.stats.chi2.ppf(0.1/2,edfs)
    top_alt = edfs*samplevar/scipy.stats.chi2.ppf(0.1/2,edfs)
    return tau_year,bottom,top,top_alt