import numpy as np
import sys
sys.path.append('../../module')
import pandas as pd
import scipy
import math

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
        df_sel = df[['MJD','RAcosDEC','Err. R.A. (micas)','BY']].copy()
        datas = 'RAcosDEC'
        datas_e = 'Err. R.A. (micas)'
    elif datatype == 'DEC':
        df_sel = df[['MJD','Dec. (micas)','Err. Dec. (micas)','BY']].copy()
        datas = 'Dec. (micas)'
        datas_e = 'Err. Dec. (micas)'
    else:
        print('Please choose RA/DEC data')

    for index, row in df_sel.iterrows():
        data_values = df_sel[(df_sel['MJD'] >= (row['MJD'] - 365)) & (df_sel['MJD'] <= (row['MJD'] + 365))]
        wd = 1/data_values[datas_e].values**2
        mean = np.average(data_values[datas],weights=wd)
        std = np.sqrt(wvar(data_values[datas].values,wd))
        if abs(row[datas] - mean) > 6*std:
            df_sel.drop(index, inplace = True)
    return df_sel

def wvar(y,w):
    # Length of data
    N = len(y)
    # N prime = number of nonzero weights
    Np = np.count_nonzero(w)
    # Weighted mean average 
    ywbar = np.average(y,weights=w)
    # Weighted variance
    return sum(w[i]*(y[i]-ywbar)**2 for i in range(N))/((1-1/Np)*sum(w[i] for i in range(N)))

# Combined EDF function
def CombinedEDF(N, m, a, d, F, S, v):
    def Log(x):
        return math.log(x) if x else 0
    
    def Sw(t, a):
        b = abs(t)
        if a == 2:
            return -b
        elif a == 1:
            return t * t * Log(b)
        elif a == 0:
            return b ** 3
        elif a == -1:
            return -t ** 4 * Log(b)
        elif a == -2:
            return -b ** 5
        elif a == -3:
            return t ** 6 * Log(b)
        elif a == -4:
            return b ** 7

    def Sx(t, F, a):
        if F > 0:
            return F ** 2 * (2 * Sw(t, a) - Sw(t - 1.0 / F, a) - Sw(t + 1.0 / F, a))
        else:
            return Sw(t, a + 2)

    def Sz(t, F, a, d):
        if d == 1:
            return 2 * Sx(t, F, a) - Sx(t - 1, F, a) - Sx(t + 1, F, a)
        elif d == 2:
            return 6 * Sx(t, F, a) - 4 * Sx(t - 1, F, a) - 4 * Sx(t + 1, F, a) + Sx(t - 2, F, a) + Sx(t + 2, F, a)
        elif d == 3:
            return 20 * Sx(t, F, a) - 15 * Sx(t - 1, F, a) - 15 * Sx(t + 1, F, a) + 6 * Sx(t - 2, F, a) + \
                   6 * Sx(t + 2, F, a) - Sx(t - 3, F, a) - Sx(t + 3, F, a)

    def Sum(J, M, S, F, a, d):
        sum_val = Sz(0, F, a, d) ** 2
        for j in range(1, J):
            z = Sz(j / S, F, a, d) ** 2 * (1 - j / M)
            sum_val += 2 * z
        z_last = Sz(J / S, F, a, d) ** 2 * (1 - J / M)
        sum_val += z_last
        return sum_val

    def bico(n, k):
        b = 1.0
        for i in range(k):
            b *= (n - i) / (k - i)
        return b

    # Local variables
    Jmax = 100
    s = float(S)
    a0, a1, b0, b1 = 0, 0, 0, 0
    edf = 0

    # Lookup tables
    T1 = [
        [0.667, 0.333, 0.778, 0.500, 0.880, 0.667],
        [0.840, 0.345, 0.997, 0.616, 1.141, 0.843],
        [1.079, 0.368, 1.033, 0.607, 1.184, 0.848],
        [0.000, 0.000, 1.048, 0.534, 1.180, 0.816],
        [0.000, 0.000, 1.302, 0.535, 1.175, 0.777],
        [0.000, 0.000, 0.000, 0.000, 1.194, 0.703],
        [0.000, 0.000, 0.000, 0.000, 1.489, 0.702]
    ]

    T2 = [
        [1.500, 0.500, 1.944, 1.000, 2.310, 1.500],
        [78.60, 25.20, 790.0, 410.0, 9950., 6520.],
        [0.667, 0.167, 0.667, 0.333, 0.778, 0.500],
        [0.000, 0.000, 0.852, 0.375, 0.997, 0.617],
        [0.000, 0.000, 1.079, 0.368, 1.033, 0.607],
        [0.000, 0.000, 0.000, 0.000, 1.053, 0.553],
        [0.000, 0.000, 0.000, 0.000, 1.302, 0.535]
    ]

    T3 = [6.000, 4.000, 15.23, 12.00, 47.80, 40.00]

    # Check arguments
    if not (1 <= d <= 3):
        return -1  # Error

    if F not in (1, m):
        return -1  # Error

    if m < 1:
        return -1  # Error

    if not (-4 <= a <= 2):
        return -1  # Error

    if S not in (1, m):
        return -1  # Error

    if a + 2 * d <= 1:
        return -1  # Error - Illegal alpha

    # Find # summands
    L = (m / F) + m * d

    # Calc M - not necessarily an int
    M = S * (N - L) / m
    M = math.floor(M) + 1.0

    J = min(int(M), (d + 1) * S)

    # Check # data points
    if N < L:
        return -1  # Error - N<L

    if v == 0:
        # Calc edf by simplified version
        edf = (Sz(0, F, a, d) ** 2 * M / Sum(J, M, S, F, a, d))
        return float(edf)

    # Calc edf by full version
    r = M / s

    if F == 1:  # Case 1. Modified variances, all alpha
        # Note: This is also the code used by unmodified
        # variances when F=m=1
        if J <= Jmax:
            # Calc edf
            edf = (Sz(0, 1, a, d) * Sz(0, 1, a, d) * M /
                   Sum(J, M, S, 1, a, d))
        else:
            if r >= d + 1:
                # Get a0 & a1 from Table 1
                a0 = T1[2 - a][2 * (d - 1) + 0]
                a1 = T1[2 - a][2 * (d - 1) + 1]

                # Calc edf
                edf = (1.0 / r) * (a0 - (a1 / r))
                edf = 1.0 / edf
            else:
                # Calc edf
                edf = (Sz(0, 1, a, d) * Sz(0, 1, a, d) * Jmax /
                       Sum(Jmax, Jmax, float(Jmax) / r, 1, a, d))
        return float(edf)  # EDF for Case 1
    else:  # Unmodified variances: F=m
        if a <= 0:  # Case 2. W FM to RR FM
            if J <= Jmax:
                if m * (d + 1) <= Jmax:  # m'=m;
                    # Calc edf
                    edf = (Sz(0, m, a, d) * Sz(0, m, a, d) * M /
                           Sum(J, M, S, m, a, d))
                else:  # m'=infinity, use F=m=0 as flag
                    edf = (Sz(0, 0, a, d) * Sz(0, 0, a, d) * M /
                           Sum(J, M, S, 0, a, d))
            else:  # J>Jmax
                if r >= d + 1:
                    # Get a0 & a1 from Table 2
                    a0 = T2[2 - a][2 * (d - 1) + 0]
                    a1 = T2[2 - a][2 * (d - 1) + 1]

                    # Calc edf
                    edf = (1.0 / r) * (a0 - (a1 / r))
                    edf = 1.0 / edf
                else:  # r<d+1
                    # Calc edf
                    edf = (Sz(0, 0, a, d) * Sz(0, 0, a, d) * Jmax /
                           Sum(Jmax, Jmax, float(Jmax) / r, 0, a, d))
            return float(edf)  # EDF for Case 2
        elif a == 1:  # Case 3. F PM
            if J <= Jmax:
                # Calc edf
                # Note: m must be <1e6 to avoid roundoff error
                edf = (Sz(0, m, 1, d) * Sz(0, m, 1, d) * M /
                       Sum(J, M, S, m, 1, d))
            else:
                if r >= d + 1:
                    # Get a0 & a1 from Table 2
                    a0 = T2[2 - a][2 * (d - 1) + 0]
                    a1 = T2[2 - a][2 * (d - 1) + 1]

                    # Get b0 & b1 from Table 3
                    b0 = T3[2 * (d - 1) + 0]
                    b1 = T3[2 * (d - 1) + 1]

                    # Calc edf
                    edf = (((b0 + b1 * Log(m)) * (b0 + b1 * Log(m)) * r) /
                           (a0 - (a1 / r)))
                else:
                    # Get b0 & b1 from Table 3
                    b0 = T3[2 * (d - 1) + 0]
                    b1 = T3[2 * (d - 1) + 1]

                    # Calc edf
                    edf = (((b0 + b1 * Log(m)) * (b0 + b1 * Log(m)) * Jmax) /
                           Sum(Jmax, Jmax, float(Jmax) / r,
                               float(Jmax) / r, 1, d))
            return float(edf)  # EDF for Case 3
        else:  # Case 4. W PM a=2
            K = int(math.ceil(r))

            if K <= d:
                # Use a0 and a1 as working variables
                a0 = bico(2 * d, d)

                # Calc sum
                edf = 0.0
                for k in range(1, K):
                    a1 = bico(2 * d, d - k)

                    edf += (1.0 - (float(k) / r)) * a1 * a1

                # Complete edf calc
                edf *= 2.0 / (a0 * a0)
                edf += 1.0
                edf = M / edf
            else:
                # Get a0 & a1 for a=2 from Table 2
                a0 = T2[0][2 * (d - 1) + 0]
                a1 = T2[0][2 * (d - 1) + 1]

                # Calc edf
                edf = (a0 - (a1 / r))
                edf = M / edf
            return float(edf)  # EDF for Case 4

def noise_id(z, dmin=0, dmax=2):
    # Initialize the variables
    d = 0
    p = 0

    # Define the function to calculate lag-1 autocorrelation
    def autocorrelation(z):
        z_bar = np.mean(z)
        numerator = np.sum((z[:-1] - z_bar) * (z[1:] - z_bar))
        denominator = np.sum((z-z_bar)**2)
        return numerator/denominator

    # Main algorithm loop
    while True:
        # Calculate lag-1 autocorrelation
        r1 = autocorrelation(z)
        delta = r1/(1+r1)

        # Check the conditions
        if d >= dmin and (delta < 0.25 or d >= dmax):
            p = -2*(delta+d)
            break
        else:
            # Redefine the data vector z
            z = z[:-1] - z[1:]
            d += 1

    return p

def allanoadev(y,t,error):
        
    # Length of observation
    N = len(y)
        
    # m array where 3 < m < N/3 
    m = np.arange(3,1+round(N//3),1)
    # Initial sampling time
    tau0 =  np.mean(np.diff(t))
    # Steps
    step = tau0
    
    # Empty variance array
    allanvar = []
    # Empty tau values array 
    tau = []

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
                
    # Calculate allan std dev
    allanstd = np.sqrt(allanvar)
        
    return tau, allanstd

from scipy import stats
def confidence_int(y,error,dev,modified,overlapping,devtype):
    # Get num of data
    N = len(y)
    # Construct weights
    w = 1/(error)**2
    # Construct m-array
    m = np.arange(3,N//3+1,1)
    # Construct empty CI lists
    low_ci = []
    high_ci = []
    # Iterate over every m value
    for k in range(len(m)):
        if modified:
            F = 1
        else:
            F = m[k]
        if overlapping:
            S = m[k]
        else:
            S = 1
        if devtype == 'allan':
            d = 2 
        elif devtype == 'hadamard':
            d = 3
        # Number of bins as floor division 
        num_bins = N // m[k]
        # Reshape the data into a 2D array
        # Each row represents a bin
        data_y = y[:num_bins*m[k]]
        data_w = w[:num_bins*m[k]]
        bins_y = data_y.reshape((num_bins, m[k]))
        bins_w = data_w.reshape((num_bins, m[k]))
        # Calculate average of each bin
        z = np.average(bins_y, axis=1, weights=bins_w)
        # Calculate alpha value 
        # Lag autocorr is reliable for N(data) >= 32 
        if N // m[k] >= 32:
            alpha = noise_id(z)
        # Defaults to alpha = 0 to match stable32 
        else:
            alpha = 0
        # Construct edfs 
        # CombinedEDF(N, m, a, d, F, S, v):
        if (alpha > -3) and (alpha < 3):
            edf = CombinedEDF(N,m[k],int(alpha),d,F,S,1)
        else:
            # Approximate value
            edf = CombinedEDF(N,m[k],int(alpha),d,F,S,0)
        # Calculate confidence intervals
        low_ci.append(np.sqrt(edf*dev[k]**2/scipy.stats.chi2.ppf(1-0.1/2,edf)))
        high_ci.append(np.sqrt(edf*dev[k]**2/scipy.stats.chi2.ppf(0.1/2,edf)))
    return low_ci, high_ci
