import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import os

# ====== Data Cleaning ======

# Create dataframe
cols = ['MJD','R.A. (mas)', 'Dec. (mas)', 'Err. R.A. (mas)', 'Err. Dec. (mas)', 'Corr.', 'Del.']
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
f = open(os.path.join(__location__, '1803+784_dem.txt'))
df = pd.read_csv(f, sep='\s+', skiprows=4, names=cols, header=None, index_col=False)

# Drop problematic row
df = df.drop(df.index[2414])

# Convert MJD index to numeric
df['MJD'] = pd.to_numeric(df['MJD'])

# Shift data values in problematic row
df.iloc[2335, :] = df.iloc[2335, :].shift()
df.iloc[2335,0] = df.iloc[2335,1]
df.iloc[2335,1], df.iloc[2335,2] = df.iloc[2335,2].split('-')

# Convert RA and Dec columns to numeric
df['R.A. (mas)'] = pd.to_numeric(df['R.A. (mas)'])
df['Dec. (mas)'] = pd.to_numeric(df['Dec. (mas)'])

# Change value to negative in problematic row
df.iloc[2335,2] *= -1

# Calculate RA*cos(dec) column
df['RAcosDEC'] = df['R.A. (mas)']*np.cos(df['Dec. (mas)'])

# Calculate BY column
df['BY'] = 2000-(51544.333981-df['MJD'])/365.242198781

# Take sample of first 500 points ~ 5 years
new_df = df.head(500)

# Scatter plot + error bars of data points
# plt.subplot(211)
# plt.title('1803+784')
# plt.errorbar(new_df['BY'], new_df['RAcosDEC'],yerr=new_df['Err. R.A. (mas)'], fmt='o', ms=1, mec='black', ecolor='lightgrey', elinewidth=1)
# plt.ylim(-2,2)
# plt.xlabel('Year')
# plt.ylabel(r'$\alpha \cos \delta$ (mas)')
# plt.grid()
# plt.subplot(212)
# plt.errorbar(new_df['BY'], new_df['Dec. (mas)'], yerr=new_df['Err. Dec. (mas)'], fmt='o', ms=1, mec='black', ecolor='lightgrey', elinewidth=1)
# plt.ylim(-4,4)
# plt.xlabel('Year')
# plt.ylabel('$\delta$ (mas)')
# plt.grid()
# plt.show()

# ====== Creating dataframe with real dates ======

# Get start and end date and truncate 
start_date = math.trunc(new_df['MJD'].values.min())
end_date = math.trunc(new_df['MJD'].values.max())

# Length of observation
N = int(round((end_date - start_date)))

# Make array of start to end date 
real_dates = np.arange(start_date,1+end_date,1)

# Create dataframe from the array
dates_df = pd.DataFrame(real_dates, columns=['MJDReal'])

# Truncate the MJD column in new_df 
new_df['MJDtrunc'] = new_df['MJD'].apply(math.trunc)

# Right merge MJDtrunc column in new_df with dataframe with all dates 
merge_df = new_df.merge(dates_df, how='right',left_on='MJDtrunc',right_on='MJDReal')

# Sort dataframe by all of the dates
merge_df = merge_df.sort_values(by=['MJDReal'])

# Fill NaN with zeroes
merge_df = merge_df.fillna(0)

# ====== Allan variance with WMA ======
# Implement the function later
# Want to see each step using Dec data

# m array where 3 < m < N/3 
m = np.arange(3,1+round(N//3),1)
# Take tau0 as 1/rate
rate = 1
tau0 = 1/rate
# steps
step = 1

# Empty variance array
allanvar = []
# Empty tau values array 
tau = []

# Create weights from Dec data
weights_std = []
for val in merge_df['Err. Dec. (mas)'].values:
    if val != 0:
        weights_std.append(1/(val**2))
    elif val == 0:
        weights_std.append(0)

# print(weights_std)


# Iterate over values of m
for k in range(len(m)): 
    # print('==========')
    # print('Current value of m: ' + str(m[k]))

    # Sliding windows 
    left = 0 
    right = m[k]
    ybar = []
    D = 0
    data_sum = 0
        
    # Check for missing windows 
    while right <= N:
        data_values = merge_df['Dec. (mas)'].values[left:right]
        data_sum = np.sum(data_values)
        # print(data_sum)
    
        if data_sum == 0:
            D +=1
        else:
            # print(weights_std[left:right])
            avg = np.average(data_values, weights=weights_std[left:right])
            # print(avg)
            ybar.append(avg)
        left += step 
        right += step    

            
    # print('Total amount of missing windows D: ' + str(D))

    # M is the number of times we can move the averaging window from the beginning to the end
    M_D = len(ybar)
    # print('Total amount of non-missing windows: ' + str(M_D))
    # print('M-2m+1-D value: ' + str(M_D-2*m[k]+1))

    # Calculate allan var 
    outersum = []
    for j in np.arange(0,M_D-2*m[k]+1,step):
        innersum = []
        for i in np.arange(j,j+m[k]-1,step):
            innersum.append(ybar[i+m[k]]-ybar[i]) #weighted mean, bikin error sendiri - dist gaussian
        outersum.append(0.5*(np.sum(innersum)/m[k])**2)

    # Calculate allan variance
    allanvar.append(np.sum(outersum)/(M_D-2*m[k]+1))
        
    # Calculate tau
    tau.append(tau0*m[k])

allanstd = np.sqrt(allanvar)
tau_year = [x/365.242198781 for x in tau]
plt.plot(tau_year, allanstd)
plt.yscale('log')
# plt.ylim(10**(-1),10**3)
plt.grid()
plt.xlabel('Timescale [yr]')
plt.ylabel('Allan std dev in $\delta$ [mas?]')