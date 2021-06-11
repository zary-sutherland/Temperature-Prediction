# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 11:39:39 2020

@author: Sutherland
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import acf, pacf, graphics
from scipy.stats import pearsonr
from scipy.stats import beta
import statsmodels.formula.api as sm
import os

dir = 'C:/Users/Sutherland/Downloads'
os.chdir(str(dir))

df_raw = pd.read_csv('C:/Users/Sutherland/Downloads/NASA_data_1995-2020_temperature_full.csv', index_col = 0)
#preprocessing
df_raw[df_raw == 0] = np.nan

NASA_temp = df_raw

NASA_temp.T2M = NASA_temp.T2M.fillna(method = 'ffill')

NASA_temp.index = pd.to_datetime(NASA_temp.index)
plt.figure(figsize=(15, 5))
plt.plot(NASA_temp.index, NASA_temp.T2M)
plt.ylabel('Temperature in degrees Celcius')
plt.xlabel('Datetime')
plt.title('Time series of Temperature from NASA/power')
plt.show()
#%%
df_raw = pd.read_csv('C:/Users/Sutherland/Desktop/solar irradiance/LoggerNet/CR1000_UWICH_CM.dat', header=0, skiprows=([1,2]))
df_raw = df_raw.rename(columns={"TIMESTAMP": "Datetime"})
df_raw.index = pd.to_datetime(df_raw.Datetime)

target = df_raw.AirTC_Avg

m = target.isnull()
target = target.mask(m)

target[ m == True] = 0

target2 = target.astype(float).resample('1 D').mean()

target2.to_csv('CR1000_UWICH_CM_2020_Sep_freq_1D_Temperature.csv')

#%%
UWICH_temp = pd.read_csv('C:/Users/Sutherland/Downloads/CR1000_UWICH_CM_2020_Sep_freq_1D_Temperature.csv', index_col=0)
UWICH_temp[UWICH_temp == 0] = np.nan

UWICH_temp.index = pd.to_datetime(UWICH_temp.index)

plt.figure(figsize=(15, 5))
plt.plot(UWICH_temp.index, UWICH_temp.AirTC_Avg)
plt.ylabel('Temperature in degrees Celcius')
plt.xlabel('Datetime')
plt.title('Time series of Temperature at UWICH')
plt.show()

#%%
df_raw = pd.read_csv('C:/Users/Sutherland/Downloads/NASA_data_1995-2020_temperature_full.csv', index_col = 0)
#preprocessing
df_raw[df_raw == 0] = np.nan

NASA_temp = df_raw

combined_data = pd.concat([UWICH_temp, NASA_temp], axis = 1)

combined_data = combined_data.dropna()

plt.figure()
plt.plot(combined_data.T2M, combined_data.AirTC_Avg, '.')
plt.xlabel('NASA Temperature °C')
plt.ylabel('UWICH Temperature °C')

Val = combined_data.values

#%%

slope, intercept = np.polyfit(Val[:,1], Val[:,0], 1)
print("Slope: "+str(slope)+ '   Intercept:    '+str(intercept))
#Slope: 1.077259928713183   Intercept:    -0.6484044312716994
#%%

derived_vals = pd.DataFrame()
derived_vals['AirTC_Avg'] =  slope * NASA_temp.T2M + intercept

UWICH_temp = pd.read_csv('C:/Users/Sutherland/Downloads/CR1000_UWICH_CM_2020_Sep_freq_1D_Temperature.csv', index_col=0)
UWICH_temp[UWICH_temp == 0] = np.nan

UWICH_temp = UWICH_temp.dropna()
UWICH_temp = UWICH_temp.combine_first(derived_vals)
UWICH_temp.index = pd.to_datetime(UWICH_temp.index)
UWICH_temp.asfreq('1d')

plt.figure(figsize=(15, 5))
plt.plot(UWICH_temp.index, UWICH_temp.AirTC_Avg)
plt.ylabel('Temperature in degrees Celcius')
plt.xlabel('Datetime')
plt.title('Time series of Temperature at UWICH with derived values')
plt.show()
UWICH_temp = UWICH_temp['2016-01-01':'2020-10-25']
#%%
UWICH_temp.AirTC_Avg[UWICH_temp.AirTC_Avg < 0] = np.nan
UWICH_temp.AirTC_Avg = UWICH_temp.AirTC_Avg.fillna(method = 'ffill')
graphics.plot_acf(UWICH_temp.AirTC_Avg, lags=100)#61 in the ar
plt.show()

graphics.plot_pacf(UWICH_temp.AirTC_Avg) #8 in the ma
plt.show()


graphics.plot_acf(UWICH_temp.AirTC_Avg.diff().dropna())#2 in ar if diff of 1 
plt.show()

graphics.plot_pacf(UWICH_temp.AirTC_Avg.diff().dropna())#7 in ma if diff of 1 
plt.show()

#%%
temp_check = 23000
bestorder = ''
for ar_order in range(3):
    for diff_order in range(2):
        for ma_order in range(9):
            aic_value = ARIMA(UWICH_temp.AirTC_Avg, order=(ar_order,diff_order,ma_order)).fit().aic
            if (aic_value <= temp_check):
                temp_check = aic_value
                bestorder = 'AR:' + str(ar_order) + '   I:' + str(diff_order)+ '   MA:' + str(ma_order)
            print('AR:' + str(ar_order) + '   I:' + str(diff_order)+ '   MA:' + str(ma_order)+ '   AIC:' + str(round(aic_value)))

print(temp_check)
print('Best Order: ' + bestorder)
#Best Order: AR:2   I:1   MA:5 with aic of 2474.3110495185233
#%%
kfold = KFold(10,True,3)
aic_values = []
i=0
for train, test in kfold.split(UWICH_temp.AirTC_Avg):
    print('iteration: '+str(i))
    mod = ARIMA(UWICH_temp.AirTC_Avg[train], order=(61,0,8))
    res = mod.fit()
    validation = res.apply(UWICH_temp.AirTC_Avg[test])
    aic_values.append(validation.aic)
    i = i + 1
average_aic = np.mean(aic_values)
print(average_aic)
#average_aic =  291.8014967345241

#%%
mod = ARIMA(UWICH_temp.AirTC_Avg, order=(61,0,8))
res = mod.fit()
print(res.summary())

sns.set_style('darkgrid')
pd.plotting.register_matplotlib_converters()
# Default figure size
sns.mpl.rc('figure',figsize=(16, 6))
# fig = res.predict(9000,9150)
# plt.figure()
# plt.plot(fig.index, fig.values)
fig, ax = plt.subplots(figsize=(15, 5))
UWICH_temp.loc['2020':].AirTC_Avg.plot(ax=ax)
fcast = res.get_forecast(30).summary_frame()
fcast['mean'].plot(ax=ax)
ax.set(xlabel ="Date", ylabel="Temperature degree celcius")
ax.fill_between(fcast.index, fcast['mean_ci_lower'], fcast['mean_ci_upper'], color='k', alpha=0.1);