import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

from matplotlib.gridspec import GridSpec
import func_empiral_EWS as func_e


label_list = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y']
absolut_address = './'


#--------------------------------------------------------------#
#GrIS
gris_time = np.arange(1855, 2013)
gris = np.loadtxt(f'{absolut_address}/data/GrIS/CWG_heigtchange.txt')
gris_lambda = []
ws_list = [80, 100, 120]
for w in ws_list:
    lamda_temp, gris_trend = func_e.lambda_wrapper_rmean(gris, ws=w, rws=10)
    gris_lambda.append(np.roll(lamda_temp, w // 2))

# -------------------------------------------------------------#
#Amoc
amoc_time1 = np.arange(1850, 2022, 1)
roll_length = 30

address = f'{absolut_address}data/AMOC/HadCRUT.5.0.1.0._amoc_mean.nc'
amoc_1 = xr.open_dataset(address).amoc.values

address = f'{absolut_address}data/AMOC/HadCRUT.5.0.1.0._amoc_ensemble_max.nc'
amoc_max_1 = xr.open_dataset(address).amoc.values

address = f'{absolut_address}data/AMOC/HadCRUT.5.0.1.0._amoc_ensemble_min.nc'
amoc_min_1 = xr.open_dataset(address).amoc.values

address = f'{absolut_address}data/AMOC/HadCRUT.5.0.1.0._lam_mean.nc'
amoc_lab_1 = np.roll(xr.open_dataset(address).lam.values,roll_length)

address = f'{absolut_address}data/AMOC/HadCRUT.5.0.1.0._lam_ensemble_max.nc'
amoc_lab_max_1 = np.roll(xr.open_dataset(address).lam.values,roll_length)

address = f'{absolut_address}data/AMOC/HadCRUT.5.0.1.0._lam_ensemble_min.nc'
amoc_lab_min_1 = np.roll(xr.open_dataset(address).lam.values,roll_length)


amoc_time2 = np.arange(1854, 2018, 1)
address = f'{absolut_address}data/AMOC/ERSSTv5_amoc_mean.nc'
data = xr.open_dataset(address).sst
amoc_2 = (data - data.mean(dim='time')).values

address = f'{absolut_address}data/AMOC/ERSSTv5_amoc_ensemble_max.nc'
amoc_max_2 = xr.open_dataset(address).amoc.values

address = f'{absolut_address}data/AMOC/ERSSTv5_amoc_ensemble_min.nc'
amoc_min_2 = xr.open_dataset(address).amoc.values

address = f'{absolut_address}data/AMOC/ERSSTv5_lambda_mean.nc'
amoc_lab_2 = np.roll(xr.open_dataset(address).lam.values, roll_length)

address = f'{absolut_address}data/AMOC/ERSSTv5_lambda_ensemble_max.nc'
amoc_lab_max_2 = np.roll(xr.open_dataset(address).lam.values,roll_length)

address = f'{absolut_address}data/AMOC/ERSSTv5_lambda_ensemble_min.nc'
amoc_lab_min_2 = np.roll(xr.open_dataset(address).lam.values,roll_length)

# -------------------------------------------------------------------------------#
#Amazon
amazon_time = np.arange(1992, 2017, 1 / 12)
amazon = np.loadtxt(f'{absolut_address}data/Amazon/amazon_vod.txt')[12:]

stl = STL(amazon, seasonal=7, period=12).fit()
amazon_ds = stl.resid + stl.trend
w = 10 * 12
ws = 15
lambda_temp, amazon_tred_1 = func_e.lambda_wrapper_rmean(amazon_ds, ws=w, rws=ws)
amazon_la = (np.roll(lambda_temp, w // 2))

amazon_time1 = np.arange(2002, 2012, 1 / 12)[6:-3]
amazon_time2 = np.arange(2012, 2021, 1 / 12)[7:]

d_amaz = xr.open_dataset(f'{absolut_address}data/Amazon/results.nc')
single1 = np.array(d_amaz['raw'].values[0, :], dtype=np.float64)[:len(amazon_time1)]
single2 = np.array(d_amaz['raw'].values[2, :], dtype=np.float64)[len(amazon_time1):]

w = 5 * 12
ws = 10
x = single1[~np.isnan(single1)]
stl1 = STL(x, seasonal=7, period=12).fit()
amazon_single1_ds = stl1.resid + stl1.trend
lamda_temp1, trend = func_e.lambda_wrapper_rmean(amazon_single1_ds, ws=w, rws=ws)
lambda_single_1 = (np.roll(lamda_temp1, w // 2))

x = single2[~np.isnan(single2)]
stl2 = STL(x, seasonal=7, period=12).fit()
amazon_single2_ds = stl2.resid + stl2.trend
lamda_temp2, trend = func_e.lambda_wrapper_rmean(amazon_single2_ds, ws=w, rws=ws)
lambda_single_2 = (np.roll(lamda_temp2, w // 2))

# ---------------------------------------------------------------------------------------------#
#SAMS
sams_time = np.arange(1979, 2020, 1 / 12)
w = 25 * 12
ws = 5 * 12
sams1 = np.loadtxt(f'{absolut_address}/data/SAMS/mean_rain_sa.txt')
lambda_temp, sams_tred_1 = func_e.lambda_wrapper_rmean(sams1, ws=w, rws=ws)
sams_la = (np.roll(lambda_temp, w // 2))

sams2 = np.loadtxt(f'{absolut_address}/data/SAMS/mean_rain_detrended_sa.txt')
lambda_temp, sams_tred_2 = func_e.lambda_wrapper_rmean(sams2, ws=w, rws=ws)
sams_la2 = (np.roll(lambda_temp, w // 2))
# ---------------------------------------------------------------------------------------------#

figsize_set = (6,6)
fig = plt.figure(figsize=figsize_set)

gs = GridSpec(2,1, figure=fig,wspace = .25,hspace = .2)
axs1 =  fig.add_subplot(gs[0, 0])
axs3 = fig.add_subplot(gs[1, 0],sharex=axs1)
func_e.draw_gris(axs1,axs3,gris_time,gris,gris_lambda,ws_list)
fig.savefig('./Gris.pdf',bbox_inches = 'tight')
plt.show()
plt.close()

fig = plt.figure(figsize=figsize_set)
gs = GridSpec(2,1, figure=fig,wspace = .25,hspace = .2)

axs1 =  fig.add_subplot(gs[0, 0])
axs3 = fig.add_subplot(gs[1, 0],sharex=axs1)
amoc_1_list = [amoc_time1, amoc_max_1, amoc_min_1, amoc_1, amoc_lab_max_1, amoc_lab_min_1,amoc_lab_1]
amoc_2_list = [amoc_time2, amoc_max_2, amoc_min_2, amoc_2, amoc_lab_max_2, amoc_lab_min_2,amoc_lab_2]
func_e.draw_AMOC(axs1,axs3,amoc_1_list,amoc_2_list)
fig.savefig('./amoc.pdf',bbox_inches = 'tight')
plt.show()
plt.close()

fig = plt.figure(figsize=figsize_set)
gs = GridSpec(2,1, figure=fig,wspace = .25,hspace = .2)
axs1 =  fig.add_subplot(gs[0, 0])
axs3 = fig.add_subplot(gs[1, 0],sharex=axs1)
amazon_list1 = [amazon_time, amazon_ds,amazon_la]
amazon_list2 = [amazon_time1, amazon_single1_ds,lambda_single_1]
amazon_list3 = [amazon_time2, amazon_single2_ds, lambda_single_2]
func_e.draw_Amaz(axs1,axs3,amazon_list1,amazon_list2,amazon_list3)
fig.savefig('./Amazon.pdf',bbox_inches = 'tight')
plt.show()
plt.close()

fig = plt.figure(figsize=figsize_set)
gs = GridSpec(2,1, figure=fig,wspace = .25,hspace = .2)
axs1 =  fig.add_subplot(gs[0, 0])
axs3 = fig.add_subplot(gs[1, 0],sharex=axs1)
func_e.draw_Sams(axs1,axs3,sams_time, sams1,sams2,sams_la,sams_la2)
fig.savefig('./sams.pdf',bbox_inches = 'tight')
plt.show()
plt.close()
