"""
Created on Thu, 03 Oct 2024
Last modified on Thu, 03 Oct 2024

@author: Thuy-Hai

Compute total expected profit for fixed layout and historical data
Expected power is sold on day-ahead market, regardless of negative or low prices
"""

#%% Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import winsound
import time

# Wind turbine
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular

# Wind site
from py_wake.site import UniformSite

# Pywake Wake models
from py_wake.deficit_models.gaussian import NiayifarGaussian
from py_wake.turbulence_models import CrespoHernandez
from py_wake.superposition_models import LinearSum
from py_wake.site.shear import PowerShear


#%% User-defined parameters

year = '2023'

# Wind farm data
wf_name = 'Northwind'
type_opt_str = 'AEP' # 'AEP', 'noR', str(r_max)
wf_layout_name = type_opt_str + '_base'
file_wt_pc = wf_name + '_opt_' + type_opt_str
data_path = 'Data\\'

# Scenarios of wind and electricity prices
scenarios_file = "data_scenarios_processed.xlsx"
nb_samples_hourly = 4 # Each sample is a quarter hour
wind_height = 100
scenarios_years = ['2021', '2022', '2023', '2024']
columns_scen = ["dateTime_elia", "dateTime", "dateTime_utc", "dateHour_utc", 
                "ws", "wd", "price_da", 
                "price_capa_up", "price_acti_up", "volume_acti_up", "volume_acti_up_pu",
                "price_capa_down", "price_acti_down", "volume_acti_down", "volume_acti_up_down",
                "price_mip", "price_neg_imb", "alpha_imb"]

# Forecast errors distribution
S = 500 # Number of sampled forecasts (S)
max_chunk = 500
# Wind
mean_error_fc_ws = 0.0 # Wind speed [m/s]
std_error_fc_ws_perc = 0.15 # [%]
mean_error_fc_wd = 0.0 # Wind distribution [°]
std_error_fc_wd = 4.2 # [°]
mean_error_fc_wp = 0.0 # Wind power [MW]
std_error_fc_wp_perc = 0.03 # [%]
# Electricity prices
mean_error_fc_price_da = 0.0 # Day-ahead prices [€/MWh]
std_error_fc_price_da_perc = 0.07 # [%]
mean_error_fc_price_imb = 0.0 # Negative imbalance prices [€/MWh]
std_error_fc_price_imb_perc = 0.1 # [%]

if wf_name == 'Northwind':
    nb_wt = 72 # number of wind turbines
    wt_name = 'V112-3'
    rotor_diam = 112
    hub_height = 71
elif wf_name == 'Rentel':
    nb_wt = 42 # number of wind turbines
    wt_name = 'SWT_7.0_154'
    rotor_diam = 154
    hub_height = 105.5

#%% Load data

data_scen = pd.read_excel(data_path + scenarios_file, sheet_name=year, names=columns_scen)

# Wind turbine data
data_turbine = pd.read_excel(data_path + file_wt_pc + '.xlsx', sheet_name=wt_name, names=['ws', 'power', 'cp', 'ct'], usecols='A:D')
pc_ws = np.array(data_turbine.ws)
pc_power = np.array(data_turbine.power)
pc_ct = np.array(data_turbine.ct)
# Wind farm data
layout_wf = pd.read_excel(data_path + file_wt_pc + '.xlsx', sheet_name=wf_layout_name, names=['name', 'x', 'y'], usecols='A:C')

#%% Setup wind farm

# Wind turbine
windTurbines = WindTurbine(name=wt_name,
                    diameter=rotor_diam,
                    hub_height=hub_height,
                    powerCtFunction=PowerCtTabular(pc_ws, pc_power,'MW', pc_ct, power_idle=0, ct_idle=0, method='linear'))

# Defining the site, wind turbines and wake model
site = UniformSite(ti=0.077, shear=PowerShear(wind_height, 0.12), interp_method='linear')
site.interp_method = 'linear' 
wake_model = NiayifarGaussian(site, windTurbines, turbulenceModel=CrespoHernandez(), superpositionModel=LinearSum())

p_farm_rated = nb_wt * windTurbines.power(20) / 1e6 # MW

x_wf = layout_wf.x
y_wf = layout_wf.y

# Plot layout
plt.figure(dpi=600)
plt.scatter(x_wf, y_wf, c='blue', marker='1')
plt.xlim(-500, 8000)
plt.ylim(-500, 8000)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.title(wf_layout_name + ' layout')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.show()

# Compute AEP
# sim_res = wake_model(x_wf, y_wf, wd=data_scen.wd, ws=data_scen.ws, time=True)
# p_wt = sim_res.Power.data / 1e6 # MW
# p_farm = np.sum(p_wt, axis=0)
# aep = np.sum(p_farm)/1e6 # TWh
# print(aep)

# winsound.Beep(500, 1000)

# aep2 = wake_model(x_wf, y_wf, wd=data_scen.wd, ws=data_scen.ws, time=True).aep().sum().values / 1e3 * 4

#%% Setup problem

x = np.array(x_wf)
y = np.array(y_wf)
  
ws, wd = np.array(data_scen.ws), np.array(data_scen.wd) 
price_da = np.array(data_scen.price_da) 
price_capa_up, price_acti_up, volume_acti_up_pu = np.array(data_scen.price_capa_up), np.array(data_scen.price_acti_up), np.array(data_scen.volume_acti_up_pu)
price_imb = np.array(data_scen.price_mip)

#%% Setup functions
    
# Standard devation is given in % 
def forecast_error_rel(X, mean_error, std_error_perc, S, val_min=-1000, val_max=1000000, sigma_min=0.01):
    x_temp = X.reshape(-1, 1) * np.ones((1, S))
    std = std_error_perc * np.abs(x_temp)
    std_error = np.where(std==0, sigma_min, std)
    error = np.random.normal(mean_error, std_error, size=(len(X), S))
    x_fc = x_temp + error
    
    x_fc_min = np.where(x_fc < val_min, val_min, x_fc)
    x_fc_min_max = np.where(x_fc_min > val_max, val_max, x_fc_min)
    
    return np.array(x_fc_min_max)

# Standard deviation is absolute value
def forecast_error(X, mean_error, std_error, S, val_min=-1000, val_max=1000000):   
    error = np.random.normal(mean_error, std_error, size=(len(X), S))
    
    x_temp = X.reshape(-1, 1) * np.ones((1, S))
    x_fc = x_temp + error
    
    x_fc_min = np.where(x_fc < val_min, val_min, x_fc)
    x_fc_min_max = np.where(x_fc_min > val_max, val_max, x_fc_min)
    
    return np.array(x_fc_min_max)


# Sub-optimization of alpha AND beta
def opti_profit(x, y, ws_kt, wd_kt, price_da_kt, price_imb_kt, S, nb_cpu=1, nb_chunks=1):
    
    KT = ws_kt.shape[0]
    
    # Forecast values
    ws_fc_kt_S = forecast_error_rel(ws_kt, mean_error_fc_ws, std_error_fc_ws_perc, S, val_min=0, val_max=30)
    wd_fc_kt_S = forecast_error(wd_kt, mean_error_fc_wd, std_error_fc_wd, S, val_min=0, val_max=360)
    price_da_fc_kt_S = forecast_error_rel(price_da_kt, mean_error_fc_price_da, std_error_fc_price_da_perc, S, sigma_min=5)
    price_imb_fc_kt_S = forecast_error_rel(price_imb_kt, mean_error_fc_price_imb, std_error_fc_price_imb_perc, S, val_min=0, sigma_min=5)
    # Ensure imbalance penalties are always higher than day-ahead prices
    price_imb_fc_kt_S = np.where(price_imb_fc_kt_S < price_da_fc_kt_S, price_da_fc_kt_S, price_imb_fc_kt_S)

    # Convert wind data to wind power using pywake
    ws_fc_list = ws_fc_kt_S.flatten() 
    wd_fc_list = wd_fc_kt_S.flatten()
    p_tot_fc_list_pw = wake_model(x, y, wd=wd_fc_list, ws=ws_fc_list, time=True, return_simulationResult=False, n_cpu=nb_cpu, wd_chunks=nb_chunks)[2].sum(axis=0) / 1e6 # MW 
    p_tot_fc_list = forecast_error_rel(p_tot_fc_list_pw, mean_error_fc_wp, std_error_fc_wp_perc, 1, val_min=0) # Modelling error on wind power
    p_tot_fc = p_tot_fc_list.reshape(KT, S)
    p_tot_fc_mean = np.mean(p_tot_fc, axis=1)
    
    # Bid all expected power to day-ahead market, regardless of price
    p_da = p_tot_fc_mean.reshape(KT, 1)

    # Identify imbalance situations    
    # If available power is lower than bidded capacity
    p_da_supplied = np.minimum(p_tot_fc, p_da)  
    delta_p_da = p_da - p_da_supplied
    
    # Compute profits and penalties 
    # Price is given in €/MWh, but timestep could be lower than an hour
    profit_pos = p_da * price_da_fc_kt_S / nb_samples_hourly
    pen = delta_p_da * price_imb_fc_kt_S / nb_samples_hourly
      
    profit = profit_pos - pen
    
    expected_profit = np.mean(profit, axis=1)
    
    return expected_profit, profit, profit_pos, pen, p_da_supplied, p_da


#%% Optimize alpha_kt for timestep k,t

# Interesting timesteps
# If > 0, reserve not profitable
cond = data_scen.price_da - (data_scen.price_capa_up + data_scen.price_acti_up * data_scen.volume_acti_up_pu)
kt_reserve = data_scen.index[cond < 0]
ws_kt_reserve = data_scen.ws[cond < 0]

# Sample timesteps kt
# kt = kt_reserve[0:2]
# len_kt = 1 if isinstance(kt, np.int64) else len(kt)

kt = 0
len_kt = 1

arr = np.ones(len_kt)
ws_kt, wd_kt = arr*np.array(data_scen.loc[kt].ws), arr*np.array(data_scen.loc[kt].wd)
price_da_kt, price_imb_kt = arr*np.array(data_scen.loc[kt].price_da), arr*np.array(data_scen.loc[kt].price_mip)

expected_profit, profit_fc, profit_pos_fc, pen_fc, p_da_s_fc, p_da = opti_profit(x, y, ws_kt, wd_kt, price_da_kt, price_imb_kt, S, nb_cpu=1, nb_chunks=1)

std_profit = np.std(np.sum(profit_fc, axis=0))

print("\nOpti AEP")
print("Total profit max:", round(np.sum(expected_profit), 3), "+/-", round(std_profit/np.sqrt(S), 3), "€") 
print("P_da_c =", round(np.sum(p_da)/nb_samples_hourly, 2), "MWh, Supplied =", round(np.sum(np.mean(p_da_s_fc, axis=1)/nb_samples_hourly), 2), "MWh")
 
    
#%% Compute yearly profit, optimizing alpha and total P offered

print(year, wf_name, wf_layout_name)

if __name__ == '__main__':  
    
    S = 500
    # max_chunk = 50
    # nb_simu_per_chunk = 100000
    nb_cpu = 4
    nb_chunks = 4
    
    KT_tot = len(data_scen)
    
    kt_reserve_chunk_index = np.arange(0, KT_tot, max_chunk)
    kt_reserve_chunk_index = np.append(kt_reserve_chunk_index, KT_tot)
    
    # Intialization 
    mean_profit_array = np.zeros(KT_tot)
    profit_fc_array = np.zeros((KT_tot, S))
    profit_pos_fc_array = np.zeros((KT_tot, S))
    pen_fc_array = np.zeros((KT_tot, S))
    p_tot_c_array = np.zeros(KT_tot)
    p_da_s_fc_array = np.zeros((KT_tot, S))
    
    start = time.time()
    
    for i_chunk in range(len(kt_reserve_chunk_index)-1):
       
        kt_chunk = data_scen.index[kt_reserve_chunk_index[i_chunk]:kt_reserve_chunk_index[i_chunk+1]]
       
        ws_chunk, wd_chunk = ws[kt_chunk], wd[kt_chunk]
        price_da_chunk, price_imb_chunk = price_da[kt_chunk], price_imb[kt_chunk]
        
        # nb_simu = len(ws_chunk) * S
        # nb_chunks = np.where(nb_simu > nb_simu_per_chunk, int(nb_simu/nb_simu_per_chunk), 4).flatten()[0]
       
        expected_profit_temp, profit_fc_temp, profit_pos_fc_temp, pen_fc_temp, p_da_s_fc_temp, p_da_temp = opti_profit(x, y, ws_chunk, wd_chunk, price_da_chunk, price_imb_chunk, S, nb_cpu=nb_cpu, nb_chunks=nb_chunks)    
            
        mean_profit_array[kt_reserve_chunk_index[i_chunk]:kt_reserve_chunk_index[i_chunk+1]] = expected_profit_temp
        profit_fc_array[kt_reserve_chunk_index[i_chunk]:kt_reserve_chunk_index[i_chunk+1], :] = profit_fc_temp
        profit_pos_fc_array[kt_reserve_chunk_index[i_chunk]:kt_reserve_chunk_index[i_chunk+1], :] = profit_pos_fc_temp
        pen_fc_array[kt_reserve_chunk_index[i_chunk]:kt_reserve_chunk_index[i_chunk+1], :] = pen_fc_temp
        p_tot_c_array[kt_reserve_chunk_index[i_chunk]:kt_reserve_chunk_index[i_chunk+1]] = p_da_temp.flatten()
        p_da_s_fc_array[kt_reserve_chunk_index[i_chunk]:kt_reserve_chunk_index[i_chunk+1], :] = p_da_s_fc_temp
        
    end = time.time()
    comp_time = end - start
    print(round(comp_time), "s")
    
    # Mean profit and std 
    total_mean_profit = np.sum(np.mean(profit_fc_array, axis=1)) / 1e6 # M€
    std_total_profit = np.std(np.sum(profit_fc_array, axis=0)) / 1e6
    total_profit_fc_array = np.sum(profit_fc_array, axis=0) / 1e6
    # Profits breakdown
    # Positive profits
    total_mean_profit_pos = np.sum(np.mean(profit_pos_fc_array, axis=1)) / 1e6 # M€
    std_total_profit_pos = np.std(np.sum(profit_pos_fc_array, axis=0)) / 1e6
    total_profit_pos_fc_array = np.sum(profit_pos_fc_array, axis=0) / 1e6
    mean_profit_pos_array = np.mean(profit_pos_fc_array, axis=1)
    # Penalties
    total_mean_pen = np.sum(np.mean(pen_fc_array, axis=1)) / 1e6 # M€
    std_total_pen = np.std(np.sum(pen_fc_array, axis=0)) / 1e6
    total_pen_fc_array = np.sum(pen_fc_array, axis=0) / 1e6
    mean_pen_array = np.mean(pen_fc_array, axis=1)
    
    print("\nTotal expected profit for", year, ":", round(total_mean_profit, 3), "+/-", round(std_total_profit/np.sqrt(S), 2),"M€")
    
    # AEP (contracted and supplied) 
    aep_c = np.sum(p_tot_c_array) / 1e3 / nb_samples_hourly # GWh
    aep_mean_s = np.mean(np.sum(p_da_s_fc_array, axis=0)) / 1e3 / nb_samples_hourly # GWh
    aep_std_s = np.std(np.sum(p_da_s_fc_array, axis=0)) / 1e3 / nb_samples_hourly # GWh
    print("Total power produced for", year, ":", round(aep_mean_s, 3), "GWh (contracted", round(aep_c, 3),"GWh)")     

    winsound.Beep(500, 1000)
    
#%% Save results

results_summary = {'mean_total_profit': total_mean_profit, 'std_total_profit': std_total_profit,
                   'mean_total_profit_pos': total_mean_profit_pos, 'std_total_profit_pos': std_total_profit_pos,
                   'mean_total_pen': total_mean_pen, 'std_total_pen': std_total_pen,
                   'aep_c': aep_c, 
                   'aep_mean_s': aep_mean_s, 'aep_std_s': aep_std_s, 
                   'comp_time': comp_time}

total_mean_profit_fc = {'total_profit_fc': total_profit_fc_array, 'total_profit_pos_fc': total_profit_pos_fc_array, 'total_pen_fc': total_pen_fc_array}

results_qh = {'p_tot_c': p_tot_c_array, 'p_tot_s': np.mean(p_da_s_fc_array, axis=1), 
              'mean_profit': mean_profit_array, 'mean_profit_pos': mean_profit_pos_array, 'mean_pen': mean_pen_array}

writer = pd.ExcelWriter('Output/post_process_' + wf_name + '_' + wf_layout_name + "_" + year + '_aep.xlsx', engine='xlsxwriter')
pd.DataFrame.from_dict(results_summary, "index").to_excel(writer, sheet_name = 'Summary', index=True)
pd.DataFrame.from_dict(total_mean_profit_fc).to_excel(writer, sheet_name = 'Total_profit_fc', index=False)
pd.DataFrame.from_dict(results_qh).to_excel(writer, sheet_name = 'Results_qh', index=False)
writer.close()


