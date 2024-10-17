"""
Created on Fri, 19 July 2024
Last modified on Thu, 10 Oct 2024

@author: Thuy-Hai

Compute total expected profit for fixed layout and historical data
Reserve penalties for missing capacity (availability tests)
Reserve penalties for missing requested reserve (kappa*R)
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
r_max = 50 # MW (50, 117, 166, 221.4)
eps_pen_r_c = 10 # Penalty multiplier for failed availablity test

# Wind farm data
wf_name = 'Northwind'
type_opt_str = str(r_max) # 'AEP', 'noR', str(r_max)
wf_layout_name = type_opt_str + '_base'
file_wt_pc = wf_name + '_opt_' + type_opt_str
data_path = 'Data\\'

# Combinatorial exploration parameters (sub-optimization problem)
alpha_step_min = 0.02
beta_step = 2/221.4 # ~2MW

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
max_chunk = 50
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
mean_error_fc_price_capa_up = 0.0 # Reserve capacity prices [€/MW/h]
std_error_fc_price_capa_up_perc = 0.07 # [%]
mean_error_fc_price_acti_up = 0.0 # Reserve activation prices [€/MWh]
std_error_fc_price_acti_up_perc = 0.1 # [%]
mean_error_fc_price_imb = 0.0 # Negative imbalance prices [€/MWh]
std_error_fc_price_imb_perc = 0.1 # [%]
# Activated volume
mean_error_fc_volume_acti_up_pu = 0.0 # Normalized activated power for upward regulation [p.u.]
std_error_fc_volume_acti_up_pu_perc = 0.1 # [%]

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

#%% Setup sub-optimization problem

x = np.array(x_wf)
y = np.array(y_wf)

# Reserve (list of possible values)
p_tot_c_list = np.array([])
alpha_list = np.array([])

# Total contracted power (list of possible values)
if r_max > 0:
    p_tot_c_kt_list = np.arange(0, 1, beta_step) * p_farm_rated
    p_tot_c_kt_list = np.append(p_tot_c_kt_list, p_farm_rated)
    
    # Make combinations, and limit reserve to R_max
    for i_p in range(len(p_tot_c_kt_list)):
        alpha_step = 1/p_tot_c_kt_list[i_p]
        if alpha_step < alpha_step_min:
            alpha_step = alpha_step_min
        if p_tot_c_kt_list[i_p] <= r_max:
            alpha_temp = np.arange(0, 1, alpha_step)
            alpha_temp = np.append(alpha_temp, 1)
        else: # If p_tot > R_max, limit R to R_max
            alpha_max_lim = r_max / p_tot_c_kt_list[i_p]
            alpha_temp = np.arange(0, alpha_max_lim, alpha_step)
            alpha_temp = np.append(alpha_temp, r_max / p_tot_c_kt_list[i_p])
        alpha_list = np.append(alpha_list, alpha_temp)
        p_tot_c_list = np.append(p_tot_c_list, np.repeat(p_tot_c_kt_list[i_p], len(alpha_temp))) 
else:
    p_tot_c_kt_list = np.arange(0, p_farm_rated, 1)
    p_tot_c_kt_list = np.append(p_tot_c_kt_list, p_farm_rated)
    p_tot_c_list = p_tot_c_kt_list
    alpha_list = np.repeat(0, len(p_tot_c_list))

   
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


def allocate_reserve(alpha_kt_list, I_alpha, p_tot_fc):
    
    r_kt = I_alpha * p_tot_fc * alpha_kt_list
    
    # if r_strat == 'percentage':
    #     # Percentage strategy and optimized alpha strategy   
    #     r_kt = I_alpha * p_tot_fc * alpha_kt_list 
    # elif r_strat == 'derating':
    #     # Derating strategy
    #     r_kt = np.where(p_tot_fc <= (1 - I_alpha * alpha_kt_list) * p_farm_rated, 0, p_tot_fc - (1 - I_alpha * alpha_kt_list) * p_farm_rated)  
    # elif r_strat == 'delta':
    #     # Delta strategy
    #     r_kt = np.where(p_tot_fc <= I_alpha * alpha_kt_list * p_farm_rated, p_tot_fc * I_alpha, I_alpha * alpha_kt_list * p_farm_rated)
    
    return r_kt


# Sub-optimization of alpha AND beta
def opti_profit(x, y, alpha_list, p_tot_c_list, ws_kt, wd_kt, price_da_kt, price_capa_up_kt, price_acti_up_kt, volume_acti_up_pu_kt, price_imb_kt, S, nb_cpu=1, nb_chunks=1):
    
    KT = ws_kt.shape[0]
    L = len(alpha_list)
    
    # Forecast values
    ws_fc_kt_S = forecast_error_rel(ws_kt, mean_error_fc_ws, std_error_fc_ws_perc, S, val_min=0, val_max=30)
    wd_fc_kt_S = forecast_error(wd_kt, mean_error_fc_wd, std_error_fc_wd, S, val_min=0, val_max=360)
    price_da_fc_kt_S = forecast_error_rel(price_da_kt, mean_error_fc_price_da, std_error_fc_price_da_perc, S, sigma_min=5)
    price_capa_up_fc_kt_S = forecast_error_rel(price_capa_up_kt, mean_error_fc_price_capa_up, std_error_fc_price_capa_up_perc, S, val_min=0, sigma_min=5) 
    price_acti_up_fc_kt_S = forecast_error_rel(price_acti_up_kt, mean_error_fc_price_acti_up, std_error_fc_price_acti_up_perc, S, val_min=0, sigma_min=5)
    volume_acti_up_pu_fc_kt_S = forecast_error_rel(volume_acti_up_pu_kt, mean_error_fc_volume_acti_up_pu, std_error_fc_volume_acti_up_pu_perc, S, val_min=0, val_max=1, sigma_min=0.01) 
    price_imb_fc_kt_S = forecast_error_rel(price_imb_kt, mean_error_fc_price_imb, std_error_fc_price_imb_perc, S, val_min=0, sigma_min=5)
    # Ensure imbalance penalties are always higher than day-ahead prices
    price_imb_fc_kt_S = np.where(price_imb_fc_kt_S < price_da_fc_kt_S, price_da_fc_kt_S, price_imb_fc_kt_S)
    
    # Reshape for every alpha value trial    
    price_da_fc = price_da_fc_kt_S.reshape(KT, S, 1)    
    price_capa_up_fc = price_capa_up_fc_kt_S.reshape(KT, S, 1)
    price_acti_up_fc = price_acti_up_fc_kt_S.reshape(KT, S, 1)
    volume_acti_up_pu_fc = volume_acti_up_pu_fc_kt_S.reshape(KT, S, 1)
    volume_acti_up_pu_fc_div = np.where(volume_acti_up_pu_fc==0, 1, volume_acti_up_pu_fc) # to avoid dividing by zero
    price_imb_fc = price_imb_fc_kt_S.reshape(KT, S, 1)
    
    # Participation to reserve only if reserve market is expected to be more profitable than day-ahead market
    # I_alpha_kt = np.where(np.mean(price_da_fc_kt_S, axis=1) < np.mean(price_capa_up_fc_kt_S, axis=1) + np.mean(price_acti_up_fc_kt_S, axis=1) * np.mean(volume_acti_up_pu_fc_kt_S, axis=1), 1, 0).reshape(-1, 1) 
    I_alpha_kt = np.ones((KT, 1)) # No need when optimizing alpha at each timestep --> 1

    # Convert wind data to wind power using pywake
    ws_fc_list = ws_fc_kt_S.flatten() 
    wd_fc_list = wd_fc_kt_S.flatten()
    p_tot_fc_list_pw = wake_model(x, y, wd=wd_fc_list, ws=ws_fc_list, time=True, return_simulationResult=False, n_cpu=nb_cpu, wd_chunks=nb_chunks)[2].sum(axis=0) / 1e6 # MW 
    p_tot_fc_list = forecast_error_rel(p_tot_fc_list_pw, mean_error_fc_wp, std_error_fc_wp_perc, 1, val_min=0) # Modelling error on wind power
    p_tot_fc = p_tot_fc_list.reshape(KT, S, 1)
    # print(np.mean(p_tot_fc))
    
    # Decisions: contracted day-ahead and reserve power  
    r_kt_L = allocate_reserve(alpha_list, I_alpha_kt, p_tot_c_list)   
    r = r_kt_L.reshape(KT, 1, L)
    
    # Do not participate to day-ahead market when negative prices
    # I_da_fc = np.where(price_da_fc <= 0, 0, 1)
    # Deduce day-ahead power from p_tot and allocated reserve
    p_da = p_tot_c_list.reshape(1, 1, L) - r #* I_da_fc

    # Identify imbalance situations    
    # If available power is lower than bidded capacity
    r_avail = np.minimum(r, p_tot_fc)
    delta_r_c = r - r_avail
    
    # If promised/contracted power is greater than actual power 
    # Prioritize reserve 
    r_requested = r * volume_acti_up_pu_fc
    r_supplied = np.minimum(r_requested, p_tot_fc)
    delta_r = r_requested - r_supplied
    p_da_supplied = np.minimum(p_tot_fc, r + p_da) -  r_avail # or r_supplied ?  
    delta_p_da = p_da - p_da_supplied
    p_tot_supplied = p_da_supplied + r_supplied
    
    # Compute profits and penalties 
    # Price is given in €/MWh, but timestep could be lower than an hour
    profit_da = (p_da * price_da_fc) / nb_samples_hourly
    profit_reserve = (r * price_capa_up_fc + r * price_acti_up_fc * volume_acti_up_pu_fc) / nb_samples_hourly
    pen_da = (delta_p_da * price_imb_fc) / nb_samples_hourly
    pen_r_c = (eps_pen_r_c * delta_r_c * price_capa_up_fc) / nb_samples_hourly
    pen_r_s = (1.3 * delta_r / volume_acti_up_pu_fc_div * (price_capa_up_fc + price_acti_up_fc * volume_acti_up_pu_fc)) / nb_samples_hourly
      
    profit = profit_da + profit_reserve - (pen_da + pen_r_c + pen_r_s)
    
    expected_profit_all_alpha = np.mean(profit, axis=1)
    expected_profit_opt = np.max(expected_profit_all_alpha, axis=1)
    profit_fc_opt = profit[np.arange(0, KT, 1), :, np.argmax(expected_profit_all_alpha, axis=1)]
    # Cost breakdown
    profit_da_fc_opt = profit_da[np.arange(0, KT, 1), :, np.argmax(expected_profit_all_alpha, axis=1)]
    pen_da_fc_opt = pen_da[np.arange(0, KT, 1), :, np.argmax(expected_profit_all_alpha, axis=1)]
    profit_r_fc_opt = profit_reserve[np.arange(0, KT, 1), :, np.argmax(expected_profit_all_alpha, axis=1)]
    pen_r_c_fc_opt = pen_r_c[np.arange(0, KT, 1), :, np.argmax(expected_profit_all_alpha, axis=1)]
    pen_r_s_fc_opt = pen_r_s[np.arange(0, KT, 1), :, np.argmax(expected_profit_all_alpha, axis=1)]

    p_da_opt = p_da[np.arange(0, KT, 1), 0, np.argmax(expected_profit_all_alpha, axis=1)]
    r_opt = r[np.arange(0, KT, 1), 0, np.argmax(expected_profit_all_alpha, axis=1)]
    p_tot_c_opt = p_da_opt + r_opt
    alpha_opt = r_opt / p_tot_c_opt
    alpha_opt = np.where(p_tot_c_opt==0, 0, alpha_opt)
    # p_tot_s_opt = p_tot_supplied[np.arange(0, KT, 1), :, np.argmax(expected_profit_all_alpha, axis=1)]
    p_da_s_opt = p_da_supplied[np.arange(0, KT, 1), :, np.argmax(expected_profit_all_alpha, axis=1)]
    p_r_s_opt = r_supplied[np.arange(0, KT, 1), :, np.argmax(expected_profit_all_alpha, axis=1)]
    
    # No reserve
    expected_profit_alpha_zero = expected_profit_all_alpha[:, alpha_list==0]
    expected_profit_no_reserve = np.max(expected_profit_alpha_zero, axis=1)
    profit_fc_alpha_zero = profit[:, :, alpha_list==0]
    profit_fc_no_reserve = profit_fc_alpha_zero[np.arange(0, KT, 1), :, np.argmax(expected_profit_alpha_zero, axis=1)]
    
    p_tot_c_alpha_zero = r[:, 0, alpha_list == 0] + p_da[:, 0, alpha_list == 0]
    p_tot_c_no_reserve = p_tot_c_alpha_zero[np.arange(0, KT, 1), np.argmax(expected_profit_alpha_zero, axis=1)]
    p_tot_s_alpha_zero = p_tot_supplied[:, :, alpha_list == 0]
    p_tot_s_no_reserve = p_tot_s_alpha_zero[np.arange(0, KT, 1), :, np.argmax(expected_profit_alpha_zero, axis=1)]
    
    results_dict = {'alpha_opt': alpha_opt, 
                    'expected_profit_opt': expected_profit_opt, 'expected_profit_no_reserve': expected_profit_no_reserve, 
                    'profit_fc_opt': profit_fc_opt, 'profit_da_fc_opt': profit_da_fc_opt, 'pen_da_fc_opt': pen_da_fc_opt, 
                    'profit_r_fc_opt': profit_r_fc_opt, 'pen_r_c_fc_opt': pen_r_c_fc_opt, 'pen_r_s_fc_opt': pen_r_s_fc_opt,
                    'profit_fc_no_reserve': profit_fc_no_reserve, 
                    'p_tot_c_opt': p_tot_c_opt, 'p_da_s_opt': p_da_s_opt, 'p_r_s_opt': p_r_s_opt, 
                    'p_tot_c_no_reserve': p_tot_c_no_reserve, 'p_tot_s_no_reserve': p_tot_s_no_reserve}
    
    return results_dict

#%% Optimize alpha_kt for timestep k,t

# Interesting timesteps
# If > 0, reserve not profitable
cond = data_scen.price_da - (data_scen.price_capa_up + data_scen.price_acti_up * data_scen.volume_acti_up_pu)
kt_reserve = data_scen.index[cond < 0]
ws_kt_reserve = data_scen.ws[cond < 0]

# Sample timesteps kt
# kt = kt_reserve[0:2]
# len_kt = 1 if isinstance(kt, np.int64) else len(kt)

kt = 22534
len_kt = 1

arr = np.ones(len_kt)
ws_kt, wd_kt = arr*np.array(data_scen.loc[kt].ws), arr*np.array(data_scen.loc[kt].wd)
price_da_kt, price_imb_kt = arr*np.array(data_scen.loc[kt].price_da), arr*np.array(data_scen.loc[kt].price_mip)
price_capa_up_kt, price_acti_up_kt = arr*np.array(data_scen.loc[kt].price_capa_up), arr*np.array(data_scen.loc[kt].price_acti_up)
volume_acti_up_pu_kt = arr*np.array(data_scen.loc[kt].volume_acti_up_pu)

results_opt = opti_profit(x, y, alpha_list, p_tot_c_list, ws_kt, wd_kt, price_da_kt, price_capa_up_kt, price_acti_up_kt, volume_acti_up_pu_kt, price_imb_kt, S, nb_cpu=1, nb_chunks=1)

p_tot_s_opt = results_opt['p_da_s_opt'] + results_opt['p_r_s_opt']
std_profit = np.std(np.sum(results_opt['profit_fc_opt'], axis=0))

alpha_opt = results_opt['alpha_opt']
mean_profit_opt = results_opt['expected_profit_opt']
p_tot_c_opt = results_opt['p_tot_c_opt']
mean_profit_no_reserve = results_opt['expected_profit_no_reserve']
p_tot_c_no_reserve = results_opt['p_tot_c_no_reserve']
p_tot_s_no_reserve = results_opt['p_tot_s_no_reserve']

print("\nOpti 2")
print("Total profit max:", round(np.sum(mean_profit_opt), 3), "+/-", round(std_profit/np.sqrt(S), 3), "€, for alpha_kt =", alpha_opt, ", R =", alpha_opt*p_tot_c_opt, "MW") 
print("P_tot_c_opt =", round(np.sum(p_tot_c_opt), 2), "MWh, Supplied =", round(np.sum(np.mean(p_tot_s_opt, axis=1)), 2), "MWh")

std_profit_no_reserve = np.std(np.sum(results_opt['profit_fc_no_reserve'], axis=0))

print("\nNo reserve (alpha = 0)")
print("Total profit:", round(np.sum(mean_profit_no_reserve), 3), "+/-", round(std_profit_no_reserve/np.sqrt(S), 3), "€")
print("P_tot_c =", round(np.sum(p_tot_c_no_reserve), 2), "MWh, Supplied =", round(np.sum(np.mean(p_tot_s_no_reserve, axis=1)), 2), "MWh", "\n")
    
    
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
    alpha_opt_array = np.zeros(KT_tot)
    mean_profit_opt_array = np.zeros(KT_tot)
    mean_profit_no_reserve_array = np.zeros(KT_tot)
    profit_fc_opt_array = np.zeros((KT_tot, S))
    profit_da_fc_opt_array = np.zeros((KT_tot, S))
    pen_da_fc_opt_array = np.zeros((KT_tot, S))
    profit_r_fc_opt_array = np.zeros((KT_tot, S))
    pen_r_c_fc_opt_array = np.zeros((KT_tot, S))
    pen_r_s_fc_opt_array = np.zeros((KT_tot, S))
    profit_fc_no_reserve_array = np.zeros((KT_tot, S))
    p_tot_c_opt_array = np.zeros(KT_tot)
    p_da_s_fc_opt_array = np.zeros((KT_tot, S))
    p_r_s_fc_opt_array = np.zeros((KT_tot, S))
    p_tot_c_no_reserve_array = np.zeros(KT_tot)
    p_tot_s_fc_no_reserve_array = np.zeros((KT_tot, S))
    
    start = time.time()
    
    for i_chunk in range(len(kt_reserve_chunk_index)-1):
       
        kt_chunk = data_scen.index[kt_reserve_chunk_index[i_chunk]:kt_reserve_chunk_index[i_chunk+1]]
       
        ws_chunk, wd_chunk = ws[kt_chunk], wd[kt_chunk]
        price_da_chunk, price_imb_chunk = price_da[kt_chunk], price_imb[kt_chunk]
        price_capa_up_chunk, price_acti_up_chunk = price_capa_up[kt_chunk], price_acti_up[kt_chunk]
        volume_acti_up_pu_chunk = volume_acti_up_pu[kt_chunk]
        
        # nb_simu = len(ws_chunk) * S
        # nb_chunks = np.where(nb_simu > nb_simu_per_chunk, int(nb_simu/nb_simu_per_chunk), 4).flatten()[0]
              
        results_temp = opti_profit(x, y, alpha_list, p_tot_c_list, ws_chunk, wd_chunk, price_da_chunk, price_capa_up_chunk, price_acti_up_chunk, volume_acti_up_pu_chunk, price_imb_chunk, S, nb_cpu=nb_cpu, nb_chunks=nb_chunks)
                    
        mean_profit_opt_array[kt_reserve_chunk_index[i_chunk]:kt_reserve_chunk_index[i_chunk+1]] = results_temp['expected_profit_opt']
        alpha_opt_array[kt_reserve_chunk_index[i_chunk]:kt_reserve_chunk_index[i_chunk+1]] = results_temp['alpha_opt']
        mean_profit_no_reserve_array[kt_reserve_chunk_index[i_chunk]:kt_reserve_chunk_index[i_chunk+1]] = results_temp['expected_profit_no_reserve']
        profit_fc_opt_array[kt_reserve_chunk_index[i_chunk]:kt_reserve_chunk_index[i_chunk+1], :] = results_temp['profit_fc_opt']
        profit_da_fc_opt_array[kt_reserve_chunk_index[i_chunk]:kt_reserve_chunk_index[i_chunk+1], :] = results_temp['profit_da_fc_opt']
        pen_da_fc_opt_array[kt_reserve_chunk_index[i_chunk]:kt_reserve_chunk_index[i_chunk+1], :] = results_temp['pen_da_fc_opt']
        profit_r_fc_opt_array[kt_reserve_chunk_index[i_chunk]:kt_reserve_chunk_index[i_chunk+1], :] = results_temp['profit_r_fc_opt']
        pen_r_c_fc_opt_array[kt_reserve_chunk_index[i_chunk]:kt_reserve_chunk_index[i_chunk+1], :] = results_temp['pen_r_c_fc_opt']
        pen_r_s_fc_opt_array[kt_reserve_chunk_index[i_chunk]:kt_reserve_chunk_index[i_chunk+1], :] = results_temp['pen_r_s_fc_opt']
        profit_fc_no_reserve_array[kt_reserve_chunk_index[i_chunk]:kt_reserve_chunk_index[i_chunk+1], :] = results_temp['profit_fc_no_reserve']
        p_tot_c_opt_array[kt_reserve_chunk_index[i_chunk]:kt_reserve_chunk_index[i_chunk+1]] = results_temp['p_tot_c_opt']
        p_da_s_fc_opt_array[kt_reserve_chunk_index[i_chunk]:kt_reserve_chunk_index[i_chunk+1], :] = results_temp['p_da_s_opt']
        p_r_s_fc_opt_array[kt_reserve_chunk_index[i_chunk]:kt_reserve_chunk_index[i_chunk+1], :] = results_temp['p_r_s_opt']
        p_tot_c_no_reserve_array[kt_reserve_chunk_index[i_chunk]:kt_reserve_chunk_index[i_chunk+1]] = results_temp['p_tot_c_no_reserve']
        p_tot_s_fc_no_reserve_array[kt_reserve_chunk_index[i_chunk]:kt_reserve_chunk_index[i_chunk+1], :] = results_temp['p_tot_s_no_reserve']
        
    end = time.time()
    comp_time = end - start
    print(round(comp_time), "s")
    
    # Mean profit and std with optimized alpha and P_tot
    total_mean_profit_opt = np.sum(np.mean(profit_fc_opt_array,axis=1)) / 1e6 # M€
    std_total_profit_opt = np.std(np.sum(profit_fc_opt_array, axis=0)) / 1e6
    total_profit_fc_opt_array = np.sum(profit_fc_opt_array, axis=0) / 1e6
    print("\nTotal expected profit for", year, ":", round(total_mean_profit_opt, 3), "+/-", round(std_total_profit_opt/np.sqrt(S), 2),"M€")
    # For day-ahead market
    # Profits
    total_mean_profit_da_opt = np.sum(np.mean(profit_da_fc_opt_array,axis=1)) / 1e6 # M€
    std_total_profit_da_opt = np.std(np.sum(profit_da_fc_opt_array, axis=0)) / 1e6
    total_profit_da_fc_opt_array = np.sum(profit_da_fc_opt_array, axis=0) / 1e6
    mean_profit_da_opt_array = np.mean(profit_da_fc_opt_array, axis=1)
    # Penalties
    total_mean_pen_da_opt = np.sum(np.mean(pen_da_fc_opt_array,axis=1)) / 1e6 # M€
    std_total_pen_da_opt = np.std(np.sum(pen_da_fc_opt_array, axis=0)) / 1e6
    total_pen_da_fc_opt_array = np.sum(pen_da_fc_opt_array, axis=0) / 1e6
    mean_pen_da_opt_array = np.mean(pen_da_fc_opt_array, axis=1)
    # For reserve markets
    # Profits
    total_mean_profit_r_opt = np.sum(np.mean(profit_r_fc_opt_array,axis=1)) / 1e6 # M€
    std_total_profit_r_opt = np.std(np.sum(profit_r_fc_opt_array, axis=0)) / 1e6
    total_profit_r_fc_opt_array = np.sum(profit_r_fc_opt_array, axis=0) / 1e6
    mean_profit_r_opt_array = np.mean(profit_r_fc_opt_array, axis=1)
    # Penalties for reserve capacity
    total_mean_pen_r_c_opt = np.sum(np.mean(pen_r_c_fc_opt_array,axis=1)) / 1e6 # M€
    std_total_pen_r_c_opt = np.std(np.sum(pen_r_c_fc_opt_array, axis=0)) / 1e6
    total_pen_r_c_fc_opt_array = np.sum(pen_r_c_fc_opt_array, axis=0) / 1e6
    mean_pen_r_c_opt_array = np.mean(pen_r_c_fc_opt_array, axis=1)
    # Penalties for reserve supplied
    total_mean_pen_r_s_opt = np.sum(np.mean(pen_r_s_fc_opt_array,axis=1)) / 1e6 # M€
    std_total_pen_r_s_opt = np.std(np.sum(pen_r_s_fc_opt_array, axis=0)) / 1e6
    total_pen_r_s_fc_opt_array = np.sum(pen_r_s_fc_opt_array, axis=0) / 1e6
    mean_pen_r_s_opt_array = np.mean(pen_r_s_fc_opt_array, axis=1)
    
    
    # AEP (contracted and supplied) with optimized alpha
    aep_c_opt = np.sum(p_tot_c_opt_array) / 1e3 / nb_samples_hourly # GWh
    p_tot_s_fc_opt_array = p_da_s_fc_opt_array + p_r_s_fc_opt_array
    aep_mean_s_opt = np.mean(np.sum(p_tot_s_fc_opt_array, axis=0)) / 1e3 / nb_samples_hourly # GWh
    aep_std_s_opt = np.std(np.sum(p_tot_s_fc_opt_array, axis=0)) / 1e3 / nb_samples_hourly # GWh
    p_tot_s_opt_array = np.mean(p_tot_s_fc_opt_array, axis=1)  # MWh
    print("Total power produced for", year, ":", round(aep_mean_s_opt, 3), "TWh (contracted", round(aep_c_opt, 3),"TWh)")   
    # Day-ahead market
    aep_da_mean_s_opt = np.mean(np.sum(p_da_s_fc_opt_array, axis=0)) / 1e3 / nb_samples_hourly # GWh
    aep_da_std_s_opt = np.std(np.sum(p_da_s_fc_opt_array, axis=0)) / 1e3 / nb_samples_hourly # GWh
    p_da_s_opt_array = np.mean(p_da_s_fc_opt_array, axis=1)  # MWh
    # Reserve market
    aep_r_mean_s_opt = np.mean(np.sum(p_r_s_fc_opt_array, axis=0)) / 1e3 / nb_samples_hourly # GWh
    aep_r_std_s_opt = np.std(np.sum(p_r_s_fc_opt_array, axis=0)) / 1e3 / nb_samples_hourly # GWh
    p_r_s_opt_array = np.mean(p_r_s_fc_opt_array, axis=1)  # MWh
    
    # Optimized alpha for each timestep
    plt.hist(alpha_opt_array)
    plt.show()
    
    print("\nNo reserve")
    # Mean profit and std with no reserve (alpha = 0, p_tot_c optimized)
    total_mean_profit_no_reserve = np.sum(mean_profit_no_reserve_array) / 1e6 # M€
    std_total_profit_no_reserve = np.std(np.sum(profit_fc_no_reserve_array, axis=0)) / 1e6
    total_profit_fc_no_reserve_array = np.sum(profit_fc_no_reserve_array, axis=0) / 1e6
    print("\nTotal expected profit for", year, ":", round(total_mean_profit_no_reserve, 3), "+/-", round(std_total_profit_no_reserve/np.sqrt(S), 2),"M€")
    # AEP (contracted and supplied) with no reserve
    aep_c_no_reserve = np.sum(p_tot_c_no_reserve_array) / 1e3 / nb_samples_hourly # GWh 
    aep_mean_s_no_reserve = np.mean(np.sum(p_tot_s_fc_no_reserve_array, axis=0)) / 1e3 / nb_samples_hourly # GWh
    aep_std_s_no_reserve = np.std(np.sum(p_tot_s_fc_no_reserve_array, axis=0)) / 1e3 / nb_samples_hourly # GWh
    p_tot_s_no_reserve_array = np.mean(p_tot_s_fc_no_reserve_array, axis=1) # MW
    print("Total power produced for", year, ":", round(aep_mean_s_no_reserve, 3), "TWh (contracted", round(aep_c_no_reserve, 3),"TWh)\n")   

    winsound.Beep(500, 1000)
    
#%% Save results

results_summary = {'mean_total_profit_opt': total_mean_profit_opt, 'std_total_profit_opt': std_total_profit_opt,
                   'mean_total_profit_da_opt': total_mean_profit_da_opt, 'std_total_profit_da_opt': std_total_profit_da_opt,
                   'mean_total_pen_da_opt': total_mean_pen_da_opt, 'std_total_pen_da_opt': std_total_pen_da_opt,
                   'mean_total_profit_r_opt': total_mean_profit_r_opt, 'std_total_profit_r_opt': std_total_profit_r_opt,
                   'mean_total_pen_r_c_opt': total_mean_pen_r_c_opt, 'std_total_pen_r_c_opt': std_total_pen_r_c_opt,
                   'mean_total_pen_r_s_opt': total_mean_pen_r_s_opt, 'std_total_pen_r_s_opt': std_total_pen_r_s_opt,
                   'mean_total_profit_noR': total_mean_profit_no_reserve, 'std_total_profit_noR': std_total_profit_no_reserve,
                   'aep_c_opt': aep_c_opt, 'aep_c_no_reserve': aep_c_no_reserve,
                   'aep_mean_s_opt': aep_mean_s_opt, 'aep_std_s_opt': aep_std_s_opt, 
                   'aep_da_mean_s_opt': aep_da_mean_s_opt, 'aep_da_std_s_opt': aep_da_std_s_opt, 
                   'aep_r_mean_s_opt': aep_r_mean_s_opt, 'aep_r_std_s_opt': aep_r_std_s_opt, 
                   'aep_mean_s_noR': aep_mean_s_no_reserve, 'aep_std_s_noR': aep_std_s_no_reserve,
                   'comp_time': comp_time}

total_mean_profit_fc = {'total_profit_fc_opt': total_profit_fc_opt_array, 
                        'total_profit_da_fc_opt': total_profit_da_fc_opt_array, 'total_pen_da_fc_opt': total_pen_da_fc_opt_array, 
                        'total_profit_r_fc_opt': total_profit_r_fc_opt_array, 'total_pen_r_c_fc_opt': total_pen_r_c_fc_opt_array, 'total_pen_r_s_fc_opt': total_pen_r_s_fc_opt_array, 
                        'total_profit_fc_noR': total_profit_fc_no_reserve_array}

results_qh = {'alpha_opt': alpha_opt_array, 'p_tot_c_opt': p_tot_c_opt_array, 'p_tot_c_noR': p_tot_c_no_reserve_array,
              'p_tot_s_opt': p_tot_s_opt_array, 'p_da_s_opt': p_da_s_opt_array, 'p_r_s_opt': p_r_s_opt_array, 'p_tot_s_noR': p_tot_s_no_reserve_array, 
              'mean_profit_opt': mean_profit_opt_array, 
              'mean_profit_da_opt': mean_profit_da_opt_array, 'mean_pen_da_opt': mean_pen_da_opt_array, 
              'mean_profit_r_opt': mean_profit_r_opt_array, 'mean_pen_r_c_opt': mean_pen_r_c_opt_array, 'mean_pen_r_s_opt': mean_pen_r_s_opt_array, 
              'mean_profit_noR': mean_profit_no_reserve_array}

writer = pd.ExcelWriter('Output/post_process_' + wf_name + '_' + wf_layout_name + "_" + year + '_' + str(r_max) + '.xlsx', engine='xlsxwriter')
pd.DataFrame.from_dict(results_summary, "index").to_excel(writer, sheet_name = 'Summary', index=True)
pd.DataFrame.from_dict(total_mean_profit_fc).to_excel(writer, sheet_name = 'Total_profit_fc', index=False)
pd.DataFrame.from_dict(results_qh).to_excel(writer, sheet_name = 'Results_qh', index=False)
writer.close()


#%% Test

# kt_chunk = np.arange(0, 50, 1)
# nb_cpu = 1
# nb_chunk = 4

# ws_fc_kt_S, wd_fc_kt_S = ws_fc_KT_S[kt_chunk, :], wd_fc_KT_S[kt_chunk, :]
# price_da_fc_kt_S, price_imb_fc_kt_S = price_da_fc_KT_S[kt_chunk, :], price_imb_fc_KT_S[kt_chunk, :]
# price_capa_up_fc_kt_S, price_acti_up_fc_kt_S = price_capa_up_fc_KT_S[kt_chunk, :], price_acti_up_fc_KT_S[kt_chunk, :]
# volume_acti_up_pu_fc_kt_S = volume_acti_up_pu_fc_KT_S[kt_chunk, :]

