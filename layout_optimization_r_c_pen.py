# -*- coding: utf-8 -*-
"""
Created on Wed, 03 Apr 2024, 14:45:53 
Last modified on Wed, 31 July 2024

@author: Thuy-Hai

Reserve penalties for missing capacity (availability tests)
Reserve penalties for missing requested reserve (kappa*R)

!!! Set n_cpu = 1 
Parallelization of gradients of WS, TI, Power and custom functions is not implemented yet
"""

#%% Imports

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from datetime import datetime
import os
import winsound
import pickle
# import random
from shapely.geometry import Point, Polygon

import importlib # import TopFarm
from topfarm import TopFarmProblem
# from py_wake.examples.data.hornsrev1 import HornsrevV80
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.site import UniformSite

from topfarm.cost_models.cost_model_wrappers import CostModelComponent
# Constraints
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.constraint_components.constraint_aggregation import DistanceConstraintAggregation
# Drivers
from topfarm.easy_drivers import EasySGDDriver #, EasyScipyOptimizeDriver
# Plotting
from topfarm.plotting import XYPlotComp #, NoPlot 
# Wake models
from py_wake.deficit_models.gaussian import NiayifarGaussian
from py_wake.turbulence_models import CrespoHernandez
from py_wake.superposition_models import LinearSum
from py_wake.site.shear import PowerShear
# Gradient
from py_wake.utils.gradients import autograd as autograd_pw
import autograd.numpy as anp

#%% User-defined parameters

# Max reserve bids
r_max = 50 # MW (50, 117, 166, 221.4)
reserve = True # Flag for participation to reserve markets
eps_pen_r_c = 10 # Penalty multiplier for failed availablity test

# Wind farm data
wf_name = 'Northwind'
init_layout = 'random' # 'base' or 'random'
min_spacing = 2 # Minimum number of rotor diameters between two turbines
file_wt_pc = wf_name 
path_data = 'Data\\'

bootstrap = False # True: replace timesteps in data after sampling, False: do not replace, '<name>': already sampled timesteps

# SGD parameters
K = 20 # Day sampling
T = 1 # Timestep sampling
S = 10 # Forecasts
sgd_iterations = 2000

# Combinatorial exploration parameters (sub-optimization problem)
p_step = 1 # ~2MW

# Scenarios of wind and electricity prices
scenarios_file = "data_scenarios"
scenarios_years = ['2023'] # ['2021', '2022', '2023'] 
nb_days = 365 # 365 + 365 + 365
nb_samples_hourly = 4 # Each sample is a quarter hour
wind_height = 100
columns_scen = ["dateTime_elia", "dateTime", "dateTime_utc", "dateHour_utc", 
                "ws", "wd", "price_da", 
                "price_capa_up", "price_acti_up", "volume_acti_up", "volume_acti_up_pu",
                "price_capa_down", "price_acti_down", "volume_acti_down", "volume_acti_up_down",
                "price_mip", "price_neg_imb", "alpha_imb"]

# Forecast errors distribution
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

output_name = wf_name + '_K' + str(K) + '_T' + str(T) + '_S' + str(S) + '_R' + str(r_max) + '_' + init_layout + '_' + str(int(time.time()))

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

data_scen = pd.DataFrame(columns=columns_scen)
for year in scenarios_years:
    data_temp = pd.read_excel(path_data + scenarios_file +'.xlsx', sheet_name=year, names=columns_scen)
    data_scen = pd.concat([data_scen, data_temp])
    
data_scen.reset_index(inplace=True, drop=True)
    
days_list = list(np.arange(0, nb_days, 1)) # Day sampling k
timestep_list = list(np.arange(0, 24 * nb_samples_hourly, 1)) # Timestep sampling t

# Add day column (for day sampling)
days = np.repeat(days_list, 24 * nb_samples_hourly)
timesteps = np.tile(timestep_list, nb_days)
day_timestep_list = list(zip(list(days), list(timesteps)))
data_scen['day_timestep'] = day_timestep_list

# Wind turbine data
data_turbine = pd.read_excel(path_data + file_wt_pc + '.xlsx', sheet_name=wt_name, names=['ws', 'power', 'cp', 'ct'], usecols='A:D')
pc_ws = np.array(data_turbine.ws)
pc_power = np.array(data_turbine.power)
pc_ct = np.array(data_turbine.ct)
# Wind farm data
layout_wf = pd.read_excel(path_data + file_wt_pc + '.xlsx', sheet_name='WT_coord', 
                              names=['lon', 'lat', 'name', '', 'x', 'y', 'edge_flag', 'edge_name', 'edge_x', 'edge_y'], usecols='A:J')
x_wf = layout_wf.x
y_wf = layout_wf.y

#%% Setup wind farm

# Wind turbine
windTurbines = WindTurbine(name=wt_name,
                    diameter=rotor_diam,
                    hub_height=hub_height,
                    powerCtFunction=PowerCtTabular(pc_ws, pc_power,'MW', pc_ct, power_idle=0, ct_idle=0, method='linear'))

# Defining the site, wind turbines and wake model
site = UniformSite(ti=0.077, shear=PowerShear(wind_height, 0.12), interp_method='linear' )
site.interp_method = 'linear' 
wake_model = NiayifarGaussian(site, windTurbines, turbulenceModel=CrespoHernandez(), superpositionModel=LinearSum())

p_farm_rated = nb_wt * windTurbines.power(20) / 1e6 # MW

# Initialize layout

# Farm boundary limits
if wf_name == 'Rentel':
    x1, y1 = 3302.87, 0
    x2, y2 = 7637.48, 3042.75
    x3, y3 = 4472.08, 6010.89
    x4, y4 = 0, 2980.88
elif wf_name == 'Northwind':
    x1, y1 = 5854.90, 3206.12
    x2, y2 = 3684.57, 5306.73
    x3, y3 = 0, 498.71
    x4, y4 = 645.86, 0
x_boundary = np.array([x1, x2, x3, x4])
y_boundary = np.array([y1, y2, y3, y4])
# If considering boundaries as rectangle
boundary_points_rectangle = np.array([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])
# Convex hull, encompass all wind turbines in convex shape
boundary_convex_hull = np.zeros((nb_wt, 2))
boundary_convex_hull[:, 0] = x_wf
boundary_convex_hull[:, 1] = y_wf
# Polygon, take edge turbines as boundaries
nb_edges = np.sum(layout_wf.edge_flag==True)
boundary_polygon = np.zeros((nb_edges, 2))
boundary_polygon[:, 0] = layout_wf.edge_x.loc[0:nb_edges-1]           
boundary_polygon[:, 1] = layout_wf.edge_y.loc[0:nb_edges-1] 


if init_layout == 'random':
    # Initial random layout
    x_min, x_max = np.min(x_boundary), np.max(x_boundary)
    y_min, y_max = np.min(y_boundary), np.max(y_boundary)
    polygon = Polygon(boundary_polygon)
    
    x_init = np.ones(nb_wt) * x_min
    y_init = np.ones(nb_wt) * y_min
    wt = 0
    while wt < nb_wt:
        x_temp = np.random.uniform(x_min, x_max, 1)
        y_temp = np.random.uniform(y_min, y_max, 1)
        # Check distance between turbines
        mask_dist = np.sqrt((x_init - x_temp)**2 + (y_init - y_temp)**2) < 1.5 * min_spacing * rotor_diam
        # Check if the point is inside the polygon
        point = Point(x_temp[0], y_temp[0])   
        if np.sum(mask_dist) == 0 and polygon.contains(point):     
            x_init[wt] = x_temp[0]
            y_init[wt] = y_temp[0]        
            wt += 1
            
elif init_layout == 'base':
    # Initial base layout
    x_init = np.array(x_wf)
    y_init = np.array(y_wf)
    
else:
    xy_init = pd.read_excel(path_data + wf_name + '_random.xlsx', sheet_name=init_layout, names=['name', 'x', 'y'], usecols='A:C')
    x_init = xy_init.x
    y_init = xy_init.y

# Plot initial layout
plt.figure(dpi=1200)
plt.scatter(x_init, y_init, c='blue', marker='1')
plt.scatter(x_boundary, y_boundary, c='red', marker='.') # boundaries
plt.xlim(-500, 8000)
plt.ylim(-500, 8000)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.title('Initial layout')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.savefig('Plots/'+ output_name +'_init.pdf', dpi=600) 
plt.show()

# # Plot actual layout
# plt.figure(dpi=1200)
# plt.scatter(x_wf, y_wf, c='blue', marker='1')
# plt.scatter(x_boundary, y_boundary, c='red', marker='.') # boundaries
# plt.xlim(-500, 8000)
# plt.ylim(-500, 8000)
# ax = plt.gca()
# ax.set_aspect('equal', adjustable='box')
# plt.title(wf_name + ' layout')
# plt.xlabel('x [m]')
# plt.ylabel('y [m]')
# plt.show()

design_vars = {'x': x_init, 'y': y_init}

#%% Setup problem constraints

# The constraints for the SGD driver are defined with the DistanceConstraintAggregation class.
# Note: as the class is specified, the order of the SpacingConstraint and XYBoundaryConstraint must be kept.

# Spacing constraint set up for minimum inter-turbine spacing
min_spacing_m = min_spacing * rotor_diam  # Minimum inter-turbine spacing in meters
# Constraint set up for the boundary type provided: 'polygon', 'convex_hull', 'rectangle'
# constraint_boundary = XYBoundaryConstraint(boundary_points_rectangle, 'rectangle')
# constraint_boundary = XYBoundaryConstraint(boundary_convex_hull, 'convex_hull') 
constraint_boundary = XYBoundaryConstraint(boundary_polygon, 'polygon') 

constraints_sgd = DistanceConstraintAggregation([SpacingConstraint(min_spacing_m), constraint_boundary], nb_wt, min_spacing_m, windTurbines)

# Plot boundary
dummy_cost = CostModelComponent(input_keys=[], n_wt=2, cost_function=lambda : 1)

def plot_boundary(constraint_boundary):
    tf = TopFarmProblem(design_vars=design_vars, # setting up the turbine positions as design variables
        cost_comp=dummy_cost, # using dummy cost model
        constraints=[constraint_boundary], # constraint set up for the boundary type provided
        plot_comp=XYPlotComp()) # support plotting function

    tf.plot_comp.plot_constraints() # plot constraints is a helper function in topfarm to plot constraints
    plt.plot(boundary_points_rectangle[:,0], boundary_points_rectangle[:,1],'.r', label='Boundary points') # plot the boundary points
    plt.xlim(-500, 8000)
    plt.ylim(-500, 8000)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.legend() # add the legend
    
plot_boundary(constraint_boundary)

#%% Setup sampling and combinatorial exploration

if reserve == True:
    # Total contracted power (list of possible values)
    p_tot_c_kt_list = np.arange(0, p_farm_rated, p_step)
    p_tot_c_kt_list = np.append(p_tot_c_kt_list, p_farm_rated)

    # Reserve (list of possible values)
    p_tot_c_list = np.array([])
    alpha_list = np.array([])

    # Make combinations, and limit reserve to R_max
    for i_p in range(len(p_tot_c_kt_list)):
        alpha_step = 1/p_tot_c_kt_list[i_p]
        if p_tot_c_kt_list[i_p] <= r_max:
            alpha_temp = np.arange(0, 1, alpha_step)
            alpha_temp = np.append(alpha_temp, 1)
        else: # If p_tot > R_max, limit R to R_max
            alpha_max_lim = r_max / p_tot_c_kt_list[i_p]
            alpha_temp = np.arange(0, alpha_max_lim, alpha_step)
            alpha_temp = np.append(alpha_temp, r_max / p_tot_c_kt_list[i_p])
        alpha_list = np.append(alpha_list, alpha_temp)
        p_tot_c_list = np.append(p_tot_c_list, np.repeat(p_tot_c_kt_list[i_p], len(alpha_temp)))
    
else: # Only participation to day-ahead market
    p_tot_c_kt_list = np.arange(0, p_farm_rated, p_step)
    p_tot_c_kt_list = np.append(p_tot_c_kt_list, p_farm_rated)
    p_tot_c_list = p_tot_c_kt_list
    alpha_list = np.repeat(0, len(p_tot_c_list))

# Random sampling process
if bootstrap == True: # Replace timesteps after sampling
    sampled_timesteps = []
    for i in range(sgd_iterations+1):
        sampled_k_list = np.repeat(np.random.choice(days_list, size=K, replace=False), T)
        sampled_t_list = np.random.choice(timestep_list, size=T*K, replace=True)
        sampled_kt_list = list(zip(list(sampled_k_list), list(sampled_t_list)))
        sampled_timesteps.append(sampled_kt_list)
elif bootstrap == False: # Remove timestep from list after sampling
    day_timestep_list_index = np.arange(0, len(day_timestep_list), 1)
    sampled_timesteps = []
    for i in range(sgd_iterations+1):
        if len(day_timestep_list_index) < T*K:
            day_timestep_list_index = np.arange(0, len(day_timestep_list), 1)
        sampled_kt_list_index = np.random.choice(day_timestep_list_index, size=T*K, replace=False)
        sampled_kt_list = [day_timestep_list[i] for i in sampled_kt_list_index]
        sampled_timesteps.append(sampled_kt_list)
        # Remove sampled timesteps from data     
        day_timestep_list_index = day_timestep_list_index[~np.isin(day_timestep_list_index, sampled_kt_list_index)]
else:
    with open('Data/' + bootstrap, "rb") as fp:   
        sampled_timesteps = pickle.load(fp) 

#%% Cost component model

# Function to compute revenues of wind farm
def scenario_sampling(it):
    
    sampled_kt_list = sampled_timesteps[it]
    data_scen_KT = data_scen[data_scen.day_timestep.isin(sampled_kt_list)]
    
    ws, wd = np.array(data_scen_KT.ws), np.array(data_scen_KT.wd) 
    price_da = np.array(data_scen_KT.price_da) 
    price_capa_up, price_acti_up, volume_acti_up_pu = np.array(data_scen_KT.price_capa_up), np.array(data_scen_KT.price_acti_up), np.array(data_scen_KT.volume_acti_up_pu)
    price_imb = np.array(data_scen_KT.price_mip)
    
    return ws, wd, price_da, price_capa_up, price_acti_up, volume_acti_up_pu, price_imb


# Standard devation is given in % 
def forecast_error_rel(X, mean_error, std_error_perc, S, val_min=0, val_max=1000000, sigma_min=0.01):
    x_temp = X.reshape(-1, 1) * np.ones((1, S))
    std = std_error_perc * np.abs(x_temp)
    std_error = np.where(std==0, sigma_min, std)
    error = np.random.normal(mean_error, std_error, size=(len(X), S))
    x_fc = x_temp + error
    
    x_fc_min = np.where(x_fc < val_min, val_min, x_fc)
    x_fc_min_max = np.where(x_fc_min > val_max, val_max, x_fc_min)
    
    return np.array(x_fc_min_max)

# Standard deviation is absolute value
def forecast_error(X, mean_error, std_error, S, val_min=0, val_max=1000000):
    x_temp = X.reshape(-1, 1) * np.ones((1, S))
    x_fc = x_temp + np.random.normal(mean_error, std_error, size=(len(X), S))
    
    x_fc_min = np.where(x_fc < val_min, val_min, x_fc)
    x_fc_min_max = np.where(x_fc_min > val_max, val_max, x_fc_min)
    
    return x_fc_min_max
    
    
def func_compute_profit(x, y, K, T, alpha_list, p_tot_c_list, S, nb_chunks=1):
    
    global it # counter for optimiztion iteration (used for sampling)
    
    if isinstance(K, int) == False: # topfarm evalulation convert int values to nd_arrays
        K = int(K[0])
        T = int(T[0])
        S = int(S[0])
        nb_chunks = int(nb_chunks[0])
    
    ws, wd, price_da, price_capa_up, price_acti_up, volume_acti_up_pu, price_imb = scenario_sampling(it)
    it = it + 1
    
    KT = len(ws)
    S = S
    L = len(alpha_list)
    
    # Forecast values
    ws_fc_KT_S = forecast_error_rel(ws, mean_error_fc_ws, std_error_fc_ws_perc, S, val_min=0, val_max=30)
    wd_fc_KT_S = forecast_error(wd, mean_error_fc_wd, std_error_fc_wd, S, val_min=0, val_max=360)
    price_da_fc_KT_S = forecast_error_rel(price_da, mean_error_fc_price_da, std_error_fc_price_da_perc, S, sigma_min=5)
    price_capa_up_fc_KT_S = forecast_error_rel(price_capa_up, mean_error_fc_price_capa_up, std_error_fc_price_capa_up_perc, S, val_min=0, sigma_min=5) 
    price_acti_up_fc_KT_S = forecast_error_rel(price_acti_up, mean_error_fc_price_acti_up, std_error_fc_price_acti_up_perc, S, val_min=0, sigma_min=5)
    volume_acti_up_pu_fc_KT_S = forecast_error_rel(volume_acti_up_pu, mean_error_fc_volume_acti_up_pu, std_error_fc_volume_acti_up_pu_perc, S, val_min=0, val_max=1, sigma_min=0.01) 
    price_imb_fc_KT_S = forecast_error_rel(price_imb, mean_error_fc_price_imb, std_error_fc_price_imb_perc, S, val_min=0, sigma_min=5)
    
    # Reshape for every alpha value trial    
    price_da_fc = price_da_fc_KT_S.reshape(KT, S, 1)    
    price_capa_up_fc = price_capa_up_fc_KT_S.reshape(KT, S, 1)
    price_acti_up_fc = price_acti_up_fc_KT_S.reshape(KT, S, 1)
    volume_acti_up_pu_fc = volume_acti_up_pu_fc_KT_S.reshape(KT, S, 1)
    volume_acti_up_pu_fc_div = anp.where(volume_acti_up_pu_fc==0, 1, volume_acti_up_pu_fc) # to avoid dividing by zero
    price_imb_fc = price_imb_fc_KT_S.reshape(KT, S, 1)
    
    # Participation to reserve only if reserve market is expected to be more profitable than day-ahead market
    # I_alpha_KT = anp.where(anp.mean(price_da_fc_KT_S, axis=1) < anp.mean(price_capa_up_fc_KT_S, axis=1) + anp.mean(price_acti_up_fc_KT_S, axis=1) * anp.mean(volume_acti_up_pu_fc_KT_S, axis=1), 1, 0).reshape(-1, 1) 
    I_alpha_KT = np.ones((KT, 1)) # No need when optimizing alpha at each timestep --> 1

    # Convert wind data to wind power using pywake
    ws_fc_list = ws_fc_KT_S.flatten() 
    wd_fc_list = wd_fc_KT_S.flatten()
    # Parallelization of gradients of WS, TI, Power and custom functions is not implemented yet 
    p_tot_fc_list_pw = wake_model(x, y, wd=wd_fc_list, ws=ws_fc_list, time=True, return_simulationResult=False, wd_chunks=nb_chunks)[2].sum(axis=0) / 1e6 # MW 
    # p_tot_fc_list = forecast_error_rel(p_tot_fc_list_pw, mean_error_fc_wp, std_error_fc_wp_perc, 1, val_min=0) # Modelling error on wind power (does not work with anp)
    p_tot_fc_KT_S = p_tot_fc_list_pw.reshape(len(ws), S)
    p_tot_fc = anp.repeat(p_tot_fc_KT_S.reshape(KT, S, 1), L, axis=2) # Repeat for all values of alpha
    # p_tot_fc = p_tot_fc_KT_S.reshape(KT, S, 1)
    
    # Decisions: contracted day-ahead and reserve power  
    # r_KT_L = allocate_reserve(alpha_list, I_alpha_KT, p_tot_c_list)  
    r_KT_L = I_alpha_KT * p_tot_c_list * alpha_list
    r = anp.repeat(r_KT_L.reshape(KT, 1, L), S, axis=1) # Repeat for all values of forecasts
    # r = r_KT_L.reshape(KT, 1, L)
    
    # Do not participate to day-ahead market when negative prices
    # I_da_fc = anp.where(price_da_fc <= 0, 0, 1)
    # Deduce day-ahead power from p_tot and allocated reserve
    p_da = p_tot_c_list.reshape(1, 1, L) - r #* I_da_fc

    # Identify imbalance situations    
    # If available power is lower than bidded capacity
    r_avail = anp.minimum(r, p_tot_fc)
    delta_r_c = r - r_avail
    
    # If promised/contracted power is greater than actual power 
    # Prioritize reserve 
    r_requested = r * volume_acti_up_pu_fc
    r_supplied = anp.minimum(r_requested, p_tot_fc)
    delta_r = r_requested - r_supplied
    p_da_supplied = anp.minimum(p_tot_fc, r + p_da) - r_avail # or r_supplied ?  
    delta_p_da = p_da - p_da_supplied
    # p_tot_supplied = p_da_supplied + r_supplied
    
    # Compute profits and penalties 
    profit_da = p_da * price_da_fc
    profit_reserve = r * price_capa_up_fc + r * price_acti_up_fc * volume_acti_up_pu_fc
    pen_da = delta_p_da * price_imb_fc
    pen_r_c = eps_pen_r_c * delta_r_c * price_capa_up_fc
    pen_r_s = 1.3 * delta_r / volume_acti_up_pu_fc_div * (price_capa_up_fc + price_acti_up_fc * volume_acti_up_pu_fc)
      
    profit_hourly = profit_da + profit_reserve - (pen_da + pen_r_c + pen_r_s)
    # Price is given in €/MWh, but timestep could be lower than an hour
    profit = profit_hourly / nb_samples_hourly
    
    profit_mean_all_alpha = anp.mean(profit, axis=1)
    
    profit_max = anp.max(profit_mean_all_alpha, axis=1)
    profit_max_tot = anp.sum(profit_max)
    
    return profit_max_tot # Only one scalar output with autograd


def profit_gradients(gradient_method=autograd_pw, wrt_arg=['x', 'y'], **kwargs):
    """Method to compute the gradients of the AEP with respect to wrt_arg using the gradient_method

    Note, this method has two behaviours:
    1) Without specifying additional key-word arguments, kwargs, the method returns the function to
    compute the gradients of the aep:
    gradient_function = wfm.aep_gradients(autograd, ['x','y'])
    gradients = gradient_function(x,y)
    This behaviour only works when wrt_arg is one or more of ['x','y','h','wd', 'ws']

    2) With additional key-word arguments, kwargs, the method returns the gradients of the aep:
    gradients = wfm.aep_gradients(autograd,['x','y'],x=x,y=y)
    This behaviour also works when wrt_arg is a keyword argument, e.g. yaw

    Parameters
    ----------
    gradient_method : gradient function, {fd, cs, autograd}
        gradient function
    wrt_arg : {'x', 'y', 'h', 'wd', 'ws', 'yaw','tilt'} or list of these arguments, e.g. ['x','y']
        argument to compute gradients of AEP with respect to
    """
    if kwargs:
        wrt_arg = np.atleast_1d(wrt_arg)

        def wrap_aep(*args, **kwargs):
            kwargs.update({n: v for n, v in zip(wrt_arg, args)})
            return func_compute_profit(**kwargs)

        f = gradient_method(wrap_aep, True, tuple(range(len(wrt_arg))))
        return np.array(f(*[kwargs.pop(n) for n in wrt_arg], **kwargs))
    else:
        argnum = [['x', 'y', 'h', 'type', 'wd', 'ws'].index(a) for a in np.atleast_1d(wrt_arg)]
        f = gradient_method(func_compute_profit, True, argnum)
        return f
    
def func_dummy_profit(x, y, K, T, alpha_list, p_tot_c_list, S, nb_chunks=1):
    return 1

#%% Evaluate cost function and gradient with initial layout

it = 0

start = time.time()
profit_tot_init = func_compute_profit(x_init, y_init, K, T, alpha_list, p_tot_c_list, S=S, nb_chunks=1)
print("Initial profit:", round(profit_tot_init, 2), "€")
end = time.time()
print(round(end - start, 3), 's')

it = 0

start = time.time()
grad_init = profit_gradients(gradient_method=autograd_pw, wrt_arg=['x','y'], x=x_init, y=y_init, 
                              K=K, T=T, alpha_list=alpha_list, p_tot_c_list=p_tot_c_list, S=S, nb_chunks=1)
end = time.time()
print(round(end - start, 3), 's')

np.mean(np.abs(grad_init))
      
#%% Setup optimization
              
profit_comp_pw = CostModelComponent(input_keys=['x','y'],
                              n_wt=nb_wt,
                              cost_function=func_dummy_profit, # no function --> dummy cost
                              cost_gradient_function=profit_gradients,
                              output_keys="total_profit",
                              output_unit="€",
                              additional_input=[('K', K), ('T', T), ('alpha_list', alpha_list), ('p_tot_c_list', p_tot_c_list), ('S', S), ('nb_chunks', 1)],
                              objective=True,
                              maximize=True)


# Don't forget to reinitialize driver
driver_sgd = EasySGDDriver(maxiter=sgd_iterations, learning_rate=rotor_diam, max_time=1000000, gamma_min_factor=0.1, 
                           disp=True)

tf_problem_sgd = TopFarmProblem(
            design_vars=design_vars,
            cost_comp=profit_comp_pw,
            constraints=constraints_sgd,
            driver=driver_sgd,
            plot_comp=XYPlotComp(save_plot_per_iteration=False),
            expected_cost=1
            )

tf_problem_sgd.evaluate()

if not os.path.exists('Figures/'):
    os.mkdir('Figures/')

#%% Run optimization

it = 0

start = time.time()
profit_opt, layout_opt, recorder = tf_problem_sgd.optimize()
end = time.time()
exec_time = round(end - start, 2)

#%% Save results

# Save recorder
recorder.save(output_name)
# os.rename('Figures/', 'Figures_' + output_name + '/')
os.mkdir('Output/' + output_name)

tf_problem_sgd.evaluate()
plot_boundary(constraint_boundary)
plt.title("Optimal layout, Profit: " + str(round(profit_opt, 3)) + " €")
plt.savefig('Plots/'+ output_name +'_opt.pdf', dpi=600)
plt.show()

print('Optimization with SGD (', str(sgd_iterations), 'iterations) took: {:.0f}s'.format(exec_time), ' with a total constraint violation of ', recorder['sgd_constraint'][-1])

winsound.Beep(500, 1000)


