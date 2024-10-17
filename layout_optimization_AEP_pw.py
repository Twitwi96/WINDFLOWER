# -*- coding: utf-8 -*-
"""
Created on Thu, 11 July 2024, 14:45
Last modified on Mon, 16 Sep 2024

@author: Thuy-Hai

Wind farm layout optimization 
Maximization of AEP

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

#%% User-defined parameters

# Wind farm data
wf_name = 'Northwind'
init_layout = '50_random_B_K150' # 'base' or 'random'
min_spacing = 2 # Minimum number of rotor diameters between two turbines
file_wt_pc = wf_name 
path_data = 'Data\\'

bootstrap = 'random_B' # init_layout # True: replace timesteps in data after sampling, False: do not replace, '<name>': already sampled timesteps

# SGD parameters
K = 150 # Day sampling
T = 1 # Timestep sampling
sgd_iterations = 2000

# Scenarios of wind and electricity prices
scenarios_file = "Data/data_scenarios_processed"
scenarios_years = ['2023'] # ['2021', '2022', '2023'] 
nb_days = 365 # 365 + 365 + 365
nb_samples_hourly = 4 # Each sample is a quarter hour
wind_height = 100
columns_scen = ["dateTime_elia", "dateTime", "dateTime_utc", "dateHour_utc", 
                "ws", "wd", "price_da", 
                "price_capa_up", "price_acti_up", "volume_acti_up", "volume_acti_up_pu",
                "price_capa_down", "price_acti_down", "volume_acti_down", "volume_acti_up_down",
                "price_mip", "price_neg_imb", "alpha"]


output_name = wf_name + '_AEP_K' + str(K) + '_T' + str(T) + '_' + init_layout + '_' + str(int(time.time()))

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
    data_temp = pd.read_excel(scenarios_file +'.xlsx', sheet_name=year, names=columns_scen)
    
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
# Convex hull, encompass all wind turbines in conex shape
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

# # Plot actual wf layout
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


#%% Cost component model

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
    
    # Save sampled timesteps
    # init_layout = 'random_E'
    # with open('Data/sampled_timesteps_K' + str(K) + '_' + init_layout, "wb") as variables:   
    #     pickle.dump(sampled_timesteps, variables)

else:
    with open('Data/sampled_timesteps_K' + str(K) + '_' + bootstrap, "rb") as fp:   
        sampled_timesteps = pickle.load(fp) 

def wind_sampling(it):
    sampled_kt_list = sampled_timesteps[it]
    data_scen_KT = data_scen[data_scen.day_timestep.isin(sampled_kt_list)]
    
    ws, wd = np.array(data_scen_KT.ws), np.array(data_scen_KT.wd) 
    
    return ws, wd

# aep function - SGD
def aep_func_pw(x, y, full=False, **kwargs):
    global it # counter for optimiztion iteration (used for sampling)
    ws, wd = wind_sampling(it)
    it = it + 1
    # Each value is AEP of wind turbine for wind situation: array(nb_wt, nb_wind) 
    aep_sgd = wake_model(x, y, wd=wd, ws=ws, time=True).aep().sum().values * 1e2
    return aep_sgd

# gradient function - SGD
def aep_jac_pw(x, y, **kwargs):
        
    global it # counter for optimiztion iteration (used for sampling)
    ws, wd = wind_sampling(it)
    it = it + 1
    
    jx, jy = wake_model.aep_gradients(gradient_method=autograd_pw, wrt_arg=['x', 'y'], x=x, y=y, ws=ws, wd=wd, time=True)
    daep_sgd = np.array([np.atleast_2d(jx), np.atleast_2d(jy)]) * 1e2
    return daep_sgd

    
def func_dummy_aep(x, y):
    return 1

#%% Evaluate cost function and gradient with initial layout

it = 0

start = time.time()
AEP_init = aep_func_pw(x_init, y_init)
print("Initial AEP:", np.round(AEP_init, 2), "10MWh")
end = time.time()
print(round(end - start, 3), 's')

it = 0

start = time.time()
grad_init = aep_jac_pw(x=x_init, y=y_init)
end = time.time()
print(round(end - start, 3), 's')

np.mean(np.abs(grad_init))
      
#%% Setup optimization
              
aep_comp_pw = CostModelComponent(input_keys=['x','y'],
                              n_wt=nb_wt,
                              cost_function=func_dummy_aep, # no function --> dummy cost
                              cost_gradient_function=aep_jac_pw,
                              output_keys="AEP",
                              output_unit="10MWh",
                              objective=True,
                              maximize=True)


# Don't forget to reinitialize driver
driver_sgd = EasySGDDriver(maxiter=sgd_iterations, learning_rate=rotor_diam, max_time=1000000, gamma_min_factor=0.1, 
                           disp=True)

tf_problem_sgd = TopFarmProblem(
            design_vars=design_vars,
            cost_comp=aep_comp_pw,
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
AEP_opt, layout_opt, recorder = tf_problem_sgd.optimize()
end = time.time()
exec_time = round(end - start, 2)

recorder.save(output_name)
# os.rename('Figures/', 'Figures_' + output_name + '/')
os.mkdir('Output/' + output_name)

tf_problem_sgd.evaluate()
plot_boundary(constraint_boundary)
plt.title("Optimal AEP, AEP: " + str(round(AEP_opt, 3)) + " 10MWh")
plt.savefig('Plots/'+ output_name +'_opt.pdf', dpi=600)
plt.show()

print('Optimization with SGD (', str(sgd_iterations), 'iterations) took: {:.0f}s'.format(exec_time), ' with a total constraint violation of ', recorder['sgd_constraint'][-1])

#%% Save results

# Save recorder
# recorder.save(output_name)

# Save computation time
perf_log =  open("Output/Computation_time_" + wf_name + ".txt", "a")
perf_log.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + ": " + str(exec_time) + " sec"
               + ' (' + output_name + ')\n')
perf_log.close()

winsound.Beep(500, 1000)

