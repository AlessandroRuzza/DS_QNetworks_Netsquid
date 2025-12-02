def label_loss(params):
    return  f"{params['name']} ({params['p_loss_init']} init, " + \
            f"{params['p_loss_length']}dB/km)"

def label_noise(params):
    return  f"{params['name']} (T1={params['t1']/1e3}us, T2={params['t2']/1e3}us)"

def label_full(params):
    return  f"{params['name']} ({params['p_loss_init']} init, " + \
            f"{params['p_loss_length']}dB/km, T1={params['t1']/1e3}us, T2={params['t2']/1e3}us)"

# Define different parameter sets to test
travel_ns_km = 1e9 / 2e5 # [ns/km] travel time
param_sets = [ 
    # {
    #     "name": "Ideal case",
    #     "shots": 20,
    #     "distances": [5, 20, 50],
    #     "p_loss_init": 0.0,
    #     "p_loss_length": 0.0,
    #     "t1": 0,
    #     "t2": 0,
    # },
    # {
    #     "name": "Realistic fibre",
    #     "shots": 1000,
    #     "distances": [5, 20, 50],
    #     "p_loss_init": 0.5,
    #     "p_loss_length": 0.2,
    #     "t1": travel_ns_km * 50,
    #     "t2": travel_ns_km * 22,
    # },
    {
        "name": "Zero length loss fibre",
        "shots": 200,
        "distances": [5, 20, 50],
        "p_loss_init": 0.5,
        "p_loss_length": 0.0,
        "t1": 0,
        "t2": 0,
    },
    {
        "name": "High length loss fibre",
        "shots": 1_000,
        "distances": [5, 20, 50],
        "p_loss_init": 0.0,
        "p_loss_length": 0.5,
        "t1": 0,
        "t2": 0,
    },{
        "name": "Low-Noise fibre (T1 = 500km travel time)",
        "shots": 1000,
        "distances": [5, 20, 50],
        "p_loss_init": 0.0,
        "p_loss_length": 0.2,
        "t1": travel_ns_km * 500,
        "t2": travel_ns_km * 50,
    },
    {
        "name": "High-Noise fibre (T1 = 50km travel time)",
        "shots": 1000,
        "distances": [5, 20, 50],
        "p_loss_init": 0.0,
        "p_loss_length": 0.2,
        "t1": travel_ns_km * 50,
        "t2": travel_ns_km * 5,
    },   
    {
        "name": "Extreme-Noise fibre (T1 = 5km travel time)",
        "shots": 1000,
        "distances": [5, 20, 50],
        "p_loss_init": 0.0,
        "p_loss_length": 0.2,
        "t1": travel_ns_km * 5,
        "t2": travel_ns_km * 0.5,
    },
]

import numpy as np
for params in param_sets:
    params['p_ge'] = {}
    for dist in params["distances"]:
        params['p_ge'][f"{dist}km"] = (1 - params['p_loss_init']) * np.power(10, -dist * params['p_loss_length'] / 10)
        print(params['name'], f"p_ge = {params['p_ge'][f'{dist}km']}")
