from pathlib import Path
import numpy as np
import re

def get_img_path(label: str):
    base = Path(__file__).resolve().parent.parent
    subfolder = re.sub(r"\s+", "_", label.splitlines()[0])
    outdir = base / "img" / subfolder
    label = label.split(maxsplit=2)[0] + "_" + label.splitlines()[1]
    outdir.mkdir(parents=True, exist_ok=True)
    label = re.sub(r"\s+", "_", str(label)).strip()
    label = re.sub(r"[^A-Za-z0-9._-]", "_", label)
    label = re.sub(r"[()]", "", label)
    label = label[:190] + ".png"
    imgpath = outdir / label
    return imgpath

def unique_and_probs(data):
    count_times = [(v, data.count(v)) for v in set(data)]
    sorted_times = sorted(count_times)
    unique_sorted = [v[0] for v in sorted_times]
    counts_sorted = [v[1] for v in sorted_times]
    total = sum(counts_sorted)
    probs = [c / total for c in counts_sorted] if total > 0 else counts_sorted
    return unique_sorted, probs

def label_loss(params):
    return  f"{params['name']} ({params['p_loss_init']} init, " + \
            f"{params['p_loss_length']}dB/km) ({params['shots']} shots)"

def label_noise(params):
    return  f"{params['name']} (T1={params['t1']/1e3}us, T2={params['t2']/1e3}us) ({params['shots']} shots)"

def label_full(params):
    return  f"{params['name']} ({params['p_loss_init']} init, " + \
            f"{params['p_loss_length']}dB/km, T1={params['t1']/1e3}us, T2={params['t2']/1e3}us) ({params['shots']} shots)"

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
        "shots": 2000,
        "distances": [5, 20, 50],
        "p_loss_init": 0.5,
        "p_loss_length": 0.0,
        "t1": 0,
        "t2": 0,
    },
    {
        "name": "High length loss fibre",
        "shots": 1000,
        "distances": [5, 20, 50],
        "p_loss_init": 0.0,
        "p_loss_length": 0.5,
        "t1": 0,
        "t2": 0,
    },
    # {
    #     "name": "Low-Noise fibre (T1 = 500km travel time) (ideal memories)",
    #     "shots": 1000,
    #     "distances": [5, 20, 50],
    #     "p_loss_init": 0.0,
    #     "p_loss_length": 0.2,
    #     "t1": travel_ns_km * 500,
    #     "t2": travel_ns_km * 50,
    #     "t1_mem": 0,
    #     "t2_mem": 0,
    # },
    {
        "name": "Low-Noise fibre (T1 = 500km travel time) (noisy memories)",
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
    # {
    #     "name": "Extreme-Noise fibre (T1 = 5km travel time)",
    #     "shots": 1000,
    #     "distances": [5, 20, 50],
    #     "p_loss_init": 0.0,
    #     "p_loss_length": 0.2,
    #     "t1": travel_ns_km * 5,
    #     "t2": travel_ns_km * 0.5,
    # },
]

def autofill_params(param_sets:list[dict]):
    for params in param_sets:
        params['p_ge'] = {}
        for dist in params["distances"]:
            params['p_ge'][f"{dist}km"] = (1 - params['p_loss_init']) * np.power(10, -dist * params['p_loss_length'] / 10)
            print(params['name'], f"({dist}km)", f"p_ge = {params['p_ge'][f'{dist}km']}")

        if params.get('t1_mem') is None:
            params["t1_mem"] = params["t1"]
        if params.get('t2_mem') is None:
            params["t2_mem"] = params["t2"]

autofill_params(param_sets)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--multiplyShotsBy", "--multiply", "--multiplyShots", type=float, default=1, help="Multiply shots by this factor")
parser.add_argument("--noisyACMemories", "--noisyMem", action="store_true", help="Use noisy AC memories (default: False)")
parser.add_argument("--skipThreshold", "--skip", type=float, default=0, help="Use noisy AC memories (default: False)")
args = parser.parse_args()

print(f"Using noisy memories in end-nodes: {args.noisyACMemories}")
print(f"Multiplying shots by: {args.multiplyShotsBy}")
print(f"Skipping executions with pge <= {args.skipThreshold}")

if args.multiplyShotsBy > 0:
    for params in param_sets:
        params["shots"] = round(params["shots"] * args.multiplyShotsBy)