from directComm import *
import matplotlib.pyplot as plt
import numpy as np
import os, re
from pathlib import Path

############## SUCCESS TIME ##################################################
def get_img_path(label:str):
    Path("img").mkdir(parents=True, exist_ok=True)
    label = re.sub(r'\s+', '_', str(label)).strip()          # collapse whitespace/newlines
    label = re.sub(r'[^A-Za-z0-9._-]', '_', label)           # keep a safe subset of chars
    label = label[:190] + ".png"
    imgpath = Path("img") / label
    return imgpath 

def plot_pmf_arrival_times(arrival_times, bins, title:str):
    plt.figure(figsize=(10, 6))
    plt.hist(
        arrival_times,
        bins=bins,
        density=False,
        weights=[1 / len(arrival_times)] * len(arrival_times),
        alpha=0.7,
        edgecolor="black",
        color="steelblue",
    )
    plt.xlabel("Arrival Time (μs)", fontsize=12)
    plt.ylabel("Probability", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_img_path(title), dpi=300)
    plt.show()
    plt.close()

def plot_cdf_arrival_times(arrival_times, title):
    arrival_times_sorted = sorted(arrival_times)
    n = len(arrival_times_sorted)
    probabilities = np.arange(1, n + 1) / n

    plt.figure(figsize=(10, 6))
    plt.plot(arrival_times_sorted, probabilities, linewidth=2, color="steelblue")
    plt.xlabel("Time (μs)", fontsize=12)
    plt.ylabel("P(Arrival Time ≤ t)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_img_path(title), dpi=300)
    plt.show()
    plt.close()

################## FIDELITY ##################################################

def plot_fidelity_distribution(fidelities, bins, title):
    plt.figure(figsize=(10, 6))
    plt.hist(
        fidelities,
        bins=bins,
        density=False,
        weights=[1 / len(fidelities)] * len(fidelities),
        alpha=0.7,
        edgecolor="black",
        color="coral",
    )
    plt.axvline(
        x=0.9,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Quality threshold (F=0.9)",
    )
    plt.xlabel("Fidelity", fontsize=12)
    plt.ylabel("Probability", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_img_path(title), dpi=300)
    plt.show()
    plt.close()

# Define different parameter sets to test
param_sets = [ 
    # Base case
    {
        "distance": 20,
        "p_loss_init": 0.5,
        "p_loss_length": 0.05,
        "depolar_freq": 5_000,
        "name": "Base case",
    },
    # Strong loss values
    {
        "distance": 20,
        "p_loss_init": 0.9,
        "p_loss_length": 0.02,
        "depolar_freq": 5_000,
        "name": "Strong init loss",
    },
    # High-loss fibre (short distance)
    {
        "distance": 10,
        "p_loss_init": 0.0,
        "p_loss_length": 0.2,
        "depolar_freq": 3_000,
        "name": "High-loss fibre",
    },
    # High-loss fibre (long distance)
    {
        "distance": 100,
        "p_loss_init": 0.0,
        "p_loss_length": 0.2,
        "depolar_freq": 3_000,
        "name": "High-loss fibre",
    },
    # Noise-dominated (bad fidelity, short distance)
    {
        "distance": 10,
        "p_loss_init": 0.5,
        "p_loss_length": 0.05,
        "depolar_freq": 10_000,
        "name": "Noise-dominated",
    },
    # Noise-dominated (bad fidelity, long distance)
    {
        "distance": 100,
        "p_loss_init": 0.5,
        "p_loss_length": 0.05,
        "depolar_freq": 10_000,
        "name": "Noise-dominated",
    },
]

# Run simulations for all parameter sets
all_results = {}
shots = 2000  # Reduce for faster testing

for params in param_sets:
    params['label'] = f"{params['name']} ({params['distance']}km, {params['p_loss_init']} init, \
                        {params['p_loss_length']}/km, {params['depolar_freq']}kHz)"

print("Running simulations with different parameters...")
for i, params in enumerate(param_sets):
    print(f"Running set {i+1}/{len(param_sets)}: {params['label']}")

    results = setup_sim(
        shots=shots,
        distance=params["distance"],
        p_loss_init=params["p_loss_init"],
        p_loss_length=params["p_loss_length"],
        depolar_freq=params["depolar_freq"],
    )

    all_results[params["label"]] = {
        "params": params,
        "sim_end_times": [res[0] for res in results],
        "total_qubits_sent": [res[1] for res in results],
        "arrival_times": [res[2] for res in results],
        "fidelities": [res[3] for res in results],
    }

print("All simulations completed!")

for label, data in all_results.items():
    print(f"\n=== {label} ===")
    sim_end_times = data["sim_end_times"]
    total_qubits_sent = data["total_qubits_sent"]
    arrival_times = data["arrival_times"]
    fidelities = data["fidelities"]

    # Print stats
    # sim_duration_stats(sim_end_times)
    # qubits_stats(total_qubits_sent)
    # arrival_times_stats(arrival_times)
    # fidelity_stats(fidelities)

    # Plots (one figure per metric per set)
    plot_pmf_arrival_times(
        arrival_times, bins=50, title=f"PMF of arrival times\n{label}"
    )
    plot_cdf_arrival_times(
        arrival_times, title=f"CDF of arrival times\n{label}"
    )
    plot_fidelity_distribution(
        fidelities, bins=30, title=f"Fidelity distribution\n{label}"
    )