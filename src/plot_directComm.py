from directComm import *
import matplotlib.pyplot as plt
import numpy as np
import re, os
from pathlib import Path

############## SUCCESS TIME ##################################################
def get_img_path(label:str):
    # resolve output directory relative to this file (avoid changing cwd)
    base = Path(__file__).resolve().parent.parent  # repo root
    subfolder = re.sub(r'\s+', '_', label.splitlines()[0])
    outdir = base / "img" / subfolder
    label = label.split(maxsplit=2)[0] + "_" + label.splitlines()[1]
    outdir.mkdir(parents=True, exist_ok=True)
    label = re.sub(r'\s+', '_', str(label)).strip()          # collapse whitespace/newlines
    label = re.sub(r'[^A-Za-z0-9._-]', '_', label)           # keep a safe subset of chars
    label = re.sub(r'[()]', '', label)
    label = label[:190] + ".png"
    imgpath = outdir / label
    return imgpath 

def unique_and_probs(data):
    count_times = [(v, data.count(v)) for v in set(data)]
    sorted_times = sorted(count_times)
    
    unique_sorted = [v[0] for v in sorted_times] 
    counts_sorted = [v[1] for v in sorted_times]
    # normalize counts to probabilities
    total = sum(counts_sorted)
    probs = [c / total for c in counts_sorted] if total > 0 else counts_sorted

    return unique_sorted, probs

def plot_pmf_arrival_times(arrival_times:dict, title:str):
    plt.figure(figsize=(10, 6))
    for name, data in arrival_times.items():
        unique_sorted, probs = unique_and_probs(data)
        x = np.array(unique_sorted, dtype=float)
        y = np.array(probs, dtype=float)
        plt.plot(x, y, label=name, linewidth=2, marker='o', linestyle='-')

    plt.xscale('log')
    plt.xlabel("Arrival Time (#attempts)", fontsize=12)
    plt.ylabel("Probability", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(get_img_path(title), dpi=300)
    plt.show()
    plt.close()

def plot_cdf_arrival_times(arrival_times:dict, title):
    plt.figure(figsize=(10, 6))

    for name, data in arrival_times.items():
        unique_sorted, probs = unique_and_probs(data)
        cumulative_probs = [sum(probs[:i+1]) for i in range(len(probs))]
        plt.plot(unique_sorted, cumulative_probs, label=name,
                 linewidth=2, marker='o')

    plt.xscale('log')
    plt.xlabel("Time (#attempts)", fontsize=12)
    plt.ylabel("P(Arrival Time ≤ t)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_img_path(title), dpi=300)
    plt.show()
    plt.close()

################## FIDELITY ##################################################

def plot_fidelity_distribution(arrival_times:dict, fidelities:dict, title):
    plt.figure(figsize=(10, 6))
    for name, data in fidelities.items():
        timed_fidelities = zip(arrival_times[name], data)
        sort = sorted(set(timed_fidelities), key=lambda f: f[0])
        unique_times = [t[0] for t in sort]
        unique_fids = [t[1] for t in sort]
        plt.plot(unique_times, unique_fids, label=name,
                 linewidth=2, marker='o')

    plt.axhline(
        y=0.9,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Quality threshold (F=0.9)",
    )
    plt.xlabel("Arrival time (#attempts)", fontsize=12)
    plt.ylabel("Fidelity", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_img_path(title), dpi=300)
    plt.show()
    plt.close()

# Define different parameter sets to test
param_sets = [ 
    {
        "name": "Ideal case",
        "shots": 20,
        "distances": [5, 10, 20, 50, 100, 200],
        "p_loss_init": 0.0,
        "p_loss_length": 0.0,
        "depolar_freq": 0,
    },
    {
        "name": "High initial loss fibre",
        "shots": 1_000,
        "distances": [5, 10, 20, 50, 100, 200],
        "p_loss_init": 0.9,
        "p_loss_length": 0.02,
        "depolar_freq": 5_000,
    },
    {
        "name": "Zero length loss fibre",
        "shots": 200,
        "distances": [5, 10, 20, 50, 100, 200],
        "p_loss_init": 0.5,
        "p_loss_length": 0.0,
        "depolar_freq": 5_000,
    },
    {
        "name": "High length loss fibre",
        "shots": 1000,
        "distances": [5, 10, 20, 40],
        "p_loss_init": 0.0,
        "p_loss_length": 0.5,
        "depolar_freq": 3_000,
    },
    {
        "name": "Noise-dominated fibre",
        "shots": 100,
        "distances": [5, 10, 20, 50, 100, 200],
        "p_loss_init": 0.2,
        "p_loss_length": 0.02,
        "depolar_freq": 10_000,
    },
]

# Run simulations for all parameter sets
if __name__ == "__main__":
    all_results = {}

    def label_param(params):
        return  f"{params['name']} ({params['p_loss_init']} init, " + \
                f"{params['p_loss_length']}dB/km, {params['depolar_freq']/1000}kHz)"

    print("Running simulations with different parameters...")
    for i, params in enumerate(param_sets):
        label = label_param(params)
        all_results[label] = []
        results = []
        for dist in params["distances"]:
            print(f"Running set {i+1}/{len(param_sets)}: {dist}km - {label}")

            results.append(setup_sim(
                shots=params["shots"],
                distance=dist,
                p_loss_init=params["p_loss_init"],
                p_loss_length=params["p_loss_length"],
                depolar_freq=params["depolar_freq"],
            ))

        all_results[label] = {}
        for d in params["distances"]:
            all_results[label] = {
                "sim_end_times": {},
                "total_qubits_sent": {},
                "arrival_times": {},
                "fidelities": {},
            }
        for d, run in zip(params["distances"], results):
            all_results[label]["sim_end_times"][f"{d}km"] = [res[0] for res in run]
            all_results[label]["total_qubits_sent"][f"{d}km"] = [res[1] for res in run]
            all_results[label]["arrival_times"][f"{d}km"] = [res[2] for res in run]
            all_results[label]["fidelities"][f"{d}km"] = [res[3] for res in run]
            
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
            total_qubits_sent, title=f"PMF of arrival times\n{label}"
        )
        plot_cdf_arrival_times(
            total_qubits_sent, title=f"CDF of arrival times\n{label}"
        )
        plot_fidelity_distribution(
            total_qubits_sent, fidelities, title=f"Fidelity distribution\n{label}"
        )