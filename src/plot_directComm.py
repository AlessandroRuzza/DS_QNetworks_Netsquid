from directComm import *
from scenarios import *
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

################## PMF, CDF PLOTS ##################################################

def plot_pmf_cdf_arrival_times_with_analytic(arrival_times: dict,
                                             title: str,
                                             params: dict | None = None,
                                             max_distances: int | None = None):
    
    fig, (ax_pmf, ax_cdf) = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, (name, data) in enumerate(arrival_times.items()):
        if max_distances is not None and idx >= max_distances:
            break

        unique_sorted, probs = unique_and_probs(data)
        x = np.asarray(unique_sorted, dtype=float)
        y = np.asarray(probs, dtype=float)
        color = colors[idx % len(colors)]

        ax_pmf.plot(
            x, y, markersize=2, linestyle="-", linewidth=0.9,
            color=color, label=name
        )
        cdf_sim = np.cumsum(y)
        ax_cdf.plot(
            x, cdf_sim, markersize=2, linestyle="-", linewidth=0.9,
            color=color
        )

        if params is not None:
            p_ge = params['p_ge'][name]
        else:
            mean_attempts = np.mean(data)
            if mean_attempts <= 0:
                continue
            p_ge = 1.0 / mean_attempts

        t_vals = np.arange(1, int(x.max()) + 1)
        pmf_analytic = p_ge * (1 - p_ge) ** (t_vals - 1)
        cdf_analytic = 1 - (1 - p_ge) ** t_vals

        ax_pmf.plot(
            t_vals, pmf_analytic,
            linestyle="--", linewidth=0.8, color=color
        )
        ax_cdf.plot(
            t_vals, cdf_analytic,
            linestyle="--", linewidth=0.8, color=color
        )

    for ax in (ax_pmf, ax_cdf):
        ax.set_xscale("log")
        ax.grid(True, alpha=0.25, which="both")
        ax.tick_params(axis="both", labelsize=9)

    ax_pmf.set_xlabel("Time units (L/c)", fontsize=11)
    ax_pmf.set_ylabel("Probability", fontsize=11)
    ax_pmf.set_title("PMF", fontsize=12, pad=6)

    ax_cdf.set_xlabel("Time units (L/c)", fontsize=11)
    ax_cdf.set_ylabel("Probability", fontsize=11)
    ax_cdf.set_title("CDF", fontsize=12, pad=6)

    ax_pmf.legend(
        title="distance (L)", fontsize=9, title_fontsize=9,
        loc="upper right", frameon=False
    )

    sim_handle = Line2D([0], [0], color="black", linestyle="-",
                        markersize=2, linewidth=0.9, label="simulation")
    ana_handle = Line2D([0], [0], color="black", linestyle="--",
                        linewidth=0.8, label="analytic")
    fig.legend(
        handles=[sim_handle, ana_handle],
        loc="lower center", ncol=2, fontsize=9, frameon=False,
        bbox_to_anchor=(0.5, 0.03)
    )
    fig.suptitle(title, fontsize=14, y=0.97)
    fig.tight_layout(rect=(0.03, 0.08, 1.0, 0.9))

    plt.savefig(get_img_path(title), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

################## FIDELITY PLOT ##################################################

def plot_fidelity_distribution(arrival_times:dict, fidelities:dict, title, expected=None):
    plt.figure(figsize=(10, 6))
    if expected is None: expected = {}
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for idx, (name, data) in enumerate(fidelities.items()):
        color = colors[idx % len(colors)]
        timed_fidelities = zip(arrival_times[name], data)
        sort = sorted(set(timed_fidelities), key=lambda f: f[0])
        unique_times = [t[0] for t in sort]
        unique_fids = [t[1] for t in sort]
        plt.plot(unique_times, unique_fids, label=name,
                 linewidth=2, color=color)
        if name not in expected.keys():
            expected[name] = np.mean(data)
        plt.axhline(expected[name], linestyle='--', color=color)
        
    plt.xlabel("Time units (L/c)", fontsize=12)
    plt.ylabel("Fidelity (A~B)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_img_path(title), dpi=300)
    # plt.show()
    plt.close()

################## MULTIPLE SCENARIO PARAMETERS ##################################################
def run_sims(param_sets:list[dict], pge_skip_threshold=0.0):
    all_results = {}
    print("Running simulations with different parameters...")
    for i, params in enumerate(param_sets):
        label = label_full(params)
        print(f"\n=== (Direct) Set {i+1}/{len(param_sets)}: {label} ===")

        all_results[label] = {
            "label_loss": label_loss(params),
            "label_noise": label_noise(params),
            "sim_end_times": {},
            "attempts_total": {},
            "arrival_times": {},
            "fidelities": {},
            "params": params,
        }

        results = []
        for dist in params["distances"]:
            if params['p_ge'][f'{dist}km'] < pge_skip_threshold: 
                print(
                    f"  (SKIP) Distance {dist} km, shots = {params['shots']}, p_ge = {params['p_ge'][f'{dist}km']:.5f}"
                )    
                continue
            print(
                f"  Distance {dist} km, shots = {params['shots']}, p_ge = {params['p_ge'][f'{dist}km']:.5f}"
            )

            results.append(setup_sim(
                shots=params["shots"],
                distance=dist,
                p_loss_init=params["p_loss_init"],
                p_loss_length=params["p_loss_length"],
                t1=params["t1"],
                t2=params["t2"],
            ))
            
            key = f"{dist}km"

        for d, run in zip(params["distances"], results):
            all_results[label]["sim_end_times"][f"{d}km"] = [res[0] for res in run]
            all_results[label]["attempts_total"][f"{d}km"] = [res[1] for res in run]
            all_results[label]["arrival_times"][f"{d}km"] = [res[2] for res in run]
            all_results[label]["fidelities"][f"{d}km"] = [res[3] for res in run]
            
    print("All simulations completed!")
    return all_results

def plot_sims(all_results):
    for label, data in all_results.items():
        print(f"\n=== {label} ===")
        sim_end_times = data["sim_end_times"]
        attempts_total = data["attempts_total"]
        arrival_times = data["arrival_times"]
        fidelities = data["fidelities"]

        # Print stats
        # sim_duration_stats(sim_end_times)
        # qubits_stats(attempts_total)
        # arrival_times_stats(arrival_times)
        # fidelity_stats(fidelities)

        # Plots (one figure per metric per set)
        plot_pmf_cdf_arrival_times_with_analytic(
            attempts_total, 
            title=f"PMF_CDF of direct transmission A~B\n{data['label_loss']}",
            params=data['params'],
        )
        plot_fidelity_distribution(
            attempts_total, fidelities, 
            title=f"Fidelity direct transmission A~B\n{data['label_noise']}",
        )

# Run simulations for all parameter sets
if __name__ == "__main__":
    all_res = run_sims(param_sets)
    plot_sims(all_res)
