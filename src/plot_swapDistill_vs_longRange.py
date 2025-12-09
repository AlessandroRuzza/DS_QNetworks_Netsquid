"""
Compare distillation (swap then distill) vs simple swap (long range) protocols.
"""
from swap_then_distill import setup_swap_then_distill_sim
import plot_longRange as longRange
from scenarios import *

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def run_distillation_sims(param_sets):
    """
    Run distillation simulations for all parameter sets.
    Returns: dict[label] -> {params, results, attempts, fidelities, ...}
    """
    all_results = {}
    print("Running distillation simulations...")
    
    for i, params in enumerate(param_sets):
        label = label_full(params)
        print(f"\n=== (Swap-then-Distill) Set {i+1}/{len(param_sets)}: {label} ===")
        
        results_by_distance = defaultdict(list)
        attempts_by_distance = defaultdict(list)
        fidelities_by_distance = defaultdict(list)
        keyRates_by_distance = defaultdict(list)
        
        for dist in params["distances"]:
            print(
                f"  Distance {dist} km, shots = {params['shots']}, p_ge = {params['p_ge'][f'{dist}km']:.5f}"
            )
            results = setup_swap_then_distill_sim(
                shots=params["shots"],
                distance=dist,
                p_loss_init=params["p_loss_init"],
                p_loss_length=params["p_loss_length"],
                T1_channel=params.get("t1", 0),
                T2_channel=params.get("t2", 0),
                T1_mem=params.get("T1_mem", 0),
                T2_mem=params.get("T2_mem", 0),
            )
            
            dist_key = f"{dist}km"
            for sim_end_time, attempts_total, F_AC, keyRate in results:
                results_by_distance[dist_key].append((sim_end_time, attempts_total, F_AC))
                attempts_by_distance[dist_key].append(attempts_total)
                fidelities_by_distance[dist_key].append(F_AC)
                keyRates_by_distance[dist_key].append(keyRate)
        
        all_results[label] = {
            "params": params,
            "results": results_by_distance,
            "attempts_total": attempts_by_distance,
            "fidelities": fidelities_by_distance,
            "keyRates": keyRates_by_distance,
            "label_full": label_full(params),
            "label_loss": label_loss(params),
            "label_noise": label_noise(params),
        }
    
    return all_results


def plot_distil_vs_longrange(all_res_distil, all_res_long):
    """
    Compare distillation-based vs simple swap (long range):
    - PMF/CDF of attempts
    - Fidelity distributions
    """
    for label_distil in all_res_distil.keys():
        # Find matching long range result
        label_long = None
        for ll in all_res_long.keys():
            if label_distil == ll:
                label_long = ll
                break
        
        if label_long is None:
            print(f"{label_distil} NOT found in long range results.")
            continue

        data_distil = all_res_distil[label_distil]
        data_long = all_res_long[label_long]
        
        # Collect all distance keys
        dist_keys_distil = sorted(data_distil["attempts_total"].keys(), 
                                   key=lambda x: int(x.replace("km", "")))
        
        # Filter valid distance pairs
        valid_distances = []
        for dist_key in dist_keys_distil:
            if data_long["attempts_total"].get(dist_key) is None:
                print(f"{dist_key} NOT found in long range results for {data_distil['params']['name']}.")
                continue
            
            valid_distances.append(dist_key)
        
        if not valid_distances:
            print(f"No valid distance pairs found for {label_distil}")
            continue
        
        num_distances = len(valid_distances)
        
        # Create PMF/CDF comparison plot with subplots for each distance
        fig_pmf_cdf, axes = plt.subplots(2, num_distances, figsize=(6 * num_distances, 10), sharey=True)
        if num_distances == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, dist_key in enumerate(valid_distances):
            dist_km = int(dist_key.replace("km", ""))
            attempts_distil = data_distil["attempts_total"][dist_key]
            attempts_long = data_long["attempts_total"][dist_key]
            
            ax_pmf = axes[0, idx]
            ax_cdf = axes[1, idx]
            
            for attempts, name, color in [
                (attempts_distil, f"Swap+Distill", "C2"),
                (attempts_long, f"Simple Swap", "C0")
            ]:
                unique_sorted, probs = unique_and_probs(attempts)
                x = np.asarray(unique_sorted, dtype=float)
                y = np.asarray(probs, dtype=float)
                
                ax_pmf.plot(x, y, linestyle="-", linewidth=1.2, color=color, label=name)
                cdf = np.cumsum(y)
                ax_cdf.plot(x, cdf, linestyle="-", linewidth=1.2, color=color, label=name)
            
            for ax in (ax_pmf, ax_cdf):
                ax.set_xscale("log")
                ax.set_xlabel("Time units (L/c)", fontsize=11)
                ax.grid(True, alpha=0.25, which="both")
                ax.legend(fontsize=9, frameon=False)
            
            ax_pmf.set_title(f"PMF - {dist_km}km", fontsize=12)
            ax_cdf.set_title(f"CDF - {dist_km}km", fontsize=12)
        
        axes[0,0].set_ylabel("Probability", fontsize=11)
        axes[1,0].set_ylabel("Probability", fontsize=11)
        fig_pmf_cdf.suptitle(f"Swap+Distill vs Simple Swap - PMF/CDF\n{data_distil['label_loss']}", fontsize=14)
        fig_pmf_cdf.tight_layout(rect=(0.02, 0.02, 1.0, 0.96))
        plt.savefig(get_img_path(f"Comparison_Distil_vs_LongRange_PMF_CDF\n{data_distil['label_loss']}"), dpi=300, bbox_inches="tight")
        plt.close()
        
        # Create fidelity comparison plot with subplots for each distance
        fig_fid, axes_fid = plt.subplots(1, num_distances, figsize=(6 * num_distances, 5), sharey=True)
        if num_distances == 1:
            axes_fid = [axes_fid]
        
        for idx, dist_key in enumerate(valid_distances):
            dist_km = int(dist_key.replace("km", ""))
            fidelities_distil = data_distil["fidelities"][dist_key]
            fidelities_long = data_long["fidelities"][dist_key]
            
            ax = axes_fid[idx]
            
            fid_distil_clean = [f for f in fidelities_distil if f is not None]
            fid_long_clean = [f for f in fidelities_long if f is not None]
            
            positions = [1, 2]
            ax.violinplot(
                [fid_distil_clean, fid_long_clean],
                positions=positions,
                widths=0.8,
                showmeans=True,
                showextrema=False,
            )
            ax.set_xticks(positions, [f"Swap+Distill\n({dist_km}km)", f"Simple Swap\n({dist_km}km)"])
            ax.set_title(f"{dist_km}km", fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
        
        axes_fid[0].set_ylabel("Bell Fidelity A~C", fontsize=12)
        fig_fid.suptitle(f"Fidelity Comparison: Swap+Distill vs Simple Swap\n{data_distil['label_noise']}", fontsize=14)
        fig_fid.tight_layout(rect=(0.02, 0.02, 1.0, 0.96))
        plt.savefig(get_img_path(f"Comparison_Distil_vs_LongRange_Fidelity\n{data_distil['label_noise']}"), dpi=300, bbox_inches="tight")
        plt.close()


def print_comparison_table(all_res_distil, all_res_long):
    """
    Print a comparison table of swap+distill vs simple swap.
    Rows: scenario-distance pairs
    Columns: avg time units, avg fidelity for both methods
    """
    print("\n" + "="*130)
    print("COMPARISON TABLE: Swap+Distill vs Simple Swap")
    print("="*130)
    print(f"{'Scenario':<30} {'Distance':<12} {'Method':<12} {'Avg Time Units':<18} {'Avg Fidelity':<15} {'Secret-Key Rate':<18}")
    print("-"*130)
    
    for label_distil in sorted(all_res_distil.keys()):
        # Find matching long range result
        label_long = None
        for ll in all_res_long.keys():
            if label_distil == ll:
                label_long = ll
                break
        
        if label_long is None:
            continue
        
        data_distil = all_res_distil[label_distil]
        data_long = all_res_long[label_long]
        
        # Get scenario name
        scenario_name: str = data_distil['params']['name']
        
        # Get distance keys
        dist_keys = sorted(data_distil["attempts_total"].keys(), 
                          key=lambda x: int(x.replace("km", "")))
        
        for dist_key in dist_keys:
            if dist_key not in data_long["attempts_total"]:
                continue
            
            # Distillation stats
            attempts_distil = data_distil["attempts_total"][dist_key]
            fidelities_distil = [f for f in data_distil["fidelities"][dist_key] if f is not None]
            keyRates_distil = [k for k in data_distil["keyRates"][dist_key] if k is not None]
            
            avg_time_distil = np.mean(attempts_distil) if attempts_distil else 0
            avg_fid_distil = np.mean(fidelities_distil) if fidelities_distil else 0
            avg_skr_distil = np.mean(keyRates_distil) if keyRates_distil else 0
            
            # Long range stats
            attempts_long = data_long["attempts_total"][dist_key]
            fidelities_long = [f for f in data_long["fidelities"][dist_key] if f is not None]
            keyRates_long = [k for k in data_long["keyRates"][dist_key] if k is not None]
            
            avg_time_long = np.mean(attempts_long) if attempts_long else 0
            avg_fid_long = np.mean(fidelities_long) if fidelities_long else 0
            avg_skr_long = np.mean(keyRates_long) if keyRates_long else 0
            
            # Print distillation row
            print(f"{scenario_name:<30} {dist_key:<12} {'Swap+Distill':<12} {avg_time_distil:<18.2f} {avg_fid_distil:<15.4f} {avg_skr_distil:<18.6f}")
            print(f"{'':<30} {'':<12} {'Simple Swap':<12} {avg_time_long:<18.2f} {avg_fid_long:<15.4f} {avg_skr_long:<18.6f}")
            print("-"*130)
    
    print("="*130 + "\n")


if __name__ == "__main__":
    all_res_distil = run_distillation_sims(param_sets)
    all_res_long = longRange.run_longrange_sims(param_sets)
    
    # print_comparison_table(all_res_distil, all_res_long)
    plot_distil_vs_longrange(all_res_distil, all_res_long)