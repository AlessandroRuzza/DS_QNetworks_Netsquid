import plot_directComm as direct
import plot_longRange as longRange
from scenarios import *

import matplotlib.pyplot as plt
import numpy as np

def plot_comparison(all_res_long, all_res_direct):
    """
    Compare repeater-based vs direct communication:
    - PMF/CDF of attempts
    - Fidelity distributions
    """
    for label_long in all_res_long.keys():
        # Find matching direct result
        label_direct = None
        for ld in all_res_direct.keys():
            if label_long == ld:
                label_direct = ld
                break
        
        if label_direct is None:
            print(f"{label_long} NOT found in direct results.")
            continue

        data_long = all_res_long[label_long]
        data_direct = all_res_direct[label_direct]
        
        # Collect all distance keys
        dist_keys_long = sorted(data_long["attempts_total"].keys(), 
                                key=lambda x: int(x.replace("km", "")))
        
        # Filter valid distance pairs
        valid_distances = []
        for dist_key_long in dist_keys_long:
            dist_km = int(dist_key_long.replace("km", ""))
            dist_key_direct = f"{int(dist_km * 2)}km"
            
            if data_direct["attempts_total"].get(dist_key_direct) is None:
                print(f"{dist_key_direct} NOT found in direct results for {data_long['params']['name']}.")
                continue
            
            valid_distances.append((dist_key_long, dist_key_direct, dist_km))
        
        if not valid_distances:
            print(f"No valid distance pairs found for {label_long}")
            continue
        
        num_distances = len(valid_distances)
        
        # Create PMF/CDF comparison plot with subplots for each distance
        fig_pmf_cdf, axes = plt.subplots(2, num_distances, figsize=(6 * num_distances, 10), sharey=True)
        if num_distances == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, (dist_key_long, dist_key_direct, dist_km) in enumerate(valid_distances):
            attempts_long = data_long["attempts_total"][dist_key_long]
            attempts_direct = data_direct["attempts_total"][dist_key_direct]
            
            ax_pmf = axes[0, idx]
            ax_cdf = axes[1, idx]
            
            for attempts, name, color in [
                (attempts_long, f"Repeater ({dist_km}km + {dist_km}km)", "C0"),
                (attempts_direct, f"Direct ({dist_km * 2}km)", "C1")
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
            
            ax_pmf.set_title(f"PMF - {dist_km * 2}km", fontsize=12)
            ax_cdf.set_title(f"CDF - {dist_km * 2}km", fontsize=12)
        
        axes[0,0].set_ylabel("Probability", fontsize=11)
        axes[1,0].set_ylabel("Probability", fontsize=11)
        fig_pmf_cdf.suptitle(f"Repeater vs Direct Communication - PMF/CDF\n{data_long['label_loss']}", fontsize=14)
        fig_pmf_cdf.tight_layout(rect=(0.02, 0.02, 1.0, 0.96))
        plt.savefig(get_img_path(f"Comparison_PMF_CDF\n{data_long['label_loss']}"), dpi=300, bbox_inches="tight")
        plt.close()
        
        # Create fidelity comparison plot with subplots for each distance
        fig_fid, axes_fid = plt.subplots(1, num_distances, figsize=(6 * num_distances, 5), sharey=True)
        if num_distances == 1:
            axes_fid = [axes_fid]
        
        for idx, (dist_key_long, dist_key_direct, dist_km) in enumerate(valid_distances):
            fidelities_long = data_long["fidelities"][dist_key_long]
            fidelities_direct = data_direct["fidelities"][dist_key_direct]
            
            ax = axes_fid[idx]
            
            fid_long_clean = [f for f in fidelities_long if f is not None]
            fid_direct_clean = [f for f in fidelities_direct if f is not None]
            
            positions = [1, 2]
            ax.violinplot(
                [fid_long_clean, fid_direct_clean],
                positions=positions,
                widths=0.8,
                showmeans=True,
                showextrema=False,
            )
            ax.set_xticks(positions, [f"Repeater\n({dist_km}km + {dist_km}km)", f"Direct\n({dist_km * 2}km)"])
            ax.set_title(f"{dist_km * 2}km", fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
        
        axes_fid[0].set_ylabel("Bell Fidelity A~C", fontsize=12)
        fig_fid.suptitle(f"Fidelity Comparison: Repeater vs Direct\n{data_long['label_noise']}", fontsize=14)
        fig_fid.tight_layout(rect=(0.02, 0.02, 1.0, 0.96))
        plt.savefig(get_img_path(f"Comparison_Fidelity\n{data_long['label_noise']}"), dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

def print_comparison_table(all_res_long, all_res_direct):
    """
    Print a comparison table of repeater vs direct communication.
    Rows: scenario-distance pairs
    Columns: avg time units, avg fidelity, p_gen for both methods
    """
    print("\n" + "="*120)
    print("COMPARISON TABLE: Repeater vs Direct Communication")
    print("="*120)
    print(f"{'Scenario':<30} {'Distance':<12} {'Method':<20} {'Avg Time Units':<18} {'Avg Fidelity':<15} {'P_gen':<10}")
    print("-"*120)
    
    for label_long in sorted(all_res_long.keys()):
        # Find matching direct result
        label_direct = None
        for ld in all_res_direct.keys():
            if label_long == ld:
                label_direct = ld
                break
        
        if label_direct is None:
            continue
        
        data_long = all_res_long[label_long]
        data_direct = all_res_direct[label_direct]
        
        # Get scenario name
        scenario_name:str = data_long['params']['name']
        
        # Collect all distance keys
        dist_keys_long = sorted(data_long["attempts_total"].keys(), 
                                key=lambda x: int(x.replace("km", "")))
        
        for dist_key_long in dist_keys_long:
            dist_km = int(dist_key_long.replace("km", ""))
            dist_key_direct = f"{int(dist_km * 2)}km"
            
            if data_direct["attempts_total"].get(dist_key_direct) is None:
                continue
            
            # Calculate statistics for repeater
            attempts_long = data_long["attempts_total"][dist_key_long]
            fidelities_long = [f for f in data_long["fidelities"][dist_key_long] if f is not None]
            avg_time_long = np.mean(attempts_long) if len(attempts_long) > 0 else 0
            avg_fid_long = np.mean(fidelities_long) if len(fidelities_long) > 0 else 0
            p_gen_long = data_long["params"]["p_ge"].get(dist_key_long, 0)
            
            # Calculate statistics for direct
            attempts_direct = data_direct["attempts_total"][dist_key_direct]
            fidelities_direct = [f for f in data_direct["fidelities"][dist_key_direct] if f is not None]
            avg_time_direct = np.mean(attempts_direct) if len(attempts_direct) > 0 else 0
            avg_fid_direct = np.mean(fidelities_direct) if len(fidelities_direct) > 0 else 0
            p_gen_direct = data_direct["params"]["p_ge"].get(dist_key_direct, 0)
            
            # Print repeater row
            split = scenario_name.split('(')
            if len(split)>=2: split.pop(1)
            scenario_disp = "(".join(split).replace("memories", "mem")
            print(f"{scenario_disp[:30]:<30} {dist_km*2:>4}km{'':<6} {'Repeater':<20} {avg_time_long:<18.2f} {avg_fid_long:<15.4f} {p_gen_long:<10.4f}")
            
            # Print direct row
            print(f"{'':<30} {'':<12} {'Direct':<20} {avg_time_direct:<18.2f} {avg_fid_direct:<15.4f} {p_gen_direct:<10.4f}")
            print("-"*120)
    
    print("="*120 + "\n")

def run_comparison(param_sets:list[dict]):
    all_res_long = longRange.run_longrange_sims(param_sets)

    import copy
    param_direct = copy.deepcopy(param_sets)
    for p in param_direct:
        p["distances"] = [2*d for d in p["distances"]]
    autofill_params(param_direct)

    all_res_direct = direct.run_sims(param_direct)
    
    return all_res_long, all_res_direct

if __name__ == "__main__":
    all_res_long, all_res_direct = run_comparison(param_sets)
    plot_comparison(all_res_long, all_res_direct)
    print_comparison_table(all_res_long, all_res_direct)