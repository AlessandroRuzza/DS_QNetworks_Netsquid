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
        
        # For each distance pair (L vs 2L)
        for dist_key_long in data_long["attempts_total"].keys():
            dist_km = int(dist_key_long.replace("km", ""))
            dist_key_direct = f"{int(dist_km * 2)}km"
            
            if data_direct["attempts_total"].get(dist_key_direct) is None:
                print(f"{dist_key_direct} NOT found in data direct.")
                continue
            
            attempts_long = data_long["attempts_total"][dist_key_long]
            attempts_direct = data_direct["attempts_total"][dist_key_direct]
            fidelities_long = data_long["fidelities"][dist_key_long]
            fidelities_direct = data_direct["fidelities"][dist_key_direct]
            
            # Plot PMF/CDF comparison
            fig, (ax_pmf, ax_cdf) = plt.subplots(1, 2, figsize=(12, 5))
            
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
                ax.set_ylabel("Probability", fontsize=11)
                ax.grid(True, alpha=0.25, which="both")
                ax.legend(fontsize=9, frameon=False)
            
            ax_pmf.set_title("PMF", fontsize=12)
            ax_cdf.set_title("CDF", fontsize=12)
            
            fig.suptitle(f"Repeater vs Direct Communication\n{data_long['label_loss']}", fontsize=14)
            fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.95))
            plt.savefig(get_img_path(f"Comparison_PMF_CDF {dist_km}km\n{data_long['label_loss']}"), dpi=300, bbox_inches="tight")
            plt.close()
            
            # Plot fidelity comparison
            fig, ax = plt.subplots(figsize=(8, 5))
            
            fid_long_clean = [f for f in fidelities_long if f is not None]
            fid_direct_clean = [f for f in fidelities_direct if f is not None]
            
            positions = [1, 2]
            bp = ax.boxplot(
                [fid_long_clean, fid_direct_clean],
                positions=positions,
                widths=0.6,
                patch_artist=True,
                label=[f"Repeater\n({dist_km}km + {dist_km}km)", f"Direct\n({dist_km * 2}km)"]
            )
            
            for patch, color in zip(bp['boxes'], ['C0', 'C1']):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            
            ax.set_ylabel("Bell Fidelity A~C", fontsize=12)
            ax.set_title(f"Fidelity Comparison: Repeater vs Direct\n{data_long['label_loss']}", fontsize=14)
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(get_img_path(f"Comparison_Fidelity {dist_km}km\n{data_long['label_loss']}"), dpi=300, bbox_inches="tight")
            plt.close()

if __name__ == "__main__":
    all_res_long = longRange.run_longrange_sims(param_sets)

    import copy
    param_direct = copy.deepcopy(param_sets)
    for p in param_direct:
        p["distances"] = [2*d for d in p["distances"]]
    autofill_params(param_direct)

    all_res_direct = direct.run_sims(param_direct)

    plot_comparison(all_res_long, all_res_direct)