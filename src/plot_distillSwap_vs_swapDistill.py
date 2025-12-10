"""
Compare two repeater strategies:
- Distill-then-Swap (distill each link first, then perform the swap)
- Swap-then-Distill (perform swap, then distill the resulting pair)
"""
from distill_then_swap import setup_distill_then_swap_sim
from swap_then_distill import setup_swap_then_distill_sim
from scenarios import *

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def run_distill_then_swap_sims(param_sets):
    """Run distill-then-swap simulations for all parameter sets."""
    all_results = {}
    print("Running distill-then-swap simulations...")

    for i, params in enumerate(param_sets):
        label = label_full(params)
        print(f"\n=== (Distill-then-Swap) Set {i+1}/{len(param_sets)}: {label} ===")

        results_by_distance = defaultdict(list)
        attempts_by_distance = defaultdict(list)
        fidelities_by_distance = defaultdict(list)
        keyRates_by_distance = defaultdict(list)

        for dist in params["distances"]:
            p_ge = params.get("p_ge", {}).get(f"{dist}km")
            p_ge_msg = f", p_ge = {p_ge:.5f}" if p_ge is not None else ""
            print(f"  Distance {dist} km, shots = {params['shots']}{p_ge_msg}")

            results = setup_distill_then_swap_sim(
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


def run_swap_then_distill_sims(param_sets):
    """Run swap-then-distill simulations for all parameter sets."""
    all_results = {}
    print("Running swap-then-distill simulations...")

    for i, params in enumerate(param_sets):
        label = label_full(params)
        print(f"\n=== (Swap-then-Distill) Set {i+1}/{len(param_sets)}: {label} ===")

        results_by_distance = defaultdict(list)
        attempts_by_distance = defaultdict(list)
        fidelities_by_distance = defaultdict(list)
        keyRates_by_distance = defaultdict(list)

        for dist in params["distances"]:
            p_ge = params.get("p_ge", {}).get(f"{dist}km")
            p_ge_msg = f", p_ge = {p_ge:.5f}" if p_ge is not None else ""
            print(f"  Distance {dist} km, shots = {params['shots']}{p_ge_msg}")

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


def plot_distill_then_vs_swap_then(all_res_distill_then, all_res_swap_then):
    """
    Compare Distill-then-Swap vs Swap-then-Distill:
    - PMF/CDF of attempts
    - Fidelity distributions
    """
    for label_distil in all_res_distill_then.keys():
        label_swap = None
        for ls in all_res_swap_then.keys():
            if label_distil == ls:
                label_swap = ls
                break

        if label_swap is None:
            print(f"{label_distil} NOT found in swap-then-distill results.")
            continue

        data_distil = all_res_distill_then[label_distil]
        data_swap = all_res_swap_then[label_swap]

        dist_keys = sorted(data_distil["attempts_total"].keys(), key=lambda x: int(x.replace("km", "")))
        valid_distances = [dk for dk in dist_keys if dk in data_swap["attempts_total"]]

        if not valid_distances:
            print(f"No valid distance pairs found for {label_distil}")
            continue

        num_distances = len(valid_distances)
        fig_pmf_cdf, axes = plt.subplots(2, num_distances, figsize=(6 * num_distances, 10), sharey=True)
        if num_distances == 1:
            axes = axes.reshape(-1, 1)

        for idx, dist_key in enumerate(valid_distances):
            dist_km = int(dist_key.replace("km", ""))
            attempts_distil = data_distil["attempts_total"][dist_key]
            attempts_swap = data_swap["attempts_total"][dist_key]

            ax_pmf = axes[0, idx]
            ax_cdf = axes[1, idx]

            for attempts, name, color in [
                (attempts_distil, "Distill->Swap", "C2"),
                (attempts_swap, "Swap->Distill", "C3"),
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

        axes[0, 0].set_ylabel("Probability", fontsize=11)
        axes[1, 0].set_ylabel("Probability", fontsize=11)
        fig_pmf_cdf.suptitle(
            f"Distill->Swap vs Swap->Distill - PMF/CDF\n{data_distil['label_loss']}", fontsize=14
        )
        fig_pmf_cdf.tight_layout(rect=(0.02, 0.02, 1.0, 0.96))
        plt.savefig(
            get_img_path(f"Comparison_DistillThenSwap_vs_SwapThenDistill_PMF_CDF\n{data_distil['label_loss']}"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Fidelity plot
        fig_fid, axes_fid = plt.subplots(1, num_distances, figsize=(6 * num_distances, 5), sharey=True)
        if num_distances == 1:
            axes_fid = [axes_fid]

        for idx, dist_key in enumerate(valid_distances):
            dist_km = int(dist_key.replace("km", ""))
            fidelities_distil = data_distil["fidelities"][dist_key]
            fidelities_swap = data_swap["fidelities"][dist_key]

            ax = axes_fid[idx]

            fid_distil_clean = [f for f in fidelities_distil if f is not None]
            fid_swap_clean = [f for f in fidelities_swap if f is not None]

            positions = [1, 2]
            ax.violinplot(
                [fid_distil_clean, fid_swap_clean],
                positions=positions,
                widths=0.8,
                showmeans=True,
                showextrema=False,
            )
            ax.set_xticks(positions, [f"Distill->Swap\n({dist_km}km)", f"Swap->Distill\n({dist_km}km)"])
            ax.set_title(f"{dist_km}km", fontsize=12)
            ax.grid(True, alpha=0.3, axis="y")

        axes_fid[0].set_ylabel("Bell Fidelity A~C", fontsize=12)
        fig_fid.suptitle(
            f"Fidelity Comparison: Distill->Swap vs Swap->Distill\n{data_distil['label_noise']}", fontsize=14
        )
        fig_fid.tight_layout(rect=(0.02, 0.02, 1.0, 0.96))
        plt.savefig(
            get_img_path(
                f"Comparison_DistillThenSwap_vs_SwapThenDistill_Fidelity\n{data_distil['label_noise']}"
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def print_comparison_table(all_res_distil_then, all_res_swap_then):
    """Print comparison of distill→swap vs swap→distill."""
    print("\n" + "=" * 138)
    print("COMPARISON TABLE: Distill->Swap vs Swap->Distill")
    print("=" * 138)
    print(f"{'Scenario':<30} {'Distance':<12} {'Method':<20} {'Avg Time Units':<18} {'Avg Fidelity':<15} {'Secret Key Rate':<18}")
    print("-" * 138)

    for label_distil in sorted(all_res_distil_then.keys()):
        label_swap = None
        for ls in all_res_swap_then.keys():
            if label_distil == ls:
                label_swap = ls
                break

        if label_swap is None:
            continue

        data_distil = all_res_distil_then[label_distil]
        data_swap = all_res_swap_then[label_swap]

        scenario_name: str = data_distil["params"]["name"]
        dist_keys = sorted(data_distil["attempts_total"].keys(), key=lambda x: int(x.replace("km", "")))

        for dist_key in dist_keys:
            if dist_key not in data_swap["attempts_total"]:
                continue

            attempts_distil = data_distil["attempts_total"][dist_key]
            fidelities_distil = [f for f in data_distil["fidelities"][dist_key] if f is not None]
            skr_distil = data_distil["keyRates"][dist_key]

            attempts_swap = data_swap["attempts_total"][dist_key]
            fidelities_swap = [f for f in data_swap["fidelities"][dist_key] if f is not None]
            skr_swap = data_swap["keyRates"][dist_key]

            avg_skr_swap = np.mean(skr_swap) if len(skr_swap) > 0 else 0
            avg_skr_distil = np.mean(skr_distil) if len(skr_distil) > 0 else 0

            avg_time_distil = np.mean(attempts_distil) if attempts_distil else 0
            avg_fid_distil = np.mean(fidelities_distil) if fidelities_distil else 0

            avg_time_swap = np.mean(attempts_swap) if attempts_swap else 0
            avg_fid_swap = np.mean(fidelities_swap) if fidelities_swap else 0

            print(
                f"{scenario_name:<30} {dist_key:<12} {'Distill->Swap':<20} {avg_time_distil:<18.2f} {avg_fid_distil:<15.4f} {avg_skr_distil:<18.3f}"
            )
            print(
                f"{'':<30} {'':<12} {'Swap->Distill':<20} {avg_time_swap:<18.2f} {avg_fid_swap:<15.4f} {avg_skr_swap:<18.3f}"
            )
            print("-" * 138)

    print("=" * 138 + "\n")


if __name__ == "__main__":
    all_res_distil_then = run_distill_then_swap_sims(param_sets)
    all_res_swap_then = run_swap_then_distill_sims(param_sets)

    print_comparison_table(all_res_distil_then, all_res_swap_then)
    plot_distill_then_vs_swap_then(all_res_distil_then, all_res_swap_then)