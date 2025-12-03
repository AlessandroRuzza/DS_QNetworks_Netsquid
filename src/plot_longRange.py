from longRange import setup_longrange_sim
from scenarios import *
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import defaultdict
import numpy as np

def plot_pmf_cdf_attempts(attempts_dict: dict, title: str, params: dict):
    fig, (ax_pmf, ax_cdf) = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, (name, data) in enumerate(attempts_dict.items()):
        unique_sorted, probs = unique_and_probs(data)
        x = np.asarray(unique_sorted, dtype=float)
        y = np.asarray(probs, dtype=float)
        color = colors[idx % len(colors)]

        ax_pmf.plot(
            x,
            y,
            # marker="o",
            markersize=2,
            linestyle="-",
            linewidth=0.9,
            color=color,
            label=name,
        )

        cdf_sim = np.cumsum(y)
        ax_cdf.plot(
            x,
            cdf_sim,
            # marker="o",
            markersize=2,
            linestyle="-",
            linewidth=0.9,
            color=color,
        )

        p_ge = params["p_ge"][name]

        t_vals = np.arange(1, int(x.max()) + 1)
        pmf_analytic = 2 * p_ge * (1 - p_ge) ** (t_vals - 1) * (
            1 - (1 - p_ge) ** t_vals
        ) - (p_ge**2) * (1 - p_ge) ** (2 * (t_vals - 1))
        cdf_analytic = (1 - (1 - p_ge) ** t_vals) ** 2

        ax_pmf.plot(t_vals, pmf_analytic, linestyle="--", linewidth=0.8, color=color)
        ax_cdf.plot(t_vals, cdf_analytic, linestyle="--", linewidth=0.8, color=color)

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
        title="distance AB (L)",
        fontsize=9,
        title_fontsize=9,
        loc="upper right",
        frameon=False,
    )

    sim_handle = Line2D(
        [0],
        [0],
        color="black",
        linestyle="-",
        # marker="o",
        markersize=2,
        linewidth=0.9,
        label="simulation",
    )
    ana_handle = Line2D(
        [0],
        [0],
        color="black",
        linestyle="--",
        linewidth=0.8,
        label="analytic",
    )
    fig.legend(
        handles=[sim_handle, ana_handle],
        loc="lower center",
        ncol=2,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, 0.03),
    )
    fig.suptitle(title, fontsize=14, y=0.97)
    fig.tight_layout(rect=(0.03, 0.08, 1.0, 0.9))

    plt.savefig(get_img_path(title), dpi=300, bbox_inches="tight")
    # plt.show()
    plt.close()


def plot_fidelity_vs_distance(fidelities_dict: dict, title: str):
    plt.figure(figsize=(7, 5))
    items = sorted(
        fidelities_dict.items(),
        key=lambda kv: float(kv[0].replace("km", "")),
    )

    distances = []
    means = []
    for name, data in items:
        d = float(name.replace("km", ""))
        vals = [f for f in data if f is not None]
        if not vals:
            continue
        distances.append(d)
        means.append(np.mean(vals))

    plt.plot(distances, means, linewidth=2)

    plt.xlabel("Distance [km]", fontsize=12)
    plt.ylabel("Fidelity A~C", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_img_path(title), dpi=300, bbox_inches="tight")
    # plt.show()
    plt.close()


def plot_violin_fidelity_binned(attempts_dict, fidelities_dict, title, params: dict):
    keys = sorted(fidelities_dict.keys(), key=lambda k: float(k.replace("km", "")))
    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)

    if n == 1:
        axes = [axes]

    # Define bins for attempts (T)
    bins = [
        (1, 2),
        (3, 4),
        (5, 7),
        (8, 12),
        (13, 20),
        (21, 40),
        (41, 999),
    ]
    bin_labels = [f"{lo}-{hi}" for (lo, hi) in bins]

    for ax, key in zip(axes, keys):
        attempts = attempts_dict[key]
        fidelities = fidelities_dict[key]

        # bucket data into bins
        binned = [[] for _ in bins]
        for a, f in zip(attempts, fidelities):
            for i, (lo, hi) in enumerate(bins):
                if lo <= a <= hi:
                    binned[i].append(f)
                    break

        # keep only bins with enough samples
        data = []
        positions = []
        for i, vals in enumerate(binned):
            if len(vals) >= 5:
                data.append(vals)
                positions.append(i + 1)

        parts = ax.violinplot(
            data,
            positions=positions,
            showmeans=True,
            showmedians=False,
            showextrema=True,
        )

        for pc in parts["bodies"]:
            pc.set_alpha(0.4)

        # plot means
        means = [np.mean(vals) for vals in data]
        ax.plot(positions, means, color="black", linewidth=1)

        ax.set_xticks(positions)
        ax.set_xticklabels([bin_labels[p - 1] for p in positions], rotation=30)
        ax.set_xlabel("Time units (L/c)", fontsize=10)
        ax.set_title(key, fontsize=11)  # + f" (p_ge = {params['p_ge'][key]:.3f})"

        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("Fidelity A~C")
    fig.suptitle(title)
    fig.tight_layout(rect=(0.02, 0.05, 1.0, 0.95))

    # plt.show()
    plt.savefig(get_img_path(title), dpi=300, bbox_inches="tight")
    plt.close()

def plot_violin_fidelity(
    attempts_dict: dict,
    fidelities_dict: dict,
    title: str,
    params:dict,
    max_T: int = 100,
    min_count: int = 0,
):
    """
    For each distance (e.g. '5km', '20km', '50km') create a subplot with
    violins over T = time units, showing the distribution of fidelities
    of shots that finished at that T.

    max_T: ignore attempts > max_T (tail is almost flat at 0.5 anyway)
    min_count: require at least this many samples per T to draw a violin
    """
    keys = sorted(fidelities_dict.keys(), key=lambda k: float(k.replace("km", "")))
    n = len(keys)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        attempts = attempts_dict[key]
        fidelities = fidelities_dict[key]

        # bucket fidelities by attempt count
        buckets = defaultdict(list)
        for a, f in zip(attempts, fidelities):
            if f is None:
                continue
            if a > max_T:
                continue
            buckets[int(a)].append(float(f))

        # keep only T with enough data
        Ts = sorted(t for t, vals in buckets.items() if len(vals) >= min_count)
        if not Ts:
            ax.set_title(f"{key}")
            continue

        data = [buckets[t] for t in Ts]

        parts = ax.violinplot(
            data,
            positions=Ts,
            showmeans=True,
            showmedians=False,
            showextrema=True,
            widths=0.8,
        )

        # make violins semi-transparent
        for pc in parts["bodies"]:
            pc.set_alpha(0.3)

        # means as small markers for clarity
        means = [np.mean(buckets[t]) for t in Ts]
        ax.plot(Ts, means, linestyle="-", linewidth=1.0)

        ax.set_title(key, fontsize=11)
        ax.set_xlabel("Time units (L/c)", fontsize=10)
        # Pick sample tick values
        if len(Ts) > 20:
            step = max(1, len(Ts) // 15)
            tick_positions = Ts[::step]
            if Ts[-1] not in tick_positions:
                tick_positions.append(Ts[-1])
            ax.set_xticks(tick_positions)
        else:
            ax.set_xticks(Ts)
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("Fidelity A~C", fontsize=11)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0.02, 0.05, 1.0, 0.92))

    plt.savefig(get_img_path(title), dpi=300, bbox_inches="tight")
    # plt.show()
    plt.close()

def run_longrange_sims():
    all_results = {}
    print("Running long-range simulations...")

    for i, params in enumerate(param_sets):
        label = label_full(params)
        print(f"\n=== (Long-range) Set {i+1}/{len(param_sets)}: {label} ===")

        all_results[label] = {
            "label_loss": label_loss(params),
            "label_noise": label_noise(params),
            "attempts_total": {},
            "fidelities": {},
            "params": params,
        }

        for dist in params["distances"]:
            print(
                f"  Distance {dist} km, shots = {params['shots']}, p_ge = {params['p_ge'][f'{dist}km']:.3f}"
            )
            res = setup_longrange_sim(
                shots=params["shots"],
                distance=dist,
                p_loss_init=params["p_loss_init"],
                p_loss_length=params["p_loss_length"],
                t1_channel=params["t1"],
                t2_channel=params["t2"],
                T1_mem=params["t1_mem"],
                T2_mem=params["t2_mem"],
            )

            key = f"{dist}km"
            attempts_total = [r[3] for r in res]
            fidelities = [r[5] for r in res]

            all_results[label]["attempts_total"][key] = attempts_total
            all_results[label]["fidelities"][key] = fidelities

    print("All long-range simulations completed!")
    return all_results


def plot_longrange(all_results):
    for label, data in all_results.items():
        print("#" * 5 + label + "#" * 5)
        attempts_total = data["attempts_total"]
        fidelities = data["fidelities"]

        plot_pmf_cdf_attempts(
            attempts_total,
            title=f"PMF_CDF of A~C\n{data['label_loss']}",
            params=data["params"],
        )

        plot_violin_fidelity(
            attempts_total,
            fidelities,
            title=f"Fidelity of long-range attempts (A~C, 1 repeater)\n{data['label_loss']}",
            params=data["params"],
        )

        plot_violin_fidelity_binned(
            attempts_total,
            fidelities,
            title=f"Fidelity of long-range attempts (A~C, 1 repeater) (binned)\n{data['label_loss']}",
            params=data["params"],
        )

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
                print(f"{dist_key_direct}km NOT found in data direct.")
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

import plot_directComm as direct

if __name__ == "__main__":
    all_res_long = run_longrange_sims()
    plot_longrange(all_res_long)

    param_direct = [p for p in param_sets]
    for p in param_direct:
        p["distances"]=[2*d for d in p["distances"]]
    all_res_direct = direct.run_sims(param_direct)

    plot_comparison(all_res_long, all_res_direct)
