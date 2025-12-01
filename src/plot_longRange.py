from longRange import setup_longrange_sim
from scenarios import param_sets, label_loss, label_noise
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path
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
            marker="o",
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
            marker="o",
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

    ax_pmf.set_xlabel("Time units (A~C)", fontsize=11)
    ax_pmf.set_ylabel("Probability", fontsize=11)
    ax_pmf.set_title("PMF", fontsize=12, pad=6)

    ax_cdf.set_xlabel("Time units (A~C)", fontsize=11)
    ax_cdf.set_ylabel("Probability", fontsize=11)
    ax_cdf.set_title("CDF", fontsize=12, pad=6)

    ax_pmf.legend(
        title="distance",
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
        marker="o",
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

    plt.plot(distances, means, marker="o", linewidth=2)

    plt.xlabel("Distance [km]", fontsize=12)
    plt.ylabel("Fidelity A~C", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_img_path(title), dpi=300, bbox_inches="tight")
    # plt.show()
    plt.close()

def plot_violin_fidelity_binned(attempts_dict, fidelities_dict, title, params:dict):
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
        ax.plot(positions, means, marker="o", color="black", linewidth=1)

        ax.set_xticks(positions)
        ax.set_xticklabels([bin_labels[p - 1] for p in positions], rotation=30)
        ax.set_xlabel("Time units (A~C)", fontsize=10)
        ax.set_title(key, fontsize=11) # + f" (p_ge = {params['p_ge'][key]:.3f})"

        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("Fidelity A~C")
    fig.suptitle(title)
    fig.tight_layout(rect=(0.02, 0.05, 1.0, 0.95))

    # plt.show()
    plt.savefig(get_img_path(title), dpi=300, bbox_inches="tight")
    plt.close()


def run_longrange_sims():
    all_results = {}
    print("Running long-range simulations...")

    for i, params in enumerate(param_sets):
        label = label_loss(params)
        print(f"\n=== Set {i+1}/{len(param_sets)}: {label} ===")

        all_results[label] = {
            "label_loss": label_loss(params),
            "label_noise": label_noise(params),
            "attempts_total": {},
            "fidelities": {},
            "params": params,
        }

        for dist in params["distances"]:
            print(
                f"  Distance {dist} km, shots = {params['shots']}, p_ge = {params['p_ge'][f"{dist}km"]:.3f}"
            )
            res = setup_longrange_sim(
                shots=params["shots"],
                distance=dist,
                p_loss_init=params["p_loss_init"],
                p_loss_length=params["p_loss_length"],
                t1_channel=params["t1"],
                t2_channel=params["t2"],
                T1_mem=params["t1"],
                T2_mem=params["t2"],
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
        attempts_total = data["attempts_total"]
        fidelities = data["fidelities"]

        plot_pmf_cdf_attempts(
            attempts_total,
            title=f"PMF_CDF of attempts (A~C)\n{data['label_loss']}",
            params=data["params"],
        )
        # plot_fidelity_vs_distance(
        #     fidelities,
        #     title=f"Fidelity A~C vs distance\n{data['label_noise']}",
        # )

        plot_violin_fidelity_binned(
            attempts_total,
            fidelities,
            title=f"Violin PMF of attempts (A~C)\n{data['label_loss']}",
            params=data['params'], 
        )
        plot_fidelity_vs_distance(
            fidelities,
            title=f"Fidelity of Long Range Entanglement (1 Repeater)\n{data['label_noise']}",
        )


if __name__ == "__main__":
    all_res = run_longrange_sims()
    plot_longrange(all_res)
