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


def plot_pmf_cdf_attempts(attempts_dict: dict, title: str):
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

        mean_attempts = np.mean(data)
        if mean_attempts <= 0:
            continue
        p_ge = 1.0 / mean_attempts

        t_vals = np.arange(1, int(x.max()) + 1)
        pmf_analytic = p_ge * (1 - p_ge) ** (t_vals - 1)
        cdf_analytic = 1 - (1 - p_ge) ** t_vals

        ax_pmf.plot(t_vals, pmf_analytic, linestyle="--", linewidth=0.8, color=color)
        ax_cdf.plot(t_vals, cdf_analytic, linestyle="--", linewidth=0.8, color=color)

    for ax in (ax_pmf, ax_cdf):
        ax.set_xscale("log")
        ax.grid(True, alpha=0.25, which="both")
        ax.tick_params(axis="both", labelsize=9)

    ax_pmf.set_xlabel("# attempts (A~C)", fontsize=11)
    ax_pmf.set_ylabel("Probability", fontsize=11)
    ax_pmf.set_title("PMF", fontsize=12, pad=6)

    ax_cdf.set_xlabel("# attempts (A~C)", fontsize=11)
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
    plt.show()
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
    plt.show()
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
        }

        for dist in params["distances"]:
            print(f"  Distance {dist} km, shots = {params['shots']}")
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
        )
        plot_fidelity_vs_distance(
            fidelities,
            title=f"Fidelity A~C vs distance\n{data['label_noise']}",
        )


if __name__ == "__main__":
    all_res = run_longrange_sims()
    plot_longrange(all_res)