import plot_directComm as direct
import plot_longRange as longRange
import plot_comparison as comparison
from scenarios import *

if __name__ == "__main__":
    all_res_direct = direct.run_sims(param_sets)
    all_res_long = longRange.run_longrange_sims(param_sets)

    direct.plot_sims(all_res_direct)
    longRange.plot_longrange(all_res_long)

    comp_res_long = all_res_long
    comp_res_direct = direct.run_sims(comparison.make_direct_params_from_long(param_sets), args.skipThreshold)
    for k, res in comp_res_direct.items():
        for key in res["attempts_total"].keys():
            res["attempts_total"][key] = [v*2 for v in res["attempts_total"][key]] # Adjust for time unit being travel time of 1 short link of longRange (1/2 of long link)

    comparison.plot_comparison(comp_res_long, comp_res_direct)
