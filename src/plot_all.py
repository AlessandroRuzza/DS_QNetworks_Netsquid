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
    comp_res_direct = direct.run_sims(comparison.make_direct_params_from_long(param_sets))
    comparison.plot_comparison(comp_res_long, comp_res_direct)
