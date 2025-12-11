import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from scenarios import *

def params_to_latex_table(param_sets):
    """Convert parameter sets to LaTeX table format."""
    
    # Start the table
    latex = r"\begin{table}[h]" + "\n"
    latex += r"\centering" + "\n"
    latex += r"\begin{tabular}{|l|c|c|c|c|c|c|}" + "\n"
    latex += r"\hline" + "\n"
    
    # Header row
    latex += r"\textbf{Scenario} & \textbf{Shots} & \textbf{Distances (km)} & "
    latex += r"\textbf{$p_\mathrm{init}$} & \textbf{$p_\mathrm{length} \big[\frac{\mathrm{dB}}{\mathrm{km}}\big]$} & "
    latex += r"\textbf{$T_1$ ($\mu$s)} & \textbf{$T_2$ ($\mu$s)} \\" + "\n"
    latex += r"\hline" + "\n"
    
    # Data rows
    for params in param_sets:
        name = params['name'].replace('_', r'\_')
        shots = params['shots']
        distances = ', '.join(map(str, params['distances']))
        p_loss_init = params['p_loss_init']
        p_loss_length = params['p_loss_length']
        
        # Convert T1 and T2 from ns to μs
        t1_us = params['t1'] / 1e3 if params['t1'] > 0 else 0
        t2_us = params['t2'] / 1e3 if params['t2'] > 0 else 0
        
        # Format T1 and T2 for display
        t1_str = f"{t1_us:.2f}" if t1_us > 0 else r"$\infty$"
        t2_str = f"{t2_us:.2f}" if t2_us > 0 else r"$\infty$"
        
        latex += f"{name} & {shots} & {distances} & {p_loss_init} & {p_loss_length} & {t1_str} & {t2_str} \\\\\n"
        latex += r"\hline" + "\n"
    
    # End the table
    latex += r"\end{tabular}" + "\n"
    latex += r"\caption{Simulation parameters for different scenarios}" + "\n"
    latex += r"\label{tab:scenarios}" + "\n"
    latex += r"\end{table}" + "\n"
    
    return latex

def params_to_detailed_latex_table(param_sets):
    """Convert parameter sets to detailed LaTeX table with p_ge values."""
    
    latex = r"\begin{table}[h]" + "\n"
    latex += r"\centering" + "\n"
    latex += r"\small" + "\n"
    latex += r"\begin{tabular}{|l|c|c|c|c|c|c|c|}" + "\n"
    latex += r"\hline" + "\n"
    
    # Header
    latex += r"\textbf{Scenario} & \textbf{Distance} & \textbf{Shots} & "
    latex += r"\textbf{$p_\mathrm{init}$} & \textbf{$p_\mathrm{length} \big[\frac{\mathrm{dB}}{\mathrm{km}}\big]$} & "
    latex += r"\textbf{$p_{ge}$} & \textbf{$T_1$ ($\mu$s)} & \textbf{$T_2$ ($\mu$s)} \\" + "\n"
    latex += r"\hline" + "\n"
    
    # Data rows
    for params in param_sets:
        name = params['name'].replace('_', r'\_')
        shots = params['shots']
        p_loss_init = params['p_loss_init']
        p_loss_length = params['p_loss_length']
        
        t1_us = params['t1'] / 1e3 if params['t1'] > 0 else 0
        t2_us = params['t2'] / 1e3 if params['t2'] > 0 else 0
        t1_str = f"{t1_us:.2f}" if t1_us > 0 else r"$\infty$"
        t2_str = f"{t2_us:.2f}" if t2_us > 0 else r"$\infty$"
        
        # One row per distance
        for i, dist in enumerate(params['distances']):
            p_ge = params['p_ge'][f"{dist}km"]
            
            if i == 0:
                # First row includes scenario name with multirow
                rows_count = len(params['distances'])
                latex += f"\\multirow{{{rows_count}}}{{*}}{{{name}}} & "
            else:
                latex += " & "
            
            latex += f"{dist} km & {shots} & {p_loss_init} & {p_loss_length} dB/km & "
            latex += f"{p_ge:.4f} & {t1_str} & {t2_str} \\\\\n"
        
        latex += r"\hline" + "\n"
    
    latex += r"\end{tabular}" + "\n"
    latex += r"\caption{Detailed simulation parameters with $p_{ge}$ for each distance}" + "\n"
    latex += r"\label{tab:scenarios_detailed}" + "\n"
    latex += r"\end{table}" + "\n"
    
    return latex

# Generate both tables
print("=" * 80)
print("BASIC TABLE:")
print("=" * 80)
basic_table = params_to_latex_table(param_sets)
print(basic_table)

print("\n" + "=" * 80)
print("DETAILED TABLE (with p_ge for each distance):")
print("=" * 80)
detailed_table = params_to_detailed_latex_table(param_sets)
print(detailed_table)

# Optionally save to file
output_file = Path(__file__).resolve().parent.parent / "latex" / "scenarios_table.tex"
output_file.parent.mkdir(parents=True, exist_ok=True)

with open(output_file, 'w') as f:
    f.write("% Basic table\n")
    f.write(basic_table)
    f.write("\n\n% Detailed table\n")
    f.write(detailed_table)

print(f"\n\nTables saved to: {output_file}")

