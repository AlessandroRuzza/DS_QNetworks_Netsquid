from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

def convert_comparison_table_to_latex(text_table: str) -> str:
    """
    Convert the text output from print_comparison_table to LaTeX format.
    Input format:
        Scenario                       Distance     Method               Avg Time Units     Avg Fidelity    P_gen      Secret-Key Rate   
        ------------------------------ ------------ -------------------- ------------------ --------------- ---------- ------------------
        scenario_name                  10km         Repeater             1.52               0.9510          0.7943     0.123456
                                                    Direct               1.61               0.9510          0.6310     0.123456
    """
    lines = text_table.strip().split('\n')
    
    # Skip header lines (first 4 lines: =, title, =, column headers, -)
    data_start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('-') and i > 0:
            data_start = i + 1
            break
    
    if data_start == 0:
        return ""
    
    # Parse data rows
    rows = []
    current_scenario = ""
    current_distance = ""
    
    for line in lines[data_start:]:
        if line.strip().startswith('='):
            break
        if line.strip().startswith('-'):
            continue
        
        # Split by whitespace, preserving structure
        parts = line.split()
        if len(parts) < 4:
            continue
        
        # Determine if this is a new scenario or continuation
        # If first column has text, it's a new scenario
        if line[0] != ' ':
            # New scenario row
            # Parse: scenario (multiple words possible), distance, method, time, fidelity, pgen, skr
            # Find where numeric data starts (Avg Time Units column)
            
            # Extract scenario (can have multiple words)
            scenario_end = line.find('km') - 3
            if scenario_end < 0:
                continue
            scenario = line[:scenario_end].strip()
            
            # Find distance
            rest = line[scenario_end:].strip()
            parts = rest.split()
            if len(parts) < 5:
                continue
            
            distance = parts[0]  # e.g., "10km"
            current_distance = distance.replace('km', '')
            method = parts[1]  # "Repeater" or "Direct"
            avg_time = parts[2]
            avg_fidelity = parts[3]
            skr = parts[4]
            
            rows.append({
                'scenario': scenario,
                'distance': current_distance,
                'method': method,
                'avg_time': avg_time,
                'avg_fidelity': avg_fidelity,
                'skr': skr
            })
        else:
            # Continuation row (Direct method)
            parts = [p for p in parts if p]
            if len(parts) < 3:
                continue
            
            method = parts[0]
            avg_time = parts[1]
            avg_fidelity = parts[2]
            skr = parts[3]
            
            rows.append({
                'scenario': '',
                'distance': '',
                'method': method,
                'avg_time': avg_time,
                'avg_fidelity': avg_fidelity,
                'skr': skr
            })
    
    # Generate LaTeX
    latex_lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\begin{tabular}{p{4.5cm} |c l r rr}",
        r"\hline",
        r"\textbf{Scenario} & \textbf{Distance} & \textbf{Method} & \textbf{Average} & \textbf{Average} & \textbf{Secret Key} \\",
        r" & \textbf{(km)} &  & $T_\mathrm{units}$ & \textbf{Fidelity} & \textbf{Rate} $[\mathrm{bits}/s]$ \\",
        r"\hline"
    ]
    
    prev_scenario = None
    for i, row in enumerate(rows):
        scenario = row['scenario']
        distance = row['distance']
        method = row['method']
        avg_time = row['avg_time']
        avg_fidelity = row['avg_fidelity']
        skr = row['skr']
        
        # Format scenario and distance (empty if continuation)
        scenario_col = scenario if scenario else ''
        distance_col = distance if distance else ''
        
        latex_line = f"{scenario_col} & {distance_col} & {method} & {avg_time} & {avg_fidelity} & {skr} \\\\"
        latex_lines.append(latex_line)
        
        # Add cline between distance groups
        if i + 1 < len(rows):
            next_row = rows[i + 1]
            # If next row has a scenario or we're at a Swap->Distill row and next has scenario
            if method == "Swap->Distill" and next_row['scenario']:
                latex_lines.append(r"\hline")
            elif method == "Swap->Distill" and next_row['method'] == "Distill->Swap" and not next_row['scenario']:
                latex_lines.append(r"\cline{2-7}")
    
    latex_lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\caption{Comparison table: Distill$\rightarrow$Swap vs Swap$\rightarrow$Distill}",
        r"\label{tab:comparison}",
        r"\end{table}"
    ])

    for i, line in enumerate(latex_lines):
        latex_lines[i] = line.replace("->", r"$\rightarrow$")
    
    return '\n'.join(latex_lines)

if __name__ == "__main__":
    # Example usage - sample output from print_comparison_table
    text_table = """
==========================================================================================================================================
COMPARISON TABLE: Distill->Swap vs Swap->Distill
==========================================================================================================================================
Scenario                       Distance     Method               Avg Time Units     Avg Fidelity    Secret Key Rate
------------------------------------------------------------------------------------------------------------------------------------------
High length loss fibre         5km          Distill->Swap        2.59               1.0000          9.367e+03
                                            Swap->Distill        2.40               1.0000          1.084e+04
------------------------------------------------------------------------------------------------------------------------------------------
High length loss fibre         20km         Distill->Swap        13.33              1.0000          5.949e+02
                                            Swap->Distill        12.98              1.0000          5.240e+02
------------------------------------------------------------------------------------------------------------------------------------------
High length loss fibre         50km         Distill->Swap        204.55             1.0000          1.423e+01
                                            Swap->Distill        214.65             1.0000          1.357e+01
------------------------------------------------------------------------------------------------------------------------------------------
High-Noise fibre               5km          Distill->Swap        5.62               0.6175          0.000e+00
                                            Swap->Distill        3.57               0.5395          0.000e+00
------------------------------------------------------------------------------------------------------------------------------------------
High-Noise fibre               20km         Distill->Swap        15.60              0.3009          0.000e+00
                                            Swap->Distill        9.18               0.3000          0.000e+00
------------------------------------------------------------------------------------------------------------------------------------------
High-Noise fibre               50km         Distill->Swap        91.95              0.2546          0.000e+00
                                            Swap->Distill        42.31              0.2548          0.000e+00
------------------------------------------------------------------------------------------------------------------------------------------
Low-Noise fibre                5km          Distill->Swap        2.14               0.9847          1.170e+04
                                            Swap->Distill        2.30               0.9794          1.076e+04
------------------------------------------------------------------------------------------------------------------------------------------
Low-Noise fibre                20km         Distill->Swap        9.89               0.8856          4.040e+02
                                            Swap->Distill        8.54               0.8286          1.468e+02
------------------------------------------------------------------------------------------------------------------------------------------
Low-Noise fibre                50km         Distill->Swap        63.06              0.6175          0.000e+00
                                            Swap->Distill        43.05              0.5400          0.000e+00
------------------------------------------------------------------------------------------------------------------------------------------
==========================================================================================================================================    
"""
    result = convert_comparison_table_to_latex(text_table)
    print(result)
