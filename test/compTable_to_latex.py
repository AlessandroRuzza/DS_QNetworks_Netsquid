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
        if len(parts) < 5:
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
            current_scenario = scenario
            
            # Find distance
            rest = line[scenario_end:].strip()
            parts = rest.split()
            if len(parts) < 6:
                continue
            
            distance = parts[0]  # e.g., "10km"
            current_distance = distance.replace('km', '')
            method = parts[1]  # "Repeater" or "Direct"
            avg_time = parts[2]
            avg_fidelity = parts[3]
            p_gen = parts[4]
            skr = parts[5]
            
            rows.append({
                'scenario': scenario,
                'distance': current_distance,
                'method': method,
                'avg_time': avg_time,
                'avg_fidelity': avg_fidelity,
                'p_gen': p_gen,
                'skr': skr
            })
        else:
            # Continuation row (Direct method)
            parts = [p for p in parts if p]
            if len(parts) < 5:
                continue
            
            method = parts[0]
            avg_time = parts[1]
            avg_fidelity = parts[2]
            p_gen = parts[3]
            skr = parts[4]
            
            rows.append({
                'scenario': '',
                'distance': '',
                'method': method,
                'avg_time': avg_time,
                'avg_fidelity': avg_fidelity,
                'p_gen': p_gen,
                'skr': skr
            })
    
    # Generate LaTeX
    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\begin{tabular}{l|cccccc}",
        r"\hline",
        r"\textbf{Scenario} & \textbf{Distance} & \textbf{Method} & \textbf{Average} & \textbf{Average} & \textbf{$P_\mathrm{gen}$} & \textbf{Secret Key} \\",
        r" & \textbf{(km)} &  & $T_\mathrm{units}$ & \textbf{Fidelity} &  & \textbf{Rate} $[\mathrm{bits}/s]$ \\",
        r"\hline"
    ]
    
    prev_scenario = None
    for i, row in enumerate(rows):
        scenario = row['scenario']
        distance = row['distance']
        method = row['method']
        avg_time = row['avg_time']
        avg_fidelity = row['avg_fidelity']
        p_gen = row['p_gen']
        skr = row['skr']
        
        # Format scenario and distance (empty if continuation)
        scenario_col = scenario if scenario else ''
        distance_col = distance if distance else ''
        
        latex_line = f"{scenario_col} & {distance_col} & {method} & {avg_time} & {avg_fidelity} & {p_gen} & {skr} \\\\"
        latex_lines.append(latex_line)
        
        # Add cline between distance groups
        if i + 1 < len(rows):
            next_row = rows[i + 1]
            # If next row has a scenario or we're at a Direct row and next has scenario
            if method == "Direct" and next_row['scenario']:
                latex_lines.append(r"\hline")
            elif method == "Repeater" and next_row['method'] == "Repeater" and not next_row['scenario']:
                pass  # Same scenario, different distance
            elif method == "Direct" and next_row['method'] == "Repeater" and not next_row['scenario']:
                latex_lines.append(r"\cline{2-7}")
    
    latex_lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\caption{Comparison of Repeater vs Direct Communication Methods}",
        r"\label{tab:comparison}",
        r"\end{table}"
    ])
    
    return '\n'.join(latex_lines)

if __name__ == "__main__":
    # Example usage - sample output from print_comparison_table
    text_table = """
========================================================================================================================
COMPARISON TABLE: Repeater vs Direct Communication
========================================================================================================================
Scenario                       Distance     Method               Avg Time Units     Avg Fidelity    P_gen      Secret-Key Rate   
------------------------------------------------------------------------------------------------------------------------------------------
High length loss fibre           10km       Repeater             2.31               1.0000          0.5623     19631.886    
                                            Direct               6.39               1.0000          0.3162     5359.716       
------------------------------------------------------------------------------------------------------------------------------------------
High length loss fibre           40km       Repeater             14.94              1.0000          0.1000     726.756        
                                            Direct               194.59             1.0000          0.0100     101.741        
------------------------------------------------------------------------------------------------------------------------------------------
High-Noise fibre                 10km       Repeater             1.51               0.4842          0.7943     0.000          
                                            Direct               3.10               0.5224          0.6310     0.000          
------------------------------------------------------------------------------------------------------------------------------------------
High-Noise fibre                 40km       Repeater             3.46               0.2981          0.3981     0.000          
                                            Direct               12.29              0.3625          0.1585     0.000          
------------------------------------------------------------------------------------------------------------------------------------------
High-Noise fibre                100km       Repeater             15.55              0.2524          0.1000     0.000          
                                            Direct               194.39             0.2838          0.0100     0.000          
------------------------------------------------------------------------------------------------------------------------------------------
Low-Noise fibre                  10km       Repeater             1.48               0.8720          0.7943     13108.695      
                                            Direct               3.16               0.9044          0.6310     3774.790       
------------------------------------------------------------------------------------------------------------------------------------------
Low-Noise fibre                  40km       Repeater             3.41               0.5520          0.3981     0.000          
                                            Direct               12.38              0.7054          0.1585     0.000          
------------------------------------------------------------------------------------------------------------------------------------------
Low-Noise fibre                 100km       Repeater             14.92              0.3245          0.1000     0.000          
                                            Direct               206.65             0.5224          0.0100     0.000          
------------------------------------------------------------------------------------------------------------------------------------------
Zero length loss fibre           10km       Repeater             2.62               1.0000          0.5000     17311.209      
                                            Direct               3.98               1.0000          0.5000     6966.841       
------------------------------------------------------------------------------------------------------------------------------------------
Zero length loss fibre           40km       Repeater             2.65               1.0000          0.5000     4194.050       
                                            Direct               3.98               1.0000          0.5000     1732.471       
------------------------------------------------------------------------------------------------------------------------------------------
Zero length loss fibre          100km       Repeater             2.66               1.0000          0.5000     1687.558       
                                            Direct               4.06               1.0000          0.5000     683.143        
------------------------------------------------------------------------------------------------------------------------------------------
==========================================================================================================================================
"""
    
    result = convert_comparison_table_to_latex(text_table)
    print(result)
