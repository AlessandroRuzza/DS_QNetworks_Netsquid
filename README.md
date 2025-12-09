# NetSquid Quantum Network Simulation

This project simulates quantum repeater networks using NetSquid, comparing different communication strategies including direct transmission, simple swap repeaters, and distillation-based protocols.

## Installation

### Prerequisites
- Python 3.12+
- NetSquid credentials (required for NetSquid installation)

### Setup

1. **Install NetSquid** (requires credentials):
```bash
# Run the installation script
# (insert your credentials when prompted)
bash scripts/install_netsquid.sh

# Or manually install with your NetSquid credentials:
python -m pip install --extra-index-url https://username:password@pypi.netsquid.org netsquid
```

2. **Install other dependencies**:
```bash
pip install -r requirements.txt
```

## Project Structure

```
├── src/
│   ├── core.py                  # Core quantum network components
│   ├── directComm.py            # Direct communication protocol
│   ├── longRange.py             # Long-range repeater with simple swap
│   ├── distillation.py          # Swap-then-distill protocol
│   ├── distill_then_swap.py     # Distill-then-swap protocol
│   ├── scenarios.py             # Simulation parameter configurations
│   ├── plot_*.py                # Plotting and simulation runners
├── test/                        # Test files
├── img/                         # Generated plots output directory
└── requirements.txt             # Python dependencies
```

**Key Files:**
- [`scripts/install_netsquid.sh`](scripts/install_netsquid.sh) - NetSquid installation helper
- [`requirements.txt`](requirements.txt) - Python dependencies
- [`src/scenarios.py`](src/scenarios.py) - Simulation parameter configurations
- [`src/core.py`](src/core.py) - Core quantum network components
- [`src/directComm.py`](src/directComm.py) - Direct communication protocol
- [`src/longRange.py`](src/longRange.py) - Long-range repeater protocol
- [`src/distillation.py`](src/distillation.py) - Swap-then-distill protocol
- [`src/distill_then_swap.py`](src/distill_then_swap.py) - Distill-then-swap protocol
- [`qConnection.ipynb`](qConnection.ipynb) - Interactive Jupyter notebook

## Running Simulations

All simulation scripts can be run from the repository root directory. They accept command-line arguments to customize the simulation parameters.

### Command-Line Arguments

All plotting scripts support these arguments (defined in [`scenarios.py`](src/scenarios.py)):

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--multiplyShotsBy` | `-m` | float | 1.0 | Multiply the number of shots by this factor |
| `--noisyACMemories` | | flag | False | Use noisy end-node (A/C) memories |
| `--skipThreshold` | `--skip` | float | 0.0 | Skip scenarios where p_ge < threshold |

**Examples:**
```bash
# Run with 5x more shots for better statistics
python src/plot_longRange.py -m 5

# Use noisy end-node memories
python src/plot_all.py --noisyACMemories

# Skip low probability scenarios (p_ge < 0.001)
python src/plot_comparison.py --skipThreshold 1e-3

# Combine multiple arguments
python src/plot_all.py -m 2.5 --noisyACMemories --skip 0.01
```

### Available Simulation Scripts

#### 1. **Complete Comparison Suite** ([`plot_all.py`](src/plot_all.py))
Runs direct communication [[sec.2]](#2-direct-communication-plot_directcommpy), 
1-repeater swap (longRange) [[sec.3]](#3-long-range-repeater-plot_longrangepy)
and comparison (direct vs longRange). [[sec.4]](#4-repeater-vs-direct-comparison-plot_comparisonpy)

```bash
python src/plot_all.py [OPTIONS]
```

**What it does:**
- Simulates direct communication
- Simulates long-range repeater (simple swap)
- Generates PMF/CDF plots for each protocol
- Generates fidelity distributions
- Creates comparison plots (repeater vs direct)

**Output:** Plots saved to `img/` subdirectories

---

#### 2. **Direct Communication** ([`plot_directComm.py`](src/plot_directComm.py))
Simulates direct quantum communication without repeaters.

```bash
python src/plot_directComm.py [OPTIONS]
```

**What it does:**
- Direct transmission of entangled pairs over distance
- No intermediate repeater nodes

**Output:** `img/Fidelity_direct_transmission_A~B/`, `img/PMF_CDF_of_direct_transmission_A~B/`

---

#### 3. **Long-Range Repeater** ([`plot_longRange.py`](src/plot_longRange.py))
Simulates quantum repeaters using simple entanglement swapping.

```bash
python src/plot_longRange.py [OPTIONS]
```

**What it does:**
- Creates two short entangled links (A-B and B-C)
- Performs Bell-state measurement (swap) at repeater node B
- Establishes long-range entanglement (A-C)

**Metrics:**
- Attempt counts (time units)
- Bell state fidelity
- PMF/CDF distributions

**Output:** `img/Fidelity_of_long-range_attempts_(A~C,_1_repeater)/`, `img/PMF_CDF_of_A~C/`

---

#### 4. **Repeater vs Direct Comparison** ([`plot_comparison.py`](src/plot_comparison.py))
Compares repeater-based vs direct communication strategies.

```bash
python src/plot_comparison.py [OPTIONS]
```

**What it does:**
- Runs both repeater and direct protocols
- Compares time units (attempts) required
- Compares fidelity distributions
- Generates comparison tables

**Output:** 
- Console: Comparison table with metrics
- Files: `img/Comparison_PMF_CDF/`, `img/Comparison_Fidelity/`

---

#### 5. **Swap-then-Distill vs Long-Range** ([`plot_swapDistill_vs_longRange.py`](src/plot_swapDistill_vs_longRange.py))
Compares distillation-enhanced repeaters with simple swap repeaters.

```bash
python src/plot_swapDistill_vs_longRange.py [OPTIONS]
```

**What it does:**
- **Swap-then-Distill**: Creates multiple pairs, swaps them, then applies distillation
- **Simple Swap**: Standard entanglement swap (baseline)
- Compares fidelity improvement vs time cost

**Output:** `img/Comparison_Distil_vs_LongRange_*/`

---

#### 6. **Distill-then-Swap vs Swap-then-Distill** ([`plot_distillSwap_vs_swapDistill.py`](src/plot_distillSwap_vs_swapDistill.py))
Compares two distillation strategies.

```bash
python src/plot_distillSwap_vs_swapDistill.py [OPTIONS]
```

**What it does:**
- **Distill->Swap**: Distills A-B and B-C links separately, then swaps
- **Swap->Distill**: Swaps first, then distills the A-C pair
- Compares which ordering is more efficient

**Output:** `img/Comparison_DistillThenSwap_vs_SwapThenDistill_*/`

---

## Scenario Configuration

Simulation parameters are defined in [`src/scenarios.py`](src/scenarios.py) in the `param_sets` list. Each scenario specifies:

### Parameter Dictionary Keys

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Scenario name |
| `shots` | int | Number of simulation runs |
| `distances` | list[int] | Distances to simulate (km) |
| `p_loss_init` | float | Initial photon loss probability |
| `p_loss_length` | float | Fiber loss coefficient (dB/km) |
| `t1` | float | T1 relaxation time for channels (ns) |
| `t2` | float | T2 dephasing time for channels (ns) |
| `t1_mem` | float | T1 for quantum memories (optional) |
| `t2_mem` | float | T2 for quantum memories (optional) |

### Default Scenarios

The following scenarios are pre-configured:

1. **Zero length loss fibre**: No distance-dependent loss, only initial loss
2. **High length loss fibre**: High distance-dependent attenuation
3. **Low-Noise fibre (noisy memories)**: Long coherence times with realistic memory noise
4. **High-Noise fibre**: Short coherence times

### Customizing Scenarios

Edit [`src/scenarios.py`](src/scenarios.py) to add/modify scenarios:

```python
param_sets = [
    {
        "name": "My Custom Scenario",
        "shots": 500,
        "distances": [10, 25, 50],
        "p_loss_init": 0.3,
        "p_loss_length": 0.15,
        "t1": travel_ns_km * 100,  # 100km equivalent
        "t2": travel_ns_km * 10,
    },
    # ... more scenarios
]
```

## Output

All plots are saved to the `img/` directory, organized by:
- Scenario name (e.g., `Low-Noise_fibre`)
- Plot type (e.g., `PMF_CDF`, `Fidelity`, `Comparison`)

### Plot Types

- **PMF/CDF plots**: Probability mass/cumulative distribution functions of attempt counts
- **Fidelity plots**: Bell state fidelity distributions (violin plots)
- **Comparison plots**: Side-by-side comparisons of different protocols

## Running on HPC (SLURM)

Example batch scripts are provided, with shell scripts to execute them:

```bash
# Run all plots with custom shot multiplier
M=5 bash run_all_plots.sh

# Run comparison analysis
bash run_comparison.sh
```

**Batch Scripts:**
- [`run_all_plots.sbatch`](scripts/run_all_plots.sbatch) - Run complete simulation suite
- [`run_comparison.sbatch`](scripts/run_comparison.sbatch) - Run comparison analysis
- [`run_distill_compare.sbatch`](scripts/run_distill_compare.sbatch) - Run distillation comparisons

## Jupyter Notebook

An interactive notebook is available at [`qConnection.ipynb`](qConnection.ipynb):

```bash
jupyter notebook qConnection.ipynb
```

## Troubleshooting

### NetSquid Import Errors
If you get import errors for NetSquid, ensure you've installed it with your credentials:
```bash
python -m pip install --extra-index-url https://pypi.netsquid.org netsquid
```

### GLIBC Version Issues (HPC)
On older HPC systems, use a container:
```bash
apptainer exec docker://python:3.12 python src/plot_all.py
```

## Contributing

When adding new protocols:
1. Implement the protocol in `src/` (e.g., `my_protocol.py`)
2. Create a corresponding plotter (e.g., `src/plot_my_protocol.py`)
3. Add comparison functions following existing patterns
4. Update this README

## Citation

If you use this code, please cite the relevant NetSquid papers and your own work.

## License

[Add your license information here]
