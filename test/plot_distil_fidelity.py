import numpy as np
import matplotlib.pyplot as plt

def p_distil(F):
    return F**2 + (2/3)*F*(1-F) + (5/9)*(1-F)**2

def g(F):
    return (F**2 + 1/9 * (1-F)**2) / p_distil(F)

# Generate F values from 0 to 1
F = np.linspace(0, 1, 1000)

# Calculate g(F)
g_values = g(F)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(F, g_values, 'b-', linewidth=2, label=r'$g(F)$')
plt.xlabel('F (Fidelity)', fontsize=12)
plt.ylabel('g(F)', fontsize=12)
plt.title(r'Plot of $g(F) =$ fidelity post distillation', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.xlim(0, 1)
plt.ylim(0, 1)

# Add reference line y=x to compare with identity
plt.plot([0, 1], [0, 1], 'r--', alpha=0.5, linewidth=1, label='y = F')
plt.legend(fontsize=11)

plt.tight_layout()
plt.savefig("distil_fidelity_plot.png", dpi=300)
plt.show()
