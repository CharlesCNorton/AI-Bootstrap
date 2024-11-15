import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import os

save_path = r"D:\GitHub\AI-Bootstrap\K-invariant"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Create multiple subplots to show different aspects
fig = plt.figure(figsize=(15, 10))

def create_visualization():
    # Plot 1: Persistence Diagram Evolution
    ax1 = fig.add_subplot(221)
    dimensions = [2, 3, 4, 5]
    persistence_features = [64, 79, 105, 121]  # Average features from our test
    ax1.plot(dimensions, persistence_features, 'b-', label='Topological Features')
    ax1.set_title('Growth of Topological Features')
    ax1.set_xlabel('Dimension')
    ax1.set_ylabel('Number of Features')

    # Plot 2: K-invariant vs Theoretical Bound (Log Scale)
    ax2 = fig.add_subplot(222)
    k_values = np.array([66.4774, 94.3203, 120.3063, 134.9263])
    theoretical = 2**np.array(dimensions)
    ax2.semilogy(dimensions, k_values, 'b-', label='K-invariant')
    ax2.semilogy(dimensions, theoretical, 'r--', label='Theoretical Bound')
    ax2.set_title('K-invariant vs Theoretical Bound (Log Scale)')
    ax2.set_xlabel('Dimension')
    ax2.set_ylabel('Value (Log Scale)')
    ax2.legend()

    # Plot 3: Ratio Plot (K-invariant / Theoretical)
    ax3 = fig.add_subplot(223)
    ratio = k_values / theoretical
    ax3.plot(dimensions, ratio, 'g-', label='K/Theoretical Ratio')
    ax3.set_title('Stability of K-invariant Bound')
    ax3.set_xlabel('Dimension')
    ax3.set_ylabel('K/Theoretical Ratio')
    ax3.axhline(y=1, color='r', linestyle='--', label='Minimum Valid Ratio')
    ax3.legend()

    # Plot 4: Growth Rate Comparison
    ax4 = fig.add_subplot(224)
    k_growth = np.diff(np.log(k_values))
    t_growth = np.diff(np.log(theoretical))
    ax4.plot(dimensions[1:], k_growth, 'b-', label='K-invariant Growth Rate')
    ax4.plot(dimensions[1:], t_growth, 'r--', label='Theoretical Growth Rate')
    ax4.set_title('Growth Rate Comparison')
    ax4.set_xlabel('Dimension')
    ax4.set_ylabel('Log Growth Rate')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'k_invariant_mathematical_analysis2.png'), dpi=300)
    plt.close()

create_visualization()
