import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Define groups, SOE values, and clustering spread based on SOE
groups = ['G2', 'F4', 'E6', 'E7', 'E8', 'SU(2)', 'Cover G2', 'Cover F4', 'Cover E8']
soe_values = [-3, -5, -7, -8, -10, -2, -3.5, -6, -9]  # Lower SOE values mean higher clustering
colors = ['blue']*5 + ['red', 'green', 'green', 'green']  # Blue for exceptional, Red for base, Green for covering
scales = [0.4, 0.3, 0.2, 0.15, 0.1, 0.5, 0.35, 0.25, 0.2]  # Scale inversely proportional to SOE

# Set up the figure and 3D axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-2, 2])
ax.set_title("Symmetry Orbit Entropy and Clustering in Exceptional, Base, and Covering Groups")
ax.set_xlabel("Clustering Dimension")
ax.set_ylabel("Symmetry Depth")
ax.set_zlabel("Entropy Spread")

# Generate particles for each group based on SOE and plot initial clusters
particle_data = [
    np.random.normal(loc=0, scale=scales[i], size=(100, 3)) for i in range(len(groups))
]
scatters = [
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], color=colors[i], s=20, alpha=0.6, label=groups[i])
    for i, data in enumerate(particle_data)
]

# Function to calculate clustering metrics for each group
def calculate_clustering_metrics(data):
    # Calculate Euclidean distances from center (0, 0, 0)
    distances = np.linalg.norm(data, axis=1)

    # Calculate Entropy of Distance Distribution
    hist, bin_edges = np.histogram(distances, bins=10, density=True)
    dist_entropy = entropy(hist)

    # Calculate KL Divergence from a uniform distribution over the same range
    uniform_dist = np.ones_like(hist) / len(hist)
    kl_divergence = entropy(hist, uniform_dist)

    return dist_entropy, kl_divergence

# Compute clustering metrics for each group in advance
metrics = [calculate_clustering_metrics(data) for data in particle_data]

# Text box for displaying metrics dynamically, justified more to the left
text_box = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, fontsize=12, verticalalignment='top')

# Add a legend to clarify group types with new color scheme
blue_proxy = plt.Line2D([0], [0], linestyle="none", color="blue", marker='o', markersize=10)
red_proxy = plt.Line2D([0], [0], linestyle="none", color="red", marker='o', markersize=10)
green_proxy = plt.Line2D([0], [0], linestyle="none", color="green", marker='o', markersize=10)
ax.legend([blue_proxy, red_proxy, green_proxy], ["Exceptional Groups (Low SOE)", "Base Groups", "Covering Groups"], loc='upper right')

# Animation function to adjust opacity cycling among group types
def update(frame):
    # Adjust rotation speed and direction for smoothness
    ax.view_init(elev=20, azim=(frame / 6) % 360)  # Slower rotation for smooth fidelity

    # Set opacity levels for each group type based on the frame count
    opacity_cycle = frame // 120 % 3  # Cycles every 120 frames for opacity change

    # Define which groups to keep fully opaque based on the cycle and display their metrics
    for i, scatter in enumerate(scatters):
        if (opacity_cycle == 0 and i < 5):  # Exceptional groups fully opaque
            scatter.set_alpha(1.0)
            current_group = i
        elif (opacity_cycle == 1 and i == 5):  # Base group fully opaque
            scatter.set_alpha(1.0)
            current_group = i
        elif (opacity_cycle == 2 and i > 5):  # Covering groups fully opaque
            scatter.set_alpha(1.0)
            current_group = i
        else:
            scatter.set_alpha(0.3)  # Faded for non-highlighted groups

    # Update the text box with the current group's metrics
    group_name = groups[current_group]
    dist_entropy, kl_divergence = metrics[current_group]
    text_box.set_text(
        f"Group: {group_name}\n"
        f"Entropy of Distance Distribution: {dist_entropy:.4f}\n"
        f"KL Divergence from Uniform: {kl_divergence:.4f}"
    )

# Increase frames to 2160 for smooth rendering, interval to 10ms for ultra-high fidelity
ani = FuncAnimation(fig, update, frames=2160, interval=5, repeat=True)

# Display the animation
plt.show()
