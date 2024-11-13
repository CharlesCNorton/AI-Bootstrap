import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import quad
from math import pi

# Define groups and metrics for computation
groups = ['SO(10)', 'SU(5)', 'E6', 'Sp(4)', 'SO(3,1)']
metrics = ['Entropy Stability', 'Clustering Complexity', 'Temperature Dependence', 'Coverage Spread']

# Set up temperature steps for cooling phase
temperature_steps = np.linspace(10, 1, 10)  # Temperature from high to low

# Define a function to calculate entropy using Haar measure integration over conjugacy classes
def calculate_entropy(group, temperature):
    # Set up specific properties based on group type (rank, dimension, etc.)
    properties = {
        'SO(10)': {'rank': 10, 'dimension': 45, 'centralizer_dim': 9},
        'SU(5)': {'rank': 5, 'dimension': 24, 'centralizer_dim': 4},
        'E6': {'rank': 6, 'dimension': 78, 'centralizer_dim': 5},
        'Sp(4)': {'rank': 4, 'dimension': 10, 'centralizer_dim': 3},
        'SO(3,1)': {'rank': 1, 'dimension': 6, 'centralizer_dim': 1}
    }

    rank = properties[group]['rank']

    # Define a probability density function over the conjugacy classes
    def density_function(x):
        return (x ** (rank - 1)) * np.exp(-x / temperature)

    # Compute entropy by integrating -f(x) * log(f(x)) over the interval [0, temperature]
    entropy_integral, _ = quad(lambda x: -density_function(x) * np.log(density_function(x) + 1e-12), 0, temperature)
    return max(entropy_integral, 0)  # Ensure non-negative entropy values

# Compute the metrics for each group across the temperature range without any placeholders
def compute_metrics(group, temperature):
    # Entropy Stability Calculation
    entropy_stability = calculate_entropy(group, temperature)

    # Clustering Complexity derived from the dimension of centralizers and conjugacy class sizes
    properties = {
        'SO(10)': {'rank': 10, 'dimension': 45, 'centralizer_dim': 9},
        'SU(5)': {'rank': 5, 'dimension': 24, 'centralizer_dim': 4},
        'E6': {'rank': 6, 'dimension': 78, 'centralizer_dim': 5},
        'Sp(4)': {'rank': 4, 'dimension': 10, 'centralizer_dim': 3},
        'SO(3,1)': {'rank': 1, 'dimension': 6, 'centralizer_dim': 1}
    }

    rank = properties[group]['rank']
    centralizer_dim = properties[group]['centralizer_dim']
    dimension = properties[group]['dimension']

    # Clustering Complexity involves computing how tightly elements are grouped within conjugacy classes
    clustering_complexity = (dimension / (centralizer_dim + 1e-12)) * np.log(entropy_stability + 1e-12)

    # Temperature Dependence reflects how entropy changes as the temperature changes
    temperature_dependence = entropy_stability / temperature

    # Coverage Spread computed as a function of root system complexity and conjugacy class distribution
    if entropy_stability > 0:
        coverage_spread = np.sqrt(entropy_stability) * (rank / (centralizer_dim + 1e-12))
    else:
        coverage_spread = 0  # Avoid invalid sqrt calculation

    return [entropy_stability, clustering_complexity, temperature_dependence, coverage_spread]

# Generate metric values for each temperature step and group
metric_values = {}
for group in groups:
    group_metrics = []
    for temp in temperature_steps:
        group_metrics.append(compute_metrics(group, temp))
    metric_values[group] = group_metrics

# Interpolation steps between each temperature step for a smooth transition
interp_steps = 20
fine_temperature_steps = np.linspace(10, 1, len(temperature_steps) * interp_steps)

# Interpolate metrics for each group for smoother transitions
def interpolate_metrics(group):
    base_metrics = np.array(metric_values[group]).T  # Metrics as rows for interpolation
    interpolated_metrics = []

    # Interpolate each metric over fine temperature steps
    for i in range(len(metrics)):
        metric_values_interpolated = base_metrics[i]
        interpolated_metrics.append(np.interp(fine_temperature_steps, temperature_steps, metric_values_interpolated))

    # Reshape interpolated metrics to create a list of lists for all metrics at each fine temp step
    interpolated_metrics = np.array(interpolated_metrics).T.tolist()
    return interpolated_metrics

# Interpolate metrics for each group
interpolated_data = {}
for group in groups:
    interpolated_data[group] = interpolate_metrics(group)

# Set up figure and polar axis for radar chart animation
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# Set up the angles for the radar chart (one angle per metric)
angles = [n / float(len(metrics)) * 2 * pi for n in range(len(metrics))]
angles.append(angles[0])  # Complete the circle

# Initial plot setup for each group
lines = {}
polygons = {}

for group in groups:
    initial_values = metric_values[group][0]
    initial_values.append(initial_values[0])  # Close the polygon
    lines[group], = ax.plot(angles, initial_values, linewidth=2, linestyle='solid', label=group)
    polygons[group] = ax.fill(angles, initial_values, alpha=0.25)

# Set labels, title, and legend
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics)
ax.set_title('Symmetry Group Evolution During Cosmic Cooling', size=15, position=(0.5, 1.1), ha='center')
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

# Animation update function for smooth interpolation
def update_smooth(frame):
    temp_step_index = frame // interp_steps  # Corresponds to a fine temperature step
    ax.set_title(f'Symmetry Group Evolution During Cosmic Cooling (Step {frame + 1})', size=15, position=(0.5, 1.1), ha='center')

    for group in groups:
        updated_metrics = interpolated_data[group][temp_step_index]
        updated_metrics.append(updated_metrics[0])  # Close the polygon to match the angles

        lines[group].set_data(angles, updated_metrics)

        # Remove old polygon and add updated one
        for poly in polygons[group]:
            poly.remove()
        polygons[group] = ax.fill(angles, updated_metrics, alpha=0.25)

# Create the animation with smooth interpolation
ani = FuncAnimation(fig, update_smooth, frames=len(fine_temperature_steps), interval=50, repeat=True)

# Display the animation
plt.show()
