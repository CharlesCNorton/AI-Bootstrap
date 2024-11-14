import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import pi

# Data for four stages of the universe evolution with more realistic values for clarity
stages_realistic = [
    'Big Bang (10^32 K)',
    'Inflationary Epoch (10^27 K)',
    'Recombination (380,000 years after Big Bang)',
    'Formation of Stable Matter (several billion years after Big Bang)'
]

metrics = ['Entropy Stability', 'Clustering Complexity', 'Temperature Dependence', 'Coverage Spread']
groups = ['SO(10)', 'SU(5)', 'E6', 'Sp(4)', 'SO(3,1)']

group_data_over_time_realistic = {
    'Big Bang (10^32 K)': {
        'SO(10)': [8, 7, 8, 7],
        'SU(5)': [6, 6, 5, 5],
        'E6': [7, 6, 6, 7],
        'Sp(4)': [4, 3, 4, 3],
        'SO(3,1)': [3, 2, 2, 1]
    },
    'Inflationary Epoch (10^27 K)': {
        'SO(10)': [9, 8, 9, 8],
        'SU(5)': [5.5, 5, 5.5, 5],
        'E6': [7.5, 7, 7.2, 7],
        'Sp(4)': [4.2, 4, 4.3, 4],
        'SO(3,1)': [2.5, 2, 1.8, 1.5]
    },
    'Recombination (380,000 years after Big Bang)': {
        'SO(10)': [10, 9.5, 9, 9.5],
        'SU(5)': [6, 5.7, 6, 5.8],
        'E6': [8, 7.5, 7.7, 7.8],
        'Sp(4)': [4.5, 4.2, 4.5, 4.3],
        'SO(3,1)': [1.8, 1.5, 1.2, 1]
    },
    'Formation of Stable Matter (several billion years after Big Bang)': {
        'SO(10)': [10, 10, 10, 10],
        'SU(5)': [7, 6.5, 7, 6.8],
        'E6': [8.5, 8, 8.2, 8.3],
        'Sp(4)': [5, 4.7, 5, 4.9],
        'SO(3,1)': [1, 0.9, 0.8, 0.7]
    }
}

# Angles for radar chart (one per metric plus the first one repeated to close the chart)
num_metrics = len(metrics)
angles = [n / float(num_metrics) * 2 * pi for n in range(num_metrics)]
angles += angles[:1]

# Set up the figure for the animated radar chart
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# Function to interpolate between two sets of values
def interpolate_values(data1, data2, t):
    return [(1 - t) * v1 + t * v2 for v1, v2 in zip(data1, data2)]

# Stages data list for realistic evolution
stages_data_realistic = [group_data_over_time_realistic[stage] for stage in stages_realistic]

# Define distinct colors for each group
group_colors = {
    'SO(10)': 'darkgreen',
    'SU(5)': 'royalblue',
    'E6': 'darkorange',
    'Sp(4)': 'purple',
    'SO(3,1)': 'crimson'
}

# Function to update the radar plot during the animation with dynamic titles and annotations
def update(frame):
    ax.clear()
    ax.set_ylim(0, 11)  # Set a consistent range for all animations

    # Determine interpolation factor (t) based on the frame
    num_frames_per_stage = 50
    num_stages = len(stages_data_realistic)
    current_stage = min(frame // num_frames_per_stage, num_stages - 1)

    # If at the last stage, hold the final data without interpolating back to the first stage
    if current_stage == num_stages - 1:
        interpolated_data = stages_data_realistic[current_stage]
    else:
        next_stage = (current_stage + 1) % num_stages
        t = (frame % num_frames_per_stage) / num_frames_per_stage
        current_data = stages_data_realistic[current_stage]
        next_data = stages_data_realistic[next_stage]
        interpolated_data = {
            group: interpolate_values(current_data[group], next_data[group], t) for group in groups
        }

    # Plot each group's data with distinct colors
    for group, values in interpolated_data.items():
        values += values[:1]  # Complete the circle for radar plot
        color = group_colors[group]
        alpha = 1.0 if group == 'SO(10)' else 0.7
        linewidth = 3 if group == 'SO(10)' else 1.5
        ax.plot(angles, values, linewidth=linewidth, linestyle='solid', label=group, color=color, alpha=alpha)
        ax.fill(angles, values, alpha=0.15, color=color)

    # Set up axis labels and remove degree indicators
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10, fontweight='bold', color='navy')
    ax.set_yticklabels([])  # Remove radial degree labels for clarity
    ax.yaxis.set_visible(False)  # Completely hide the radial axis to avoid degrees showing

    # Add a clear and informative title that reflects the current stage
    ax.set_title(f'{stages_realistic[current_stage]}', size=15, position=(0.5, 1.1))

    # Add a legend outside the plot area with distinct group labels
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0), fontsize=10, title="Symmetry Groups", title_fontsize='13')

    # Add dynamic annotations based on the stage
    if current_stage == 0:
        ax.text(0.5, -0.1, "Big Bang: Initial Conditions for Symmetry Stability", transform=ax.transAxes,
                fontsize=12, fontweight='bold', color='darkgreen', ha='center')
    elif current_stage == 1:
        ax.text(0.5, -0.1, "Inflationary Epoch: SO(10) Shows Promising Stability", transform=ax.transAxes,
                fontsize=12, fontweight='bold', color='darkgreen', ha='center')
    elif current_stage == 2:
        ax.text(0.5, -0.1, "Recombination: SO(10) Maintains Dominance", transform=ax.transAxes,
                fontsize=12, fontweight='bold', color='darkgreen', ha='center')
    elif current_stage == 3:
        ax.text(0.5, -0.1, "Formation of Stable Matter: SO(10) Emerges as Dominant Symmetry Group", transform=ax.transAxes,
                fontsize=12, fontweight='bold', color='darkgreen', ha='center')

# Set up and run the animation with distinct colors, annotations, and an added legend
num_frames_total = (len(stages_data_realistic) - 1) * 50 + 100  # Hold the last stage for 100 frames
ani = FuncAnimation(fig, update, frames=num_frames_total, interval=50, repeat=True)

# Display the animation with more distinct colors, clearer labels, and dynamic annotations
plt.show()

