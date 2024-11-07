import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
import tkinter as tk
from tkinter import filedialog
import os

# Initialize tkinter root and hide main window
root = tk.Tk()
root.withdraw()

# Prompt user to select output directory
output_directory = filedialog.askdirectory(title="Select Output Directory")

# Define parameters
num_levels = 15  # Number of homotopy levels for cascade visualization
frequency = 0.1  # Frequency for oscillatory perturbations
epsilon = 0.3  # Perturbation amplitude
a0 = 1.0  # Base value

# Helper function to generate oscillatory perturbations
def generate_level(n, epsilon, frequency):
    # Loop Space oscillation
    loop_space = ((a0 + (a0 + epsilon * np.sin(frequency * n))) / 2) ** (1 / n) + np.cos(n * (a0 + epsilon * np.sin(frequency * n)))
    # Product Type oscillation
    product_type = ((a0 + epsilon * np.cos(frequency * n)) ** (1 / n) + np.cos(n * (a0 + epsilon * np.cos(frequency * n))) +
                    (a0 - epsilon * np.sin(frequency * n)) ** (1 / n) + np.sin(n * (a0 - epsilon * np.sin(frequency * n)))) / 2
    # Fibration with cohomological effect
    fibration = ((a0 + epsilon * np.sin(frequency * n)) ** (1 / n) + np.cos(n * a0)) / 2
    return loop_space, product_type, fibration

# Prepare figure and 3D axis
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1, 1, 2])  # Taller aspect for visual effect
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(0, num_levels + 2)

# Generate color gradient for stability visualization
colors = plt.cm.viridis(np.linspace(0, 1, num_levels))

# Function to update the frame for animation
def update(frame):
    ax.clear()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(0, num_levels + 2)

    for n in range(1, num_levels + 1):
        loop_space, product_type, fibration = generate_level(n, epsilon, frequency + 0.02 * frame)

        # Visualizing each level with concentric circles to represent Loop, Product, and Fibration stability
        theta = np.linspace(0, 2 * np.pi, 100)

        # Draw Loop Space as inner circle
        x_loop = loop_space * np.cos(theta)
        y_loop = loop_space * np.sin(theta)
        z = np.full_like(theta, n)
        ax.plot(x_loop, y_loop, z, color=colors[n-1], linewidth=2, label=f'Loop Space Level {n}' if n == 1 else "")

        # Draw Product Type as outer circle
        x_product = product_type * np.cos(theta)
        y_product = product_type * np.sin(theta)
        ax.plot(x_product, y_product, z, color=colors[n-1], linewidth=2, linestyle='--', label=f'Product Type Level {n}' if n == 1 else "")

        # Draw Fibration Type as innermost circle with cohomological effect
        x_fibration = fibration * np.cos(theta) * 0.8  # Slightly smaller for visual distinction
        y_fibration = fibration * np.sin(theta) * 0.8
        ax.plot(x_fibration, y_fibration, z, color=colors[n-1], linewidth=2, linestyle=':', label=f'Fibration Level {n}' if n == 1 else "")

    # Set labels for clarity
    ax.set_xlabel('X Axis (Stability Component)')
    ax.set_ylabel('Y Axis (Stability Component)')
    ax.set_zlabel('Homotopy Level (Recursive Structure)')

# Create the animation object
ani = FuncAnimation(fig, update, frames=200, interval=100)

# Define PillowWriter with metadata to enable looping
writer = PillowWriter(fps=20, metadata={'loop': 0})

# Build full path for the output file
output_path = os.path.join(output_directory, "homotopy_stability_cascade.gif")

# Save the animation as a looping GIF to the selected directory
ani.save(output_path, writer=writer)

print(f"Animation saved as {output_path}")
