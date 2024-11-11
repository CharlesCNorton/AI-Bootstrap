import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

groups = ['G2', 'F4', 'E6', 'E7', 'E8', 'SU(2)', 'Cover G2', 'Cover F4', 'Cover E8']
soe_values = [-3, -5, -7, -8, -10, -2, -3.5, -6, -9]
colors = ['blue']*5 + ['red', 'green', 'green', 'green']
scales = [0.4, 0.3, 0.2, 0.15, 0.1, 0.5, 0.35, 0.25, 0.2]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-2, 2])
ax.set_title("Visualizing Entropy, Clustering, and Symmetry in Exceptional, Base, and Covering Groups")
ax.set_xlabel("Clustering Dimension")
ax.set_ylabel("Symmetry Depth")
ax.set_zlabel("Entropy Spread")

particle_data = [
    np.random.normal(loc=0, scale=scales[i], size=(100, 3)) for i in range(len(groups))
]
scatters = [
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], color=colors[i], s=20, alpha=0.6, label=groups[i])
    for i, data in enumerate(particle_data)
]

blue_proxy = plt.Line2D([0], [0], linestyle="none", color="blue", marker='o', markersize=10)
red_proxy = plt.Line2D([0], [0], linestyle="none", color="red", marker='o', markersize=10)
green_proxy = plt.Line2D([0], [0], linestyle="none", color="green", marker='o', markersize=10)
ax.legend([blue_proxy, red_proxy, green_proxy], ["Exceptional Groups (Low SOE)", "Base Groups", "Covering Groups"], loc='upper right')

def rotate_group(data, angle):
    cos_angle, sin_angle = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([
        [cos_angle, -sin_angle, 0],
        [sin_angle, cos_angle, 0],
        [0, 0, 1]
    ])
    return data @ rotation_matrix.T

def update(frame):
    angle = np.radians(frame / 3)
    opacity_cycle = frame // 60 % 3
    for i, scatter in enumerate(scatters):
        rotated_data = rotate_group(particle_data[i], angle)
        scatter._offsets3d = (rotated_data[:, 0], rotated_data[:, 1], rotated_data[:, 2])
        if opacity_cycle == 0 and i < 5:
            scatter.set_alpha(1.0)
        elif opacity_cycle == 1 and i == 5:
            scatter.set_alpha(1.0)
        elif opacity_cycle == 2 and i > 5:
            scatter.set_alpha(1.0)
        else:
            scatter.set_alpha(0.3)

ani = FuncAnimation(fig, update, frames=3000, interval=5, repeat=True)

plt.show()
