import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import os

# Ensure we're saving to the correct directory
save_path = r"D:\GitHub\AI-Bootstrap\K-invariant"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Create figure
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Data from our test results
dimensions = np.array([2, 3, 4, 5])
k_values = np.array([66.4774, 94.3203, 120.3063, 134.9263])
theoretical = 2**dimensions

def animate(frame):
    ax.clear()

    # Plot the main comparison line
    ax.plot(dimensions, k_values, theoretical, 'r-', linewidth=3,
            label='K-invariant vs Theoretical')

    # Plot the actual points
    ax.scatter(dimensions, k_values, theoretical, c='blue', s=100)

    # Labels and title
    ax.set_xlabel('Dimension', fontsize=12, labelpad=10)
    ax.set_ylabel('K-invariant Value', fontsize=12, labelpad=10)
    ax.set_zlabel('Theoretical Complexity', fontsize=12, labelpad=10)
    ax.set_title('K-invariant Bounds Theoretical Complexity\n' +
                f'Rotation angle: {frame}°', fontsize=14, pad=20)

    # Rotate view
    ax.view_init(elev=20., azim=frame)

    # Add legend
    ax.legend(fontsize=10)

    # Add annotations for each point
    for i in range(len(dimensions)):
        ax.text(dimensions[i], k_values[i], theoretical[i],
               f'Dim {dimensions[i]}\nK={k_values[i]:.1f}\nT={theoretical[i]}',
               fontsize=8)

    # Set consistent axis limits
    ax.set_xlim(1.5, 5.5)
    ax.set_ylim(50, 150)
    ax.set_zlim(0, 35)

# Create and save animation
anim = animation.FuncAnimation(fig, animate, frames=360, interval=50, blit=False)
anim.save(os.path.join(save_path, 'k_invariant_visualization.gif'),
         writer='pillow', fps=30)

print(f"Animation saved to: {os.path.join(save_path, 'k_invariant_visualization.gif')}")
plt.close()
