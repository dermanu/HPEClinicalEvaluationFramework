import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

# Define the path to your TRC file
trc_file_path = 'MoCap/par4/Par4proc.trc'

# Read the TRC file, skipping the header
with open(trc_file_path, 'r') as trc_file:
    lines = trc_file.readlines()

# Find the line that starts with "Frame#" to identify the marker names
for i, line in enumerate(lines):
    if line.startswith("Frame#"):
        marker_names = line.split()[2:]
        marker_count = len(marker_names)
        start_line = i + 1
        break

# Extract marker coordinates
marker_data = np.genfromtxt(lines[start_line:], delimiter='\t', dtype=float, usecols=range(2, 2 + 3 * marker_count))
marker_data = marker_data.reshape(-1, marker_count, 3)
marker_data = marker_data[:, 33:, :]
frame = 30000
marker_names = marker_names[33:]
marker_count = len(marker_names)

# Create a 3D plot for each marker's trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Motion Capture Marker Trajectories')

# Plot each marker's trajectory
for i in range(marker_count):
    marker_name = marker_names[i]
    x = marker_data[frame, i, 0]
    y = marker_data[frame, i, 1]
    z = marker_data[frame, i, 2]
    print(marker_name, x, y, z)
    ax.scatter(x, y, z, label=marker_name)

# Define the connections between markers
connections = [('RWJC', 'REJC'), ('REJC', 'RSJC'), ('RAJC', 'RKJC'), ('RKJC', 'RHJC'),  ('LWJC', 'LEJC'), ('LEJC', 'LSJC'), ('LAJC', 'LKJC'), ('LKJC', 'LHJC'), ('RHJC', 'LHJC'), ('RSJC', 'LSJC'), ('RSJC', 'RHJC'), ('LSJC', 'LHJC')]
# Connect the specified markers
for connection in connections:
    marker1, marker2 = connection
    i1 = marker_names.index(marker1)
    print(i1)
    i2 = marker_names.index(marker2)
    x1, y1, z1 = marker_data[frame, i1, 0], marker_data[frame, i1, 1], marker_data[frame, i1, 2]
    x2, y2, z2 = marker_data[frame, i2, 0], marker_data[frame, i2, 1], marker_data[frame, i2, 2]
    ax.plot([x1, x2], [y1, y2], [z1, z2], 'k-')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.legend()

# Adjust aspect ratio
ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio

# Set the same range for all axes to make them isotropic
max_range = max(np.max(marker_data[frame], axis=0)) + 1
min_range = min(np.min(marker_data[frame], axis=0))
ax.set_xlim([min_range, max_range])
ax.set_ylim([min_range, max_range])
ax.set_zlim([min_range, max_range])

plt.show()
