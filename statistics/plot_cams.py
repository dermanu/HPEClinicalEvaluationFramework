"""
Plotting camera positions for publication
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Room dimensions
room_width = 5.4
room_height = 5.7
middle_of_room = (room_width / 2, room_height / 2)

# Square dimensions and position
square_size = 1.6
square_bottom_left = ((room_width - square_size) / 2, (room_height - square_size) / 2)

# Camera positions based on specified distances and angles
camera_positions = {
    "bl": {"distance": 3.0, "angle": -3 * np.pi / 4},  # Bottom-left corner
    "br": {"distance": 3.0, "angle": -np.pi / 4},  # Bottom-right corner
    "fr": {"distance": 3.0, "angle": np.pi / 4},  # Top-right corner
    "fl": {"distance": 3.0, "angle": 3 * np.pi / 4},  # Top-left corner
    "fm": {"distance": 2.7, "angle": np.pi / 2},  # Middle-front
    "sl": {"distance": 2.55, "angle": np.pi},  # Left side middle
}

# Recalculate camera positions
cameras = {
    cam_name: {
        "pos": (
            middle_of_room[0] + cam_data["distance"] * np.cos(cam_data["angle"]),
            middle_of_room[1] + cam_data["distance"] * np.sin(cam_data["angle"])
        ),
        "metric": f"m{idx + 1}: {0.85 - 0.01 * idx:.2f}"
    }
    for idx, (cam_name, cam_data) in enumerate(camera_positions.items())
}

# Create figure
fig, ax = plt.subplots(figsize=(8, 8), dpi=300, facecolor='white')

# Add grid
ax.grid(True, linestyle='--', alpha=0.6, zorder=0)

# Plot room boundary
ax.set_xlim(0, room_width)
plt.xticks(fontsize=16)
ax.set_ylim(0, room_height)
plt.yticks(fontsize=16)
ax.set_aspect('equal', adjustable='box')
ax.plot([0, room_width, room_width, 0, 0],
        [0, 0, room_height, room_height, 0],
        color='black')


# Add a square in the middle of the room
middle_square = patches.Rectangle(square_bottom_left, square_size, square_size, color='lightblue', alpha=0.7, zorder=1)
ax.add_patch(middle_square)

# Add a rectangle at the front of the room
front_rect_width = room_width * 0.4  # 80% of the room width
front_rect_height = 0.1  # Height of the rectangle
front_rect_bottom_left = ((room_width - front_rect_width) / 2, room_height - front_rect_height)
front_rectangle = patches.Rectangle(front_rect_bottom_left, front_rect_width, front_rect_height,
                                    color='blue', alpha=0.8, zorder=4)
ax.add_patch(front_rectangle)
ax.text(room_width / 2, room_height - front_rect_height / 2, '',
        fontsize=10, ha='center', va='center', color='white', zorder=2)

# Add cameras and their labels
for cam_name, cam_data in cameras.items():
    x, y = cam_data["pos"]
    # Calculate angle towards the center
    angle = np.arctan2(middle_of_room[1] - y, middle_of_room[0] - x) + np.pi / 2
    # Rotate triangle so a flat side faces the center
    camera_icon = patches.RegularPolygon((x, y), numVertices=3, radius=0.2,
                                         orientation=angle, color='black', zorder=3)
    ax.add_patch(camera_icon)
    ax.text(x + 0.03, y - 0.4, f"{cam_name}", fontsize=18, weight='bold', color='black', ha='center', zorder=3)

# Add labels and title
ax.set_xlabel("Anteroposterior Length (m)", zorder=4, fontsize=16)
ax.set_ylabel("Mediolateral Length (m)", zorder=4, fontsize=16)

# Show plot
plt.show()
