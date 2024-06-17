import plotly.graph_objects as go
import wandb
import numpy as np

def plot_3d_keypoints(keypoints, model_name, wandb_name, epoch):
    # Extract X, Y, and Z coordinates from keypoints
    x, z, y = zip(*keypoints)

    # Define connections between related keypoints
    if model_name == 'mediapipe':
        connections = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (6, 7), (0, 6),
                        (1, 7), (6, 8), (7, 9), (8, 10), (9, 11), (10, 12),
                        (11, 13), (10, 14), (11, 15), (12, 14), (13, 15)]

    # Create a Plotly 3D scatter plot
    fig = go.Figure()

    # Scatter plot for keypoints
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(color='red', size=5)))

    # Plot connections
    for connection in connections:
        x_vals = [x[connection[0]], x[connection[1]]]
        y_vals = [y[connection[0]], y[connection[1]]]
        z_vals = [z[connection[0]], z[connection[1]]]
        fig.add_trace(go.Scatter3d(x=x_vals, y=y_vals, z=z_vals, mode='lines', line=dict(color='blue')))

    # Combine x, y, z values into a single list
    all_values = x + y + z

    # Find the minimum and maximum values
    min_value = min(all_values) - abs(min(all_values) * 0.1)
    max_value = max(all_values) + abs(max(all_values) * 0.1)

    # Update layout to set axis limits
    fig.update_layout(
        scene=dict(
            aspectmode='cube',
            xaxis=dict(title='X', range=[min_value, max_value]),
            yaxis=dict(title='Y', range=[min_value, max_value]),
            zaxis=dict(title='Z', range=[min_value, max_value])
        )
    )

    # Log the 3D scatter plot using WandB
    if wandb_name == 'morphed':
        wandb.log({'Model Output: Morphed Keypoints': fig, "epoch": epoch+1})
    elif wandb_name == 'ground_truth':
        wandb.log({'Ground Truth: Vizlab Dataset Keypoints': fig, "epoch": epoch+1})
    elif wandb_name == 'hpe_truth':
        wandb.log({'Model Input: HPE Keypoints': fig, "epoch": epoch+1})
    else:
        raise ValueError(f"Invalid wandb_name: {wandb_name}")

def plot_3d_keypoints_all(keypoints_morphed, keypoints_ground_truth, keypoints_hpe_truth, model_name, epoch):
    colors = ['red', 'green', 'blue']
    names = ['Morphed KeyPoints', 'Ground Truth KeyPoints', 'HPE Truth KeyPoints']
    # Define connections between related keypoints
    if model_name == 'mediapipe':
        connections = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (6, 7), (0, 6),
                        (1, 7), (6, 8), (7, 9), (8, 10), (9, 11), (10, 12),
                        (11, 13), (10, 14), (11, 15), (12, 14), (13, 15)]

    # Create a Plotly 3D scatter plot
    fig = go.Figure()

    all_values = ()
    idx = 0
    for keypoints in [keypoints_morphed, keypoints_ground_truth, keypoints_hpe_truth]:
        # Extract X, Y, and Z coordinates from keypoints
        x, z, y = zip(*keypoints)

        # Scatter plot for keypoints
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(color=colors[idx], size=5)))

        # Plot connections
        for connection in connections:
            x_vals = [x[connection[0]], x[connection[1]]]
            y_vals = [y[connection[0]], y[connection[1]]]
            z_vals = [z[connection[0]], z[connection[1]]]
            fig.add_trace(go.Scatter3d(x=x_vals, y=y_vals, z=z_vals, mode='lines', line=dict(color=colors[idx])))

        idx += 1

        # Combine x, y, z values into a single list
        all_values = all_values + x + y + z

    # Find the minimum and maximum values
    min_value = min(all_values) - abs(min(all_values) * 0.1)
    max_value = max(all_values) + abs(max(all_values) * 0.1)

    # Update layout to set axis limits
    fig.update_layout(
        scene=dict(
            aspectmode='cube',
            xaxis=dict(title='X', range=[min_value, max_value]),
            yaxis=dict(title='Y', range=[min_value, max_value]),
            zaxis=dict(title='Z', range=[min_value, max_value])
        )
    )

    # Log the 3D scatter plot using WandB
    wandb.log({"3D Keypoints Comparison": fig, "epoch": epoch+1})


def plot_3d_keypoints_validation(keypoints_ground_truth, keypoints_hpe_truth, model_name):
    colors = ['red', 'blue']
    names = ['Morphed KeyPoints', 'Ground Truth KeyPoints', 'HPE Truth KeyPoints']

    keypoints_ground_truth = np.squeeze(np.array(list(keypoints_ground_truth[0].values()))[:, 0, :])
    keypoints_hpe_truth = np.squeeze(np.array(list(keypoints_hpe_truth[0].values()))[:, 0, :])

    # Define connections between related keypoints
    if model_name == 'mediapipe':
        connections = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (6, 7), (0, 6),
                        (1, 7), (6, 8), (7, 9), (8, 10), (9, 11), (10, 12),
                        (11, 13), (10, 14), (11, 15), (12, 14), (13, 15)]

    # Create a Plotly 3D scatter plot
    fig = go.Figure()

    all_values = ()
    idx = 0
    for keypoints in [keypoints_ground_truth, keypoints_hpe_truth]:
        # Extract X, Y, and Z coordinates from keypoints
        x, z, y = zip(*keypoints)

        # Scatter plot for keypoints
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(color=colors[idx], size=5)))

        # Plot connections
        for connection in connections:
            x_vals = [x[connection[0]], x[connection[1]]]
            y_vals = [y[connection[0]], y[connection[1]]]
            z_vals = [z[connection[0]], z[connection[1]]]
            fig.add_trace(go.Scatter3d(x=x_vals, y=y_vals, z=z_vals, mode='lines', line=dict(color=colors[idx])))

        idx += 1

        # Combine x, y, z values into a single list
        all_values = all_values + x + y + z

    # Find the minimum and maximum values
    min_value = min(all_values) - abs(min(all_values) * 0.1)
    max_value = max(all_values) + abs(max(all_values) * 0.1)

    # Update layout to set axis limits
    fig.update_layout(
        scene=dict(
            aspectmode='cube',
            xaxis=dict(title='X', range=[min_value, max_value]),
            yaxis=dict(title='Y', range=[min_value, max_value]),
            zaxis=dict(title='Z', range=[min_value, max_value])
        )
    )

    # Log the 3D scatter plot using WandB
    wandb.log({"3D Keypoints Comparison": fig})


def plot_3d_keypoints_gt_pred_single_frame(keypoints_gt, keypoints_pred, model_name, frame_idx=0):
    colors = ['green', 'red']
    names = ['Ground Truth KeyPoints', 'HPE Pred KeyPoints']

    # Define connections between related keypoints for MediaPipe
    if model_name == 'mediapipe':
        connections = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (6, 7), (0, 6),
                       (1, 7), (6, 8), (7, 9), (8, 10), (9, 11), (10, 12),
                       (11, 13), (10, 14), (11, 15), (12, 14), (13, 15)]

    # Create a Plotly 3D scatter plot
    fig = go.Figure()

    all_values = []
    idx = 0

    for keypoints_list, color in zip([keypoints_gt, keypoints_pred], colors):
        # Extract the specified frame from keypoints in all dictionaries
        frame_keypoints = []
        for keypoints in keypoints_list:
            frame_points = np.array([keypoints[key][frame_idx] for key in keypoints.keys()])
            frame_keypoints.append(frame_points)

        concatenated_keypoints = np.concatenate(frame_keypoints, axis=0)

        # Extract X, Y, and Z coordinates from keypoints
        x, y, z = concatenated_keypoints[:, 0], concatenated_keypoints[:, 1], concatenated_keypoints[:, 2]

        # Scatter plot for keypoints
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(color=color, size=5), name=names[idx]))

        # Plot connections
        for connection in connections:
            x_vals = [x[connection[0]], x[connection[1]]]
            y_vals = [y[connection[0]], y[connection[1]]]
            z_vals = [z[connection[0]], z[connection[1]]]
            fig.add_trace(go.Scatter3d(x=x_vals, y=y_vals, z=z_vals, mode='lines', line=dict(color=color)))

        idx += 1

        # Combine x, y, z values into a single list
        all_values.extend(x)
        all_values.extend(y)
        all_values.extend(z)

    # Find the minimum and maximum values
    min_value = min(all_values) - abs(min(all_values) * 0.1)
    max_value = max(all_values) + abs(max(all_values) * 0.1)

    # Update layout to set axis limits
    fig.update_layout(
        scene=dict(
            aspectmode='cube',
            xaxis=dict(title='X', range=[min_value, max_value]),
            yaxis=dict(title='Y', range=[min_value, max_value]),
            zaxis=dict(title='Z', range=[0, max_value])
        )
    )

    # Log the 3D scatter plot using WandB
    wandb.log({"3D Keypoints Comparison Single Frame": fig})
