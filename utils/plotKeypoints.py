import plotly.graph_objects as go
import wandb
import numpy as np
def plot_3d_keypoints(keypoints, model_name, wandb_name, epoch):

    if wandb_name == 'morphed':
        color = "red"
        name = "Morphed KeyPoints"
        # wandb.log({'Test': fig})
    elif wandb_name == 'ground_truth':
        color = "green"
        name = "Ground Truth KeyPoints"

    elif wandb_name == 'hpe_truth':
        color = "blue"
        name = "HPE Truth KeyPoints"

    else:
        raise Exception("Invalid wandb name")



    colors = ['red', 'green', 'blue']
    # Define connections between related keypoints
    if model_name == 'mediapipe':
        connections = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (6, 7), (0, 6),
                        (1, 7), (6, 8), (7, 9), (8, 10), (9, 11), (10, 12),
                        (11, 13), (10, 14), (11, 15), (12, 14), (13, 15)]

    # Create a Plotly 3D scatter plot
    fig = go.Figure()

    # Extract X, Y, and Z coordinates from keypoints
    x, z, y = zip(*keypoints)

    # Scatter plot for keypoints
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(color=color, size=5)))

    # Plot connections
    for connection in connections:
        x_vals = [x[connection[0]], x[connection[1]]]
        y_vals = [y[connection[0]], y[connection[1]]]
        z_vals = [z[connection[0]], z[connection[1]]]
        fig.add_trace(go.Scatter3d(x=x_vals, y=y_vals, z=z_vals, mode='lines', line=dict(color=color)))

    # Combine x, y, z values into a single list
    #all_values = x + y + z

    # Find the minimum and maximum values
    min_value = np.min(keypoints) - abs(np.min(keypoints) * 0.1)
    max_value = np.max(keypoints) + abs(np.max(keypoints) * 0.1)

    # Update layout to set axis limits
    fig.update_layout(
        scene=dict(
            aspectmode='cube',
            xaxis=dict(title='X', range=[min_value, max_value]),
            yaxis=dict(title='Y', range=[min_value, max_value]),
            zaxis=dict(title='Z', range=[min_value, max_value])
        )
    )

    wandb.log({name: fig, "epoch": epoch+1})

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
        trace_name = 'Keypoints ' + names[idx]
        # Extract X, Y, and Z coordinates from keypoints
        x, z, y = zip(*keypoints)

        # Scatter plot for keypoints
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(color=colors[idx], size=5), name = trace_name))

        # Plot connections
        for connection in connections:
            x_vals = [x[connection[0]], x[connection[1]]]
            y_vals = [y[connection[0]], y[connection[1]]]
            z_vals = [z[connection[0]], z[connection[1]]]
            fig.add_trace(go.Scatter3d(x=x_vals, y=y_vals, z=z_vals, mode='lines', line=dict(color=colors[idx]), name = trace_name))
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


def plot_3d_keypoints_gt_pred(keypoints_gt, keypoints_pred, model_name):
    colors = ['green', 'red']
    names = ['Ground Truth KeyPoints', 'HPE Pred KeyPoints']
    # Define connections between related keypoints
    if model_name == 'mediapipe':
        connections = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (6, 7), (0, 6),
                        (1, 7), (6, 8), (7, 9), (8, 10), (9, 11), (10, 12),
                        (11, 13), (10, 14), (11, 15), (12, 14), (13, 15)]

    # Create a Plotly 3D scatter plot
    fig = go.Figure()

    all_values = ()
    idx = 0
    for keypoints in [keypoints_gt, keypoints_pred]:
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
            zaxis=dict(title='Z', range=[0, max_value])
        )
    )

    # Log the 3D scatter plot using WandB
    wandb.log({"3D Keypoints Comparison": fig})
