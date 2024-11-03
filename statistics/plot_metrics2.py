import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from setuptools.command.rotate import rotate

# Load the necessary data (adjust the file paths as necessary)
with open('multi/all_metrics_single.pkl', 'rb') as f:
    all_metrics_single = pickle.load(f)

with open('multi/keypoints_metrics.pkl', 'rb') as f:
    keypoints_metrics = pickle.load(f)

with open('multi/p_values.pkl', 'rb') as f:
    p_values = pickle.load(f)

with open('multi/angle_errors_metrics.pkl', 'rb') as f:
    angle_errors_metrics = pickle.load(f)


# Preparing a dictionary to store data for each augmentation
metrics_dict = {}

for condition, metrics in all_metrics_single.items():
    # Extract the augmentation name
    augmentation = condition  # This assumes `condition` is the augmentation name

    # Create a dictionary for this specific augmentation
    augmentation_metrics = {}

    # Extract 'pmpjpe' (mean) values
    augmentation_metrics['pmpjpe'] = [pmpjpe_value[0] for pmpjpe_value in metrics['pmpjpe']]

    # Extract 'angular' (angle metrics)
    augmentation_metrics['angular'] = [angular_value for angular_value in metrics['angle']]

    # Extract 'velocity' (mean) values
    augmentation_metrics['velocity'] = [velocity_value[0] for velocity_value in metrics['velocity']]

    # Extract 'pcc' values
    augmentation_metrics['pcc'] = [pcc_value for pcc_value in metrics['pcc']]

    # Add this augmentation's metrics to the main dictionary
    metrics_dict[augmentation] = augmentation_metrics

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Predefine custom y-axis labels for each metric
y_labels = {
    'pmpjpe': 'Overall PMPJE in [mm]',
    'angular': 'Overall joint angle in [°]',
    'velocity': 'Overall joint speed in [m/s]',
    'pcc': 'Overall PCC'
}

# Define the custom order for augmentations (using original names)
augmentation_order = [
    'background', 'defocus', 'occlusion', 'underexposure', 'desynchronize', 'decalibration',
    'cameras_4_0', 'cameras_4_2', 'cameras_4_3', 'cameras_5_1', 'cameras_0_4_3', 'cameras_5_4_1', 'cameras_0_4_3_2',
    'upper', 'lower', 'complex', 'sitting'
]

# Define a mapping from the original names to the display names
augmentation_display_names = {
    'background': 'Background',
    'defocus': 'Defocus',
    'occlusion': 'Occlusion',
    'underexposure': 'Underexposure',
    'desynchronize': 'Desynchronized',
    'decalibration': 'Decalibrated',
    'cameras_4_0': 'Camera 0, 4',
    'cameras_4_2': 'Camera 2, 4',
    'cameras_4_3': 'Camera 3, 4',
    'cameras_5_1': 'Camera 1, 5',
    'cameras_0_4_3': 'Camera 0, 3, 4',
    'cameras_5_4_1': 'Camera 1, 4, 5',
    'cameras_0_4_3_2': 'Camera 0, 2, 3, 4',
    'upper': 'Upper',
    'lower': 'Lower',
    'complex': 'Complex',
    'sitting': 'Sitting'
}

# Create a dictionary to store DataFrames for each metric
metric_dfs = {}

# Prepare separate DataFrames for each metric
for metric_name in ['pmpjpe', 'angular', 'velocity', 'pcc']:
    augmentation_list = []
    value_list = []

    # Loop through the dictionary and unpack data into lists
    for augmentation, metrics in metrics_dict.items():
        if augmentation != 'none':
            values = metrics[metric_name]
            augmentation_list.extend([augmentation_display_names[augmentation]] * len(values))  # Apply display names
            value_list.extend(values)

    # Create a DataFrame for each metric and set augmentation as a categorical variable with a specific order
    metric_dfs[metric_name] = pd.DataFrame({
        'augmentation': pd.Categorical(augmentation_list,
                                       categories=[augmentation_display_names[aug] for aug in augmentation_order],
                                       ordered=True),
        'value': value_list
    })

# Define a color palette for the plots
palette = sns.color_palette("Set2")

# Plot separate boxplots for each metric with enhanced readability
metrics_to_plot = ['pmpjpe', 'angular', 'velocity', 'pcc']
f, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))

for i, metric in enumerate(metrics_to_plot):
    row, col = divmod(i, 2)
    # Create the boxplot with the specified order
    sn = sns.boxplot(
            x='augmentation',
            y='value',
            data=metric_dfs[metric],
            palette=palette,
            showfliers=False,
            order=[augmentation_display_names[aug] for aug in augmentation_order],
            ax = axes[row, col]
    )


    # Set the title and labels on the correct axis
    axes[row, col].set_title(metric.capitalize(), fontsize=16)
    axes[row, col].set_ylabel(y_labels[metric], fontsize=14)  # Use predefined y-axis labels
    axes[row, col].set_xlabel('', fontsize=14)  # Use predefined y-axis labels
    # Set the x-ticks to the augmentation names explicitly
    #sn.set_xticks(range(len(augmentation_order)), labels = [augmentation_display_names[aug] for aug in augmentation_order], rotation=45, fontsize=12)

    # Rotate x-tick labels for better readability
    axes[row, col].tick_params(axis='x', rotation=45, labelsize=12)


    # Tighten layout if needed
    plt.tight_layout()

# Show all plots
plt.show()

# Select the condition
condition = 'cameras_5'  # Replace with your condition

# Get the angular errors for the selected condition
angle_error_data = angle_errors_metrics[condition]
angle_means = angle_error_data['all_angle_error_m']
angle_stds = angle_error_data['all_angle_error_s']

# Define joint positions
joints = {
    'head': (0, 10),
    'neck': (0, 8),
    'left_shoulder': (-2, 8),
    'right_shoulder': (2, 8),
    'left_elbow': (-3, 6),
    'right_elbow': (3, 6),
    'left_wrist': (-4, 4),
    'right_wrist': (4, 4),
    'spine': (0, 6),
    'left_hip': (-1, 4),
    'right_hip': (1, 4),
    'left_knee': (-1, 2),
    'right_knee': (1, 2),
    'left_ankle': (-1, 0),
    'right_ankle': (1, 0),
}

# Define bones
bones = [
    ('head', 'neck'),
    ('neck', 'left_shoulder'),
    ('neck', 'right_shoulder'),
    ('left_shoulder', 'left_elbow'),
    ('right_shoulder', 'right_elbow'),
    ('left_elbow', 'left_wrist'),
    ('right_elbow', 'right_wrist'),
    ('neck', 'spine'),
    ('spine', 'left_hip'),
    ('spine', 'right_hip'),
    ('left_hip', 'left_knee'),
    ('right_hip', 'right_knee'),
    ('left_knee', 'left_ankle'),
    ('right_knee', 'right_ankle'),
]

# Map angles to joints
angle_to_joint_map = {
    'left_shoulder_angles': 'left_shoulder',
    'right_shoulder_angles': 'right_shoulder',
    'left_elbow_angles': 'left_elbow',
    'right_elbow_angles': 'right_elbow',
    'left_hip_angles': 'left_hip',
    'right_hip_angles': 'right_hip',
    'left_knee_angles': 'left_knee',
    'right_knee_angles': 'right_knee',
}

# Create the plot
fig, ax = plt.subplots(figsize=(8, 12))

# Plot bones
for bone in bones:
    joint1, joint2 = bone
    x_values = [joints[joint1][0], joints[joint2][0]]
    y_values = [joints[joint1][1], joints[joint2][1]]
    ax.plot(x_values, y_values, 'k-', linewidth=2)

# Plot joints
for joint_name, (x, y) in joints.items():
    ax.plot(x, y, 'ko', markersize=8)

# Annotate joints with angular errors
for angle_name, joint_name in angle_to_joint_map.items():
    if angle_name in angle_means:
        mean_error = angle_means[angle_name]
        std_error = angle_stds[angle_name]
        x, y = joints[joint_name]
        ax.text(x + 0.3, y, f"{mean_error:.2f}° ({std_error:.2f}°)", fontsize=10, color='blue')

# Adjust plot aesthetics
ax.set_aspect('equal')
ax.set_xlim(-5, 5)
ax.set_ylim(-1, 11)
ax.axis('off')
ax.set_title(f"Mean (Std) Angular Errors for Condition: {condition}", fontsize=14)

# Show the plot
plt.tight_layout()
plt.show()




#########################################################
## 2. Similarity of Joint Center Position Measurements ##
#########################################################
## 2.1 Figure with exemplary movement of x,y, and z axes for ground truth and prediction
def plot_joint_center_similarity(all_metrics_single):
    for condition, metrics in all_metrics_single.items():
        plt.figure(figsize=(10, 6))
        for axis in ['x', 'y', 'z']:
            # Assuming 'joint_position_gt' and 'joint_position_pred' contain ground truth and prediction
            gt = np.array([pos[axis] for pos in metrics['joint_position_gt']])
            pred = np.array([pos[axis] for pos in metrics['joint_position_pred']])

            plt.plot(gt, label=f'{axis.upper()} Ground Truth', linestyle='--')
            plt.plot(pred, label=f'{axis.upper()} Prediction')

        plt.title(f'Joint Center Position for Condition {condition}')
        plt.xlabel('Frames')
        plt.ylabel('Position')
        plt.legend()
        plt.tight_layout()
        plt.show()

## 2.2 Effect Size Forest Plot
def plot_effect_size_forest(p_values):
    data = []
    for metric, tests in p_values.items():
        for test in tests:
            data.append({
                'Metric': metric,
                'Condition': test['augmentation'],
                'Effect Size': test['effect_size'],
                'CI Lower': test['effect_size_ci'][0],
                'CI Upper': test['effect_size_ci'][1],
                'Significant': test['Significant']
            })

    df = pd.DataFrame(data)
    df = df.sort_values(by='Effect Size')

    plt.figure(figsize=(10, 6))
    for i, row in df.iterrows():
        plt.plot([row['CI Lower'], row['CI Upper']], [i, i], 'k-', lw=2)
        plt.scatter(row['Effect Size'], i, color='red' if row['Significant'] else 'blue')

    plt.yticks(range(len(df)), df['Condition'] + ' (' + df['Metric'] + ')')
    plt.axvline(0, color='grey', linestyle='--')
    plt.xlabel('Effect Size (Cohen\'s d)')
    plt.title('Effect Sizes with Confidence Intervals')
    plt.tight_layout()
    plt.show()


##########################################################
## 3. Relevance of Setup Errors on Measurement Accuracy ##
##########################################################
## 3.1 Boxplot on movement error for different augmentations (grouped for algorithms)
def plot_boxplot_augmentations(all_metrics_single):
    data = []
    for condition, metrics in all_metrics_single.items():
        for aug, errors in metrics['augmentation_errors'].items():
            for error in errors:
                data.append({'Condition': condition, 'Augmentation': aug, 'Error': error})

    df = pd.DataFrame(data)

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Augmentation', y='Error', hue='Condition')
    plt.title('Movement Error Across Augmentations')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

## 3.2 Heatmap for per-keypoint PMPJPE, angular error, and velocity error
def plot_heatmap_augmentations(keypoints_metrics, metric='pmpjpe_m'):
    data = []
    for condition, keypoints in keypoints_metrics.items():
        for keypoint, metrics in keypoints.items():
            data.append({
                'Condition': condition,
                'Keypoint': keypoint,
                metric.capitalize(): metrics[metric]
            })

    df = pd.DataFrame(data)
    pivot_df = df.pivot(index='Keypoint', columns='Condition', values=metric.capitalize())

    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title(f'Per-Keypoint {metric.upper()} Across Conditions')
    plt.tight_layout()
    plt.show()

## 3.3 Effect Size Forest Plot


###################################################################
## 4. Relevance of Camera Placement on Pose Estimation Precision ##
###################################################################
## 4.1 Boxplot on the difference of camera placement (grouped for algorithms)
def plot_boxplot_camera_placement(all_metrics_single):
    # Similar structure as boxplot for augmentations
    plot_boxplot_augmentations(all_metrics_single)

## 4.2 Heatmap for per-keypoint PMPJPE, angular error, and velocity error
def plot_heatmap_camera_placement(keypoints_metrics):
    plot_heatmap_augmentations(keypoints_metrics, metric='pmpjpe_m')

# Running the plotting functions
plot_joint_center_similarity(all_metrics_single)
plot_effect_size_forest(p_values)
plot_boxplot_augmentations(all_metrics_single)
plot_heatmap_augmentations(keypoints_metrics)

## 4.3 Effect Size Forest Plot


#####################################################################
## 5. Analysis of Different Movement Types on Measurement Accuracy ##
#####################################################################
## 5.1 Boxplot on the difference of camera placement (grouped for algorithms)
def plot_boxplot_movement_types(all_metrics_single):
    data = []
    for condition, metrics in all_metrics_single.items():
        for movement, errors in metrics['movement_errors'].items():
            for error in errors:
                data.append({'Condition': condition, 'Movement': movement, 'Error': error})

    df = pd.DataFrame(data)

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Movement', y='Error', hue='Condition')
    plt.title('Movement Error Across Movement Types')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

## 5.2 Heatmap for per-keypoint PMPJPE, angular error, and velocity error
def plot_heatmap_movement_types(keypoints_metrics, metric='pmpjpe_m'):
    data = []
    for condition, keypoints in keypoints_metrics.items():
        for keypoint, metrics in keypoints.items():
            data.append({
                'Condition': condition,
                'Keypoint': keypoint,
                metric.capitalize(): metrics[metric]
            })

    df = pd.DataFrame(data)
    pivot_df = df.pivot(index='Keypoint', columns='Condition', values=metric.capitalize())

    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title(f'Per-Keypoint {metric.upper()} for Different Movements')
    plt.tight_layout()
    plt.show()

## 5.3 Effect Size Forest Plot
def plot_effect_size_forest_movement_types(p_values):
    data = []
    for movement_type, tests in p_values.items():
        for test in tests:
            data.append({
                'Movement Type': movement_type,
                'Condition': test['augmentation'],
                'Effect Size': test['effect_size'],
                'CI Lower': test['effect_size_ci'][0],
                'CI Upper': test['effect_size_ci'][1],
                'Significant': test['Significant']
            })

    df = pd.DataFrame(data)
    df = df.sort_values(by='Effect Size')

    plt.figure(figsize=(10, 6))
    for i, row in df.iterrows():
        plt.plot([row['CI Lower'], row['CI Upper']], [i, i], 'k-', lw=2)
        plt.scatter(row['Effect Size'], i, color='red' if row['Significant'] else 'blue')

    plt.yticks(range(len(df)), df['Movement Type'] + ' (' + df['Condition'] + ')')
    plt.axvline(0, color='grey', linestyle='--')
    plt.xlabel('Effect Size (Cohen\'s d)')
    plt.title('Effect Sizes with Confidence Intervals for Movement Types')
    plt.tight_layout()
    plt.show()


# Running the plotting functions

plot_joint_center_similarity(all_metrics_single)
plot_effect_size_forest(p_values)
plot_boxplot_augmentations(all_metrics_single)
plot_heatmap_augmentations(keypoints_metrics)
plot_boxplot_movement_types(all_metrics_single)
plot_heatmap_movement_types(keypoints_metrics)
plot_effect_size_forest_movement_types(p_values)