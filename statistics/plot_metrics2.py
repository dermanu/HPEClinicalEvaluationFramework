import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the necessary data (adjust the file paths as necessary)
with open('multi/all_metrics_single.pkl', 'rb') as f:
    all_metrics_single = pickle.load(f)

with open('multi/keypoints_metrics.pkl', 'rb') as f:
    keypoints_metrics = pickle.load(f)

with open('multi/p_values.pkl', 'rb') as f:
    p_values = pickle.load(f)


############################################
## 1. Comparison of Joint-Specific Errors ##
############################################
## 1.1 Grouped Boxplot, for each joint (make left-right difference visible)
def plot_boxplot_joint_errors(all_metrics_single):
    data = []
    for condition, metrics in all_metrics_single.items():
        for joint, values in metrics['joint_errors'].items():
            for value in values:
                data.append(
                    {'Condition': condition, 'Joint': joint, 'Error': value[0]})  # Assuming the first element is error

    df = pd.DataFrame(data)

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Joint', y='Error', hue='Condition')
    plt.title('Joint-Specific Error Comparison')
    plt.ylabel('Error')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

## 1.2 Skeleton Diagram with Keypoints Color-Coded
def plot_skeleton_diagram(keypoints_metrics, condition_to_plot='Condition1'):
    keypoint_coords = {
        'head': (0, 10), 'neck': (0, 8),
        'left_shoulder': (-2, 8), 'right_shoulder': (2, 8),
        'left_elbow': (-3, 6), 'right_elbow': (3, 6),
        'left_hand': (-4, 4), 'right_hand': (4, 4),
        'torso': (0, 6), 'left_hip': (-1, 4), 'right_hip': (1, 4),
        'left_knee': (-1, 2), 'right_knee': (1, 2),
        'left_foot': (-1, 0), 'right_foot': (1, 0),
    }

    keypoints = list(keypoints_metrics[condition_to_plot].keys())
    metric_values = [keypoints_metrics[condition_to_plot][kp]['pmpjpe_m'] for kp in keypoints]

    norm = plt.Normalize(min(metric_values), max(metric_values))
    colors = plt.cm.viridis(norm(metric_values))

    plt.figure(figsize=(8, 8))
    for i, kp in enumerate(keypoints):
        x, y = keypoint_coords[kp]
        plt.scatter(x, y, color=colors[i], s=100)
        plt.text(x + 0.1, y, kp, fontsize=9, ha='left', va='center')

    skeleton_edges = [('head', 'neck'), ('neck', 'left_shoulder'), ('neck', 'right_shoulder'),
                      ('left_shoulder', 'left_elbow'), ('right_shoulder', 'right_elbow'),
                      ('left_elbow', 'left_hand'), ('right_elbow', 'right_hand'),
                      ('neck', 'torso'), ('torso', 'left_hip'), ('torso', 'right_hip'),
                      ('left_hip', 'left_knee'), ('right_hip', 'right_knee'),
                      ('left_knee', 'left_foot'), ('right_knee', 'right_foot')]

    for edge in skeleton_edges:
        kp1, kp2 = edge
        x1, y1 = keypoint_coords[kp1]
        x2, y2 = keypoint_coords[kp2]
        plt.plot([x1, x2], [y1, y2], 'k-', lw=2)

    plt.title(f'Per-Keypoint PMPJPE Mean for {condition_to_plot}')
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), label='PMPJPE Mean')
    plt.axis('off')
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
plot_boxplot_joint_errors(all_metrics_single)
plot_skeleton_diagram(keypoints_metrics)
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
plot_boxplot_joint_errors(all_metrics_single)
plot_skeleton_diagram(keypoints_metrics)
plot_joint_center_similarity(all_metrics_single)
plot_effect_size_forest(p_values)
plot_boxplot_augmentations(all_metrics_single)
plot_heatmap_augmentations(keypoints_metrics)
plot_boxplot_movement_types(all_metrics_single)
plot_heatmap_movement_types(keypoints_metrics)
plot_effect_size_forest_movement_types(p_values)