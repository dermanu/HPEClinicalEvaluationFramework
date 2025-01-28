import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the necessary data
with open('../statistics/multi/all_metrics_single.pkl', 'rb') as f:
    all_metrics_single = pickle.load(f)

with open('../statistics/multi/p_values.pkl', 'rb') as f:
    p_values = pickle.load(f)

with open('../statistics/multi/keypoints_metrics.pkl', 'rb') as f:
    keypoints_metrics = pickle.load(f)


# 1. Box Plot for PMPJPE
def plot_boxplot_pmpjpe(all_metrics_single):
    data = []
    for condition, metrics in all_metrics_single.items():
        pmpjpe_values = metrics['pmpjpe']  # Assuming 'pmpjpe' is stored as a list of tuples
        for value in pmpjpe_values:
            data.append({'Condition': condition, 'PMPJPE Mean': value[0]})  # Use mean from tuple

    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Condition', y='PMPJPE Mean')
    plt.title('PMPJPE Distribution Across Conditions')
    plt.ylabel('PMPJPE Mean')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('boxplot_pmpjpe.png')
    plt.show()


# 2. Bar Plot with Error Bars for other metrics
def plot_barplot_metrics(all_metrics_single):
    data = []
    metrics_to_plot = ['pmpjpe_m', 'angle_m', 'velocity_m', 'pcc']

    for condition, metrics in all_metrics_single.items():
        for metric in metrics_to_plot:
            if metric in metrics:
                mean_value = np.mean([x[0] for x in metrics[metric]])  # Extract mean
                std_value = np.std([x[0] for x in metrics[metric]])  # Extract std
                data.append({'Condition': condition, 'Metric': metric, 'Mean': mean_value, 'Std': std_value})

    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Metric', y='Mean', hue='Condition', capsize=.1, ci=None)
    for i in range(df.shape[0]):
        plt.errorbar(x=i, y=df.iloc[i]['Mean'], yerr=df.iloc[i]['Std'], fmt='none', c='black', capsize=5)
    plt.title('Comparison of Metrics Across Conditions with Std Error')
    plt.ylabel('Mean Value')
    plt.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('barplot_metrics.png')
    plt.show()


# 3. Heatmap for per-keypoint PMPJPE
def plot_heatmap_keypoints(keypoints_metrics):
    data = []
    for condition, keypoints in keypoints_metrics.items():
        for keypoint, metrics in keypoints.items():
            data.append({
                'Condition': condition,
                'Keypoint': keypoint,
                'PMPJPE Mean': metrics['pmpjpe_m']  # Assuming 'pmpjpe_m' exists
            })

    df = pd.DataFrame(data)
    pivot_df = df.pivot(index='Keypoint', columns='Condition', values='PMPJPE Mean')

    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap='viridis')
    plt.title('Per-Keypoint PMPJPE Mean Across Conditions')
    plt.ylabel('Keypoint')
    plt.xlabel('Condition')
    plt.tight_layout()
    plt.savefig('heatmap_keypoints.png')
    plt.show()


# 4. Skeleton Diagram with Keypoints Color-Coded
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
    plt.savefig('skeleton_diagram.png')
    plt.show()


# 5. Effect Size Forest Plot
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
    plt.savefig('effect_size_forest.png')
    plt.show()


# Running the plotting functions
plot_boxplot_pmpjpe(all_metrics_single)
plot_barplot_metrics(all_metrics_single)
plot_heatmap_keypoints(keypoints_metrics)
plot_skeleton_diagram(keypoints_metrics, condition_to_plot='Condition1')
plot_effect_size_forest(p_values)
