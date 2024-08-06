# Plot example movement (ground truth and HPE results) and adding pcc and pvalue in the legend for each augmentation
# Plot example movement (ground truth and HPE results) and adding pcc and pvalue in the legend for each movement type
# Plot boxplot (mean and std) for angular error overall, inference, r2_overall, pmpjpe overall, velocity overall for each augmentation
# Plot boxplot (mean and std) for angular error overall, inference, r2_overall, pmpjpe overall, velocity overall for each movement type
# Heatmap of different correlations between different augmentations
# Example images of all augmentations
# Example of morphing (before and after)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the structured data from the uploaded file
structured_df = pd.read_csv('results/structured_project.csv')


# Function to plot example movements with PCC and p-value
def plot_example_movement(df, condition_col, metric_col, title, output_file):
    plt.figure(figsize=(12, 6))
    conditions = df[condition_col].unique()

    for condition in conditions:
        condition_data = df[df[condition_col] == condition]
        plt.plot(condition_data[metric_col],
                 label=f"{condition} (PCC: {condition_data['pcc'].iloc[0]:.2f}, p-value: {condition_data['pvalue'].iloc[0]:.2f})")

    plt.title(title)
    plt.xlabel("Sample Number")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.savefig(output_file)
    plt.close()


# Function to plot boxplots for specified metrics
def plot_boxplot(df, metrics, group_col, title, output_file):
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 8))

    for i, metric in enumerate(metrics):
        sns.boxplot(x=group_col, y=metric, data=df, ax=axes[i])
        axes[i].set_title(f"{metric} by {group_col}")

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_file)
    plt.close()


# Function to plot heatmap of correlations
def plot_correlation_heatmap(df, metrics, title, output_file):
    corr_matrix = df[metrics].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(title)
    plt.savefig(output_file)
    plt.close()


# Extract unique metrics for boxplots and heatmaps
unique_metrics = ['angular_m_ankle_angle_r', 'pmpjpe_m_joint_right_elbow', 'velocity_m_joint_left_ankle']

# Plotting example movements for augmentations
#plot_example_movement(
#    df=structured_df,
#    condition_col='condition',
#    metric_col='mean',
#    title='Example Movement with Augmentations',
#    output_file='plots/example_movement_augmentations.png'
#)

# For movement_type, we assume it is a column in the dataframe. If not, it needs to be created or extracted.
# structured_df['movement_type'] = <some logic to define movement type>

# Plotting example movements for movement types
#plot_example_movement(
#    df=structured_df,
#    condition_col='movement_type',
#    metric_col='mean',
#    title='Example Movement by Type',
#    output_file='plots/example_movement_types.png'
#)

# Metrics to plot in boxplots
metrics = ['angular_m_ankle_angle_r', 'inference', 'r2_overall', 'pmpjpe_m_joint_right_elbow',
           'velocity_m_joint_left_ankle']

# Plotting boxplots for augmentations
plot_boxplot(
    df=structured_df,
    metrics=metrics,
    group_col='condition',
    title='Boxplot of Metrics by Augmentation',
    output_file='plots/boxplot_augmentations.png'
)

# Plotting boxplots for movement types
plot_boxplot(
    df=structured_df,
    metrics=metrics,
    group_col='movement_type',
    title='Boxplot of Metrics by Movement Type',
    output_file='plots/boxplot_movement_types.png'
)

# Plotting correlation heatmap
plot_correlation_heatmap(
    df=structured_df,
    metrics=metrics,
    title='Correlation Heatmap of Augmentations',
    output_file='plots/correlation_heatmap_augmentations.png'
)
