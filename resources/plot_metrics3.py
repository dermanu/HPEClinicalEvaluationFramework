import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data_type = 'mono' # Change to mono or multi respectively

# Define colors for each group using the colorblind palette
palette = sns.color_palette("colorblind", 4)  # Generate 3 distinct colors

metric_titles = {
    'pmpjpe': 'MPJPE',
    'angle': 'MPJAE',
    'velocity': 'MPJAVE',
    'pcc': 'PCC'}

# Predefine custom y-axis labels for each metric
y_labels = {
    'pmpjpe': 'Overall MPJPE in [mm]',
    'angle': 'Overall MPJAE [°]',
    'velocity': 'Overall MPJAVE [°/s]',
    'pcc': 'Overall PCC'
}

if data_type == 'multi':
    # Define the custom order for augmentations (using original names)
    augmentation_order = [
        'background', 'defocus', 'occlusion', 'underexposure', 'desynchronize', 'decalibration',
        'cameras_4_0', 'cameras_4_3', 'cameras_5_1', 'cameras_5_4_1', 'cameras_0_4_5', 'cameras_5_4_1_3',
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
        'cameras_4_0': 'Cam fl-fr',
        'cameras_4_3': 'Cam fl-bl',
        'cameras_5_1': 'Cam fm-sl',
        'cameras_5_4_1': 'Cam fl-fm-sl',
        'cameras_0_4_5': 'Cam fl-fm-fr',
        'cameras_5_4_1_3': 'Cam fl-fm-bl-sl',
        'upper': 'Upper',
        'lower': 'Lower',
        'complex': 'Whole-Body',
        'sitting': 'Sitting'
    }

    box_colors = {
        'Background': palette[0],
        'Defocus': palette[0],
        'Occlusion': palette[0],
        'Underexposure': palette[0],
        'Desynchronized': palette[0],
        'Decalibrated': palette[0],
        'Cam fl-fr': palette[1],
        'Cam fl-bl': palette[1],
        'Cam fm-sl': palette[1],
        'Cam fl-fm-sl': palette[1],
        'Cam fl-fm-fr': palette[1],
        'Cam fl-fm-bl-sl': palette[1],
        'Upper': palette[2],
        'Lower': palette[2],
        'Whole-Body': palette[2],
        'Sitting': palette[2],
    }

else:
    augmentation_order = [
         'background', 'defocus', 'occlusion', 'underexposure',
         'cameras_0', 'cameras_1', 'cameras_2', 'cameras_3', 'cameras_4', 'cameras_5',
         'upper', 'lower', 'complex', 'sitting'
    ]

    augmentation_display_names = {
         'background': 'Background',
         'defocus': 'Defocus',
         'occlusion': 'Occlusion',
         'underexposure': 'Underexposure',
         'cameras_4': 'Cam fr',
         'cameras_5': 'Cam fm',
         'cameras_0': 'Cam fl',
         'cameras_1': 'Cam bl',
         'cameras_2': 'Cam br',
         'cameras_3': 'Cam sl',
         'upper': 'Upper',
         'lower': 'Lower',
         'complex': 'Whole-Body',
         'sitting': 'Sitting'
    }

    box_colors = {
        'Background': palette[0],
        'Defocus': palette[0],
        'Occlusion': palette[0],
        'Underexposure': palette[0],
        'Cam fr': palette[1],
        'Cam fm': palette[1],
        'Cam fl': palette[1],
        'Cam bl': palette[1],
        'Cam br': palette[1],
        'Cam sl': palette[1],
        'Upper': palette[2],
        'Lower': palette[2],
        'Whole-Body': palette[2],
        'Sitting': palette[2],
    }

# Define a function to prepare bxp data
def prepare_bxp_data(condition, metric_data):
    """Prepares data in the format required by matplotlib.axes.Axes.bxp."""
    return {
        "label": condition,
        "med": metric_data["median"],
        "q1": metric_data["Q1"],
        "q3": metric_data["Q3"],
        "whislo": metric_data["loval"],  # Lower whisker
        "whishi": metric_data["hival"],  # Upper whisker
        "fliers": []  # No outliers for simplicity; can be added if needed
    }

# Generate bxp data for all metrics
def generate_bxp_stats(boxplot_df, metrics_to_plot):
    bxp_data = {}
    for metric in metrics_to_plot:
        bxp_data[metric] = [
            prepare_bxp_data(row["condition"], row)
            for _, row in boxplot_df[boxplot_df["metric"] == metric].iterrows()
        ]
    return bxp_data

# Plot the boxplots using the bxp method
def plot_bxp(bxp_data):
    axis_limit = {
        'pmpjpe': (-100, 350),
        'angle': (-15, 40),
        'velocity': (-15 ,20),
    }

    for metric, stats in bxp_data.items():

        stats = [
            stat for stat in stats
            if stat["label"].lower().replace(" ", "_") in augmentation_order
        ]
        stats = sorted(stats, key=lambda x: augmentation_order.index(x["label"].lower().replace(" ", "_")))

        # Replace x-labels with display names
        for stat in stats:
            original_label = stat["label"].lower().replace(" ", "_")
            stat["label"] = augmentation_display_names[original_label]

        fig, ax = plt.subplots(figsize=(8, 6))
        boxplot_elements = ax.bxp(stats, showfliers=False, patch_artist=True)
        for box, stat in zip(boxplot_elements['boxes'], stats):
            label = stat["label"]
            box.set_facecolor(box_colors[label])

        for median in boxplot_elements['medians']:
            median.set_color('black')
            median.set_linewidth(2)

        #ax.set_title(metric_titles[metric], fontsize=18)
        ax.set_ylabel(y_labels[metric], fontsize=14)
        ax.set_xlabel("", fontsize=14)
        ax.tick_params(axis="x", rotation=45)
        ax.yaxis.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
        for label in ax.get_xticklabels():
            label.set_ha('right')
        tick_labels = ax.get_xticklabels()
        for label in tick_labels:
            if label.get_text() in ('Cam fl-fr', 'Cam fm'):
                label.set_fontweight('bold')
        plt.ylim(axis_limit[metric])
        plt.tight_layout()
        plt.show()

# Load the all_metrics.pkl file
file_path = data_type + '/all_metrics.pkl'  # Replace with your actual file path
all_metrics = pd.read_pickle(file_path)

# Metrics of interest
metrics_to_plot = ['pmpjpe', 'angle', 'velocity']

# Prepare the data for boxplots
columns = ['condition', 'metric', 'Q1', 'median', 'Q3', 'loval', 'hival', 'actual_loval', 'actual_hival']
data_for_boxplots = []

for condition, metrics in all_metrics.items():
    for metric in metrics_to_plot:
        if f"{metric}_Qs" in metrics:
            Qs = metrics[f"{metric}_Qs"]
            data_for_boxplots.append([condition, metric] + Qs)

# Create a DataFrame
boxplot_df = pd.DataFrame(data_for_boxplots, columns=columns)

# Prepare boxplot data from your DataFrame (boxplot_df)
bxp_data = generate_bxp_stats(boxplot_df, metrics_to_plot)

# Plot the boxplots
plot_bxp(bxp_data)