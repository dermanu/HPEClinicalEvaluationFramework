import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import argparse

def generate_and_plot_boxplots(data_type='mono'):
    """
    Generate and plot boxplots for various metrics from a preprocessed metrics file.

    Parameters:
    data_type (str): 'mono' or 'multi' to specify the type of data being analyzed.

    Returns:
    None
    """

    # Define the color palette for groups
    palette = sns.color_palette("colorblind", 4)  # Generate distinct colors

    # Metric titles and y-axis labels
    metric_titles = {
        'pmpjpe': 'MPJPE',
        'angle': 'MPJAE',
        'velocity': 'MPJAVE',
        'pcc': 'PCC'
    }

    y_labels = {
        'pmpjpe': 'Overall MPJPE [mm]',
        'angle': 'Overall MPJAE [°]',
        'velocity': 'Overall MPJAVE [°/s]',
        'pcc': 'Overall PCC'
    }

    # Define augmentation orders and display names based on data type
    if data_type == 'multi':
        augmentation_order = [
            'background', 'defocus', 'occlusion', 'underexposure', 'desynchronize', 'decalibration',
            'cameras_4_0', 'cameras_4_3', 'cameras_5_1', 'cameras_5_4_1', 'cameras_0_4_5', 'cameras_5_4_1_3',
            'upper', 'lower', 'complex', 'sitting'
        ]

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

    else:  # mono
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

    def prepare_bxp_data(condition, metric_data):
        """Prepare data for matplotlib's bxp function."""
        return {
            "label": condition,
            "med": metric_data["median"],
            "q1": metric_data["Q1"],
            "q3": metric_data["Q3"],
            "whislo": metric_data["loval"],
            "whishi": metric_data["hival"],
            "fliers": []  # No outliers for simplicity
        }

    def generate_bxp_stats(boxplot_df, metrics_to_plot):
        """Generate bxp data for each metric."""
        bxp_data = {}
        for metric in metrics_to_plot:
            bxp_data[metric] = [
                prepare_bxp_data(row["condition"], row)
                for _, row in boxplot_df[boxplot_df["metric"] == metric].iterrows()
            ]
        return bxp_data

    def plot_bxp(bxp_data):
        """Plot boxplots for each metric."""
        # The axis limits are static so multi and mono or different algorithms can be better compared.
        axis_limit = {
            'pmpjpe': (-100, 350),
            'angle': (-15, 40),
            'velocity': (-15, 20),
        }

        for metric, stats in bxp_data.items():
            # Filter and sort stats
            stats = [
                stat for stat in stats
                if stat["label"].lower().replace(" ", "_") in augmentation_order
            ]
            stats = sorted(stats, key=lambda x: augmentation_order.index(x["label"].lower().replace(" ", "_")))

            # Replace labels with display names
            for stat in stats:
                original_label = stat["label"].lower().replace(" ", "_")
                stat["label"] = augmentation_display_names[original_label]

            # Create the plot
            fig, ax = plt.subplots(figsize=(8, 6))
            boxplot_elements = ax.bxp(stats, showfliers=False, patch_artist=True)

            # Set colors for boxes
            for box, stat in zip(boxplot_elements['boxes'], stats):
                label = stat["label"]
                box.set_facecolor(box_colors[label])

            # Style medians
            for median in boxplot_elements['medians']:
                median.set_color('black')
                median.set_linewidth(2)

            # Style axes
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

    # Load the data
    all_metrics = pd.read_pickle("statistics/" + data_type + "/all_metrics.pkl")

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

    # Prepare boxplot data
    bxp_data = generate_bxp_stats(boxplot_df, metrics_to_plot)

    # Plot the boxplots
    plot_bxp(bxp_data)


# Command-line interface
def main():
    parser = argparse.ArgumentParser(description="Generate and plot boxplots for metrics data.")
    parser.add_argument("--data_type", type=str, choices=["mono", "multi"], default="mono", help="Specify data type: 'mono' or 'multi'.")
    args = parser.parse_args()

    generate_and_plot_boxplots(data_type=args.data_type)

if __name__ == "__main__":
    main()

# Example command
# python statistics/plot_metrics.py --data_type mono

