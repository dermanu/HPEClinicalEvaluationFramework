import pandas as pd
import wandb
import ast
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import numpy as np


# Function to extract means and standard deviations from the 'summary' column
def extract_stats(summary):
    if isinstance(summary, str):
        try:
            summary_dict = ast.literal_eval(summary)
        except (ValueError, SyntaxError):
            return {}, {}
    elif isinstance(summary, dict):
        summary_dict = summary
    else:
        return {}, {}

    means = {k: v for k, v in summary_dict.items() if 'mean' in k or '_m_' in k}
    stds = {k: v for k, v in summary_dict.items() if 'std' in k or '_s_' in k}
    return means, stds


# Function to fetch data from W&B
def fetch_wandb_data(entity, project, sweep_id):
    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    runs = sweep.runs
    summary_list, config_list, name_list = [], [], []
    for run in runs:
        summary_list.append(run.summary._json_dict)
        config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})
        name_list.append(run.name)
    return pd.DataFrame({"summary": summary_list, "config": config_list, "name": name_list})


# Function to pair mean and std keys based on a common prefix
def pair_keys(means_keys, stds_keys):
    paired_keys = []
    for mean_key in means_keys:
        for std_key in stds_keys:
            if mean_key.replace('_m_', '_s_').replace('_mean', '_std') == std_key:
                paired_keys.append((mean_key, std_key))
    return paired_keys


# Main function to perform the analysis
def main():
    entity, project, sweep_id = "dermanu", "HPE_framework", "3xtcz0q1"
    runs_df = fetch_wandb_data(entity, project, sweep_id)

    # Apply the extraction function to the data
    runs_df['means'], runs_df['stds'] = zip(*runs_df['summary'].apply(extract_stats))

    # Identify common keys across all 'means' and 'stds' dictionaries, skipping the first run
    common_means_keys = set.intersection(*(set(d.keys()) for i, d in enumerate(runs_df['means']) if d and i != 0))
    common_stds_keys = set.intersection(*(set(d.keys()) for i, d in enumerate(runs_df['stds']) if d and i != 0))

    # Pair mean and std keys based on a common prefix
    paired_keys = pair_keys(common_means_keys, common_stds_keys)

    # Ensure that we only process rows where the key is present in both means and stds
    structured_data = []
    for idx, row in runs_df.iterrows():
        if idx == 0:
            continue
        for mean_key, std_key in paired_keys:
            if mean_key in row['means'] and std_key in row['stds']:
                structured_data.append({
                    'condition': row['name'],
                    'metric': mean_key,
                    'mean': row['means'][mean_key],
                    'std': row['stds'][std_key],
                    'sample_number': row['summary'].get('sample_number', None)  # Using 'sample_number' as a sample number
                })

    structured_df = pd.DataFrame(structured_data)

    structured_df['mean'] = pd.to_numeric(structured_df['mean'], errors='coerce')
    structured_df['std'] = pd.to_numeric(structured_df['std'], errors='coerce')


    # Perform ANOVA tests for each metric
    anova_results = {}
    for metric in structured_df['metric'].unique():
        metric_data = structured_df[structured_df['metric'] == metric]
        conditions = metric_data['condition'].unique()
        samples = [metric_data[metric_data['condition'] == condition]['mean'].dropna().values for condition in
                   conditions]

        # Ensure samples are numerical arrays
        try:
            samples = [np.asarray(sample, dtype=float) for sample in samples]

            # Perform ANOVA
            f_val, p_val = stats.f_oneway(*samples)
            anova_results[metric] = {'f_val': f_val, 'p_val': p_val}
        except TypeError as e:
            print(f"Skipping ANOVA for {metric} due to error: {e}")


    # Convert ANOVA results to DataFrame for better visualization
    anova_results_df = pd.DataFrame.from_dict(anova_results, orient='index')

    # Display the ANOVA results
    print("ANOVA Results")
    print(anova_results_df)

    # Summarize the ANOVA results
    significant_anova = anova_results_df[anova_results_df['p_val'] < 0.05]
    print("Significant ANOVA Results")
    print(significant_anova)

    # Save the ANOVA results to a CSV file
    anova_results_df.to_csv("anova_results.csv", index=True)

    # Prepare to perform post-hoc tests
    significant_metrics = significant_anova.index

    posthoc_results = {}
    for metric in significant_metrics:
        metric_data = structured_df[structured_df['metric'] == metric]
        print(f"Processing Tukey HSD for metric: {metric}")
        print(metric_data[['mean', 'condition']])  # Print the data being passed to Tukey HSD
        try:
            tukey = pairwise_tukeyhsd(endog=metric_data['mean'], groups=metric_data['condition'], alpha=0.05)
            posthoc_results[metric] = tukey.summary()

            # Print Tukey HSD results
            print(f"Tukey HSD results for {metric}")
            print(tukey.summary())

            # Convert Tukey HSD results to DataFrame and save to CSV
            tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
            tukey_df.to_csv(f"tukey_hsd_results_{metric}.csv", index=False)
        except Exception as e:
            print(f"Skipping Tukey HSD test for {metric} due to error: {e}")

    # Save the structured dataframe
    structured_df.to_csv("structured_project.csv", index=False)


if __name__ == "__main__":
    main()
