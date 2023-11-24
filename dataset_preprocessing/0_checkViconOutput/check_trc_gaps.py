import os
import pandas as pd

# Define the main folder path where the TRC files and subfolders are located
main_folder_path = 'MoCap'

# Initialize an empty dictionary to store column gap information for each file
all_column_gaps = {}

# Recursively search for TRC files in the main folder and its subfolders
for root, dirs, files in os.walk(main_folder_path):
    for filename in files:
        if filename.endswith(".trc"):
            file_path = os.path.join(root, filename)

            # Read the TRC file to extract headers
            try:
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                    header_row1 = lines[3].strip().split('\t')
                    header_row1 = header_row1[2:]
                    for i in range(len(header_row1)):
                        if header_row1[i] != '':
                            prev_value = header_row1[i]
                        elif prev_value != '':
                            header_row1[i] = prev_value
                    header_row1.extend([header_row1[-1], header_row1[-1]])
                    header_row2 = lines[4].strip().split('\t')
            except FileNotFoundError:
                print(f"File not found: {file_path}")
                continue

            # Create a dictionary to map column indices to column headers
            column_map = {index: header_row1[index] + ' ' + header_row2[index] for index in range(len(header_row2))}

            # Load the TRC file into a DataFrame, skipping the first five rows (header)
            df = pd.read_csv(file_path, delimiter='\t', header=None, skiprows=5)

            # Initialize an empty dictionary to store the column gaps for this file
            column_gaps = {}

            # Iterate through columns 2 to 136
            for col in range(2, 135):
                # Find rows with empty cells in the current column
                empty_cells = df[df[col].isnull()]

                # If there are empty cells, identify the start index and length of each gap
                if not empty_cells.empty:
                    gap_indices = empty_cells.index.tolist()
                    start_index = gap_indices[0]
                    gap_length = len(gap_indices)
                    column_gaps[column_map[col]] = {'Start Index': start_index, 'Gap Length': gap_length}

            # Add the column gap information for this file to the overall dictionary
            all_column_gaps[file_path] = column_gaps

# Display the results for all files
for file_path, column_gaps in all_column_gaps.items():
    if column_gaps:
        print(f"File: {file_path}")
        print("Columns with gaps:")
        for col, gap_info in column_gaps.items():
            start_index = gap_info['Start Index']
            gap_length = gap_info['Gap Length']
            print(f"{col}: Start Index - {start_index}, Gap Length - {gap_length}")
    else:
        print(f"File: {file_path}")
        print("No gaps found in columns 2 to 136.")
    print()
