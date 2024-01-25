import os
import pandas as pd
from trc import TRCData

def read_csv_files(folder_path, save_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith("_1proc.trc"):
                file_path = os.path.join(root, file)
                try:
                    print(f"Processing file {file} in {root}.")

                    # Create the full path for the input file
                    input_file_path = os.path.join(root, file)

                    # Read TRC
                    mocap_data = TRCData()
                    mocap_data.load(file_path)

                    # Rename headers to a combination of the first two rows
                    header_names = mocap_data['Markers']
                    expanded_header_names = [item for header in header_names for item in expand_header(header)]
                    expanded_header_names.insert(0, 'Frame')
                    expanded_header_names.insert(1, 'Time')

                    df = pd.DataFrame(columns=expanded_header_names)
                    df['Frame'] = mocap_data['Frame#']
                    df['Time'] = mocap_data['Time']

                    for header in header_names:
                        df[header + '_X'] = [sublist[0] for sublist in mocap_data[header]]
                        df[header + '_Y'] = [sublist[1] for sublist in mocap_data[header]]
                        df[header + '_Z'] = [sublist[2] for sublist in mocap_data[header]]

                    # Append data from _2proc.trc, _3proc.trc, _4proc.trc files
                    for i in range(2, 5):
                        additional_file = file.replace("_1proc.trc", f"_{i}proc.trc")
                        additional_file_path = os.path.join(root, additional_file)
                        print(f"Append file {i}.")

                        if os.path.exists(additional_file_path):
                            additional_data = TRCData()
                            additional_data.load(additional_file_path)

                            additional_df = pd.DataFrame(columns=expanded_header_names)
                            additional_df['Frame'] = additional_data['Frame#'] + df['Frame'].max()
                            additional_df['Time'] = additional_data['Time'] + df['Time'].max()

                        for header in header_names:
                                additional_df[header + '_X'] = [sublist[0] for sublist in additional_data[header]]
                                additional_df[header + '_Y'] = [sublist[1] for sublist in additional_data[header]]
                                additional_df[header + '_Z'] = [sublist[2] for sublist in additional_data[header]]

                        # Concatenate additional data to the main DataFrame
                        df = pd.concat([df, additional_df], ignore_index=True)

                    # Create the corresponding subfolder structure in the new folder
                    relative_path = os.path.relpath(input_file_path, folder_path)
                    new_subfolder = os.path.join(save_path, os.path.dirname(relative_path))

                    # Ensure the new subfolder exists, create it if not
                    os.makedirs(new_subfolder, exist_ok=True)

                    # Create the full path for the output file
                    output_file_path = os.path.join(new_subfolder, file)
                    output_file_path = os.path.splitext(output_file_path)[0] + ".csv"
                    output_file_path = output_file_path.replace("_1proc.csv", "proc.csv")

                    # Save the processed DataFrame to the new folder
                    df.to_csv(output_file_path, index=False)
                    print('Data saved successfully at: ' + output_file_path)

                except pd.errors.EmptyDataError:
                    print(f"File {file} in {root} is empty.")


# Function to expand header names
def expand_header(header):
    return [f'{header}_X', f'{header}_Y', f'{header}_Z']


# Replace 'your_folder_path' with the actual path to the root folder containing subfolders
read_csv_files('/home/emanu/Desktop/MoCap/vicon_data_complete', '/home/emanu/Desktop/MoCap/complete_raw/vicon')
