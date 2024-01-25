import os
import pandas as pd

def read_csv_files(folder_path, save_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith("_1proc.csv"):
                file_path = os.path.join(root, file)
                try:
                    print(f"Processing file {file} in {root}.")
                    # Create the full path for the input file
                    input_file_path = os.path.join(root, file)

                    # Read CSV skipping the first two rows and setting low_memory=False
                    df = pd.read_csv(file_path, skiprows=[0, 1], header=None, low_memory=False)

                    # Rename headers to a combination of the first two rows
                    header = df.iloc[0].ffill()
                    subheader = df.iloc[1]
                    combined_header = header + '_' + subheader
                    combined_header = combined_header.str.replace('NewSubject:', '')
                    combined_header[0] = 'Frame'
                    combined_header[1] = 'Subframe'

                    df.columns = combined_header

                    # Drop the first two rows and the last column
                    df = df.iloc[1:]
                    df.reset_index(drop=True, inplace=True)
                    df.drop(index=[0], inplace=True)
                    df.drop(columns=['Subframe'])
                    df = df.iloc[:, :-1]
                    df.reset_index(drop=True, inplace=True)

                    # Append data from _2proc.csv, _3proc.csv, _4proc.csv files
                    for i in range(2, 5):
                        additional_file = file.replace("_1proc.csv", f"_{i}proc.csv")
                        additional_file_path = os.path.join(root, additional_file)
                        print(f"Append file {i}.")

                        if os.path.exists(additional_file_path):
                            additional_df = pd.read_csv(file_path, skiprows=[0, 1], header=None, low_memory=False)

                            # Rename headers to a combination of the first two rows
                            additional_df.columns = combined_header

                            # Drop the first two rows and the last column
                            additional_df = additional_df.iloc[2:]
                            additional_df.reset_index(drop=True, inplace=True)
                            additional_df.drop(index=[0], inplace=True)
                            additional_df.drop(columns=['Subframe'])
                            additional_df = additional_df.iloc[:, :-1]
                            additional_df.reset_index(drop=True, inplace=True)

                            additional_df['Frame'].iloc[1:] = pd.to_numeric(df['Frame'].iloc[-1]) + additional_df['Frame'].iloc[1:].astype(int)

                            # Concatenate additional data to the main DataFrame
                            df = pd.concat([df, additional_df], ignore_index=True)

                    # Create the corresponding subfolder structure in the new folder
                    relative_path = os.path.relpath(input_file_path, folder_path)
                    new_subfolder = os.path.join(save_path, os.path.dirname(relative_path))

                    # Ensure the new subfolder exists, create it if not
                    os.makedirs(new_subfolder, exist_ok=True)

                    # Create the full path for the output file
                    output_file_path = os.path.join(new_subfolder, file)
                    output_file_path = os.path.splitext(output_file_path)[0] + "Extended.csv"
                    output_file_path = output_file_path.replace("_1procExtended.csv", "Extended.csv")


                    # Save the processed DataFrame to the new folder
                    df.to_csv(output_file_path, index=False)
                    print('Data saved successfully at: ' + output_file_path)

                except pd.errors.EmptyDataError:
                    print(f"File {file} in {root} is empty.")

# Replace 'your_folder_path' with the actual path to the root folder containing subfolders
read_csv_files('/home/emanu/Desktop/MoCap/vicon_data_complete', '/home/emanu/Desktop/MoCap/complete_raw/vicon')
