import os
import pandas as pd
from trc import TRCData


def read_csv_files(folder_path, save_path):
    # Select only the desired columns
    desired_columns = ['Frame', 'Time', 'C7_X', 'C7_Y', 'C7_Z', 'T10_X', 'T10_Y', 'T10_Z', 'CLAV_X', 'CLAV_Y', 'CLAV_Z',
                       'STRN_X', 'STRN_Y', 'STRN_Z', 'RBAK_X', 'RBAK_Y', 'RBAK_Z', 'LSHO_X', 'LSHO_Y', 'LSHO_Z',
                       'LUPA_X', 'LUPA_Y', 'LUPA_Z', 'LELB_X', 'LELB_Y', 'LELB_Z', 'LFRM_X', 'LFRM_Y', 'LFRM_Z',
                       'LWRA_X', 'LWRA_Y', 'LWRA_Z', 'LWRB_X', 'LWRB_Y', 'LWRB_Z', 'RSHO_X', 'RSHO_Y', 'RSHO_Z',
                       'RUPA_X', 'RUPA_Y', 'RUPA_Z', 'RELB_X', 'RELB_Y', 'RELB_Z', 'RFRM_X', 'RFRM_Y', 'RFRM_Z',
                       'RWRA_X', 'RWRA_Y', 'RWRA_Z', 'RWRB_X', 'RWRB_Y', 'RWRB_Z', 'LASI_X', 'LASI_Y', 'LASI_Z',
                       'RASI_X', 'RASI_Y', 'RASI_Z', 'LPSI_X', 'LPSI_Y', 'LPSI_Z', 'RPSI_X', 'RPSI_Y', 'RPSI_Z',
                       'LTHI_X', 'LTHI_Y', 'LTHI_Z', 'LKNE_X', 'LKNE_Y', 'LKNE_Z', 'LTIB_X', 'LTIB_Y', 'LTIB_Z',
                       'LANK_X', 'LANK_Y', 'LANK_Z', 'LHEE_X', 'LHEE_Y', 'LHEE_Z', 'LTOE_X', 'LTOE_Y', 'LTOE_Z',
                       'RTHI_X', 'RTHI_Y', 'RTHI_Z', 'RKNE_X', 'RKNE_Y', 'RKNE_Z', 'RTIB_X', 'RTIB_Y', 'RTIB_Z',
                       'RANK_X', 'RANK_Y', 'RANK_Z', 'RHEE_X', 'RHEE_Y', 'RHEE_Z', 'RTOE_X', 'RTOE_Y', 'RTOE_Z',
                       'LHJC_X', 'LHJC_Y', 'LHJC_Z', 'RHJC_X', 'RHJC_Y', 'RHJC_Z', 'LKJC_X', 'LKJC_Y', 'LKJC_Z',
                       'RKJC_X', 'RKJC_Y', 'RKJC_Z', 'LAJC_X', 'LAJC_Y', 'LAJC_Z', 'RAJC_X', 'RAJC_Y', 'RAJC_Z',
                       'LSJC_X', 'LSJC_Y', 'LSJC_Z', 'RSJC_X', 'RSJC_Y', 'RSJC_Z', 'LEJC_X', 'LEJC_Y', 'LEJC_Z',
                       'REJC_X', 'REJC_Y', 'REJC_Z', 'LWJC_X', 'LWJC_Y', 'LWJC_Z', 'RWJC_X', 'RWJC_Y', 'RWJC_Z']

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

                    # Only keep the desired columns
                    df = df[desired_columns]

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
                                if header in additional_data:
                                    additional_df[header + '_X'] = [sublist[0] for sublist in additional_data[header]]
                                    additional_df[header + '_Y'] = [sublist[1] for sublist in additional_data[header]]
                                    additional_df[header + '_Z'] = [sublist[2] for sublist in additional_data[header]]
                                else:
                                    print(f"Header {header} not found in {additional_file}. Ignoring.")

                            # Only keep the desired columns
                            additional_df = additional_df[desired_columns]

                            # Concatenate additional data to the main DataFrame
                            df = pd.concat([df, additional_df], ignore_index=True)

                    # Check if the first row in the "Time" column is -0.01
                    if df['Time'].iloc[0] == -0.01:
                        # Generate Time values starting from 0.00 increasing by 0.01 for each frame
                        df['Time'] = [(0.0 + 0.01 * j) for j in range(len(df))]
                        df['Time'] = df['Time'].round(2)


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
read_csv_files('/media/emanu/LaCie/MoCap/Broken', '/home/emanu/Desktop/MoCap/complete_raw/vicon')