import os
import pandas as pd

def check_csv_for_nan(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith("proc.csv"):
                file_path = os.path.join(root, file)
                try:
                    # Read CSV skipping the first 5 rows as headers and units are in rows 3, 4, and 5
                    df = pd.read_csv(file_path, low_memory=False)

                    # Check for NaN values or empty cells
                    nan_columns = df.columns[df.isnull().any()].tolist()
                    empty_columns = df.columns[df.applymap(lambda x: x == '').any()].tolist()

                    if df.isnull().values.any() or df.empty:
                        print(f"File {file} in {root} contains NaN values or empty cells.")

                        if nan_columns:
                            print(f"Columns with NaN values: {', '.join(nan_columns)}")

                            # Rename columns based on values in row 4 (units)
                            mapping = {column: unit for column, unit in zip(df.columns, df.iloc[1, :])}
                            df.rename(columns=mapping, inplace=True)

                        if empty_columns:
                            print(f"Columns with empty cells: {', '.join(empty_columns)}")

                except pd.errors.EmptyDataError:
                    print(f"File {file} in {root} is empty.")

# Replace 'your_folder_path' with the actual path to the root folder containing subfolders
check_csv_for_nan('/home/emanu/Desktop/MoCap/MoCapData')
