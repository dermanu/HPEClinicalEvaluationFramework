import cv2
import pandas as pd
import os

def get_csv_frame_count(file_path):
    df = pd.read_csv(file_path)
    return len(df)

def get_avi_frame_count(file_path):
    video = cv2.VideoCapture(file_path)
    frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    return frames

def compare_frame_counts(folder_path):
    mismatched_files = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            csv_path = os.path.join(folder_path, filename)
            avi_path = os.path.join(folder_path, filename.replace(".csv", ".avi"))

            if os.path.exists(avi_path):
                csv_frame_count = get_csv_frame_count(csv_path)
                avi_frame_count = get_avi_frame_count(avi_path)

                print(f"File: {filename}")
                print(f"CSV Frame Count: {csv_frame_count} frames")
                print(f"AVI Frame Count: {avi_frame_count} frames")

                if csv_frame_count == avi_frame_count:
                    print("Frame counts match.")
                else:
                    print("Frame counts do not match.")
                    mismatched_files.append(filename)

                print("\n")

    if mismatched_files:
        print("Warning: Some files have unequal frame counts.")
        print(f"Mismatched files: {', '.join(mismatched_files)}")

# Example usage
folder_path = "/home/emanu/Desktop/SegmentedData/par4"
compare_frame_counts(folder_path)