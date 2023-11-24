import os
import glob
import numpy as np
import pyxdf
import pandas as pd
import time
from datetime import datetime
from scipy.interpolate import griddata
from scipy.signal import correlate

main_folder = '/media/emanu/LaCie/VICON_DATA_PROC'

for root, dirs, files in os.walk(main_folder):
    for folder_name in dirs:
        print(f"Processing files in subfolder: {folder_name}")

        for file_extension in ('csv', 'trc', 'xdf'):
            try:
                file_path = glob.glob(os.path.join(root, folder_name, f'*.{file_extension}'))[0]
                if file_extension == 'csv':
                    print(file_path)
                    csv_stream = pd.read_csv(file_path, delimiter=',', header=None, skiprows=5)
                    with open(file_path, 'r') as file:
                        lines = file.readlines()
                        csv_header = []
                        # Loop through the header rows and split each line into a list of values
                        for i in range(5):
                            if i == 2:
                                # Replace "NewSubject" in the second row with the replacement variable
                                header_line = lines[i].replace("NewSubject", folder_name).split(',')
                            else:
                                header_line = lines[i].split(',')
                            csv_header.append(header_line)
                elif file_extension == 'trc':
                    print(file_path)
                    trc_stream = pd.read_csv(file_path, delimiter='\t', header=None, skiprows=5)
                    with open(file_path, 'r') as file:
                        lines = file.readlines()
                        trc_header = []
                        trc_header = [line.split(',') for line in lines[1:5]]
                elif file_extension == 'xdf':
                    print(file_path)
                    xdf_streams, xdf_header = pyxdf.load_xdf(file_path)
            except IndexError:
                print(f"Warning: {file_extension.upper()} file not found in subfolder {folder_name}")
                continue

        isQualisys = []
        isUnity = []
        isCam = []
        CameraData = {}

        tol = 10
        tol = tol * 1000 / (tol ** 2)

        for stream in range(len(xdf_streams)):
            tempTime = np.array(xdf_streams[stream]['time_stamps'].T)
            tempTime = [datetime.utcfromtimestamp(t) for t in tempTime]
            tempTime = [t.strftime('%H:%M:%S.%f') for t in tempTime]
            tempData = pd.DataFrame(data=np.array(xdf_streams[stream]['time_series']).T, columns=tempTime)

            if xdf_streams[stream]['info']['name'] == 'Qualisys':
                label = xdf_streams[stream]['info']['desc']['channels']['channel']
                for j in range(len(label)):
                        tempData.Propertiesrename(columns={j: label[j]['label']}, inplace=True)
                isQualisys.append(stream)
                tempData['Time'] = np.round(tempData['Time'] * tol).astype(int) / tol

                QualisysData = tempData

            elif xdf_streams[stream]['info']['name'] == 'Unity.TaskNumber':
                isUnity.append(stream)
                tempData.rename(columns={0: 'UnityTaskNumber'}, inplace=True)
                tempData['Time'] = np.round(tempData['Time'] * tol).astype(int) / tol
                UnityData = tempData

            else:
                tempData.rename(columns={0: xdf_streams[stream]['info']['name']}, inplace=True)

                if xdf_streams[stream]['info']['name'] == 'FrameMarker2':
                    tempData['Time'] = tempData['Time'] - 0.01
                elif xdf_streams[stream]['info']['name'] == 'FrameMarker0':
                    tempData['Time'] = tempData['Time'] - 0.017
                else:
                    tempData['Time'] = tempData['Time'] - 0.024

                isCam.append(stream)
            CameraData[len(CameraData) + 1] = tempData

        # Sort Cameras according to their Number
        CameraDataNew = [None] * 7
        for fieldNr in range(6):
            for camNr in range(6):
                if CameraData[1][fieldNr]['Time'][0] == f'FrameMarker{camNr - 1}':
                    CameraDataNew[camNr] = CameraData[1][fieldNr]

        CameraData = CameraDataNew

        # Combine inputs
        QualisysData, delay, _ = alignQTM(QTMDataRaw, QualisysData, UnityData)
        plotData(QualisysData, delay)
        inputComplete = [UnityData] + list(CameraData.values()) + [QualisysData]

        # No resampling
        TT = pd.concat(inputComplete, axis=1)
        TT['Time'] = TT['Time'] - TT['Time'].iloc[0]

        # Write file
        filePath = os.path.join(dataPath, currD)
        fileName = os.path.splitext(os.path.basename(TRCfilePath))[0]
        TT.to_csv(os.path.join(filePath, f'{fileName}.csv'), index=False)
        TT.to_pickle(os.path.join(filePath, f'{fileName}.pkl'))