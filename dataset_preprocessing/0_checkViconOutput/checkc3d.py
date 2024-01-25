import c3d

with open('/home/emanu/Desktop/MoCap/vicon_data/par4/Par4proc.c3d', 'rb') as handle:
    reader = c3d.Reader(handle)
    for i, (points, analog) in enumerate(reader.read_frames()):
        print('Frame {}: {}'.format(i, points.round(2)))