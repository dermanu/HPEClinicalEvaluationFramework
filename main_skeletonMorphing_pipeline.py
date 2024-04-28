from skeletonMorphing.loadMorphDatasets import run_load
from skeletonMorphing.trainSkeletonMorphing import train
import argparse
import numpy as np



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--par', default=12, type=int, help='foo help')
    parser.add_argument('--load', default=False, type=bool, help='foo help')
    datapath = "E:\MoCap"
    args = parser.parse_args()
    if args.load:
        run_load(datapath)

    pars = np.array([args.par])
    pars = np.array([12, 14, 15, 16])
    train(datapath, pars)
