
from skeletonMorphing.loadMorphDatasets import run_load
from skeletonMorphing.trainSkeletonMorphing import init_sweep
import argparse
import numpy as np


if __name__ == '__main__':

    #pars = np.array([10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
    pars = np.array([10, 11, 12, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25])
    datapath = "E:\MoCap"
    fold_id = [20, 21]
    init_sweep(datapath, pars, fold_id)
