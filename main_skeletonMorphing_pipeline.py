from skeletonMorphing.loadMorphDatasets import run_load
from skeletonMorphing.trainSkeletonMorphing import train
import argparse
import numpy as np



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', default=True, type=bool, help='foo help')
    parser.add_argument('--par', default=12, type=int, help='foo help')
    parser.add_argument('--load', default=False, type=bool, help='foo help')
    parser.add_argument('--random', default=False, type=bool, help='foo help')
    parser.add_argument('--wandb', default=True, type=bool, help='foo help')
    datapath = "E:\MoCap"
    args = parser.parse_args()
    if args.load:
        run_load(datapath)

        exit()

    rnd = args.random
    pars = np.array([args.par])
    #pars = np.array([10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
    pars = np.array([10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 25])
    # par 5 and 6 missing!
    debug = True
    wandb =True
    train(datapath, pars, rnd, wandb, debug)
