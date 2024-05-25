from skeletonMorphing.loadMorphDatasets import run_load
from skeletonMorphing.trainSkeletonMorphing import train
import argparse
import numpy as np



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', default=False, type=bool, help='foo help')
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
    pars = np.array([12, 15 ,16])
    pars = np.array([10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
    debug = args.debug
    wandb = args.wandb
    train(datapath, pars, rnd, wandb, debug)
