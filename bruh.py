from skeletonMorphing.loadMorphDatasets import run_load
from skeletonMorphing.trainSkeletonMorphing import train

if __name__ == '__main__':
    datapath = "E:\MoCap"
    #run_load(datapath)
    train(datapath)
