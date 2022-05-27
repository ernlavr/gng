import src.GrowingNeuralGas as Gng
import cv2
import argparse
import numpy as np
import os

# Create commandline arguments for passing location of image
def getCmdLineArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default=os.path.join('data', 'dog.png'), help='Input data')
    parser.add_argument('-e', '--epocs', type=int, default=10000, help='Number of training iterations')
    args = parser.parse_args()
    return args


def main():
    args = getCmdLineArgs()
    img = cv2.imread(args.input)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img,127,255,0)
    gng = Gng.GrowingNeuralGas(np.swapaxes(img[1], 0, 1), args.epocs)
    

if __name__ == "__main__":
    main()