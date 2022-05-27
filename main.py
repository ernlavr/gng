import src.GrowingNeuralGas as Gng
import cv2
import argparse
import numpy as np

# Create commandline arguments for passing location of image
def getCmdLineArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', help='Location of image to be processed')
    args = parser.parse_args()
    return args


def main():
    args = getCmdLineArgs()
    img = cv2.imread(args.image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img,10,255,0)
    gng = Gng.GrowingNeuralGas(img[1])
    

if __name__ == "__main__":
    main()