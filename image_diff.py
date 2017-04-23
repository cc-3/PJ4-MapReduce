"""
  ___                       ___  _  __  __
 |_ _|_ __  __ _ __ _ ___  |   \(_)/ _|/ _|
  | || '  \/ _` / _` / -_) | |) | |  _|  _|
 |___|_|_|_\__,_\__, \___| |___/|_|_| |_|
                |___/

    Comparador de imagenes
"""

import cv2
import numpy as np
import argparse


def main(args):
    im1 = cv2.imread(args.input1, cv2.IMREAD_UNCHANGED)
    im2 = cv2.imread(args.input2, cv2.IMREAD_UNCHANGED)
    diff = np.round(im1.astype(np.float32) - im2.astype(np.float32))
    if np.max(np.abs(diff)) > 1:
        print "Images do not match"
    else:
        print "Images match"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Compara dos imagenes")
    parser.add_argument("-I1", "--input1", type=str,
            help="imagen de entrada 1")
    parser.add_argument("-I2", "--input2", type=str,
            help="imagen de entrada 2")
    args = parser.parse_args()
    main(args)
