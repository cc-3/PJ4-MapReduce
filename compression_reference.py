# coding=utf-8

"""
  ___      __
 | _ \___ / _|___ _ _ ___ _ _  __ ___
 |   / -_)  _/ -_) '_/ -_) ' \/ _/ -_)
 |_|_\___|_| \___|_| \___|_||_\__\___|

    Algoritmo de referencia para hacer la compresion
"""

import os
import argparse
import cv2
import matplotlib.pyplot as plt
from helper_functions import *


def naive_compress(image):
    image = truncate((None, image))[1]
    Y, crf, cbf = convert_to_YCrCb(image)
    channels = [Y, crf, cbf]
    height, width = image.shape[0:2]
    reimg = np.zeros((height, width, 3), dtype='uint8')
    for idx, channel in enumerate(channels):
        no_rows = channel.shape[0]
        no_cols = channel.shape[1]
        dst = np.zeros((no_rows, no_cols), dtype='float32')
        no_vert_blocks = no_cols / B_SIZE
        no_horz_blocks = no_rows / B_SIZE
        for j in range(no_vert_blocks):
            for i in range(no_horz_blocks):
                i_start = i * B_SIZE
                i_end = i_start  + B_SIZE
                j_start = j * B_SIZE
                j_end = j_start + B_SIZE
                cur_block = channel[i_start : i_end, j_start : j_end]
                dct = dct_block(cur_block.astype(np.float32) - 128)
                q = quantize_block(dct, idx==0, QF)
                inv_q = quantize_block(q, idx==0, QF, inverse = True)
                inv_dct = dct_block(inv_q, inverse = True)
                dst[i_start : i_end, j_start : j_end] = inv_dct
        dst = dst + 128
        dst[dst>255] = 255
        dst[dst<0] = 0
        dst = resize_image(dst, width, height)
        reimg[:,:,idx] = dst
    return to_rgb(reimg)


def main(args):
    global B_SIZE, QF
    B_SIZE = 8
    QF = args.QF
    input_name = os.path.basename(args.input)
    img = cv2.imread(args.input)
    result = naive_compress(img)
    output = os.path.join('ref-out', 'naive_QF'+ str(QF) + '_' + input_name)
    cv2.imwrite(output, result)
    if args.show == 1:
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        plt.imshow(result)
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="realiza la compresion forma secuencial.")
    parser.add_argument("-I", "--input", type=str, default="test/",
            help="imagen de entrada")
    parser.add_argument("-C", "--QF", type=int, default=99,
            help="taza de compresion de la imagen")
    parser.add_argument("-s", "--show", type=int, default=0,
            help="0 muestra la imagen | 1 no muestra la imagen")
    args = parser.parse_args()
    main(args)
