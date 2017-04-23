# coding=utf-8

"""
  _  _     _                 ___             _   _
 | || |___| |_ __  ___ _ _  | __|  _ _ _  __| |_(_)___ _ _  ___
 | __ / -_) | '_ \/ -_) '_| | _| || | ' \/ _|  _| / _ \ ' \(_-<
 |_||_\___|_| .__/\___|_|   |_| \_,_|_||_\__|\__|_\___/_||_/__/
            |_|

     Funciones de ayuda que hacen las transformaciones
"""

import cv2
import numpy as np


def quantize_block(block, is_luminance, QF=99, inverse=False):
    """
    Aplica la cuantización o la cuantización inversa a un bloque de 8x8

    si is_luminance es True (significa que es un bloque del canal Y),
    de lo contrario deberia ser False

    QF nos indica que tanto vamos a comprimir y que tantas perdidas pueden
    haber si QF=99 entonces practicamente se dejaria la imagen como la original
    y no estariamos comprimiendo nada

    inverse si el bloque viene de la inversa DCT
    """
    QY=np.array([[16,11,10,16,24,40,51,61],
                     [12,12,14,19,26,48,60,55],
                     [14,13,16,24,40,57,69,56],
                     [14,17,22,29,51,87,80,62],
                     [18,22,37,56,68,109,103,77],
                     [24,35,55,64,81,104,113,92],
                     [49,64,78,87,103,121,120,101],
                     [72,92,95,98,112,100,103,99]])

    QC=np.array([[17,18,24,47,99,99,99,99],
                     [18,21,26,66,99,99,99,99],
                     [24,26,56,99,99,99,99,99],
                     [47,66,99,99,99,99,99,99],
                     [99,99,99,99,99,99,99,99],
                     [99,99,99,99,99,99,99,99],
                     [99,99,99,99,99,99,99,99],
                     [99,99,99,99,99,99,99,99]])

    scale = 1.0
    if QF < 50 and QF >= 1:
        scale = np.floor( 5000 / QF )
    elif QF < 100:
        scale = 200 - 2 * QF
    else:
        scale = 200 - 2 * 99
    scale = scale / 100.0
    Q = QC * scale
    if is_luminance:
        Q = QY * scale
    if inverse:
        return block * Q
    else:
        return np.round(block / Q)


def dct_block(block, inverse=False):
    """
    Aplica DCT o la Inversa DCT a un bloque de 8x8
    """
    block = block.astype(np.float32)
    if inverse:
        return cv2.idct(block)
    else:
        return cv2.dct(block)


def convert_to_YCrCb(img):
    """
    Convierte una imagen al espacio de color Y, Cr, Cb y se hace un subsample
    de los canales Cr y Cb

    Y.shape es de la forma (height, width)
    mientras que los canales Crf y Cbf son de la forma (height / 2, width / 2)

    Ustedes van a tener que utilizar resize_image despues de que hayan
    terminado de realizar las transformaciones para reconstruir la imagen al
    tamano original
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Hscale = 2
    Vscale = 2
    Y = img[:,:,0]
    crf = cv2.boxFilter(img[:,:,1], ddepth=-1, ksize=(2,2))
    cbf = cv2.boxFilter(img[:,:,2], ddepth=-1, ksize=(2,2))
    crf = crf[::Vscale, ::Hscale]
    cbf = cbf[::Vscale, ::Hscale]
    return (Y, crf, cbf)


def resize_image(img, width, height):
    """
    Cambia el tamano de una imagen a un width y height dado.
    Automaticamente va a ser rellenado si se va de un tamaño pequeño
    a uno grande.
    """
    return cv2.resize(img, (width, height))


def to_rgb(img):
    """
    Convierte una imagen del espacio de color YCbCr de vuelta a RGB
    """
    return cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)


def truncate(pair):
    """
    Asegura que las imagenes tengan el tamano correcto para generar
    completamente bloques de 8x8. Para esto se asegura que las dimensiones
    sean multiplos de 8. Si es necesario se trunca la imagen.
    """
    k = pair[0]
    img, QF = pair[1]
    height, width = np.array(img.shape[:2])/8 * 8
    img = img[:height, :width]
    return (k, (img, QF))
