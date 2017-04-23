# coding=utf-8

"""
  ___             ___
 | _ \_  _ _ _   |_ _|_ __  __ _ __ _ ___
 |   / || | ' \   | || '  \/ _` / _` / -_)
 |_|_\._,_|_||_| |___|_|_|_\__,_\__, \___|
  / __|___ _ __  _ __ _ _ ___ __|___/___ _ _
 | (__/ _ \ '  \| '_ \ '_/ -_|_-<_-</ _ \ '_|
  \___\___/_|_|_| .__/_| \___/__/__/\___/_|
                |_|

    script que corre su algoritmo MapReduce
"""

import os
import time
import cv2
import argparse
import numpy as np
from spark_image_compressor import run
from moviepy.editor import VideoFileClip
from moviepy.editor import ImageSequenceClip


def main(args):
    QF = args.QF
    bSz = args.batch_size
    threads = args.threads
    if args.test:
        # para siempre obtener las mismas imagenes random
        np.random.seed(223)
        # creamos una coleccion de imagenes random
        image_collection = []
        for idx in xrange(100):
            img = np.round(np.random.rand(400, 400, 3) * 255).astype('uint8')
            image_collection.append((idx, img))
        # corremos su algoritmo
        result = run(image_collection, QF=QF, batch_size=bSz, threads=threads)
        # ordenamos por key
        result.sort(key=lambda x: x[0])
        # creamos una secuencia de strings para usar writelines
        result = [str(x[0]) + ": " + str(x[1]) + "\n" for x in result]
        # escribimos el resultado a un archivo de texto
        with open("test_output.txt", 'w') as f:
            f.writelines(result)
    elif args.input and not args.video:
        # leemos la imagen
        image = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
        # hacemos de que la imagen se copie 10 veces para que tenga sentido
        # correrlo con spark
        images = [(idx, image) for idx in range(10)]
        # corremos su algoritmo
        result = run(images, QF=QF, batch_size=bSz, threads=threads)
        # nombre de salida
        input_name = os.path.basename(args.input)
        output = 'spark_QF' + str(QF) + '_' + input_name
        cv2.imwrite(output, result[0][1])
    else:
        # leemos el video que queremos comprimir
        clip = VideoFileClip(args.input)
        # obtenemos los frames y los enumeramos, importante enumerar asi lo
        # podemos ordenar ya procesados
        frames = [(idx, frame) for idx, frame in
                  enumerate(clip.iter_frames(fps=30))]
        # corremos el algoritmo
        result = run(frames, QF=QF, batch_size=bSz, threads=threads)
        # ordenamos los frames
        result.sort(key=lambda x: x[0])
        # solo dejamos los frames
        result = [frame for idx, frame in result]
        # creamos un video a partir de una secuencia de imagenes a 30fps
        video = ImageSequenceClip(result, fps=30)
        # nombre de la salida
        input_name = os.path.basename(args.input)
        output = 'spark_QF' + str(QF) + '_' + input_name
        # creamos el video
        video.write_videofile(output, audio=False, fps=30)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="realiza la compresion utilizando MapReduce")
    parser.add_argument("-T", "--test", action="store_true",
            help="si quieren probar su algoritmo con imagenes random")
    parser.add_argument("-I", "--input", type=str, default="test/test1.jpg",
            help="imagen de entrada")
    parser.add_argument("-C", "--QF", type=int, default=99,
            help="taza de compresion de la imagen")
    parser.add_argument("-B", "--batch_size", type=int, default=64,
            help="tamaño del batch")
    parser.add_argument("-P", "--threads", type=int, default=8,
            help="numero de threads/subprocesos que tiene su computadora")
    parser.add_argument("-V", "--video", action="store_true",
            help="si el input es un video pasar esta bandera")
    args = parser.parse_args()
    main(args)
