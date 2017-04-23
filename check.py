#!/usr/bin/env python

"""
   ___ _           _
  / __| |_  ___ __| |__
 | (__| ' \/ -_) _| / /
  \___|_||_\___\__|_\_\.

    Script que verifica si su algoritmo es correcto
"""

import os
import argparse
import numpy as np
from spark_image_compressor import run


def main(args):
    QF = 99
    bSz = 64
    threads = args.threads
    # para siempre obtener las mismas imagenes random
    np.random.seed(1)
    # creamos una coleccion de imagenes random
    image_collection = []
    for idx in xrange(100):
        img = np.round(np.random.rand(400, 400, 3) * 255).astype('uint8')
        image_collection.append((idx, (img, QF)))
    # corremos su algoritmo
    result = run(image_collection, batch_size=bSz, threads=threads)
    # ordenamos por key
    result.sort(key=lambda x: x[0])
    # creamos una secuencia de strings para usar writelines
    result = [str(x[0]) + ": " + str(x[1]) + "\n" for x in result]
    # escribimos el resultado a un archivo de texto
    with open("test_output.txt", 'w') as f:
        f.writelines(result)
    # creamos dos objetos para leer los archivos
    f1 = open('ref-out/test_output_ref.txt')
    f2 = open('test_output.txt')
    # leemos su contenido
    text1 = f1.read()
    text2 = f2.read()
    # verificamos
    with open('check.log', 'w') as f:
        if text1==text2:
            f.write('TODO BIEN - SU ALGORITMO FUNCIONA CORRECTAMENTE')
        else:
            f.write('SU ALGORITMO NO FUNCIONA CORRECTAMENTE')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Verifica si el resultado es correcto")
    parser.add_argument("-T", "--threads", type=int, default=8,
            help="numero de threads/subprocesos que tiene su computadora")
    args = parser.parse_args()
    main(args)
