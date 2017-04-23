# coding=utf-8

"""
  ___                _
 / __|_ __  __ _ _ _| |__
 \__ \ '_ \/ _` | '_| / /
 |___/ .__/\__,_|_| |_\_\.
     |_|

  ___                        ___
 |_ _|_ __  __ _ __ _ ___   / __|___ _ __  _ __ _ _ ___ ______ ___ _ _
  | || '  \/ _` / _` / -_) | (__/ _ \ '  \| '_ \ '_/ -_|_-<_-</ _ \ '_|
 |___|_|_|_\__,_\__, \___|  \___\___/_|_|_| .__/_| \___/__/__/\___/_|
                |___/                     |_|

    Este archivo es que tienen que modificar para obtener el resultado
    deseado
"""

import pyspark
import numpy as np
from helper_functions import *
from pyspark import SparkContext, SparkConf


############ AQUI TIENEN QUE DEFINIR SUS FUNCIONES DE AYUDA ###############


def generate_Y_cb_cr_matrices(rdd):
    """
    Esta funcion tiene que generar un nuevo rdd transformado

    recuerden que a este nivel todavia son imagenes completas miren que es
    lo que pasa cuando utilizan la funcion convert_to_YCrCb(img) en
    helper_functions ¿deberia usar map, flatMap, o no usar ninguna de estas?
    """
    ### SU SOLUCION AQUI ###
    return rdd


def generate_sub_blocks(rdd):
    """
    Esta funcion tiene que generar un nuevo rdd transformado

    a este nivel ustedes devuelven ya un rdd con varios subblocks, piensen muy
    bien que informacion quieren guardar en los pares (key, value), recuerden
    que el key y value pueden ser una tupla con varios elementos

    ¿sera que es necesario guardar de donde viene ese bloque y en que posicion
    estaba?
    """
    ### SU SOLUCION AQUI ###
    rdd = rdd.flatMap(generate_sub_blocks_flatmap).repartition(16)
    return rdd


def apply_transformations(rdd):
    """
    Esta funcion tiene que generar un nuevo rdd transformado

    aqui cada bloque tiene que ser transformado piensen como pueden hacerlo de
    una manera mas optima aunque para este proyecto solo vamos a medir que
    utilicen Spark correctamente y que funcione, no que sea el algoritmo mas
    optimo
    """
    ### SU SOLUCION AQUI ###
    return rdd


def combine_sub_blocks(rdd):
    """
    Esta funcion tiene que generar un nuevo rdd transformado

    Cuando ya tengan un rdd de subblocks transformados esos subblocks van a
    venir de diferentes imagenes, tienen que combinar los bloques para
    re-construir la imagen. Su rdd deberia de contener valores que son
    arrays de numpy de size (height, width)
    """
    ### SU SOLUCION AQUI ###
    return rdd


def run(images, QF=99, batch_size=64, threads=8):
    """
    Esta funcion tiene que retornar una lista de python

    Retorna una lista de python donde todas las imagenes ya han sido procesadas
    el formato retornado en la lista deberia ser que por cada elemento se
    deberia de encontrar una tupla de la forma (image_id, image_matrix) donde
    image_matrix es un arreglo de numpy de size (height, width, 3)
    """

    # algunas variables globales
    global P, WIDTH, HEIGHT, QF_G, B_SIZE

    # inicializamos spark
    url = 'local[{0}]'.format(threads)
    # lo configuramos
    conf = SparkConf().setAppName("SparkImageCompressor").setMaster(url)
    # y creamos el contexto
    sc = SparkContext(conf=conf)
    # cuantas imagenes hay
    size = len(images)
    # cuantas iteraciones vamos a tener que hacer
    total = int(np.ceil(float(size)/batch_size))
    # aqui es donde iremos guardando el resultado
    output = []
    # Partitions esto ayuda a que tengamos repartido el dataset en todos los
    # threads por lo general se quiere que hayan 2 particiones por thread en
    # este caso lo hacemos con repartition despues de una transformacion
    # o en el momento de crear el RDD (VEAN LAS NOTAS DE SPARK DEL PROYECTO)
    # para ver como pueden utilizar esto para que sea mas eficiente
    P = threads * 2
    # WIDTH HEIGHT B_SIZE Y QF
    # estas las tienen que utilizar en sus funciones a la hora de transformar
    # los bloques y restaurar las imagenes asi que no las olviden
    HEIGHT, WIDTH = images[0][1].shape[0:2]
    QF_G = QF
    # block size
    B_SIZE = 8  # siempre es 8
    # iteramos
    # xrange = range solo que xrange no crea el arreglo online, xrange es mas
    # eficiente en terminos de memoria
    for i in xrange(total):
        # calculamos los limites para agarrar un nuevo batch de imagenes
        START = i * batch_size
        END = min((i + 1) * batch_size, size)
        # obtenemos el batch de imagenes
        batch = images[START:END]
        # truncamos el batch de imagenes
        rdd = sc.parallelize(batch, P).map(truncate)
        ##########################
        #    AQUI SU SOLUCION    #
        ##########################
        # pueden agregar cualquier otra funcion que quieran para hacer en el
        # rdd aqui pueden escribir todas las funciones que quieran y crean que
        # sean necesarias
        output += rdd.collect()
    # le damos stop al contexto de spark para finalizar
    sc.stop()
    # devolvemos el resultado
    return  output
