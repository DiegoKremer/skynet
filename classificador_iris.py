from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib

import tensorflow as tf
import numpy as np

#Define os arquivos com os datasets de treino e teste.

IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"


#Se os arquivos não estiverem armazenados localmente baixa os mesmos.
if not os.path.exists(IRIS_TRAINING):
    raw = urllib.urlopen(IRIS_TRAINING_URL).read()
    with open(IRIS_TRAINING,'w') as f:
        f.write(raw)

if not os.path.exists(IRIS_TEST):
    raw = urllib.urlopen(IRIS_TEST_URL).read()
    with open(IRIS_TEST,'w') as f:
        f.write(raw)

#Carrega os datasets

#O método load_csv_with_header precisa de três parametros que é o filename(caminho do arquivo
# do dataset, o target_dtype que é o valor alvo (tipo de dado numpy) e os atributos
#(tipo de dado numpy).
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TRAINING,
    target_dtype=np.int,
    features_dtype=np.float32)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TEST,
    target_dtype=np.int,
    features_dtype=np.float32)

#Configurando um modelo de classificador de Deep Neural Network (DNN).

#Especifica que todos os atributos são valores numéricos reais.
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

#Constroi 3 camadas DNN com 10, 20 e 10 unidades respectivamente.
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10,20,10],
                                            n_classes=3,
                                            model_dir="/tmp/iris_model")
