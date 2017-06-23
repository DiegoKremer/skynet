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



def main():
    # Se os arquivos não estiverem armazenados localmente baixa os mesmos.
    if not os.path.exists(IRIS_TRAINING):
        raw = urllib.request.urlopen(IRIS_TRAINING_URL).read().decode()
        with open(IRIS_TRAINING,'w') as f:
            f.write(raw)

    if not os.path.exists(IRIS_TEST):
        raw = urllib.request.urlopen(IRIS_TEST_URL).read().decode()
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

    #Constroi 3 camadas DNN com 10, 20 e 10 unidades respectivamente, sendo que os parametros
    # da classe DNNClassifier são feature_columns que é as colunas de atributos que definimos acima,
    # hidden_units são "camadas escondidas" usadas para melhorar performance do modelo, n_classses
    # que define o número de espécies de Iris que temos e por fim o caminho do diretório onde o
    # tensorflow irá salvar arquivos de checkpoints.
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units=[10,20,10],
                                                n_classes=3,
                                                model_dir="/tmp/iris_model")


    # Define a função de entrada dos dados de treino do modelo.
    def get_train_inputs():
      x = tf.constant(training_set.data)
      y = tf.constant(training_set.target)

      return x, y


    # Juntamos o classificador que configuramos mais acima com os dados de treino e indicamos quantos
    # passos (steps) de treino ele irá realizar.

    #Encaixa o modelo no dataset.
    classifier.fit(input_fn=get_train_inputs, steps=2000)


    #Agora que o modelo de treino e os dados de treino estão prontos vamos usar o método evaluate()
    # para avaliar a precisão

    #Define a função de entrada dos dados de teste do modelo.
    def get_test_inputs():
      x = tf.constant(test_set.data)
      y = tf.constant(test_set.target)


    #Avalia a precisão e printa na tela. Obs: O valor pode variar um pouco a cada treino realizado.
    accuracy_score = classifier.evaluate(input_fn=get_test_inputs,
                                         steps=1)["accuracy"]

    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))



    #Realiza a predição de duas novas amostras de Iris para dizer qual espécie ela é.
    # Para isso usamos o método predict().

    def new_samples():
        return np.array([[6.4, 3.2, 4.5, 1.5],
                        [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)

    predictions = list(classifier.predict(input_fn=new_samples()))


    print ("New Samples, Class Predictions:     {}\n"
           .format(predictions))

if __name__ == "__main__":
    main()
