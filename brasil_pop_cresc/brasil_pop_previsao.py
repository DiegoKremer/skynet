from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pandas as pd
import numpy as np

def main(unused_argv):

    DATA = "/home/diegokremer/Downloads/brazil_pop_growth.csv"
    DATA_TEST = "/home/diegokremer/Downloads/brazil_pop_growth.csv"

    COLUNAS = ["ano","pop","nasc","morte"]

    ATRIBUTOS = ["ano","pop","nasc","morte"]

    ALVO = ["ano"]


    training_set = pd.read_csv("brazil_pop_growth.csv", skipinitialspace=True,
                               skiprows=1, names=COLUNAS)
    test_set = pd.read_csv("brazil_pop_growth_test.csv", skipinitialspace=True,
                           skiprows=1, names=COLUNAS)

    prediction_set = [2015]

    def input_fn(data_set):
        feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
        labels = tf.constant(data_set[LABEL].values)
        return feature_cols, labels

    def main(unused_argv):
        # Load datasets
        training_set = pd.read_csv("boston_train.csv", skipinitialspace=True,
                                   skiprows=1, names=COLUMNS)
        test_set = pd.read_csv("boston_test.csv", skipinitialspace=True,
                               skiprows=1, names=COLUMNS)

        # Set of 6 examples for which to predict median house values
        prediction_set = pd.read_csv("boston_predict.csv", skipinitialspace=True,
                                     skiprows=1, names=COLUMNS)

        # Feature cols
        feature_cols = [tf.contrib.layers.real_valued_column(k)
                        for k in FEATURES]

        # Build 2 layer fully connected DNN with 10, 10 units respectively.
        regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                                  hidden_units=[10, 10],
                                                  model_dir="/tmp/boston_model")

        # Fit
        regressor.fit(input_fn=lambda: input_fn(training_set), steps=5000)

        # Score accuracy
        ev = regressor.evaluate(input_fn=lambda: input_fn(test_set), steps=1)
        loss_score = ev["loss"]
        print("Loss: {0:f}".format(loss_score))

        # Print out predictions
        y = regressor.predict(input_fn=lambda: input_fn(prediction_set))
        # .predict() returns an iterator; convert to a list and print predictions
        predictions = list(itertools.islice(y, 6))

    print("Predictions: {}".format(str(predictions)))

if __name__ == '__main__':
    tf.app.run()