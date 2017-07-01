from __future__ import print_function


# Faz o download do dataset do Titanic no TFLearn
from tflearn.datasets import titanic
titanic.download_dataset('titanic_dataset.csv')

# Carrega o CSV e indica que a primeira coluna é o label
# e que há dados categoricos (nao numericos).
from tflearn.data_utils import load_csv
data, labels = load_csv('titanic_dataset.csv', target_column=0,
                        categorical_labels=True, n_classes=2)

