from __future__ import print_function

import numpy as np

# Faz o download do dataset do Titanic no TFLearn
from tflearn.datasets import titanic
titanic.download_dataset('titanic_dataset.csv')

# Carrega o CSV e indica que a primeira coluna é o label
# e que há dados categoricos (nao numericos).
from tflearn.data_utils import load_csv
dados, labels = load_csv('titanic_dataset.csv', target_column=0,
                        categorical_labels=True, n_classes=2)

# Funcao de preprocessamento dos dados
def preprocess(dados, colunas_a_ignorar):
    # Organiza por ordem decrescente de ID e deleta colunas ignoradas
    for id in sorted(colunas_a_ignorar, reverse=True):
        [r.pop(id) for r in dados]
    for i in range(len(dados)):
      # Converte o campo Sex para float (numerico).
      dados[i][1] = 1. if dados[i][1] == 'female' else 0.
    return np.array(dados, dtype=np.float32)

# Ignora as colunas "name" e "ticket" do dataset
Ignorar_colunas=[1, 6]

# Preprocessa os dados carregados para remover os campos desnecessarios
# e transformar atributos categoricos em numericos
dados = preprocess(dados, Ignorar_colunas)



