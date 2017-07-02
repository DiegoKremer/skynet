from __future__ import print_function

import numpy as np
import tflearn

# Faz o download do dataset do Titanic no TFLearn
from tflearn.datasets import titanic
titanic.download_dataset('titanic_dataset.csv')

# Carrega o CSV e indica que a primeira coluna e o label
# e que ha dados categoricos (nao numericos).
from tflearn.data_utils import load_csv
dados, labels = load_csv('titanic_dataset.csv', target_column=0,
                        categorical_labels=True, n_classes=2)

# Funcao de preprocessamento dos dados
def preprocessa(dados, colunas_a_ignorar):
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
dados = preprocessa(dados, Ignorar_colunas)

# Constroi a rede neural
rede = tflearn.input_data(shape=[None, 6])
rede = tflearn.fully_connected(rede, 32)
rede = tflearn.fully_connected(rede, 32)
rede = tflearn.fully_connected(rede, 2, activation='softmax')
rede = tflearn.regression(rede)

# Define o modelo, neste caso uma Deep Neural Network (Rede Neural Profunda)
modelo = tflearn.DNN(rede)

# Inicia o treinamento
modelo.fit(dados, labels, n_epoch=10, batch_size=16, show_metric=True)

#Entradas de nova pessoa
classe = raw_input('A qual classe voce pertence?')
nome = raw_input('Qual seu nome?')
sexo = raw_input('Qual seu sexo?')
#Traduz o sexo de portugues para ingles para corresponder ao dataset
if (sexo == 'Feminino'):
    sexo = 'female'
else:
    sexo ='male'
idade = raw_input('Qual sua idade?')
irmaos_esposo = raw_input('Quantos irmaos e/ou esposa(o) levou?')
pais_filhos = raw_input('Quantos pais e/ou filhos levou?')
passagem = 'N/A'
valor_passagem = raw_input('Valor da passagem?')

#Cria uma pessoa com os dados inseridos
pessoa = [classe, nome, sexo, idade, irmaos_esposo, pais_filhos, passagem, valor_passagem]

#Processa essa pessoa para seus dados estejam corretos
pessoa = preprocessa([pessoa], Ignorar_colunas)

#Realiza a previsao baseado na pessoa criada e processada
previsao = modelo.predict(pessoa)

#Exibe a previsao na tela
print("Chance de sobrevivencia...", previsao[0][1])