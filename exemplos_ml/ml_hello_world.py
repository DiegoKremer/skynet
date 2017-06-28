from sklearn import tree
import warnings

warnings.filterwarnings("ignore")

#Define quais sao os atributos e rotulos do dataset.
atributos = [[135,1],[140, 1],[130,1],[145,1],[150,0],[170,0],[160,0],[155,0]]
rotulos = [0,0,0,0,1,1,1,1]

#Define o tipo de classificador que sera usado.
classificador = tree.DecisionTreeClassifier()


#Treina este classificador com os atributos e rotulos.
classificador = classificador.fit(atributos, rotulos)


#Recebe os atributos
peso = raw_input('Qual o peso da fruta?')
textura = raw_input('Qual a textura da fruta?')

#Transforma os atributos categoricos (texto) em numericos
if str(textura)=='irregular':
    textura=0
elif str(textura)=='suave':
    textura=1

#Cria uma variavel da nova amostra contendo seus atributos
nova_amostra = [peso, textura]

#Realiza uma predicao da nova amostra
classificador.predict(nova_amostra)

#Transforma a predicao numerica em categorica para melhor visualizacao
if classificador.predict(nova_amostra) == 1 :
    print("\n \n Laranja")
else: print("\n \n Maca")


#/home/diegokremer/PycharmProjects/exemplos_ml/ml_hello_world.py