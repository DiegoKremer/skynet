import argparse
import sys
import tempfile
import pandas as pd
import tensorflow as tf


#Ignora avisos do sistema para reduzir poluicao na tela da execucao
import warnings
warnings.filterwarnings("ignore")


#identifica as colunas (caracteristicas) dos dados presentes no dataset.
COLUMNS = ["idade", "classe_trabalho", "salariototal", "educacao", "educacao_num",
            "estado_civil", "ocupacao", "relacionamento", "raca", "genero",
           "ganho_capital", "perda_capital", "horas_por_semana", "pais_nativo",
           "renda_suporte"]


LABEL_COLUMN = "label"


#Define quais colunas são de Categoria (não numéricas com número finito de possiveis entradas )
# e quais são Contínuas (numéricas com lógica de continuidade).
CATEGORICAL_COLUMNS = ["classe_trabalho", "educacao", "estado_civil", "ocupacao","relacionamento", "raca", "genero", "pais_nativo"]
CONTINUOUS_COLUMNS = ["idade", "educacao_num", "ganho_capital", "perda_capital","horas_por_semana"]




def build_estimator(model_dir, model_type):
    #Constroi um estimador.

    #SparseColumns: Para as variaveis de categoria usamos o .sparse_column para definir valores
    # numericos para cada uma das categorias possiveis.


    # Se sabemos ja quais categorias poderemos encontrar entao usamos
    # o .sparse_column_with_keys e definimos os valores como keys.
    genero = tf.contrib.layers.sparse_column_with_keys(column_name="genero",
                                                       keys=["female", "male"])

    # Se nao sabemos quais categorias poderao aparecer podemos definir um hash com algum tamanho
    # maximo que ira automaticamente atribuir os IDs numericos para cada nova entrada que achar.

    educacao = tf.contrib.layers.sparse_column_with_hash_bucket(
        "educacao", hash_bucket_size=1000)
    relacionamento = tf.contrib.layers.sparse_column_with_hash_bucket(
        "relacionamento", hash_bucket_size=100)
    classe_trabalho = tf.contrib.layers.sparse_column_with_hash_bucket(
        "classe_trabalho", hash_bucket_size=100)
    ocupacao = tf.contrib.layers.sparse_column_with_hash_bucket(
        "ocupacao", hash_bucket_size=1000)
    pais_nativo = tf.contrib.layers.sparse_column_with_hash_bucket(
        "pais_nativo", hash_bucket_size=1000)



    # Para as colunas numericas usamos o .real_valued_column para transformar o campo em valor real.

    idade = tf.contrib.layers.real_valued_column("idade")
    educacao_num = tf.contrib.layers.real_valued_column("educacao_num")
    ganho_capital = tf.contrib.layers.real_valued_column("ganho_capital")
    perda_capital = tf.contrib.layers.real_valued_column("perda_capital")
    horas_por_semana = tf.contrib.layers.real_valued_column("horas_por_semana")

    # Em alguns casos a variavel de numero real pode trazer problemas para o modelo. No exemplo que
    # usamos o modelo pode ter problemas para fazer a correlacao do valor do salario com a idade, pois
    # o salario de uma pessoa tende a crescer na meia idade e depois diminuir quando a pessoa se
    # aposenta.Isso faz com que o sistema nao a relacao de continuidade da idade com o valor do salario,
    # por isso usamos os buckets para transformar a variavel numerica em categorica.

    idade_buckets = tf.contrib.layers.bucketized_column(idade,
                                                      boundaries=[
                                                          18, 25, 30, 35, 40, 45,
                                                          50, 55, 60, 65
                                                      ])

    # Configura as colunas de atributos para o modelo Wide
    wide_columns = [genero, pais_nativo, educacao, ocupacao, classe_trabalho,
                    relacionamento, idade_buckets,
                    tf.contrib.layers.crossed_column([educacao, ocupacao],
                                                     hash_bucket_size=int(1e4)),
                    tf.contrib.layers.crossed_column(
                        [idade_buckets, educacao, ocupacao],
                        hash_bucket_size=int(1e6)),
                    tf.contrib.layers.crossed_column([pais_nativo, ocupacao],
                                                  hash_bucket_size=int(1e4))]

    #Configura as colunas de atributos para o modelo Deep
    deep_columns = [
        tf.contrib.layers.embedding_column(classe_trabalho, dimension=8),
        tf.contrib.layers.embedding_column(educacao, dimension=8),
        tf.contrib.layers.embedding_column(genero, dimension=8),
        tf.contrib.layers.embedding_column(relacionamento, dimension=8),
        tf.contrib.layers.embedding_column(pais_nativo,
                                           dimension=8),
        tf.contrib.layers.embedding_column(ocupacao, dimension=8),
        idade,
        educacao_num,
        ganho_capital,
        perda_capital,
        horas_por_semana,
    ]

    #Define qual modelo sera usado. Wide, Deep ou combinado.

    if model_type == "wide":
        m = tf.contrib.learn.LinearClassifier(model_dir=model_dir,
                                              feature_columns=wide_columns)
    elif model_type == "deep":
        m = tf.contrib.learn.DNNClassifier(model_dir=model_dir,
                                           feature_columns=deep_columns,
                                           hidden_units=[100, 50])
    else:
        m = tf.contrib.learn.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=[100, 50],
            fix_global_step_increment_bug=True)

    return m


# Constroi a funcao de entrada dos dados para treino e teste
def input_fn(df):
    # Cria um mapa de dicionario para cada atributo continuo chamado (k) para os valores
    # daquela coluna armazenados em um Tensor constante.
    continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}

    # Cria um mapa de dicionario para cada atributo categorico chamado (k) para os valores
    # daquela coluna armazenados em um Tensor esparso.
    categorical_cols = {
        k: tf.SparseTensor(
            indices=[[i, 0] for i in range(df[k].size)],
            values=df[k].values,
            dense_shape=[df[k].size, 1])
        for k in CATEGORICAL_COLUMNS}

    # Combina os dois dicionarios (continuo e categorico) em um.
    feature_cols = dict(continuous_cols)
    feature_cols.update(categorical_cols)

    # Converte o label em um Tensor constante.
    label = tf.constant(df[LABEL_COLUMN].values)
    # Retorna as colunas de atributos e o label.
    return feature_cols, label

# Constroi a funcao de entrada dos dados para previsoes (nao recebe o Label)
def predict_fn(df):
    # Cria um mapa de dicionario para cada atributo continuo chamado (k) para os valores
    # daquela coluna armazenados em um Tensor constante.
    continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}

    # Cria um mapa de dicionario para cada atributo categorico chamado (k) para os valores
    # daquela coluna armazenados em um Tensor esparso.
    categorical_cols = {
        k: tf.SparseTensor(
            indices=[[i, 0] for i in range(df[k].size)],
            values=df[k].values,
            dense_shape=[df[k].size, 1])
        for k in CATEGORICAL_COLUMNS}
    # Combina os dois dicionarios (continuo e categorico) em um.
    feature_cols = dict(continuous_cols)
    feature_cols.update(categorical_cols)

    # Retorna as colunas de atributos.
    return feature_cols

#Treina e avalia o modelo
def train_and_eval(model_dir, model_type, train_steps):
    # Le e armazena o arquivo que contem as amostras que serao treinadas
    df_train = pd.read_csv(
        tf.gfile.Open("/home/diegokremer/tensorflow/dataset/adult.data"),
        names=COLUMNS,
        skipinitialspace=True,
        engine="python")

    # Le e armazena o arquivo que contem as amostras que serao usadas no teste de precisao
    df_test = pd.read_csv(
        tf.gfile.Open("/home/diegokremer/tensorflow/dataset/adult.test"),
        names=COLUMNS,
        skipinitialspace=True,
        skiprows=1,
        engine="python")

    # Le e armazena o arquivo que contem as amostras que serao previstas
    df_prever = pd.read_csv(
        tf.gfile.Open("/home/diegokremer/tensorflow/dataset/prever.csv"),
        names=COLUMNS,
        skipinitialspace=True,
        skiprows=1,
        engine="python")

  # Remove elementos NaN (Que nao sao numeros)
    df_train = df_train.dropna(how='any', axis=0)
    df_test = df_test.dropna(how='any', axis=0)

    df_train[LABEL_COLUMN] = (
          df_train["renda_suporte"].apply(lambda x: ">50K" in x)).astype(int)
    df_test[LABEL_COLUMN] = (
          df_test["renda_suporte"].apply(lambda x: ">50K" in x)).astype(int)

    model_dir = tempfile.mkdtemp() if not model_dir else model_dir
    print("Diretorio do modelo = %s" % model_dir)

    m = build_estimator(model_dir, model_type)
    m.fit(input_fn=lambda: input_fn(df_train), steps=train_steps)
    results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))


    #Realiza a predicao das amostras do arquivo CSV
    prediction = m.predict(input_fn=lambda: predict_fn(df_prever))

    resultado_certo = [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0]

    #Exibe os resultados da predicao e compara com o real
    print("Previsao: ",list(prediction))
    print("Real:     ",resultado_certo)


preFLAGS = None


def main(_):
  train_and_eval(FLAGS.model_dir, FLAGS.model_type, FLAGS.train_steps)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--model_dir",
      type=str,
      default="",
      help="Base directory for output models."
  )
  parser.add_argument(
      "--model_type",
      type=str,
      default="wide_n_deep",
      help="Valid model types: {'wide', 'deep', 'wide_n_deep'}."
  )
  parser.add_argument(
      "--train_steps",
      type=int,
      default=200,
      help="Number of training steps."
  )




  FLAGS, unparsed = parser.parse_known_args()
tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


