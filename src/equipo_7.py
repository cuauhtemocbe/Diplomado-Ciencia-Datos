"""
El modulo sirve para analizar la completitud de un conjunto de datos y 
realizar gráficas por tipo de variable (Discretas y continuas)

Los desarrolladores son:
  Alfaro Segura Vanessa Paola
  Izumi Sierra Saemi Marissa
  Tadeo Trejo Miguel Angel
  Cuauhtémoc Salvador Bautista Enciso
  Tamayo Guerrero Brandon
"""
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.decomposition import PCA
from varclushi import VarClusHi


pd.set_option('display.max_columns', None)



def check_time_format(time_str, format='%m/%d/%Y'):
    try:
        datetime.strptime(str(time_str), format)
        return True
    except ValueError:
        return False

class DataViz:
  """
    Una clase para la visualización de datos que proporciona métodos estáticos para analizar y visualizar
    información contenida en un DataFrame de pandas.
    """

  @staticmethod
  def completitud(data):
    """
    Función que presenta la proporcion de valores nulos que tiene cada columna.
    Parámetros:
    data (pd.DataFrame): El DataFrame que contiene los datos.
    Retorna:
    Presenta el % de valores nulos que contiene cada columna del dataframe.  
    """

    return data.isnull().sum()/len(data)

  @staticmethod
  def histograma(data: pd.DataFrame, columnas: list) -> None:
    """
      Genera histogramas para las columnas especificadas de un DataFrame.

      Parámetros:
      data (pd.DataFrame): El DataFrame que contiene los datos.
      columnas (list): Una lista de nombres de columnas continuas para las cuales se generarán los gráficos.

      Retorna:
      Visualiza el gráfico de cada columna del argumento columnas
      """
    for column  in columnas:
      fig, axs = plt.subplots(1, 1, figsize=(15, 5))
      sns.histplot(data.groupby(column)[column].count(), ax=axs).set_title(column)
      plt.tight_layout()
      plt.show()

  @staticmethod
  def caja(data: pd.DataFrame, columnas: list) -> None:
    """

      Genera diagramas de caja y bigotes para las columnas especificadas de un DataFrame.

      Parámetros:
      data (pd.DataFrame): El DataFrame que contiene los datos.
      columnas (list): Una lista de nombres de columnas continuas para las cuales se generarán los gráficos.

      Retorna:
      Visualiza el gráfico de cada columna del argumento columnas
      """
    for column  in columnas:
      fig, axs = plt.subplots(1, 1, figsize=(15, 5))
      sns.boxplot(data=data, x=column, ax=axs).set(title=column)
      plt.tight_layout()
      plt.show()

  @staticmethod
  def barras_horizontales(data: pd.DataFrame, columnas: list) -> None:
    """
    Genera un gráfico de barras horizontales para las columnas especificadas de un DataFrame.

    Parámetros:
    data (pd.DataFrame): El DataFrame que contiene los datos.
    columnas (list): Una lista de nombres de columnas discretas para las cuales se generarán los gráficos.

    Retorna:
    Visualiza el gráfico de cada columna del argumento columnas
    """

    for column  in columnas:
      fig, axs = plt.subplots(1, 1, figsize=(15, 5))
      sns.barplot(data.groupby(column)[column].count(), ax=axs, orient='h').set_title(column)
      plt.tight_layout()
      plt.show()

  @staticmethod
  def lineas(data: pd.DataFrame, columnas: list, fecha: str) -> None:
      """
      Genera un gráfico de lineas para las columnas especificadas de un DataFrame.

      Parámetros:
      data (pd.DataFrame): El DataFrame que contiene los datos.
      columnas (list): Una lista de nombres de columnas continuas para las cuales se generarán los gráficos.
      fecha (str): Es la columna que contiene la fecha de tipo datetime

      Retorna:
      Visualiza el gráfico de cada columna del argumento columnas
      """

      for column  in columnas:
          fig, axs = plt.subplots(1, 1, figsize=(15, 5))
          sns.lineplot(x=fecha , y=column, data=data, marker="o")
          plt.xticks(rotation=45)
          plt.tight_layout()
          plt.show()

  @staticmethod
  def puntos(data: pd.DataFrame, columnas: list) -> None:
      """
      Genera un gráfico de puntos para todos los pares ordenados de las columnas especificadas de un DataFrame,
      exceptuando los pares cuyos atributos sean el mismo.

      Parámetros:
      data (pd.DataFrame): El DataFrame que contiene los datos.
      columnas (list): Una lista de nombres de columnas continuas para las cuales se generarán los gráficos.

      Retorna:
      Visualiza el gráfico de cada par ordenado argumento columnas
      """
      for column_a in columnas:
        for column_b in columnas:
            if column_a == column_b:
              continue
            fig, axs = plt.subplots(1, 1, figsize=(15, 5))
            sns.scatterplot(data=data, x= column_a, y= column_b)
            plt.tight_layout()
            plt.show()

  @staticmethod
  def densidad(data: pd.DataFrame, columnas: list) -> None:
      """
      Genera un gráfico de densidad de las columnas especificadas de un DataFrame,

      Parámetros:
      data (pd.DataFrame): El DataFrame que contiene los datos.
      columnas (list): Una lista de nombres de columnas continuas para las cuales se generarán los gráficos.

      Retorna:
      Visualiza el gráfico de cada columna del argumento columnas
      """
      for column in columnas:
            fig, axs = plt.subplots(1, 1, figsize=(15, 5))
            sns.kdeplot(data=data, x=column)
            plt.tight_layout()
            plt.show()

  @staticmethod
  def violin(data: pd.DataFrame, columnas: list) -> None:
      """
      Genera un gráfico de violín de las columnas especificadas de un DataFrame,

      Parámetros:
      data (pd.DataFrame): El DataFrame que contiene los datos.
      columnas (list): Una lista de nombres de columnas continuas para las cuales se generarán los gráficos.

      Retorna:
      Visualiza el gráfico de cada columna del argumento columnas
      """
      for column in columnas:
            fig, axs = plt.subplots(1, 1, figsize=(15, 5))
            sns.violinplot(x = column, data = data)
            plt.tight_layout()
            plt.show()

  @staticmethod
  def calor(data: pd.DataFrame, columnas: list) -> None:
      """
      Genera un gráfico de calor de las columnas especificadas de un DataFrame,

      Parámetros:
      data (pd.DataFrame): El DataFrame que contiene los datos.
      columnas (list): Una lista de nombres de columnas continuas para las cuales se generarán los gráficos.

      Retorna:
      Visualiza el gráfico de cada columna del argumento columnas
      """

      correl =  data[columnas].corr()
      fig, axs = plt.subplots(1, 1, figsize=(15, 5))
      sns.heatmap(data = correl)
      plt.tight_layout()
      plt.show()


def detect_outliers_iqr(data, column):
    """
    Detecta los límites para identificar valores atípicos en un conjunto de datos usando el rango intercuartílico (IQR).

    Parámetros:
    data (DataFrame): El DataFrame que contiene los datos.
    column (str): El nombre de la columna para la cual se calcularán los límites de los valores atípicos.

    Retorna:
    tuple: Un par (lower_bound, upper_bound) donde `lower_bound` es el límite inferior y `upper_bound` es el límite superior 
           para identificar los valores atípicos en la columna especificada.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return lower_bound, upper_bound


def transform_outliers(data, column, lower_bound, upper_bound):
    """
    Transforma los valores atípicos en una columna de un DataFrame, ajustándolos a los límites inferior y superior especificados.

    Parámetros:
    data (DataFrame): El DataFrame que contiene los datos.
    column (str): El nombre de la columna en la que se van a transformar los valores atípicos.
    lower_bound (float): El límite inferior para los valores de la columna.
    upper_bound (float): El límite superior para los valores de la columna.

    Retorna:
    DataFrame: El DataFrame con los valores atípicos transformados dentro de los límites especificados.
    """
    data.loc[data[column] < lower_bound, column] = lower_bound
    data.loc[data[column] > upper_bound, column] = upper_bound
    return data


def process_outliers(data, column):
    """
    Procesa los valores atípicos en una columna de un DataFrame, detectándolos y ajustándolos dentro de los límites calculados 
    utilizando el método del rango intercuartílico (IQR).

    Parámetros:
    data (DataFrame): El DataFrame que contiene los datos.
    column (str): El nombre de la columna en la que se van a procesar los valores atípicos.

    Retorna:
    DataFrame: El DataFrame con los valores atípicos transformados dentro de los límites especificados.
    """
    lower_bound, upper_bound = detect_outliers_iqr(data, column)
    data = transform_outliers(data, column, lower_bound, upper_bound)

    return data


def get_pca(data, n_components):
    """
    Realiza Análisis de Componentes Principales (PCA) en los datos proporcionados.

    Parámetros:
    data (pd.DataFrame o np.ndarray): Los datos de entrada en los que se aplicará el PCA. 
                                      Debe ser un DataFrame de pandas o un array de numpy.
    n_components (int): El número de componentes principales a retener.

    Retorna:
    tuple: Una tupla que contiene:
        - pca (PCA): El objeto PCA ajustado.
        - pca_df (pd.DataFrame): Un DataFrame de pandas con las nuevas características transformadas por PCA.
    """
    pca = PCA(n_components=n_components)
    pca_df = pd.DataFrame(pca.fit_transform(data))

    return pca, pca_df


def get_kbest(data_features, data_target, k):
    """
    Selecciona las mejores características según la prueba estadística f_regression.

    Parámetros:
    data_features (pd.DataFrame o np.ndarray): Las características de entrada para la selección.
    data_target (pd.DataFrame o np.ndarray): El objetivo o las etiquetas asociadas a las características.
    k (int): El número de características principales a seleccionar.

    Retorna:
    pd.DataFrame: Un DataFrame de pandas con los nombres de las características y sus respectivas puntuaciones, ordenado por las puntuaciones de manera descendente.
    """
    kb = SelectKBest(score_func=f_regression, k=k)
    kb.fit(data_features, data_target.loc[:, data_target.columns[0]])
    scores_df = pd.DataFrame(list(zip(data_features.columns, kb.scores_)), columns=['Feature', 'Score']).sort_values(by='Score', ascending=False)

    return scores_df


def get_varclushi(data):
    """
    Realiza un análisis de clusters de variables (Variable Clustering) en los datos proporcionados.

    Parámetros:
    data (pd.DataFrame o np.ndarray): Los datos de entrada sobre los que se realizará el análisis de clusters de variables.

    Retorna:
    tuple: Una tupla que contiene:
        - info (pd.DataFrame): Un DataFrame de pandas con información detallada sobre los clusters de variables.
        - rsquare (pd.DataFrame): Un DataFrame de pandas con los valores de R-cuadrado para cada variable en relación a su cluster.
    """
    vc = VarClusHi(data)
    vc.varclus()

    return vc.info, vc.rsquare


