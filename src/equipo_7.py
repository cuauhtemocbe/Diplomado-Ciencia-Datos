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

pd.set_option('display.max_columns', None)



def check_time_format(time_str, format='%m/%d/%Y'):
    try:
        datetime.strptime(str(time_str), format)
        return True
    except ValueError:
        return False

class DataViz:

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