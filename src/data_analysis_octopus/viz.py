"""Plotting helpers for data_analysis_octopus (DataViz).

Extracted from the top-level module into its own submodule -- DataViz
doesn't call any of the module's other functions/classes, so it's the
smallest safe slice to split out (see issue #22).
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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

        # Calcular el porcentaje de valores nulos
        null_percent = round((data.isnull().sum() / len(data) * 100).to_frame(), 2)
        null_percent = null_percent.rename(columns={0: "% valores nulos"})

        # Calcular el conteo de valores no nulos
        count_nulls = data.isnull().sum().to_frame()
        count_nulls = count_nulls.rename(columns={0: "Total de nulos"})

        # Concatenar los resultados
        result_df = pd.concat([count_nulls, null_percent], axis=1)
        result_df = result_df.reset_index()
        result_df = result_df.rename(columns={"index": "features"})

        return result_df

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
        for column in columnas:
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
        for column in columnas:
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

        for column in columnas:
            fig, axs = plt.subplots(1, 1, figsize=(15, 5))
            sns.barplot(
                data.groupby(column)[column].count(), ax=axs, orient="h"
            ).set_title(column)
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

        for column in columnas:
            fig, axs = plt.subplots(1, 1, figsize=(15, 5))
            sns.lineplot(x=fecha, y=column, data=data, marker="o")
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
                sns.scatterplot(data=data, x=column_a, y=column_b)
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
            sns.violinplot(x=column, data=data)
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

        correl = data[columnas].corr()
        fig, axs = plt.subplots(1, 1, figsize=(15, 5))
        sns.heatmap(data=correl)
        plt.tight_layout()
        plt.show()
