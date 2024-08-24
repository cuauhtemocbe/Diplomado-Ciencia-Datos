"""
El modulo sirve para analizar la completitud de un conjunto de datos y 
realizar gráficas por tipo de variable (Discretas y continuas)
"""

__version__ = "1.0.3"

import collections
from datetime import datetime
from typing import List, Tuple, Union, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from IPython.display import clear_output
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from sklearn import metrics
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV, SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.naive_bayes import ComplementNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (MinMaxScaler, OneHotEncoder, RobustScaler,
                                   StandardScaler)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from varclushi import VarClusHi
from xgboost import XGBClassifier

ModelClassifier: TypeAlias = Union[SVC, KNeighborsClassifier, DecisionTreeClassifier,
                       GaussianNB, MultinomialNB, ComplementNB,
                       LogisticRegression, XGBClassifier]


class GroupNumericalFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.bins_dict = {}  # Diccionario para almacenar los bins calculados por columna

    def fit(self, X):
        self.bins_dict = {}
        self.numerical_columns = X.select_dtypes(include=["number"]).columns  # Obtener columnas numéricas
        
        for col in self.numerical_columns:
            # Calcular bins basados en cuantiles de la columna especificada
            quantiles = np.unique(
                X[col].quantile([0, 0.2, 0.4, 0.6, 0.8, 1]).values
            )
            bins = list(quantiles[:-1]) + [quantiles[-1] + 0.01]
            self.bins_dict[col] = bins
        return self

    def transform(self, X):
        X_transformed = pd.DataFrame(index=X.index)
        
        for col in self.bins_dict:
            bins = self.bins_dict[col]
            labels = [f"{bins[i]}_a_{bins[i+1]}" for i in range(len(bins)-1)]
            X_transformed[f"cat_{col}"] = pd.cut(X[col], bins=bins, labels=labels, right=False)
        
        return X_transformed

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


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
        result_df = result_df.rename(columns={'index': 'features'})

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


def check_time_format(time_str, format="%m/%d/%Y"):
    try:
        datetime.strptime(str(time_str), format)
        return True
    except ValueError:
        return False


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
    scores_df = pd.DataFrame(
        list(zip(data_features.columns, kb.scores_)), columns=["Feature", "Score"]
    ).sort_values(by="Score", ascending=False)

    return scores_df.reset_index(drop=True)


def get_varclushi(data):
    """
    Esta función utiliza la librería VarClusHi para realizar un análisis de clusters de variables en los datos proporcionados.
    Calcula la información detallada sobre los clusters de variables y los valores de R-cuadrado para cada variable en relación a su cluster.
    Además, selecciona las mejores características (variables) basadas en el análisis de clusters.

    Parámetros:
    data (pd.DataFrame o np.ndarray): Los datos de entrada sobre los que se realizará el análisis de clusters de variables.

    Retorna:
    tuple: Una tupla que contiene:
        - info (pd.DataFrame): Un DataFrame de pandas con información detallada sobre los clusters de variables.
        - rsquare (pd.DataFrame): Un DataFrame de pandas con los valores de R-cuadrado para cada variable en relación a su cluster.
        - best_features (pd.Series): Una serie pandas que contiene las mejores características (variables) seleccionadas basadas en el análisis de clusters.

    Ejemplo:
    >>> info, rsquare, best_features = get_varclushi(data)
    """
    vc = VarClusHi(data)
    vc.varclus()

    vc_info = vc.info
    vc_rsquare_df = vc.rsquare

    best_features = (
        vc_rsquare_df.sort_values(by=["Cluster", "RS_Ratio"], ascending=False)
        .groupby("Cluster")
        .first()["Variable"]
    )

    return vc_info, vc_rsquare_df, best_features


def get_information_value(df, var, tgt):
    """
    Calcula el Valor de Información (IV) de una variable en un DataFrame con respecto a una variable objetivo.

    Parámetros:
    - df (DataFrame): El DataFrame que contiene las variables de interés.
    - var (str): El nombre de la variable para la cual se calculará el IV.
    - tgt (str): El nombre de la variable objetivo.

    Retorna:
    - float: El Valor de Información (IV) de la variable especificada con respecto a la variable objetivo.

    Ejemplo:
    >>> IV(df, 'variable', 'objetivo')
    0.1234
    """
    # Agrupar el DataFrame por la variable de interés y calcular las frecuencias y sumas
    aux = df[[var, tgt]].groupby(var).agg(["count", "sum"])

    # Calcular el número de eventos y no eventos para cada categoría de la variable de interés
    aux["evento"] = aux[tgt, "sum"]
    aux["no_evento"] = aux[tgt, "count"] - aux[tgt, "sum"]

    # Calcular la proporción de eventos y no eventos para cada categoría
    aux["%evento"] = aux["evento"] / aux["evento"].sum()
    aux["%no_evento"] = aux["no_evento"] / aux["no_evento"].sum()

    # Calcular el Weight of Evidence (WOE) para cada categoría
    aux["WOE"] = np.log(aux["%no_evento"] / aux["%evento"])

    # Calcular el IV para cada categoría
    aux["IV"] = (aux["%no_evento"] - aux["%evento"]) * aux["WOE"]

    # Sumar todos los IVs de las categorías para obtener el IV total de la variable
    return aux["IV"].sum()


def count_percentage(df, columna):
    # Realizar el conteo de valores en la columna especificada
    conteo = df[columna].value_counts().reset_index()
    conteo.columns = [columna, "conteo"]

    # Calcular los porcentajes
    total = conteo["conteo"].sum()
    conteo["porcentaje"] = round((conteo["conteo"] / total) * 100, 2)

    return conteo


def create_feature_dataframe(data, column):
    feature = data.columns[0]
    category = data.at[0, column]
    conteo = data.at[0, "conteo"]
    porcentaje = data.at[0, "porcentaje"]

    # Constructing the final dictionary
    variable = {
        "feature": feature,
        "category": category,
        "conteo": conteo,
        "porcentaje": porcentaje,
    }

    # Converting the dictionary to a DataFrame
    return pd.DataFrame([variable])


def transform_data(X_train, X_test, numerical_features=None,
                   categorical_features=None):
    
    numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler()),  
    ])

    categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ])

    transformers = []
    if numerical_features:
        transformers.append(("num", numerical_transformer, numerical_features))

    if categorical_features:
        transformers.append(("cat", categorical_transformer, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers)

    # Ajustar y transformar los datos de entrenamiento
    X_train_transformed_data = preprocessor.fit_transform(X_train)
    X_test_transformed_data = preprocessor.transform(X_test)
    # # Obtener los nombres de las columnas después de aplicar OneHotEncoder
    # onehot_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    # categorical_features_encoded = onehot_encoder.get_feature_names_out(
    #     input_features=categorical_features)

    # if numerical_features:
    #     feature_names =  numerical_features + list(categorical_features_encoded)
    # else:
    #     feature_names = list(categorical_features)
     # Get feature names after transformation
    feature_names = []
    if numerical_features:
        feature_names += numerical_features
    if categorical_features:
        onehot_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        categorical_features_encoded = onehot_encoder.get_feature_names_out(
            input_features=categorical_features)
        feature_names += list(categorical_features_encoded)

    train_index = X_train.index
    test_index = X_test.index

    # Convertir los datos transformados a DataFrames de Pandas
    X_train_transformed_df = pd.DataFrame(X_train_transformed_data, columns=feature_names, index=train_index)
    X_test_transformed_df = pd.DataFrame(X_test_transformed_data, columns=feature_names, index=test_index)


    return X_train_transformed_df, X_test_transformed_df, preprocessor


def perform_grid_search(X_train, y_train, model, param_grid, cv=3, verbose=True):

    if verbose:
        verbose = 3
    else:
        verbose = 1

    grid_search = GridSearchCV(
        model, param_grid, cv=StratifiedKFold(n_splits=cv),
        scoring="f1_micro", n_jobs=-1, error_score=-1, verbose=verbose
    )
    # Entrenamiento
    grid_search.fit(X_train, y_train)
    # Obtener los mejores hiperparámetros y el mejor modelo
    best_model = grid_search.best_estimator_
    print("Mejores hiperparámetros encontrados GridSearchCV:")
    print(grid_search.best_params_)

    return best_model, grid_search.best_params_


def cross_validation_report(model, X_train, y_train, verbose):
    cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=10,
            scoring="f1_micro",
        )

    f1_mean_score_train = cv_scores.mean()
    std_dev_train = round(np.std(cv_scores), 4)

    if verbose:
        print(">>> F1 Macro Score de validación cruzada (train):", f1_mean_score_train)
        print(">>> Standar deviation (train):", std_dev_train)

    return f1_mean_score_train, std_dev_train


def test_report(model, X_test, y_test, verbose):
    # Predicción y evaluación en el conjunto de prueba
    y_pred = model.predict(X_test)
    # Evaluación del modelo en el conjunto de prueba
    f1_score_test = f1_score(y_test, y_pred)

    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc_test = round(roc_auc_score(y_test, y_prob), 2)

    except:
        roc_auc_test = np.nan

    if verbose:
        print(">>> F1 Macro Score en el conjunto de prueba (test):", f1_score_test)
        print("\nReporte de Clasificación en el conjunto de prueba:")
        print(classification_report(y_test, y_pred))

    return f1_score_test, roc_auc_test


def train_classifier_model(X_train, X_test, y_train, y_test, model, param_grid=None, verbose: bool = True):
    model_name = type(model).__name__

    if model_name in ["SVC"]:
        X_train = csr_matrix(X_train)
        X_test = csr_matrix(X_test)
    # Si se proporciona un grid de parámetros, realizar GridSearchCV
    if param_grid:
        best_model, best_params = perform_grid_search(
            X_train, y_train, model, param_grid, verbose=verbose)
    else:
        best_params = ""
        best_model = model
        best_model.fit(X_train, y_train)

    f1_mean_score_train, std_dev_train = cross_validation_report(
        best_model, X_train, y_train, verbose)
    f1_score_test, roc_auc_test = test_report(best_model, X_test, y_test, verbose)
    
    return (best_model, best_params, f1_mean_score_train, std_dev_train,
            f1_score_test, roc_auc_test)


def evaluate_models(params_dict, X_train, X_test, y_train, y_test):
    results_list = []
    for model_name, model in params_dict.items():
        results = train_classifier_model(
            X_train, X_test, y_train, y_test, model=model, verbose=False
        )

        _, _, f1_mean_score_train, std_dev_train, f1_score_test, roc_auc_test = results

        results_list.append([
            model_name, f1_mean_score_train, std_dev_train, f1_score_test, roc_auc_test
        ])
        
    return pd.DataFrame(results_list, columns=[
        "model", "f1-score-train", "std-dev", "f1-score-test", "roc-auc-test"])


def get_best_features_rfecv(X, y, model, scoring):
    
    rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(5), scoring=scoring)
    rfecv.fit(X, y)

    return X.columns[rfecv.support_].tolist()


def get_feature_importances(model, X):
    coefficients = pd.Series(model.coef_.flatten())
    features_df = pd.DataFrame(
        {
            "Características": pd.Series(X.columns),
            "Coeficientes": coefficients
        }
    )

    features_df["Importancia"] = features_df["Coeficientes"].abs()
    features_df = features_df.sort_values(by="Importancia", ascending=False)
    return features_df.reset_index(drop=True)


def freq_discrete(df, features):
    for feature in features:
        print(f"Feature: {feature}")
        abs_ = df[feature].value_counts(dropna=False).to_frame().rename(columns={"count": "Absolute frequency"})
        rel_ = df[feature].value_counts(dropna=False, normalize= True).to_frame().rename(columns={"proportion": "Relative frequency"})
        freq = abs_.join(rel_)
        freq["Accumulated frequency"] = freq["Absolute frequency"].cumsum()
        freq["Accumulated %"] = freq["Relative frequency"].cumsum()
        freq["Absolute frequency"] = freq["Absolute frequency"].map(lambda x: "{:,.0f}".format(x))
        freq["Relative frequency"] = freq["Relative frequency"].map(lambda x: "{:,.2%}".format(x))
        freq["Accumulated frequency"] = freq["Accumulated frequency"].map(lambda x: "{:,.0f}".format(x))
        freq["Accumulated %"] = freq["Accumulated %"].map(lambda x: "{:,.2%}".format(x))
        display(freq)


def get_features_by_xgb_importance(
        model: XGBClassifier, importance_type: str) -> List:
    """Returns a list with the features sorted by importance"""

    imp_scores_d = model.get_booster().get_score(
        importance_type=importance_type)
    sorted_imp = sorted(imp_scores_d.items(), key=lambda kv: kv[1])
    sorted_dict = collections.OrderedDict(sorted_imp)

    return [key for key in sorted_dict.keys()]

def estimate_score_metrics(y_test: pd.Series,
                           y_pred: np.ndarray,
                           y_prob: np.ndarray
                           ) -> Tuple[float, float, int, int, int, int]:
    """Returns the following evaluation metrics: ROC, ROC_AUC,
    \rF1-score, Recall, Accuracy, Brier"""
    roc = round(metrics.roc_auc_score(y_test, y_pred), 2)
    roc_auc = round(metrics.roc_auc_score(y_test, y_prob), 2)

    f1 = round(metrics.f1_score(y_test, y_pred) * 100)
    recall = round(metrics.recall_score(y_test, y_pred) * 100)
    accuracy = round(metrics.accuracy_score(y_test, y_pred) * 100)
    brier = round(metrics.brier_score_loss(y_test, y_pred) * 100)

    return roc, roc_auc, f1, recall, accuracy, brier


def get_total_iterations(model, importance_types_list: List) -> int:
    """Returns the total of iterations for the modeling by xgb
    feature importance"""
    no_elements = 1
    for imp_type in importance_types_list:
        features = get_features_by_xgb_importance(
            model=model, importance_type=imp_type)

        while len(features) > 0:
            _ = features.pop(0)
            no_elements += 1

    return no_elements

def modeling_by_subset(
        model: ModelClassifier,
        x_train: pd.DataFrame,
        y_train: Union[pd.Series, pd.DataFrame],
        features: List) -> np.array:
    """Returns an array with Machine Learning model, identifier of model,
    number of features, metrics scores, """
    return model.fit(x_train[features], y_train)


def predict_by_subset(
        predictor: ModelClassifier,
        x_test: pd.DataFrame,
        features: List) -> np.array:
    """Returns an array with Machine Learning model, identifier of model,
    number of features, metrics scores, """
    x_test_subset = x_test[features]
    y_pred = predictor.predict(x_test_subset)
    y_prob = np.around(predictor.predict_proba(x_test_subset)[:, 1], 2)

    return y_pred, y_prob


def modeling_by_xgb_importance(
        model_name: str,
        model: ModelClassifier,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_test) -> pd.DataFrame:
    """Returns a dataframe of models scores using xgboost feature
    \r importance to select the best features
    """
    gral_model = XGBClassifier(n_jobs=-1)
    gral_model_fitted = gral_model.fit(x_train, y_train)
    imp_types_lst = ['total_gain', 'total_cover', 'weight', 'gain', 'cover']

    no_elements = get_total_iterations(gral_model_fitted, imp_types_lst)
    count = 1
    row_lst: List[np.array] = []
    row_array = np.array(row_lst)

    for importance_type in imp_types_lst:
        features_list = get_features_by_xgb_importance(
            model=gral_model_fitted, importance_type=importance_type)

        while len(features_list) > 0:

            predictor = modeling_by_subset(model=clone(model),
                                           x_train=x_train,
                                           y_train=y_train,
                                           features=features_list)

            y_pred, y_prob = predict_by_subset(predictor=predictor,
                                               x_test=x_test,
                                               features=features_list)

            score_metrics = estimate_score_metrics(
                y_test=y_test, y_pred=y_pred, y_prob=y_prob)

            row_lst.append(np.array([
                model_name,
                'model_' + str(count),
                len(features_list),
                *score_metrics,
                importance_type,
                ','.join(features_list)]))

            count += 1
            _ = features_list.pop(0)

            txt = 'of ' + ' Modeling with ' + str(model_name) + ' :'
            update_progress(count / no_elements, progress_text=txt)

        row_array = np.array(row_lst)

    # Names of columns of info value dataframe
    cols_dict = {
        'Models': str, 'Id': str, 'No_features': int, 'ROC': float,
        'ROC_AUC': float, 'F1': float, 'Recall': float, 'Accuracy': float,
        'Brier': float, 'Importance_type': str, 'Best_features': str,}

    cols_name = [names for names in cols_dict]
    df = pd.DataFrame(data=row_array, columns=cols_name)
    df = df.astype(cols_dict)

    df = df.sort_values(['F1', 'ROC_AUC', 'No_features'],
                        ascending=False).reset_index(drop=True)

    clear_output(wait=False)

    return df


def update_progress(progress, progress_text=''):
    """ Print the progress of a 'FOR' inside a function """

    bar_length = 40
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    progress = max(progress, 0)
    progress = min(progress, 1)
    block = int(round(bar_length * progress))
    clear_output(wait=True)
    text = ' '.join(['Progress', progress_text, '[{0}] {1:.1f}%'])
    ouput_text = text.format("#" * block + "-" * (bar_length - block),
                             progress * 100)

    print(ouput_text)