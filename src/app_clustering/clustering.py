import os
import re
import unicodedata
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import umap
from dotenv import load_dotenv
from googleapiclient.discovery import build
from plotly.subplots import make_subplots
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from sklearn import set_config
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    calinski_harabasz_score,
    pairwise_distances,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from transformers import pipeline
from wordcloud import WordCloud

if os.getenv("RAILWAY_ENVIRONMENT") is None:
    load_dotenv()

api_key = os.getenv("youtube_api_key")

RANDOM_STATE = 333

stopwords_es = [
    "a",
    "al",
    "algo",
    "algún",
    "alguna",
    "algunas",
    "alguno",
    "algunos",
    "ante",
    "antes",
    "bajo",
    "bastante",
    "bien",
    "cada",
    "casi",
    "como",
    "con",
    "cuanto",
    "de",
    "del",
    "desde",
    "donde",
    "durante",
    "el",
    "ella",
    "ellos",
    "en",
    "encima",
    "ese",
    "eso",
    "esta",
    "estas",
    "este",
    "estos",
    "fuera",
    "hay",
    "la",
    "las",
    "le",
    "lo",
    "los",
    "más",
    "me",
    "mi",
    "mí",
    "menos",
    "mismo",
    "mucho",
    "muy",
    "nada",
    "ni",
    "no",
    "nos",
    "nuestro",
    "nuestra",
    "o",
    "os",
    "para",
    "pero",
    "poco",
    "por",
    "que",
    "quien",
    "si",
    "sólo",
    "sobre",
    "su",
    "sus",
    "te",
    "tu",
    "tus",
    "un",
    "una",
    "unas",
    "uno",
    "unos",
    "vos",
    "ya",
    "yo",
    "además",
    "alrededor",
    "aún",
    "bajo",
    "bien",
    "cada",
    "cierta",
    "ciertas",
    "como",
    "con",
    "de",
    "debe",
    "dentro",
    "dos",
    "ella",
    "en",
    "entonces",
    "entre",
    "esa",
    "esos",
    "está",
    "hasta",
    "incluso",
    "lejos",
    "lo",
    "luego",
    "medio",
    "mientras",
    "muy",
    "nunca",
    "o",
    "otro",
    "para",
    "pero",
    "poco",
    "por",
    "se",
    "si",
    "sin",
    "sobre",
    "tan",
    "te",
    "ten",
    "tendría",
    "todos",
    "total",
    "un",
    "una",
    "uno",
    "ustedes",
    "yo",
    "y",
    "es",
    "son",
    "solo",
    "les",
]


def normalize_text(text):
    text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("ASCII")
    text = text.lower()
    return text


def remove_stopwords(text, stopwords):
    # Divide el texto en palabras y elimina las stopwords
    return [word for word in text.split() if word not in stopwords]


def plot_wordcloud(data, text_column, output_filename=None):
    text = " ".join(data[text_column])

    stopwords_set = set(stopwords_es)

    normalized_text = normalize_text(text)
    cleaned_text = remove_stopwords(normalized_text, stopwords_set)
    filtered_text = replace_html_entities(" ".join(cleaned_text))

    # Crear la nube de palabras usando los conteos
    wordcloud = WordCloud(
        width=800, height=400, background_color="white", normalize_plurals=True
    ).generate(filtered_text)

    # Mostrar la nube de palabras
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")

    if output_filename:
        plt.savefig(output_filename, format="png")
        plt.close()
        return output_filename


def extract_video_id(url):
    """
    Extrae el video_id de una URL de YouTube.

    Parámetros:
    - url: str, la URL del video de YouTube.

    Retorna:
    - video_id: str, el identificador del video de YouTube.
    """
    # Expresión regular para encontrar el video_id en una URL de YouTube
    pattern = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(pattern, url)

    if match:
        return match.group(1)
    else:
        raise ValueError("No se pudo encontrar un ID de video en la URL proporcionada.")


def get_youtube_video_details(url, api_key):
    """
    Obtiene detalles de un video de YouTube usando la API de YouTube Data v3.

    :param video_id: ID del video de YouTube.
    :param api_key: Clave de API de YouTube Data v3.
    :return: Un diccionario con el nombre del video, el canal, el número de vistas y el número de comentarios.
    """
    try:
        youtube = build("youtube", "v3", developerKey=api_key)

        video_id = extract_video_id(url)

        request = youtube.videos().list(part="snippet,statistics", id=video_id)
        response = request.execute()

        if "items" in response and len(response["items"]) > 0:
            video = response["items"][0]
            details = {
                "title": video["snippet"]["title"],
                "channel_title": video["snippet"]["channelTitle"],
                "view_count": video["statistics"].get("viewCount", "No disponible"),
                "comment_count": video["statistics"].get(
                    "commentCount", "No disponible"
                ),
            }
            return details
        else:
            return {"error": "No se encontró el video con el ID proporcionado."}
    except Exception as e:
        return {"error": str(e)}


def get_youtube_comments(api_key, url, max_results=100):
    """
    Obtiene comentarios de un video de YouTube y los convierte en un DataFrame de pandas.

    Parámetros:
    - api_key: str, la clave de API de YouTube.
    - video_id: str, el ID del video de YouTube.
    - max_results: int, el número máximo de comentarios a obtener por solicitud (predeterminado es 100).

    Retorna:
    - df: pandas DataFrame, contiene los comentarios del video.
    """

    # Crear el servicio de la API de YouTube
    youtube = build("youtube", "v3", developerKey=api_key)

    # Solicitar los comentarios del video
    video_id = extract_video_id(url)
    request = youtube.commentThreads().list(
        part="snippet", videoId=video_id, maxResults=max_results
    )

    response = request.execute()

    # Lista para almacenar los datos de los comentarios
    comments_data = []

    # Procesar y almacenar los comentarios en la lista
    for item in response["items"]:
        comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        author = item["snippet"]["topLevelComment"]["snippet"]["authorDisplayName"]
        published_at = item["snippet"]["topLevelComment"]["snippet"]["publishedAt"]

        comments_data.append(
            {"author": author, "comment": comment, "published_at": published_at}
        )

    # Paginar y obtener más comentarios si hay más disponibles
    next_page_token = response.get("nextPageToken")

    while next_page_token:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            pageToken=next_page_token,
            maxResults=max_results,
        )
        response = request.execute()

        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            author = item["snippet"]["topLevelComment"]["snippet"]["authorDisplayName"]
            published_at = item["snippet"]["topLevelComment"]["snippet"]["publishedAt"]

            comments_data.append(
                {"author": author, "comment": comment, "published_at": published_at}
            )

        next_page_token = response.get("nextPageToken")

    # Convertir la lista de comentarios en un DataFrame de pandas
    df = pd.DataFrame(comments_data)

    return df


def add_normalized_embeddings_to_dataframe(
    data, text_column, model_name="paraphrase-multilingual-MiniLM-L12-v2"
):
    """
    Genera y normaliza embeddings para una columna de texto en un DataFrame y agrega estos embeddings como nuevas columnas.

    Parámetros:
    - data: pandas DataFrame, el DataFrame que contiene la columna de texto.
    - text_column: str, el nombre de la columna en el DataFrame que contiene el texto para generar embeddings.
    - model_name: str, el nombre del modelo de SentenceTransformer a utilizar (por defecto "sentence-transformers/stsb-xlm-r-multilingual").

    Retorna:
    - data: pandas DataFrame, el DataFrame original con las nuevas columnas de embeddings normalizados.
    """

    model = SentenceTransformer(model_name)
    sentences = data[text_column].tolist()
    embeddings = model.encode(sentences)
    normalized_embeddings = normalize(embeddings, norm="l2")

    data["embeddings"] = [embedding for embedding in normalized_embeddings]

    return data


def plot_k_distance(data, threshold=0.01, quantile=0.95):
    # embeddings_matrix = np.array(data["embeddings"].tolist())
    embeddings_matrix = data.copy()

    for threshold in [threshold, 0.05, 0.1, 0.2]:
        min_samples = int(round(data.shape[0] * threshold, 0))
        n_neighbors = min_samples - 1

        if n_neighbors > 2:
            nn = NearestNeighbors(
                n_neighbors=n_neighbors, algorithm="auto", metric="cosine", n_jobs=-1
            )
            nn.fit(embeddings_matrix)
            distances, _ = nn.kneighbors(embeddings_matrix)
            k_distances = distances[:, -1]
            min_eps = np.percentile(k_distances, quantile * 100)
            k_distances = np.sort(k_distances)
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=k_distances, mode="lines", name="k-distances"))
            fig.add_hline(
                y=min_eps,
                line=dict(color="red", dash="dash"),
                name=f"min_eps = {min_eps:.2f}",
            )
            fig.update_layout(
                title="k-Distance Graph",
                xaxis_title="Index",
                yaxis_title="Distance",
                width=800,
                height=600,
                template="plotly_dark",
            )
            return fig, min_eps
    return None, None


def find_most_similar_comment(cluster_data, avg_embedding):
    similarities = [
        1 - cosine(avg_embedding, emb) for emb in cluster_data["embeddings"]
    ]
    most_similar_index = np.argmax(similarities)

    return cluster_data.iloc[most_similar_index]["comment"]


def format_text(text, line_length=50):
    """
    Formatea el texto agregando saltos de línea cada 'line_length' caracteres.

    :param text: El texto a formatear.
    :param line_length: La longitud máxima de cada línea (por defecto 50 caracteres).
    :return: El texto formateado con saltos de línea.
    """
    # Divide el texto en partes de longitud 'line_length'
    formatted_text = "<br>".join(
        text[i : i + line_length] for i in range(0, len(text), line_length)
    )
    return formatted_text


def replace_html_entities(text):
    """
    Reemplaza entidades HTML conocidas en el texto con sus caracteres correspondientes.

    :param text: El texto con entidades HTML.
    :return: El texto con las entidades reemplazadas.
    """
    replacements = {
        "&quot;": '"',
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
        "<br>": "\n",  # Reemplazar <br> con salto de línea
    }

    for entity, char in replacements.items():
        text = text.replace(entity, char)

    return text


def plot_sentiment_global(
    data,
    sentimiento_col="sentimiento",
    title="Evolución de Comentarios por Sentimiento",
    width=1200,
    height=600,
):
    """"""
    df_global = data[sentimiento_col].value_counts().reset_index()
    df_global.columns = [sentimiento_col, "count"]

    fig_global = go.Figure()

    color_palette = {"positivo": "#138d75", "negativo": "#a93226", "neutro": "#909497"}

    for sentimiento in df_global[sentimiento_col].unique():
        df_sentimiento = df_global[df_global[sentimiento_col] == sentimiento]
        fig_global.add_trace(
            go.Bar(
                x=df_sentimiento[sentimiento_col],
                y=df_sentimiento["count"],
                text=df_sentimiento["count"],
                textposition="inside",
                insidetextanchor="middle",
                name=sentimiento,
                marker=dict(color=color_palette[sentimiento]),
            )
        )

    fig_global.update_layout(
        title=f"{title} - Global",
        xaxis_title="Sentimiento",
        yaxis_title="Número Total de Comentarios",
        legend_title="Sentimiento",
        template="plotly_dark",
        width=width,
        height=height,
    )

    return fig_global


def plot_sentiment_daily(
    data,
    fecha_col="published_at",
    sentimiento_col="sentimiento",
    title="Evolución de Comentarios por Sentimiento",
    width=1200,
    height=600,
):
    """"""
    data[fecha_col] = pd.to_datetime(data[fecha_col])

    df_grouped = (
        data.groupby([pd.Grouper(key=fecha_col, freq="D"), sentimiento_col])
        .size()
        .reset_index(name="count")
    )

    df_grouped["total_daily"] = df_grouped.groupby(pd.Grouper(key=fecha_col, freq="D"))[
        "count"
    ].transform("sum")
    df_grouped["percentage"] = df_grouped["count"] / df_grouped["total_daily"] * 100

    fig_daily = go.Figure()

    color_palette = {"positivo": "#138d75", "negativo": "#a93226", "neutro": "#909497"}

    for sentimiento in data[sentimiento_col].unique():
        df_sentimiento = df_grouped[df_grouped[sentimiento_col] == sentimiento]
        fig_daily.add_trace(
            go.Bar(
                x=df_sentimiento[fecha_col],
                y=df_sentimiento["total_daily"],
                name=sentimiento,
                text=df_sentimiento["count"],
                texttemplate="%{text}",
                textposition="inside",
                insidetextanchor="middle",
                customdata=df_sentimiento["percentage"],
                hovertemplate="<b>Fecha</b>: %{x}<br><b>Sentimiento</b>: %{name}<br><b>Porcentaje</b>: %{customdata:.1f}%<br><b>Total de Comentarios</b>: %{text}<extra></extra>",  # Información emergente con porcentaje y total
                marker=dict(color=color_palette[sentimiento]),
            )
        )

    fig_daily.update_layout(
        title=f"{title} - Por Día",
        xaxis_title="Fecha",
        yaxis_title="Total de Comentarios",
        legend_title="Sentimiento",
        barmode="stack",
        template="plotly_dark",
        width=width,
        height=height,
    )

    return fig_daily


def create_3d_umap_plot(data):

    def calculate_sentiment_info(data):
        cluster_sentiments = (
            data.groupby("Cluster")["sentimiento"].value_counts().unstack(fill_value=0)
        )
        total_by_cluster = cluster_sentiments.sum(axis=1)
        sentiment_percentages = (
            cluster_sentiments.div(total_by_cluster, axis=0) * 100
        ).round(2)

        sentiment_info = {}
        for cluster in total_by_cluster.index:
            info = [
                f"{sentiment}: {count} ({percent}%)"
                for sentiment, count, percent in zip(
                    cluster_sentiments.columns,
                    cluster_sentiments.loc[cluster],
                    sentiment_percentages.loc[cluster],
                )
            ]
            sentiment_info[cluster] = (
                f"Total {total_by_cluster[cluster]}<br>" + "<br>".join(info)
            )

        return sentiment_info

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=data["UMAP1"],
            y=data["UMAP2"],
            z=data["UMAP3"],
            mode="markers",
            marker=dict(
                size=3,
                color=data["Cluster"],
                colorscale="Viridis",
                colorbar=dict(title="Cluster"),
            ),
            text=data["sentimiento"],
            name="Puntos",
        )
    )

    fig.update_layout(
        scene=dict(xaxis_title="UMAP 1", yaxis_title="UMAP 2", zaxis_title="UMAP 3"),
        template="plotly_dark",
        title="Visualización 3D con UMAP y Clustering",
    )

    sentiment_info = calculate_sentiment_info(data)

    hovertemplate = (
        "Cluster: %{marker.color}<br>"
        + data["Cluster"].map(sentiment_info)
        + "<br>"
        + "<extra></extra>"
    )

    fig.update_traces(hovertemplate=hovertemplate)

    fig.show()


def perform_clustering(data, min_eps, max_eps=0.95, n=5, embeddings_col="embeddings"):

    embeddings_matrix = np.array(data[embeddings_col].tolist())
    # threshold_values = np.round(np.linspace(min_eps, max_eps, n), 2)
    threshold_values = np.linspace(min_eps, max_eps, n)

    cluster_assignments = {}
    cluster_counts = {}
    calinski_harabasz_scores = {}
    silhouette_scores = {}
    most_similar_comments = {}

    for distance_threshold in threshold_values:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            linkage="complete",
            metric="cosine",
        )
        data[f"cluster_{distance_threshold}"] = clustering.fit_predict(
            embeddings_matrix
        )
        cluster_assignments[distance_threshold] = data[f"cluster_{distance_threshold}"]
        cluster_counts[distance_threshold] = data[
            f"cluster_{distance_threshold}"
        ].value_counts()
        labels = data[f"cluster_{distance_threshold}"]

        # Calcular Calinski-Harabasz Score
        if len(np.unique(labels)) > 1:
            # Recalcular matriz de distancias con base en los clusters
            euclidean_distances = pairwise_distances(
                embeddings_matrix, metric="euclidean"
            )
            ch_score = calinski_harabasz_score(euclidean_distances, labels)
        else:
            ch_score = -1  # Valor predeterminado si solo hay un clúster
        calinski_harabasz_scores[distance_threshold] = ch_score

        # Calcular Silhouette Score
        if len(np.unique(labels)) > 1:
            sil_score = silhouette_score(embeddings_matrix, labels, metric="cosine")
        else:
            sil_score = -1  # Valor predeterminado si solo hay un clúster
        silhouette_scores[distance_threshold] = sil_score

        # Placeholder for finding the most similar comment function
        most_similar_comments[distance_threshold] = {}
        for cluster_id in np.unique(labels):
            cluster_data = data[data[f"cluster_{distance_threshold}"] == cluster_id]
            avg_embedding = np.mean(cluster_data[embeddings_col].tolist(), axis=0)
            # Replace with your actual implementation
            most_similar_comment = find_most_similar_comment(
                cluster_data, avg_embedding
            )
            most_similar_comments[distance_threshold][cluster_id] = most_similar_comment

    return (
        cluster_assignments,
        cluster_counts,
        calinski_harabasz_scores,
        silhouette_scores,
        most_similar_comments,
        data,
    )


def build_sankey_data(
    cluster_assignments,
    cluster_counts,
    most_similar_comments,
    min_items_by_cluster=10,
):
    labels = []
    source = []
    target = []
    values = []
    comments = []

    threshold_values = sorted(cluster_assignments.keys())
    valid_clusters = {}

    for threshold in threshold_values:
        valid_clusters[threshold] = [
            j
            for j in np.unique(cluster_assignments[threshold])
            if cluster_counts[threshold].get(j, 0) >= min_items_by_cluster
        ]

    for i, threshold in enumerate(threshold_values):
        for j in valid_clusters[threshold]:
            cluster_name = (
                f"{j} (d={threshold})\nTotal: {cluster_counts[threshold].get(j, 0)}"
            )
            if cluster_name not in labels:
                labels.append(cluster_name)
                comments.append(
                    format_text(
                        replace_html_entities(
                            most_similar_comments[threshold].get(j, "N/A")
                        )
                    )
                )

        if i > 0:
            prev_threshold = threshold_values[i - 1]
            for prev_cluster in valid_clusters[prev_threshold]:
                for curr_cluster in valid_clusters[threshold]:
                    count = np.sum(
                        (cluster_assignments[prev_threshold] == prev_cluster)
                        & (cluster_assignments[threshold] == curr_cluster)
                    )
                    if count > 0:
                        source_idx = labels.index(
                            f"{prev_cluster} (d={prev_threshold})\nTotal: {cluster_counts[prev_threshold].get(prev_cluster, 0)}"
                        )
                        target_idx = labels.index(
                            f"{curr_cluster} (d={threshold})\nTotal: {cluster_counts[threshold].get(curr_cluster, 0)}"
                        )
                        source.append(source_idx)
                        target.append(target_idx)
                        values.append(count)

    return (labels, source, target, values, comments)


def plot_sankey(labels, source, target, values, comments, width=None, height=None):
    fig = go.Figure(
        go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0),
                label=labels,
                hovertemplate="<b>%{label}</b><br>"
                + "<br><b>Commentario:</b><br>%{customdata}<extra></extra>",
                customdata=comments,
            ),
            link=dict(
                source=source,
                target=target,
                value=values,
                hovertemplate="<extra></extra>",
            ),
        )
    )
    fig.update_layout(
        title_text="Sankey Diagram of Agglomerative Clustering Transitions",
        font_size=14,
        width=width,
        height=height,
        template="plotly_dark",
    )

    return fig


def plot_clustering_metric(silhouette_scores, calinski_scores):
    """
    Genera un gráfico que muestra los puntajes de silhouette y Calinski-Harabasz frente a los umbrales de distancia,
    con dos ejes Y diferentes y marca el umbral con el mejor puntaje de silhouette.

    Args:
        silhouette_scores (dict): Un diccionario donde las claves son umbrales de distancia
                                  y los valores son puntajes de silhouette correspondientes.
        calinski_scores (dict): Un diccionario donde las claves son umbrales de distancia
                                y los valores son puntajes de Calinski-Harabasz correspondientes.

    Returns:
        fig (plotly.graph_objects.Figure): Un objeto Figure de Plotly con el gráfico generado.
    """
    # Obtener los umbrales de distancia y puntajes
    silhouette_thresholds = sorted(silhouette_scores.keys())
    silhouette_metric_scores = [silhouette_scores[t] for t in silhouette_thresholds]

    calinski_thresholds = sorted(calinski_scores.keys())
    calinski_metric_scores = [calinski_scores[t] for t in calinski_thresholds]

    # Determinar el mejor umbral basado en el puntaje más alto de silhouette
    best_threshold = max(silhouette_scores, key=silhouette_scores.get)

    # Crear el gráfico con dos ejes Y
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Añadir la traza para el puntaje de silhouette
    fig.add_trace(
        go.Scatter(
            x=silhouette_thresholds,
            y=silhouette_metric_scores,
            mode="lines+markers",
            name="Silhouette Score",
            marker=dict(color="red", size=10),
            line=dict(color="red", width=2),
            text=[
                f"Threshold: {t}<br>Silhouette Score: {s}"
                for t, s in zip(silhouette_thresholds, silhouette_metric_scores)
            ],
            hoverinfo="text",
        ),
        secondary_y=False,  # Eje Y izquierdo
    )

    # Añadir la traza para el puntaje de Calinski-Harabasz
    fig.add_trace(
        go.Scatter(
            x=calinski_thresholds,
            y=calinski_metric_scores,
            mode="lines+markers",
            name="Calinski-Harabasz Score",
            marker=dict(color="blue", size=10),
            line=dict(color="blue", width=2),
            text=[
                f"Threshold: {t}<br>Calinski-Harabasz Score: {s}"
                for t, s in zip(calinski_thresholds, calinski_metric_scores)
            ],
            hoverinfo="text",
        ),
        secondary_y=True,  # Eje Y derecho
    )

    # Añadir una línea vertical para el mejor umbral
    fig.add_vline(
        x=best_threshold,
        line=dict(color="green", width=2, dash="dash"),
        annotation_text=f"Best Threshold: {best_threshold}",
        annotation_position="top right",
    )

    # Configurar el diseño del gráfico
    fig.update_layout(
        title="Clustering Metrics vs. Threshold Distance",
        xaxis_title="Threshold Distance",
        yaxis_title="Silhouette Score",
        yaxis2_title="Calinski-Harabasz Score",
        font=dict(size=12),
        width=800,
        height=600,
        template="plotly_dark",
    )

    return fig, best_threshold


classifier = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    truncation=True,
)


def map_sentiment(estrella):
    if estrella in ["1 star", "2 stars"]:
        return "negativo"
    elif estrella == "3 stars":
        return "neutro"
    elif estrella in ["4 stars", "5 stars"]:
        return "positivo"


def classify_sentiment(texto):
    resultado = classifier(texto)[0]
    sentimiento = map_sentiment(resultado["label"])
    return (
        sentimiento,
        resultado["score"],
    )


def classify_sentiment_df(data, comment_col="comment"):

    def classify_sentiment(texto):
        resultado = classifier(texto)[0]
        sentimiento = map_sentiment(resultado["label"])
        return sentimiento, resultado["score"]

    data["sentimiento"], data["confianza"] = zip(
        *data[comment_col].apply(classify_sentiment)
    )

    return data


def transform_embeddings(
    data, embeddings_col="embeddings", n_components=3, random_seed=42
):
    # Convertir embeddings a matriz numpy
    embeddings_matrix = np.array(data[embeddings_col].tolist())

    # Aplicar UMAP para reducción de dimensionalidad
    umap_model = umap.UMAP(
        n_components=n_components, random_state=random_seed, metric="cosine"
    )
    data_umap = umap_model.fit_transform(embeddings_matrix)

    # Calcular distancias y percentiles para determinar min_eps y max_eps
    distances = pairwise_distances(data_umap, metric="cosine")
    min_eps = np.percentile(distances, 10)
    max_eps = np.percentile(distances, 50)

    umap_data = pd.DataFrame(
        {"embeddings": [embedding.tolist() for embedding in data_umap]}
    )
    umap_data["comment"] = data["comment"]

    return umap_data, min_eps, max_eps


def determine_min_items_by_cluster(total):
    """ """
    if total < 50:
        min_items_by_cluster = 1
    elif total < 100:
        min_items_by_cluster = 5
    elif total < 500:
        min_items_by_cluster = 10
    else:
        min_items_by_cluster = int(round(total * 0.01, 2))

    return min_items_by_cluster


def main(): ...


if __name__ == "__main__":
    main()
