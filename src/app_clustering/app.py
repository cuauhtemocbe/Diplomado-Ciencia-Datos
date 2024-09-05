import os
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from googleapiclient.discovery import build
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

RANDOM_STATE = 333

app = Flask(__name__)

if os.getenv("RAILWAY_ENVIRONMENT") is None:
    load_dotenv()

api_key = os.getenv("youtube_api_key")
print(f"El valor de la variable de entorno es: {api_key[:5]}")


def extract_video_id(url):
    """
    Extrae el video_id de una URL de YouTube.

    Parámetros:
    - url: str, la URL del video de YouTube.

    Retorna:
    - video_id: str, el identificador del video de YouTube.
    """
    # Expresión regular para encontrar el video_id en una URL de YouTube
    return url.split("=")[-1]


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
    embeddings_matrix = np.array(data["embeddings"].tolist())

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
    replacements = {"&quot;": '"', "&amp;": "&", "&lt;": "<", "&gt;": ">"}

    for entity, char in replacements.items():
        text = text.replace(entity, char)

    return text


def cluster_and_sankey(
    data, min_eps, n=5, embeddings_col="embeddings", min_items_by_cluster=10
):
    embeddings_matrix = np.array(data[embeddings_col].tolist())
    threshold_values = np.round(np.linspace(min_eps, 0.95, n), 2)

    cluster_assignments = {}
    cluster_counts = {}
    calinski_harabasz_scores = {}
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
        ch_score = calinski_harabasz_score(embeddings_matrix, labels)
        calinski_harabasz_scores[distance_threshold] = ch_score
        most_similar_comments[distance_threshold] = {}
        for cluster_id in np.unique(labels):
            cluster_data = data[data[f"cluster_{distance_threshold}"] == cluster_id]
            avg_embedding = np.mean(cluster_data[embeddings_col].tolist(), axis=0)
            most_similar_comment = find_most_similar_comment(
                cluster_data, avg_embedding
            )
            most_similar_comments[distance_threshold][cluster_id] = most_similar_comment
    labels = []
    source = []
    target = []
    values = []
    comments = []
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
                        f'Comment: {most_similar_comments[threshold].get(j, "N/A")}'
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

    return labels, source, target, values, comments, calinski_harabasz_scores, data


def plot_sankey(labels, source, target, values, comments, width=None, height=None):
    fig = go.Figure(
        go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                hovertemplate="<b>%{label}</b><br>"
                + "<br>%{customdata}<extra></extra>",
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


def plot_calinski_harabasz_scores(ch_scores):
    """
    Genera un gráfico de Calinski-Harabasz Score frente a los umbrales de distancia.

    Args:
        ch_scores (dict): Un diccionario donde las claves son umbrales de distancia
                          y los valores son puntajes Calinski-Harabasz correspondientes.

    Returns:
        fig (plotly.graph_objects.Figure): Un objeto Figure de Plotly con el gráfico generado.
    """
    # Extraer umbrales de distancia y puntajes Calinski-Harabasz
    threshold_distances = []
    calinski_harabasz_scores = []

    for threshold, scores in ch_scores.items():
        threshold_distances.append(threshold)
        calinski_harabasz_scores.append(scores)

    # Crear el gráfico
    fig = go.Figure(
        go.Scatter(
            x=threshold_distances,
            y=calinski_harabasz_scores,
            mode="lines+markers",
            marker=dict(color="blue", size=10),
            line=dict(color="blue", width=2),
            text=[
                f"Threshold: {t}<br>Score: {s}"
                for t, s in zip(threshold_distances, calinski_harabasz_scores)
            ],
            hoverinfo="text",
        )
    )

    # Configurar diseño del gráfico
    fig.update_layout(
        title="Calinski-Harabasz Score vs. Threshold Distance",
        xaxis_title="Threshold Distance",
        yaxis_title="Calinski-Harabasz Score",
        font=dict(size=12),
        width=800,
        height=600,
        template="plotly_dark",
    )

    return fig


def convert_graph_to_html(graph, full_html=False):
    return pio.to_html(graph, full_html=full_html) if graph else None


@app.route("/", methods=["GET", "POST"])
def index():
    video_details = None
    k_distance_graph = None
    scores_graph = None
    sankey_graph = None

    if request.method == "POST":
        url = request.form["url"]
        if url:
            video_details = get_youtube_video_details(url, api_key)
            comments_df = get_youtube_comments(api_key, url)
            comments_df = add_normalized_embeddings_to_dataframe(comments_df, "comment")

            k_distance_graph, min_eps = plot_k_distance(comments_df)

            total = comments_df.shape[0]

            if total < 50:
                min_items_by_cluster = 1
            elif total < 100:
                min_items_by_cluster = 5
            elif total < 500:
                min_items_by_cluster = 10
            else:
                min_items_by_cluster = int(round(total * 0.05, 2))

            (
                labels,
                source,
                target,
                values,
                comments,
                calinski_harabasz_scores,
                _,
            ) = cluster_and_sankey(
                comments_df, min_eps, min_items_by_cluster=min_items_by_cluster
            )

            sankey_graph = plot_sankey(
                labels, source, target, values, comments, height=1000
            )
            scores_graph = plot_calinski_harabasz_scores(calinski_harabasz_scores)

            k_distance_graph = convert_graph_to_html(k_distance_graph)
            sankey_graph = convert_graph_to_html(sankey_graph, full_html=True)
            scores_graph = convert_graph_to_html(scores_graph)

    return render_template(
        "index.html",
        video_details=video_details,
        k_distance_graph=k_distance_graph,
        sankey_graph=sankey_graph,
        scores_graph=scores_graph,
    )


#  gunicorn -b 0.0.0.0:5000 app:app
# http://172.20.0.2:5000/
# http://0.0.0.0:5000/
if __name__ == "__main__":
    app.run()
