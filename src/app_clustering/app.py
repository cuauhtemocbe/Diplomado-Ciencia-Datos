import os
import re
import unicodedata
from collections import Counter

import app_clustering.app as clustering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import umap
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from googleapiclient.discovery import build
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from sklearn import set_config
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from transformers import pipeline
from wordcloud import WordCloud
from sklearn.metrics import (
    calinski_harabasz_score,
    pairwise_distances,
    silhouette_score,
)

from app_clustering import clustering

if os.getenv("RAILWAY_ENVIRONMENT") is None:
    load_dotenv()

api_key = os.getenv("youtube_api_key")

app = Flask(__name__)

RANDOM_STATE = 333

def convert_graph_to_html(graph, full_html=False):
    return pio.to_html(graph, full_html=full_html) if graph else None


@app.route("/", methods=["GET", "POST"])
def index():
    video_details = None
    k_distance_graph = None
    scores_graph = None
    sankey_graph = None
    image_path = None
    sentiment_daily_graph = None
    sentiment_count = None

    if request.method == "POST":
        url = request.form["url"]
        if url:
            video_details = clustering.get_youtube_video_details(url, api_key)
            comments_df = clustering.get_youtube_comments(api_key, url)
            comments_df = clustering.add_normalized_embeddings_to_dataframe(
                comments_df, "comment"
            )

            comments_df["published_at"] = pd.to_datetime(
                comments_df["published_at"]
            ).dt.date

            comments_df = clustering.classify_sentiment_df(comments_df)
            sentiment_count = comments_df["sentimiento"].value_counts().to_dict()
            sentiment_daily_graph = clustering.plot_sentiment_global(comments_df)

            sentiment_daily_graph = convert_graph_to_html(sentiment_daily_graph)

            # image_path = os.path.join(os.getcwd(), "static/wordcloud.png")
            # print("path", image_path)

            # clustering.plot_wordcloud(
            #     comments_df, text_column="comment", output_filename=image_path
            # )

    #         k_distance_graph, min_eps = clusteringplot_k_distance(comments_df)

    #         total = comments_df.shape[0]

    #         if total < 50:
    #             min_items_by_cluster = 1
    #         elif total < 100:
    #             min_items_by_cluster = 5
    #         elif total < 500:
    #             min_items_by_cluster = 10
    #         else:
    #             min_items_by_cluster = int(round(total * 0.05, 2))

    #         (
    #             labels,
    #             source,
    #             target,
    #             values,
    #             comments,
    #             calinski_harabasz_scores,
    #             _,
    #         ) = cluster_and_sankey(
    #             comments_df, min_eps, min_items_by_cluster=min_items_by_cluster
    #         )

    #         sankey_graph = plot_sankey(
    #             labels, source, target, values, comments, height=1000
    #         )
    #         scores_graph = plot_calinski_harabasz_scores(calinski_harabasz_scores)

    #         k_distance_graph = convert_graph_to_html(k_distance_graph)
    #         sankey_graph = convert_graph_to_html(sankey_graph, full_html=True)
    #         scores_graph = convert_graph_to_html(scores_graph)

    # return render_template(
    #     "index.html",
    #     video_details=video_details,
    #     k_distance_graph=k_distance_graph,
    #     sankey_graph=sankey_graph,
    #     scores_graph=scores_graph,
    # )
    return render_template(
        "index.html",
        video_details=video_details,
        k_distance_graph=k_distance_graph,
        sankey_graph=sankey_graph,
        scores_graph=scores_graph,
        wordcloud_path=image_path,
        sentiment_daily_graph=sentiment_daily_graph,
        sentiment_count=sentiment_count,
    )


#  gunicorn -b 0.0.0.0:5000 app:app
# http://172.20.0.2:5000/
# http://0.0.0.0:5000/
if __name__ == "__main__":
    app.run()
