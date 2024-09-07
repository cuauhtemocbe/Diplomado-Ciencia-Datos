import os
import re

import app_clustering.app as clustering
import pandas as pd
import plotly.io as pio
import umap
from app_clustering import clustering
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
import logging

if os.getenv("RAILWAY_ENVIRONMENT") is None:
    load_dotenv()

api_key = os.getenv("youtube_api_key")

app = Flask(__name__)
app.logger.setLevel(logging.ERROR)
app.config['PROPAGATE_EXCEPTIONS'] = False

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
            sentiment_daily_graph = clustering.plot_sentiment_daily(comments_df)

            sentiment_daily_graph = convert_graph_to_html(sentiment_daily_graph)

            umap_df, min_eps, max_eps = clustering.transform_embeddings(
                comments_df, embeddings_col="embeddings"
            )

            # image_path = os.path.join(os.getcwd(), "static/wordcloud.png")
            # print("path", image_path)

            total = comments_df.shape[0]

            min_items_by_cluster = clustering.determine_min_items_by_cluster(total)

            (
                cluster_assignments,
                cluster_counts,
                calinski_harabasz_scores,
                silhouette_scores,
                most_similar_comments,
                umap_df,
            ) = clustering.perform_clustering(
                umap_df, min_eps, max_eps, n=10, embeddings_col="embeddings"
            )

            labels, source, target, values, comments = clustering.build_sankey_data(
                cluster_assignments,
                cluster_counts,
                most_similar_comments,
                min_items_by_cluster=min_items_by_cluster,
            )

            sankey_graph = clustering.plot_sankey(
                labels, source, target, values, comments, height=1000, width=1200
            )
            sankey_graph = convert_graph_to_html(sankey_graph)

            scores_graph, _ = clustering.plot_clustering_metric(silhouette_scores, calinski_harabasz_scores)
            scores_graph = convert_graph_to_html(scores_graph)

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
