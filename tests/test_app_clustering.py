"""Tests for app_clustering's error handling (see issue #31).

Runs only under the `nlp` Poetry group (`make test-nlp`) -- flask and
google-api-python-client aren't installed in the `core` container that
`make test` uses.

Two layers:
- `get_youtube_comments` unit tests: mock `googleapiclient.discovery.build`
  to raise, assert the function returns an {"error": ...} dict instead of
  propagating the exception (mirrors `get_youtube_video_details`'s existing
  contract).
- Flask route tests: monkeypatch the `clustering` module's functions (as
  `app.py` calls them) to simulate each Gherkin scenario from issue #31,
  using the Flask test client against the real `/` route.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from app_clustering import app as app_module
from app_clustering import clustering


# --- get_youtube_comments unit tests ---------------------------------------


def test_get_youtube_comments_returns_error_dict_on_api_failure():
    with patch("app_clustering.clustering.build") as mock_build:
        mock_build.return_value.commentThreads.return_value.list.return_value.execute.side_effect = Exception(
            "invalidKey"
        )

        result = clustering.get_youtube_comments("bad-key", "https://youtu.be/dQw4w9WgXcQ")

    assert isinstance(result, dict)
    assert "invalidKey" in result["error"]


def test_get_youtube_comments_returns_dataframe_on_success():
    mock_response = {
        "items": [
            {
                "snippet": {
                    "topLevelComment": {
                        "snippet": {
                            "textDisplay": "great video",
                            "authorDisplayName": "someone",
                            "publishedAt": "2024-01-01T00:00:00Z",
                        }
                    }
                }
            }
        ]
    }

    with patch("app_clustering.clustering.build") as mock_build:
        mock_build.return_value.commentThreads.return_value.list.return_value.execute.return_value = (
            mock_response
        )

        result = clustering.get_youtube_comments("good-key", "https://youtu.be/dQw4w9WgXcQ")

    assert isinstance(result, pd.DataFrame)
    assert result.iloc[0]["comment"] == "great video"


# --- Flask route tests -------------------------------------------------------


@pytest.fixture(name="client")
def client_fixture():
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def test_index_shows_error_when_comments_fetch_fails_invalid_api_key(client, monkeypatch):
    monkeypatch.setattr(
        app_module.clustering,
        "get_youtube_video_details",
        lambda url, api_key: {"title": "t", "channel_title": "c", "view_count": "1", "comment_count": "1"},
    )
    monkeypatch.setattr(
        app_module.clustering,
        "get_youtube_comments",
        lambda api_key, url: {"error": "API key not valid"},
    )

    response = client.post("/", data={"url": "https://youtu.be/dQw4w9WgXcQ"})

    assert response.status_code == 200
    assert b"API key not valid" in response.data
    assert b"Detalles del Video" not in response.data


def test_index_shows_error_when_comments_disabled(client, monkeypatch):
    monkeypatch.setattr(
        app_module.clustering,
        "get_youtube_video_details",
        lambda url, api_key: {"title": "t", "channel_title": "c", "view_count": "1", "comment_count": "1"},
    )
    monkeypatch.setattr(
        app_module.clustering,
        "get_youtube_comments",
        lambda api_key, url: {"error": "commentsDisabled: comments are disabled for this video"},
    )

    response = client.post("/", data={"url": "https://youtu.be/dQw4w9WgXcQ"})

    assert response.status_code == 200
    assert b"commentsDisabled" in response.data


def test_index_shows_error_on_quota_exceeded(client, monkeypatch):
    monkeypatch.setattr(
        app_module.clustering,
        "get_youtube_video_details",
        lambda url, api_key: {"title": "t", "channel_title": "c", "view_count": "1", "comment_count": "1"},
    )
    monkeypatch.setattr(
        app_module.clustering,
        "get_youtube_comments",
        lambda api_key, url: {"error": "quotaExceeded"},
    )

    response = client.post("/", data={"url": "https://youtu.be/dQw4w9WgXcQ"})

    assert response.status_code == 200
    assert b"quotaExceeded" in response.data


def test_index_shows_error_when_video_details_fetch_fails(client, monkeypatch):
    monkeypatch.setattr(
        app_module.clustering,
        "get_youtube_video_details",
        lambda url, api_key: {"error": "No se encontró el video con el ID proporcionado."},
    )
    monkeypatch.setattr(
        app_module.clustering,
        "get_youtube_comments",
        lambda api_key, url: pd.DataFrame(
            {"author": ["a"], "comment": ["hola"], "published_at": ["2024-01-01T00:00:00Z"]}
        ),
    )

    response = client.post("/", data={"url": "https://youtu.be/dQw4w9WgXcQ"})

    assert response.status_code == 200
    assert b"No se encontr\xc3\xb3 el video" in response.data
    assert b"Detalles del Video" not in response.data


def test_index_renders_results_on_success(client, monkeypatch):
    monkeypatch.setattr(
        app_module.clustering,
        "get_youtube_video_details",
        lambda url, api_key: {"title": "t", "channel_title": "c", "view_count": "1", "comment_count": "1"},
    )
    monkeypatch.setattr(
        app_module.clustering,
        "get_youtube_comments",
        lambda api_key, url: pd.DataFrame(
            {"author": ["a"], "comment": ["hola"], "published_at": ["2024-01-01T00:00:00Z"]}
        ),
    )
    monkeypatch.setattr(
        app_module.clustering,
        "add_normalized_embeddings_to_dataframe",
        lambda df, text_column: df.assign(embeddings=[[0.1, 0.2]]),
    )
    monkeypatch.setattr(
        app_module.clustering,
        "classify_sentiment_df",
        lambda df: df.assign(sentimiento=["positivo"], confianza=[0.9]),
    )
    monkeypatch.setattr(
        app_module.clustering, "plot_sentiment_daily", lambda df: go.Figure()
    )
    monkeypatch.setattr(
        app_module.clustering,
        "transform_embeddings",
        lambda df, embeddings_col="embeddings": (
            pd.DataFrame({"embeddings": [[0.1, 0.2]], "comment": ["hola"]}),
            0.1,
            0.9,
        ),
    )
    monkeypatch.setattr(
        app_module.clustering, "determine_min_items_by_cluster", lambda total: 1
    )
    monkeypatch.setattr(
        app_module.clustering,
        "perform_clustering",
        lambda umap_df, min_eps, max_eps, n, embeddings_col: ({}, {}, {}, {}, {}, umap_df),
    )
    monkeypatch.setattr(
        app_module.clustering,
        "build_sankey_data",
        lambda *args, **kwargs: (["a"], [], [], [], ["hola"]),
    )
    monkeypatch.setattr(
        app_module.clustering, "plot_sankey", lambda *args, **kwargs: go.Figure()
    )
    monkeypatch.setattr(
        app_module.clustering,
        "plot_clustering_metric",
        lambda silhouette, calinski: (go.Figure(), 0.5),
    )

    response = client.post("/", data={"url": "https://youtu.be/dQw4w9WgXcQ"})

    assert response.status_code == 200
    assert b"Detalles del Video" in response.data
    assert b"Gr\xc3\xa1fico de Sankey" in response.data
    assert b"<h2>Error</h2>" not in response.data


def test_add_normalized_embeddings_uses_sentence_transformer_api(monkeypatch):
    encoded_sentences = []

    class FakeSentenceTransformer:
        def __init__(self, model_name):
            assert model_name == "test-model"

        def encode(self, sentences):
            encoded_sentences.extend(sentences)
            return np.array([[3.0, 4.0], [0.0, 2.0]])

    monkeypatch.setattr(clustering, "SentenceTransformer", FakeSentenceTransformer)
    data = pd.DataFrame({"comment": ["hola", "adios"]})

    result = clustering.add_normalized_embeddings_to_dataframe(
        data, "comment", model_name="test-model"
    )

    assert encoded_sentences == ["hola", "adios"]
    np.testing.assert_allclose(
        np.array(result["embeddings"].tolist()),
        np.array([[0.6, 0.8], [0.0, 1.0]]),
    )


def test_classify_sentiment_df_maps_labels_and_scores(monkeypatch):
    def fake_classifier(text):
        return [
            {
                "label": "5 stars" if text == "bien" else "1 star",
                "score": 0.95 if text == "bien" else 0.88,
            }
        ]

    monkeypatch.setattr(clustering, "classifier", fake_classifier)
    data = pd.DataFrame({"comment": ["bien", "mal"]})

    result = clustering.classify_sentiment_df(data)

    assert result[["sentimiento", "confianza"]].to_dict("records") == [
        {"sentimiento": "positivo", "confianza": 0.95},
        {"sentimiento": "negativo", "confianza": 0.88},
    ]
