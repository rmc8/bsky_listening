import logging

import pandas as pd
import numpy as np
from janome.tokenizer import Tokenizer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

STOP_WORDS = [
    "の",
    "に",
    "は",
    "を",
    "た",
    "が",
    "で",
    "て",
    "と",
    "し",
    "れ",
    "さ",
    "ある",
    "いる",
    "も",
    "する",
    "から",
    "な",
    "こと",
    "として",
    "いく",
    "ない",
]
TOKENIZER = Tokenizer()


def _tokenize_japanese(text) -> list:
    return [
        token.surface
        for token in TOKENIZER.tokenize(text)
        if token.surface not in STOP_WORDS
    ]


def _cluster_embeddings(
    docs,
    embeddings,
    metadatas,
    min_cluster_size=2,
    n_components=2,
    n_topics=6,
) -> pd.DataFrame:
    logger.info(f"StaStarting clustering with {len(docs)} documents")

    try:
        # UMAP次元削減
        logger.info("Initializing UMAP dimensionality reduction")
        umap_model = UMAP(
            random_state=42,
            n_components=n_components,
            n_jobs=-1,  # 並列処理を有効化
        )

        # HDBSCAN クラスタリング
        logger.info("Initializing HDBSCAN clustering")
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=1,  # より小さいクラスターを許容
            core_dist_n_jobs=-1,  # 並列処理を有効化
        )

        # Vectorizer設定
        logger.info("Setting up CountVectorizer")
        vectorizer_model = CountVectorizer(tokenizer=_tokenize_japanese)

        # BERTopicモデル
        logger.info("Initializing BERTopic model")
        topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            verbose=True,
        )

        # モデルのフィット
        logger.info("Fitting BERTopic model")
        _, __ = topic_model.fit_transform(docs, embeddings=embeddings)

        # Spectral Clustering
        logger.info("Performing Spectral Clustering")
        n_samples = len(embeddings)
        n_neighbors = min(n_samples - 1, 10)
        spectral_model = SpectralClustering(
            n_clusters=n_topics,
            affinity="nearest_neighbors",
            n_neighbors=n_neighbors,
            random_state=42,
            n_jobs=-1,  # 並列処理を有効化
        )

        # UMAP変換と予測
        logger.info("Performing UMAP transformation")
        umap_embeds = umap_model.fit_transform(embeddings)
        logger.info("Predicting clusters")
        cluster_labels = spectral_model.fit_predict(umap_embeds)

        # 結果の生成
        logger.info("Generating document info")
        result = topic_model.get_document_info(
            docs=docs,
            metadata={
                **metadatas,
                "x": umap_embeds[:, 0],
                "y": umap_embeds[:, 1],
            },
        )

        result.columns = [c.lower() for c in result.columns]
        result = result[["index", "x", "y", "probability"]]
        result["cluster-id"] = cluster_labels

        logger.info(
            f"Clustering completed successfully with {len(set(cluster_labels))} clusters"
        )
        return result

    except Exception as e:
        logger.error(f"Error during clustering: {str(e)}", exc_info=True)
        raise


def clustering(config: dict, idf: pd.DataFrame, edf: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting clustering pipeline")
    try:
        logger.info("Loading topic data")
        arguments_array = idf["topic"].values
        embeddings_array = np.asarray(edf["embedding"].values.tolist())

        clusters = int(config["clustering"]["n_clusters"])
        logger.info(f"Configured for {clusters} clusters")

        # クラスタリングの実行
        result = _cluster_embeddings(
            docs=arguments_array,
            embeddings=embeddings_array,
            metadatas={
                "index": idf["index"].values,
                "cid": idf["cid"].values,
            },
            min_cluster_size=clusters,
            n_topics=clusters,
        )
        logger.info("Clustering pipeline completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error in clustering pipeline: {str(e)}", exc_info=True)
        raise
