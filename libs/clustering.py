import logging

import pandas as pd
import numpy as np
from janome.tokenizer import Tokenizer
from umap import UMAP
from hdbscan import HDBSCAN
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


def _tokenize_japanese(text: str) -> list[str]:
    """
    日本語のテキストをJanomeでトークン化し、ストップワードを除外します。

    Args:
        text (str): トークン化する日本語のテキスト。

    Returns:
        list[str]: ストップワードが除外されたトークンのリスト。
    """
    return [
        token.surface
        for token in TOKENIZER.tokenize(text)
        if token.surface not in STOP_WORDS
    ]


def _perform_clustering(
    documents: list[str],
    embeddings: np.ndarray,
    metadata_dict: dict,
    min_cluster_size: int = 10,
    n_components: int = 2,
    num_topics: int = 6,
) -> pd.DataFrame:
    """
    与えられたドキュメントと埋め込みを使用して、UMAP、HDBSCAN、BERTopicを組み合わせた
    クラスタリングを実行します。クラスター数はユーザーが設定した固定値を使用します。

    Args:
        documents (list[str]): クラスタリング対象のドキュメントのリスト。
        embeddings (np.ndarray): ドキュメントに対応する埋め込みのNumPy配列。
        metadata_dict (dict): ドキュメントの追加メタデータを含む辞書。
                              結果DataFrameに結合されます。
        min_cluster_size (int, optional): HDBSCANの最小クラスターサイズ。デフォルトは2。
        n_components (int, optional): UMAPの次元削減後の次元数。デフォルトは2。
        num_topics (int, optional): BERTopicで生成するトピック（クラスター）の数。デフォルトは6。

    Returns:
        pd.DataFrame: クラスタリング結果を含むDataFrame。
                      各ドキュメントのインデックス、UMAP座標(x, y)、確率、クラスターIDが含まれます。

    Raises:
        Exception: クラスタリング処理中にエラーが発生した場合。
    """
    logger.info(f"Starting clustering with {len(documents)} documents")

    try:
        # UMAP次元削減モデルの初期化
        logger.info("Initializing UMAP dimensionality reduction model")
        umap_model = UMAP(
            random_state=42,
            n_components=n_components,
            n_jobs=-1,
        )

        # HDBSCANクラスタリングモデルの初期化
        logger.info("Initializing HDBSCAN clustering model")
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=1,
            core_dist_n_jobs=-1,
        )

        # CountVectorizerのセットアップ
        logger.info("Setting up CountVectorizer")
        vectorizer_model = CountVectorizer(tokenizer=_tokenize_japanese)

        # BERTopicモデルの初期化とクラスタ数の固定
        logger.info(f"Initializing BERTopic model with nr_topics={num_topics}")
        topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            verbose=True,
            nr_topics=num_topics,
        )

        # BERTopicモデルのフィットとトピック割り当ての取得
        logger.info("Fitting BERTopic model and getting topics")
        topics, probs = topic_model.fit_transform(documents, embeddings=embeddings)

        # UMAP座標を取得
        logger.info("Getting UMAP coordinates")
        umap_embeddings = topic_model.umap_model.transform(embeddings)

        # 結果DataFrameの生成
        logger.info("Generating result DataFrame")
        result_df = pd.DataFrame(
            {
                "index": metadata_dict["index"],
                "cid": metadata_dict["cid"],
                "x": umap_embeddings[:, 0],
                "y": umap_embeddings[:, 1],
                "probability": probs,
                "cluster-id": topics,
            }
        )

        logger.info(
            f"Clustering completed successfully with {len(set(result_df['cluster-id']))} clusters"
        )
        return result_df

    except Exception as e:
        logger.error(f"Error during clustering: {str(e)}", exc_info=True)
        raise


def clustering(config: dict, idf: pd.DataFrame, edf: pd.DataFrame) -> pd.DataFrame:
    """
    クラスタリングパイプラインのエントリポイント。
    設定、入力データフレーム、埋め込みデータフレームを受け取り、クラスタリング結果を返します。

    Args:
        config (dict): クラスタリング設定を含む辞書。
                       `config["clustering"]["n_clusters"]` でクラスター数を指定します。
        idf (pd.DataFrame): トピックとメタデータを含む入力DataFrame。
                                 "topic", "index", "cid" カラムが必要です。
        edf (pd.DataFrame): 埋め込みデータを含むDataFrame。
                                     "embedding" カラムが必要です。

    Returns:
        pd.DataFrame: クラスタリング結果を含むDataFrame。
                      各ドキュメントのインデックス、UMAP座標(x, y)、確率、クラスターIDが含まれます。

    Raises:
        Exception: クラスタリングパイプライン中にエラーが発生した場合。
    """
    logger.info("Starting clustering pipeline")
    try:
        logger.info("Loading topic data")
        documents_to_cluster = idf["topic"].values.tolist()
        document_embeddings = np.asarray(edf["embedding"].values.tolist())

        num_clusters = int(config["clustering"]["n_clusters"])
        logger.info(f"Configured for {num_clusters} clusters")

        # クラスタリングの実行
        clustering_result = _perform_clustering(
            documents=documents_to_cluster,
            embeddings=document_embeddings,
            metadata_dict={
                "index": idf["index"].values,
                "cid": idf["cid"].values,
            },
            min_cluster_size=10,
            num_topics=num_clusters,
        )
        logger.info("Clustering pipeline completed successfully")
        return clustering_result
    except Exception as e:
        logger.error(f"Error in clustering pipeline: {str(e)}", exc_info=True)
        raise
