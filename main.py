import os
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any

import fire
import pandas as pd
from dotenv import load_dotenv

from libs import bsky, chart, clustering, embedding, labeling, preproc
from libs.file_name import FileName

load_dotenv()

now = datetime.now()
pj_name = f"{now:%Y%m%d}"


THIS_DIR = Path(__file__).parent
PROJECT_DIR = THIS_DIR / "project"


def get_config() -> dict[str, Any]:
    config_file = THIS_DIR / "config.toml"
    with config_file.open("rb") as b:
        return tomllib.load(b)


class BskyListening:
    """
    Bluesky ソーシャルメディアプラットフォームから特定のユーザーの投稿を取得し、
    その内容を分析・可視化するための一連の処理を提供するクラスです。
    """

    @staticmethod
    def fetch(pj_name: str = pj_name, limit: int = 500):
        """
        指定されたBlueskyアカウントから投稿データを取得し、TSV形式で保存します。
        リポストは除外されます。

        Args:
            pj_name (str): プロジェクト名。デフォルトは現在日付 (YYYYMMDD)。
            limit (int): 取得する投稿の最大件数。デフォルトは500。
        """
        df = bsky.fetch(
            config=get_config(), app_pass=str(os.getenv("BSKY_APP_PASS")), limit=limit
        )
        io_dir = PROJECT_DIR.joinpath(pj_name)
        os.makedirs(io_dir, exist_ok=True)
        output_path = io_dir.joinpath(FileName.bsky_posts.value)
        df.to_csv(output_path, sep="\t", index=False)

    @staticmethod
    def preproc(pj_name: str = pj_name):
        """
        取得した投稿テキストから主要なトピックを抽出し、分析しやすいように整形します。

        Args:
            pj_name (str): プロジェクト名。デフォルトは現在日付 (YYYYMMDD)。
        """
        io_dir = PROJECT_DIR.joinpath(pj_name)
        input_path = io_dir.joinpath(FileName.bsky_posts.value)
        idf = pd.read_csv(input_path, sep="\t")
        odf = preproc.preproc(
            config=get_config(),
            api_key=str(os.getenv("OPENAI_API_KEY")),
            idf=idf,
        )
        output_path = io_dir.joinpath(FileName.preproc.value)
        odf.to_csv(output_path, sep="\t", index=False)

    @staticmethod
    def _embedding_by_openai(idf: pd.DataFrame) -> pd.DataFrame:
        """
        OpenAIモデルを使用して埋め込みを生成します。

        Args:
            idf (pd.DataFrame): 前処理されたトピックデータを含むDataFrame。

        Returns:
            pd.DataFrame: 埋め込みデータを含むDataFrame。
        """
        return embedding.embed_by_openai(
            config=get_config(),
            api_key=str(os.getenv("OPENAI_API_KEY")),
            idf=idf,
        )

    @staticmethod
    def _embedding_by_ollama(idf: pd.DataFrame) -> pd.DataFrame:
        """
        Ollamaモデルを使用して埋め込みを生成します。

        Args:
            idf (pd.DataFrame): 前処理されたトピックデータを含むDataFrame。

        Returns:
            pd.DataFrame: 埋め込みデータを含むDataFrame。
        """
        return embedding.embed_by_ollama(
            config=get_config(),
            idf=idf,
        )

    @classmethod
    def embedding(cls, pj_name: str = pj_name, is_local: bool = True):
        """
        前処理されたトピックデータから、機械学習モデル（OpenAIまたはOllama）を使用して
        数値ベクトル（埋め込み）を生成します。

        Args:
            pj_name (str): プロジェクト名。デフォルトは現在日付 (YYYYMMDD)。
            is_local (bool): ローカルのOllamaモデルを使用するかどうか。デフォルトはTrue。
                             Falseの場合、OpenAIモデルを使用します。
        """
        io_dir = PROJECT_DIR.joinpath(pj_name)
        input_path = io_dir.joinpath(FileName.preproc.value)
        idf = pd.read_csv(input_path, sep="\t")
        embedding_func = (
            cls._embedding_by_ollama if is_local else cls._embedding_by_openai
        )
        odf = embedding_func(idf)
        output_path = io_dir.joinpath(FileName.embedding.value)
        odf.to_pickle(output_path)

    @staticmethod
    def clustering(pj_name: str = pj_name):
        """
        生成された埋め込みデータを用いて、投稿を類似性に基づいてクラスタリングします。
        UMAP、HDBSCAN、BERTopic、Spectral Clusteringを組み合わせて使用します。

        Args:
            pj_name (str): プロジェクト名。デフォルトは現在日付 (YYYYMMDD)。
        """
        io_dir = PROJECT_DIR.joinpath(pj_name)
        preproc_path = io_dir.joinpath(FileName.preproc.value)
        embedding_path = io_dir.joinpath(FileName.embedding.value)
        idf = pd.read_csv(preproc_path, sep="\t")
        edf = pd.read_pickle(embedding_path)
        odf = clustering.clustering(
            config=get_config(),
            idf=idf,
            edf=edf,
        )
        output_path = io_dir.joinpath(FileName.clustering.value)
        odf.to_csv(output_path, sep="\t", index=False)

    @staticmethod
    def labeling(pj_name: str = pj_name, threshold: float = 0.70):
        """
        クラスタリングされた各グループに対して、その特徴をもっともよく表すラベルを生成します。

        Args:
            pj_name (str): プロジェクト名。デフォルトは現在日付 (YYYYMMDD)。
            threshold (float): ラベル生成に使用するデータの確率閾値。デフォルトは0.70。
                               この値以上の確率を持つデータのみがラベル生成に使用されます。
        """
        io_dir = PROJECT_DIR.joinpath(pj_name)
        clustering_path = io_dir.joinpath(FileName.clustering.value)
        preproc_path = io_dir.joinpath(FileName.preproc.value)
        pdf = pd.read_csv(preproc_path, sep="\t")
        cdf = pd.read_csv(clustering_path, sep="\t")
        idf = pd.merge(pdf, cdf, on=["index"], how="left")
        odf = labeling.labeling(
            config=get_config(),
            api_key=str(os.getenv("OPENAI_API_KEY")),
            idf=idf,
            threshold=threshold,
        )
        output_path = io_dir.joinpath(FileName.labeling.value)
        odf.to_csv(output_path, sep="\t", index=False)

    @staticmethod
    def chart(pj_name: str = pj_name, full_html: bool = True):
        """
        クラスタリングとラベリングの結果を基に、インタラクティブな散布図（HTML形式）を生成します。

        Args:
            pj_name (str): プロジェクト名。デフォルトは現在日付 (YYYYMMDD)。
            full_html (bool): チャートを完全なHTMLとして出力するかどうか。デフォルトはTrue。
        """
        io_dir = PROJECT_DIR.joinpath(pj_name)
        clustering_path = io_dir.joinpath(FileName.clustering.value)
        preproc_path = io_dir.joinpath(FileName.preproc.value)
        pdf = pd.read_csv(preproc_path, sep="\t")
        cdf = pd.read_csv(clustering_path, sep="\t")
        idf = pd.merge(pdf, cdf, on=["index"], how="left")
        labeling_path = io_dir.joinpath(FileName.labeling.value)
        ldf = pd.read_csv(labeling_path, sep="\t")
        html = chart.chart(config=get_config(), pdf=idf, ldf=ldf, full_html=full_html)
        output_path = io_dir.joinpath(FileName.chart.value)
        with open(output_path, "w") as f:
            f.write(html)

    @classmethod
    def all(
        cls,
        pj_name: str = pj_name,
        limit: int = 500,
        is_local: bool = True,
        threshold: float = 0.70,
        full_html: bool = True,
    ):
        """
        Blueskyの投稿取得からチャート作成までの一連の処理をすべて実行します。

        Args:
            pj_name (str): プロジェクト名。デフォルトは現在日付 (YYYYMMDD)。
            limit (int): 取得する投稿の最大件数。デフォルトは500。
            is_local (bool): 埋め込み生成でローカルのOllamaモデルを使用するかどうか。
                             デフォルトはTrue。Falseの場合、OpenAIモデルを使用します。
            threshold (float): ラベル生成に使用するデータの確率閾値。デフォルトは0.70。
                               この値以上の確率を持つデータのみがラベル生成に使用されます。
            full_html (bool): チャートを完全なHTMLとして出力するかどうか。デフォルトはTrue。
        """
        cls.fetch(pj_name=pj_name, limit=limit)
        cls.preproc(pj_name=pj_name)
        cls.embedding(pj_name=pj_name, is_local=is_local)
        cls.clustering(pj_name=pj_name)
        cls.labeling(pj_name=pj_name, threshold=threshold)
        cls.chart(pj_name=pj_name, full_html=full_html)


def main():
    fire.Fire(BskyListening)


if __name__ == "__main__":
    main()
