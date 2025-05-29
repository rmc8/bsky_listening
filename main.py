import os
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any

import fire
import pandas as pd
from dotenv import load_dotenv


from libs import bsky, chart, preproc, embedding, clustering, labeling
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
    @staticmethod
    def fetch(pj_name: str = pj_name, limit: int = 500):
        df = bsky.fetch(
            config=get_config(), app_pass=str(os.getenv("BSKY_APP_PASS")), limit=limit
        )
        io_dir = PROJECT_DIR.joinpath(pj_name)
        os.makedirs(io_dir, exist_ok=True)
        output_path = io_dir.joinpath(FileName.bsky_posts.value)
        df.to_csv(output_path, sep="\t", index=False)

    @staticmethod
    def preproc(pj_name: str = pj_name):
        io_dir = PROJECT_DIR.joinpath(pj_name)
        input_path = io_dir.joinpath(FileName.bsky_posts.value)
        idf = pd.read_csv(input_path, sep="\t")
        odf = preproc.preproc(
            config=get_config(),
            api_key=str(os.getenv("XAI_API_KEY")),
            idf=idf,
        )
        output_path = io_dir.joinpath(FileName.preproc.value)
        odf.to_csv(output_path, sep="\t", index=False)

    @staticmethod
    def _embedding_by_openai(idf: pd.DataFrame) -> pd.DataFrame:
        return embedding.embed_by_openai(
            config=get_config(),
            api_key=str(os.getenv("OPENAI_API_KEY")),
            idf=idf,
        )

    @staticmethod
    def _embedding_by_ollama(idf: pd.DataFrame) -> pd.DataFrame:
        return embedding.embed_by_ollama(
            config=get_config(),
            idf=idf,
        )

    @classmethod
    def embedding(cls, pj_name: str = pj_name, is_local: bool = True):
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
    def labeling(pj_name: str = pj_name):
        io_dir = PROJECT_DIR.joinpath(pj_name)
        clustering_path = io_dir.joinpath(FileName.clustering.value)
        preproc_path = io_dir.joinpath(FileName.preproc.value)
        pdf = pd.read_csv(preproc_path, sep="\t")
        cdf = pd.read_csv(clustering_path, sep="\t")
        idf = pd.merge(pdf, cdf, on=["index"], how="left")
        odf = labeling.labeling(
            config=get_config(),
            api_key=str(os.getenv("XAI_API_KEY")),
            idf=idf,
        )
        output_path = io_dir.joinpath(FileName.labeling.value)
        odf.to_csv(output_path, sep="\t", index=False)

    @staticmethod
    def chart(pj_name: str = pj_name):
        io_dir = PROJECT_DIR.joinpath(pj_name)
        clustering_path = io_dir.joinpath(FileName.clustering.value)
        preproc_path = io_dir.joinpath(FileName.preproc.value)
        pdf = pd.read_csv(preproc_path, sep="\t")
        pdf = pdf[~pdf["topic"].str.contains("朝活", na=False)]
        cdf = pd.read_csv(clustering_path, sep="\t")
        idf = pd.merge(pdf, cdf, on=["index"], how="left")
        labeling_path = io_dir.joinpath(FileName.labeling.value)
        ldf = pd.read_csv(labeling_path, sep="\t")
        html = chart.chart(config=get_config(), pdf=idf, ldf=ldf)
        output_path = io_dir.joinpath(FileName.chart.value)
        with open(output_path, "w") as f:
            f.write(html)


def main():
    fire.Fire(BskyListening)


if __name__ == "__main__":
    main()
