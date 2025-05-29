from typing import Literal

from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from pandas import DataFrame
from pydantic import SecretStr
from tqdm import tqdm


class Embedding:
    def __init__(self, model: OllamaEmbeddings | OpenAIEmbeddings) -> None:
        self.model = model

    def _embed(self, docs: list[str]) -> list[list[float]]:
        return self.model.embed_documents(docs)

    def run(self, idf: DataFrame, batch_size: int = 1000) -> DataFrame:
        embeddings = []
        for i in tqdm(range(0, len(idf), batch_size)):
            topics = idf["topic"].tolist()[i : i + batch_size]
            embeds = self._embed(topics)
            embeddings.extend(embeds)
        df = DataFrame(
            [
                {
                    "index": idf.iloc[i]["index"],
                    "cid": idf.iloc[i]["cid"],
                    "embedding": e,
                }
                for i, e in enumerate(embeddings)
            ]
        )
        return df


def embed_by_openai(config: dict, api_key: str, idf: DataFrame) -> DataFrame:
    emb_model = OpenAIEmbeddings(
        model=config["embedding"]["openai_model"],
        api_key=SecretStr(api_key),
    )
    e = Embedding(emb_model)
    return e.run(idf)


def embed_by_ollama(config: dict, idf: DataFrame) -> DataFrame:
    emb_model = OllamaEmbeddings(
        model=config["embedding"]["ollama_model"],
        base_url=config["embedding"]["ollama_base_url"],
    )
    e = Embedding(emb_model)
    return e.run(idf)
