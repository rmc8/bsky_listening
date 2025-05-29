import random

import pandas as pd


from langchain.prompts import ChatPromptTemplate
from langchain_xai import ChatXAI
from pydantic import BaseModel, Field, SecretStr
from tqdm import tqdm


class LabelData(BaseModel):
    label: str = Field(..., description="テキストから生成したラベルを格納する。")


def _get_label(config: dict, api_key: str, topics: list[str]) -> str:
    llm = ChatXAI(
        model=config["labeling"]["model"],
        api_key=SecretStr(api_key),
        temperature=0.0,
    )
    topic_data = "\n".join(topics).replace("{", "{{").replace("}", "}}")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                config["labeling"]["system_prompt"],
            ),
            (
                "human",
                topic_data,
            ),
        ]
    )
    chain = prompt | llm.with_structured_output(LabelData)
    res: LabelData = chain.invoke({})  # type: ignore
    return res.label


def labeling(
    config: dict, api_key: str, idf: pd.DataFrame, threshold: float
) -> pd.DataFrame:
    """
    クラスタリングされた各グループに対して、その特徴をもっともよく表すラベルを生成します。

    Args:
        config (dict): 設定情報を含む辞書。
        api_key (str): XAI APIのキー。
        idf (pd.DataFrame): クラスタリングされたデータを含むDataFrame。
        threshold (float): ラベル生成に使用するデータの確率閾値。
                           この値以上の確率を持つデータのみがラベル生成に使用されます。

    Returns:
        pd.DataFrame: 各クラスタIDと対応するラベルを含むDataFrame。
    """
    data: list[dict[str, str]] = []
    cluster_ids = sorted(list(idf["cluster-id"].unique()))
    for cluster_id in tqdm(cluster_ids):
        cdf = idf[(idf["cluster-id"] == cluster_id) & (idf["probability"] >= threshold)]
        topics = cdf["topic"].tolist()
        random.shuffle(topics)
        topic_samples = [f'"""\n{t.strip()}\n"""' for t in topics[:16]]
        label = _get_label(config, api_key, topic_samples)
        data.append({"cluster-id": cluster_id, "label": label})
    return pd.DataFrame(data)
