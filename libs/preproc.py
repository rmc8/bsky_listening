import logging
import pandas as pd


from langchain.prompts import ChatPromptTemplate
from langchain_xai import ChatXAI
from pydantic import BaseModel, Field
from tqdm import tqdm


class TopicData(BaseModel):
    topics: list[str] = Field(
        ...,
        description="ユーザーのポストから抽出された主要な活動・行動トピックのリスト。複数の行動が含まれる場合は分割されます。",
    )


def _get_topics(config: dict, api_key: str, post: str):
    llm = ChatXAI(
        model=config["preproc"]["model"],
        api_key=api_key,
        temperature=0.0,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                config["preproc"]["system_prompt"],
            ),
            (
                "human",
                post.replace("{", "{{").replace("}", "}}"),
            ),
        ]
    )
    chain = prompt | llm.with_structured_output(TopicData)
    res: TopicData = chain.invoke({})
    return res.topics


def preproc(config: dict, api_key: str, idf: pd.DataFrame):
    logging.basicConfig(level=logging.WARNING)
    data = []
    index = 0
    idf = idf.dropna(subset=["text"])
    for rec in tqdm(idf.to_dict("records")):
        topics = _get_topics(config, api_key, rec["text"])
        for topic in topics:
            data.append(
                {
                    "index": index,
                    "cid": rec["cid"],
                    "topic": topic,
                }
            )
            index += 1
    return pd.DataFrame(data)
