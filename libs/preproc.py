import logging
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr
from tqdm import tqdm


class TopicData(BaseModel):
    topics: list[str] = Field(
        ...,
        description="ユーザーのポストから抽出された主要な活動・行動トピックのリスト。複数の行動が含まれる場合は分割されます。",
    )


def _get_topics(args):
    config, api_key, post = args
    llm = ChatOpenAI(
        model=config["preproc"]["model"],
        api_key=SecretStr(api_key),
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
    try:
        res: TopicData = chain.invoke({})  # type: ignore
        return res.topics
    except Exception as e:
        logging.error(f"Error processing post: {e}")
        return []


def preproc(config: dict, api_key: str, idf: pd.DataFrame, max_workers: int = 4):
    logging.basicConfig(level=logging.WARNING)
    idf = idf.dropna(subset=["text"])
    records = idf.to_dict("records")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Prepare arguments for each task
        args = [(config, api_key, rec["text"]) for rec in records]

        # Use tqdm to monitor progress
        results = list(tqdm(executor.map(_get_topics, args), total=len(records)))

    data = []
    index = 0
    for rec, topics in zip(records, results):
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
