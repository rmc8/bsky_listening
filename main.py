import os
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any

import fire
import pandas as pd
from dotenv import load_dotenv


from libs import bsky, preproc
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
            config=get_config(), app_pass=os.getenv("BSKY_APP_PASS"), limit=limit
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
            api_key=os.getenv("XAI_API_KEY"),
            idf=idf,
        )
        output_path = io_dir.joinpath(FileName.preproc.value)
        odf.to_csv(output_path, sep="\t", index=False)


def main():
    fire.Fire(BskyListening)


if __name__ == "__main__":
    main()
