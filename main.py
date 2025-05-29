import os
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any

import fire
# import pandas as pd
from dotenv import load_dotenv


from libs import bsky
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
        output_dir = PROJECT_DIR.joinpath(pj_name)
        os.makedirs(output_dir, exist_ok=True)
        output_path = output_dir.joinpath(FileName.bsky_posts.value)
        df.to_csv(output_path, sep="\t", index=False)



def main():
    fire.Fire(BskyListening)


if __name__ == "__main__":
    main()
