import logging
from time import sleep
from typing import Any

from atproto import Client
from pandas import DataFrame
from retry import retry

logger = logging.getLogger(__name__)


@retry(tries=5, delay=6.0)
def _get_timeline(c: Client, actor: str, cursor: str | None):
    res = c.get_author_feed(actor=actor, cursor=cursor, limit=100)
    return res


def fetch(config: dict[str, Any], app_pass: str, limit: int) -> DataFrame:
    logging.basicConfig(level=logging.INFO)
    handle = config["bluesky"]["handle"]
    c = Client()
    c.login(login=handle, password=app_pass)
    posts: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        res = _get_timeline(c, handle, cursor)
        for feed_view in res.feed:
            post = feed_view.post
            if post.viewer is not None and post.viewer.repost is not None:
                continue
            posts.append(
                {
                    "uri": post.uri,
                    "cid": post.cid,
                    "text": post.record.text,
                    "created_at": post.record.created_at,
                }
            )
        cursor = post.record.created_at
        if res.feed:
            cursor = res.feed[-1].post.record.created_at
            logger.info("cursor: %s, len: %s", cursor, len(posts))
            continue
        if res.cursor is None or len(posts) >= limit or len(res.feed) == 0:
            break
        sleep(1.0)
    return DataFrame(posts)
