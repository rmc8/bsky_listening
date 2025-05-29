from enum import StrEnum


class FileName(StrEnum):
    bsky_posts = "bsky_posts.tsv"
    preproc = "preproc.tsv"
    embedding = "embedding.pkl"
    clustering = "clustering.tsv"
