import plotly.express as px
import pandas as pd


def create_scatter_plot(config: dict, pdf: pd.DataFrame, ldf: pd.DataFrame):
    df = pd.merge(pdf, ldf, on="cluster-id", how="left")
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="cluster-id",
        hover_data=["topic", "cluster-id", "label"],
        title=config["chart"]["title"],
    )
    fig.update_layout(
        font=dict(family="IPAexGothic, Noto Sans CJK JP, sans-serif", size=18)
    )
    return fig


def chart(config: dict, pdf: pd.DataFrame, ldf: pd.DataFrame) -> str:
    fig = create_scatter_plot(config, pdf, ldf)
    return fig.to_html(full_html=True)
