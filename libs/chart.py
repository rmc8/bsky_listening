import plotly.express as px
import pandas as pd


def _create_scatter_plot(config: dict, pdf: pd.DataFrame, ldf: pd.DataFrame):
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


def chart(config: dict, pdf: pd.DataFrame, ldf: pd.DataFrame, full_html: bool) -> str:
    """
    散布図を生成し、HTML形式で返します。

    Args:
        config (dict): 設定情報を含む辞書。
        pdf (pd.DataFrame): 前処理されたデータを含むDataFrame。
        ldf (pd.DataFrame): ラベリングされたデータを含むDataFrame。
        full_html (bool): 完全なHTMLドキュメントとして出力するかどうか。デフォルトはTrue。

    Returns:
        str: 生成された散布図のHTML表現。
    """
    fig = _create_scatter_plot(config, pdf, ldf)
    return fig.to_html(full_html=full_html)
