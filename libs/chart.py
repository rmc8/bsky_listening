from typing import Literal
import plotly.express as px
import pandas as pd
import numpy as np


def _create_scatter_plot(
    config: dict,
    pdf: pd.DataFrame,
    ldf: pd.DataFrame,
    pallet_type: Literal[
        "high_contrast", "colorblind_friendly", "vivid"
    ] = "colorblind_friendly",
):
    df = pd.merge(pdf, ldf, on="cluster-id", how="left")

    # クラスターIDを文字列に変換（重要：これがないとカラーパレットが正しく適用されない）
    df["cluster-id"] = df["cluster-id"].astype(str)

    # より視認性の良いカラーパレットを使用
    color_palette_options = {
        # オプション1: Plotlyの標準的な高コントラストパレット
        "high_contrast": px.colors.qualitative.Plotly,
        # オプション2: 色覚多様性に配慮したパレット
        "colorblind_friendly": [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
            "#aec7e8",
            "#ffbb78",
            "#98df8a",
            "#ff9896",
            "#c5b0d5",
            "#c49c94",
            "#f7b6d3",
            "#c7c7c7",
            "#dbdb8d",
            "#9edae5",
            "#393b79",
            "#637939",
            "#8c6d31",
            "#843c39",
            "#7b4173",
            "#5254a3",
            "#8ca252",
            "#bd9e39",
            "#ad494a",
            "#a55194",
        ],
        # オプション3: より鮮明で区別しやすいカスタムパレット
        "vivid": [
            "#FF0000",
            "#00FF00",
            "#0000FF",
            "#FFFF00",
            "#FF00FF",
            "#00FFFF",
            "#800080",
            "#FFA500",
            "#008000",
            "#FF69B4",
            "#4169E1",
            "#DC143C",
            "#32CD32",
            "#FFD700",
            "#9932CC",
            "#FF4500",
            "#2E8B57",
            "#FF1493",
            "#1E90FF",
            "#228B22",
            "#B22222",
            "#4682B4",
            "#D2691E",
            "#9370DB",
            "#20B2AA",
            "#F4A460",
            "#8B008B",
            "#556B2F",
            "#CD5C5C",
            "#40E0D0",
        ],
    }
    selected_palette = color_palette_options[pallet_type]

    # クラスターの数を確認し、必要に応じてパレットを拡張
    unique_clusters = df["cluster-id"].unique()
    num_clusters = len(unique_clusters)

    if num_clusters > len(selected_palette):
        # パレットが足りない場合は繰り返し使用
        extended_palette = (
            selected_palette * ((num_clusters // len(selected_palette)) + 1)
        )[:num_clusters]
        selected_palette = extended_palette

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="cluster-id",
        color_discrete_sequence=selected_palette,
        hover_data=["topic", "cluster-id", "label"],
        title=config["chart"]["title"],
        opacity=0.8,
        size_max=12,
        # カテゴリの順序を明示的に指定
        category_orders={
            "cluster-id": sorted(
                unique_clusters, key=lambda x: int(x) if x != "-1" else -1
            )
        },
    )

    # マーカーの境界線を追加して区別しやすくする
    fig.update_traces(
        marker=dict(
            line=dict(width=0.8, color="white"),  # 境界線を少し太くして視認性向上
            size=7,  # マーカーサイズを少し大きく
        )
    )

    # cluster-idを数値に戻してグループ化処理
    df["cluster-id-numeric"] = df["cluster-id"].astype(int)

    cluster_centers = (
        df.groupby("cluster-id-numeric")
        .agg(
            x=("x", "mean"),
            y=("y", "mean"),
            label=(
                "label",
                lambda x: x.mode()[0] if not x.mode().empty else "",
            ),
            count=("cluster-id-numeric", "count"),  # クラスターのサイズも取得
        )
        .reset_index()
    )

    # クラスターサイズに基づいてラベル表示を選択的に行う
    cluster_centers = cluster_centers.sort_values("count", ascending=False)

    # 動的にラベルサイズとyshiftを調整
    max_clusters_to_label = min(20, len(cluster_centers))

    for i, row in cluster_centers.head(max_clusters_to_label).iterrows():
        if row["cluster-id-numeric"] == -1:
            continue

        # ラベルテキストをより短く調整
        label_text = row["label"]
        if len(label_text) > 12:
            label_text = label_text[:9] + "..."

        cluster_label = f"C{row['cluster-id-numeric']}: {label_text}"

        # クラスターサイズに応じてフォントサイズを調整
        font_size = max(9, min(12, 8 + row["count"] // 10))

        # 重複を避けるためにランダムなオフセットを追加
        np.random.seed(int(row["cluster-id-numeric"]))  # 再現性のためにシード固定
        x_offset = np.random.uniform(-0.1, 0.1)
        y_offset = np.random.uniform(0.1, 0.3)

        fig.add_annotation(
            x=row["x"] + x_offset,
            y=row["y"] + y_offset,
            text=cluster_label,
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor="rgba(0, 0, 0, 0.6)",
            ax=0,
            ay=-20,
            font=dict(
                size=font_size,
                color="black",
                family="IPAexGothic, Noto Sans CJK JP, sans-serif",
                weight="bold",
            ),
            bgcolor="rgba(255, 255, 255, 0.95)",
            bordercolor="rgba(0, 0, 0, 0.5)",
            borderwidth=1.5,
            xanchor="center",
            yanchor="top",
        )

    # 残りの小さなクラスターには番号のみ表示
    for i, row in cluster_centers.tail(
        len(cluster_centers) - max_clusters_to_label
    ).iterrows():
        if row["cluster-id-numeric"] == -1:
            continue

        fig.add_annotation(
            x=row["x"],
            y=row["y"],
            text=f"C{row['cluster-id-numeric']}",
            showarrow=False,
            font=dict(
                size=8,
                color="white",
                family="IPAexGothic, Noto Sans CJK JP, sans-serif",
                weight="bold",
            ),
            bgcolor="rgba(0, 0, 0, 0.8)",
            bordercolor="rgba(255, 255, 255, 0.9)",
            borderwidth=1.5,
            xanchor="center",
            yanchor="middle",
        )

    fig.update_layout(
        font=dict(family="IPAexGothic, Noto Sans CJK JP, sans-serif", size=12),
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=1080,
        height=610,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.01,
            bgcolor="rgba(255, 255, 255, 0.95)",
            bordercolor="rgba(0, 0, 0, 0.4)",
            borderwidth=1.5,
            itemsizing="constant",
            font=dict(size=10, weight="bold"),
            # 凡例のタイトルを追加
            title=dict(text="Cluster ID", font=dict(size=12, weight="bold")),
        ),
        margin=dict(l=60, r=200, t=100, b=60),
        title=dict(font=dict(size=16, weight="bold"), x=0.5, xanchor="center"),
    )

    # 軸のグリッドを削除
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    # ズーム機能を有効にして詳細確認を容易に
    fig.update_layout(xaxis=dict(fixedrange=False), yaxis=dict(fixedrange=False))

    return fig


def chart(
    config: dict,
    pdf: pd.DataFrame,
    ldf: pd.DataFrame,
    full_html: bool,
    pallet_type: Literal[
        "colorblind_friendly", "vivid", "high_contrast"
    ] = "colorblind_friendly",
) -> str:
    fig = _create_scatter_plot(config, pdf, ldf, pallet_type)
    return fig.to_html(full_html=full_html)
