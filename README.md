# bsky_listening

## 概要

`bsky_listening` は、Blueskyソーシャルメディアプラットフォームから特定のユーザーの投稿を取得し、その内容を分析・可視化するためのPythonツールです。投稿をトピックごとにクラスタリングし、その傾向を散布図として表示することで、ユーザーの活動や関心事を視覚的に把握することを目的としています。

## 機能

### 1. 投稿の取得 (`fetch`)

指定されたBlueskyアカウントから投稿データを取得し、TSV形式で保存します。リポストは除外されます。

- **モジュール**: `libs/bsky.py`
- **コマンド**: `python main.py fetch --pj_name [プロジェクト名] --limit [取得件数]`
  - `pj_name`: プロジェクト名（デフォルトは現在日付 `YYYYMMDD`）
  - `limit`: 取得する投稿の最大件数（デフォルトは500）

### 2. 投稿の前処理 (`preproc`)

取得した投稿テキストから主要なトピックを抽出し、分析しやすいように整形します。

- **モジュール**: `libs/preproc.py`
- **コマンド**: `python main.py preproc --pj_name [プロジェクト名]`

### 3. 埋め込み生成 (`embedding`)

前処理されたトピックデータから、機械学習モデル（OpenAIまたはOllama）を使用して数値ベクトル（埋め込み）を生成します。

- **モジュール**: `libs/embedding.py`
- **コマンド**: `python main.py embedding --pj_name [プロジェクト名] --is_local [True/False]`
  - `is_local`: ローカルのOllamaモデルを使用するかどうか（デフォルトはTrue）。Falseの場合、OpenAIモデルを使用します。

### 4. クラスタリング (`clustering`)

生成された埋め込みデータを用いて、投稿を類似性に基づいてクラスタリングします。UMAP、HDBSCAN、BERTopic、Spectral Clusteringを組み合わせて使用します。

- **モジュール**: `libs/clustering.py`
- **コマンド**: `python main.py clustering --pj_name [プロジェクト名]`

### 5. ラベリング (`labeling`)

クラスタリングされた各グループに対して、その特徴をもっともよく表すラベルを生成します。

- **モジュール**: `libs/labeling.py`
- **コマンド**: `python main.py labeling --pj_name [プロジェクト名]`

### 6. チャート作成 (`chart`)

クラスタリングとラベリングの結果を基に、インタラクティブな散布図（HTML形式）を生成します。

- **モジュール**: `libs/chart.py`
- **コマンド**: `python main.py chart --pj_name [プロジェクト名]`

### 7. すべての処理を一括実行 (`all`)

Blueskyの投稿取得からチャート作成までの一連の処理をすべて実行します。

- **モジュール**: `main.py`
- **コマンド**: `python main.py all --pj_name [プロジェクト名] --limit [取得件数] --is_local [True/False]`
  - `pj_name`: プロジェクト名（デフォルトは現在日付 `YYYYMMDD`）
  - `limit`: 取得する投稿の最大件数（デフォルトは500）
  - `is_local`: 埋め込み生成でローカルのOllamaモデルを使用するかどうか（デフォルトはTrue）。Falseの場合、OpenAIモデルを使用します。

## 環境設定

### `config.toml`

`config.toml.example` を参考に `config.toml` ファイルを作成し、Blueskyのハンドル、使用するLLMモデル、チャートのタイトルなどを設定してください。

```toml
[bluesky]
handle = "your-handle.bsky.social" # あなたのBlueskyハンドル

[preproc]
model = "grok-3-fast-beta" # 前処理に使用するLLMモデル
system_prompt = """
# ここに前処理用のシステムプロンプトを記述
"""

[embedding]
ollama_base_url = "http://localhost:11434" # OllamaのベースURL
ollama_model = "jeffh/intfloat-multilingual-e5-large-instruct:f16" # Ollamaモデル名
openai_model = "text-embedding-3-small" # OpenAIモデル名

[clustering]
n_clusters = 44 # クラスタリングするクラスター数

[labeling]
model="grok-3-beta" # ラベリングに使用するLLMモデル
system_prompt = """
# ここにラベリング用のシステムプロンプトを記述
"""

[chart]
title="わたしのアカウントのポスト分析" # チャートのタイトル
```

### `.env`

`BSKY_APP_PASS`, `XAI_API_KEY`, `OPENAI_API_KEY` などのAPIキーやパスワードを `.env.example` を参考に `.env` ファイルに設定してください。

```dotenv
BSKY_APP_PASS="your_bluesky_app_password"
XAI_API_KEY="your_xai_api_key"
OPENAI_API_KEY="your_openai_api_key"
```

## 実行方法

各機能は `fire` コマンドラインツールを使用して実行できます。

```bash
# 投稿の取得
python main.py fetch --limit 1000

# 投稿の前処理
python main.py preproc

# 埋め込み生成 (Ollamaを使用する場合)
python main.py embedding --is_local True

# 埋め込み生成 (OpenAIを使用する場合)
python main.py embedding --is_local False

# クラスタリング
python main.py clustering

# ラベリング
python main.py labeling

# チャート作成
python main.py chart

# 全ての処理を一括実行
python main.py all --limit 1000 --is_local True
```

各ステップは、前のステップが完了していることを前提としています。

## ライセンス

このプロジェクトは［LICENSE］ファイルに記載されているライセンスの下で公開されています。
