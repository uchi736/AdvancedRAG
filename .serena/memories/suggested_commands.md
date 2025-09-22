# 推奨コマンド一覧

## 開発環境セットアップ
```bash
# 仮想環境の作成と有効化
python -m venv myenv
myenv\Scripts\activate  # Windows

# 依存関係のインストール
pip install -r requirements.txt

# 環境変数の設定
copy .env.example .env
```

## アプリケーション実行
```bash
# Streamlitアプリの起動
streamlit run app.py
```

## 専門用語処理
```bash
# 専門用語抽出
python src/scripts/term_extractor_embeding.py

# クラスタリング分析
python src/scripts/term_clustering_analyzer.py

# DBへのインポート
python src/scripts/import_terms_to_db.py
```

## 評価システム
```bash
# RAGシステムの評価実行
python src/evaluation/evaluator.py
```

## Git操作（Windows）
```bash
git status
git add .
git commit -m "メッセージ"
git push origin main
```

## システムコマンド（Windows）
- `dir`: ディレクトリ内容表示
- `type`: ファイル内容表示
- `findstr`: テキスト検索
- `cd`: ディレクトリ移動

## データベース
```bash
# PostgreSQL接続（環境変数PG_URLを使用）
psql $env:PG_URL
```