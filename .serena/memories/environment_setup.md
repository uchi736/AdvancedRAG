# 環境設定情報

## 必須環境変数
```
# Azure OpenAI
AZURE_OPENAI_API_KEY=<APIキー>
AZURE_OPENAI_ENDPOINT=<エンドポイント>
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=<GPT-4oデプロイメント名>
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=<埋め込みモデル名>

# PostgreSQL + pgvector
PG_URL=<接続URL>
# または個別設定
DB_HOST=<ホスト>
DB_PORT=<ポート>
DB_DATABASE=<データベース名>
DB_USER=<ユーザー>
DB_PASSWORD=<パスワード>

# PDF処理（オプション）
PDF_PROCESSOR_TYPE=legacy|pymupdf|azure_di

# Azure Document Intelligence（azure_di使用時）
AZURE_DI_ENDPOINT=<エンドポイント>
AZURE_DI_API_KEY=<APIキー>
AZURE_DI_MODEL=prebuilt-layout
SAVE_MARKDOWN=false
```

## 開発環境
- **OS**: Windows
- **Python**: 3.9+
- **仮想環境**: myenv/
- **IDE推奨**: VSCode（Python拡張機能）

## データベース要件
- PostgreSQL 14+
- pgvector拡張機能
- 必要なテーブル:
  - documents
  - chunks
  - jargon_dictionary
  - knowledge_nodes
  - knowledge_edges