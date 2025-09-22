# 高度なRAGシステム (iRAG) プロジェクト概要

## プロジェクトの目的
文書検索とSQL検索を統合した高度なRAG（Retrieval-Augmented Generation）システム。Streamlitベースの直感的なUIを提供し、Azure OpenAI Serviceを活用して多言語対応の自然言語処理による強力な情報検索と質問応答を実現する。

## 技術スタック
- **フロントエンド**: Streamlit
- **バックエンド**: Python 3.9+
- **ベクトルデータベース**: PostgreSQL + pgvector
- **言語モデル**: Azure OpenAI (GPT-4o, text-embedding-ada-002)
- **日本語処理**: SudachiPy
- **検索エンジン**: LangChain + カスタムハイブリッドリトリーバー
- **PDF処理**: PyMuPDF, Azure Document Intelligence

## 主な依存ライブラリ
- langchain==0.3.25
- streamlit==1.45.1
- pgvector==0.2.4
- sudachipy==0.6.8
- azure-ai-documentintelligence==1.0.0b4
- PyMuPDF==1.26.0

## 主な機能
1. **ハイブリッド検索**: ベクトル検索とキーワード検索を組み合わせたRRF
2. **日本語特化**: SudachiPyによる形態素解析
3. **Text-to-SQL**: 自然言語からSQLクエリ生成
4. **専門用語辞書**: Golden-Retriever機能
5. **複数PDF処理エンジン**: PyMuPDF（高速）とAzure Document Intelligence（高精度）
6. **評価システム**: Recall、Precision、MRR、nDCG、Hit Rate
7. **ナレッジグラフ**: 専門用語の関係性を可視化