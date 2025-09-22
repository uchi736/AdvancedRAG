# プロジェクト構成

## ルートディレクトリ
- `app.py`: Streamlitアプリケーションのエントリポイント
- `requirements.txt`: Python依存関係
- `.env.example`: 環境変数テンプレート
- `README.md`: プロジェクトドキュメント

## src/ ディレクトリ構成

### core/ - コアビジネスロジック
- `rag_system.py`: RAGシステムのファサード

### rag/ - RAGシステムのコアモジュール
- `chains.py`: LangChainのチェーンとプロンプト設定
- `config.py`: 設定ファイル
- `document_parser.py`: レガシーPDF処理
- `evaluator.py`: 評価システム
- `ingestion.py`: ドキュメント取り込み
- `jargon.py`: 専門用語辞書管理
- `prompts.py`: プロンプト管理
- `retriever.py`: ハイブリッド検索
- `sql_handler.py`: Text-to-SQL機能
- `text_processor.py`: 日本語テキスト処理
- `pdf_processors/`: PDF処理プロセッサ群

### ui/ - UIコンポーネント
- `chat_tab.py`: チャット画面
- `data_tab.py`: データ管理画面
- `dictionary_tab.py`: 辞書管理画面
- `documents_tab.py`: ドキュメント管理画面
- `evaluation_tab.py`: 評価システム画面
- `settings_tab.py`: 設定画面
- `sidebar.py`: サイドバー
- `state.py`: セッション状態管理

### evaluation/ - 評価システム
- `evaluator.py`: 評価メインスクリプト
- `test_scenarios.py`: テストシナリオ

### scripts/ - 拡張スクリプト
- `term_extract.py`: 専門用語抽出
- `term_extractor_embeding.py`: 埋め込みベース抽出
- `term_clustering_analyzer.py`: クラスタリング分析
- `knowledge_graph/`: ナレッジグラフ関連

### utils/ - ユーティリティ
- `helpers.py`: ヘルパー関数
- `style.py`: UIスタイル設定

## その他のディレクトリ
- `docs/`: ドキュメント
- `output/`: 出力ファイル
- `myenv/`: Python仮想環境