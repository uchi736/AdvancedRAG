# コードスタイルと規約

## Python コーディング規約
- **型ヒント**: 使用（pydantic、typing_extensionsを活用）
- **docstring**: 関数・クラスに日本語での説明を記載
- **命名規則**: 
  - クラス名: PascalCase（例: RAGSystem, HybridRetriever）
  - 関数/変数: snake_case（例: get_retriever, text_processor）
  - 定数: UPPER_SNAKE_CASE（例: ENV_DEFAULTS）

## インポート順序
1. 標準ライブラリ
2. サードパーティライブラリ
3. ローカルモジュール

## エラーハンドリング
- try-exceptブロックで適切にラップ
- ログ出力とユーザーフレンドリーなエラーメッセージ

## 設計パターン
- **ファサードパターン**: RAGSystemクラスで複雑な処理を隠蔽
- **戦略パターン**: PDF処理エンジンの切り替え（PDFProcessorStrategy）
- **シングルトン**: Config管理

## 日本語処理
- SudachiPy Mode.Cを使用（最長一致）
- UTF-8エンコーディング徹底