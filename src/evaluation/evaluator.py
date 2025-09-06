"""
RAG System Evaluation Script
============================
This script demonstrates how to evaluate the RAG system using the integrated evaluation module.
"""

import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from dotenv import load_dotenv
from src.rag.config import Config
from src.core.rag_system import RAGSystem

load_dotenv()


def load_test_questions_from_csv(csv_path: str) -> List[Dict[str, Any]]:
    """
    Load test questions from CSV file
    
    Expected CSV format:
    - 質問: The question column
    - 想定の引用元1, 想定の引用元2, etc.: Expected source columns
    """
    df = pd.read_csv(csv_path, encoding='utf-8')
    test_questions = []
    
    for _, row in df.iterrows():
        question = row.get('質問', '')
        if not question:
            continue
            
        # Extract expected sources from numbered columns
        expected_sources = []
        for i in range(1, 10):  # Check up to 10 expected sources
            source_col = f'想定の引用元{i}'
            if source_col in row and pd.notna(row[source_col]):
                expected_sources.append(str(row[source_col]))
        
        if expected_sources:
            test_questions.append({
                'question': question,
                'expected_sources': expected_sources
            })
    
    return test_questions


def create_sample_test_questions() -> List[Dict[str, Any]]:
    """
    Create sample test questions for demonstration
    """
    return [
        {
            'question': 'RAGシステムのベクトル検索について教えてください',
            'expected_sources': [
                'ベクトル検索はドキュメントの意味的類似性を利用した検索手法です',
                'Embeddingを使用して文書をベクトル化し、コサイン類似度で検索します'
            ]
        },
        {
            'question': 'ハイブリッド検索の仕組みは？',
            'expected_sources': [
                'ハイブリッド検索はキーワード検索とベクトル検索を組み合わせた手法',
                'BM25などの従来手法と埋め込みベースの検索を融合'
            ]
        },
        {
            'question': 'リランキングの効果について説明してください',
            'expected_sources': [
                'リランキングは初期検索結果を再順位付けして精度を向上させる',
                'Cross-Encoderモデルを使用してより正確な関連性スコアを計算'
            ]
        }
    ]


async def evaluate_rag_system(rag_system: RAGSystem, test_questions: List[Dict[str, Any]]):
    """
    Run evaluation on the RAG system
    """
    print("=" * 80)
    print("RAGシステム評価開始")
    print("=" * 80)
    
    # Single method evaluation
    print("\n1. 単一手法での評価（Azure Embedding）")
    metrics = await rag_system.evaluate_system(
        test_questions=test_questions,
        similarity_method="azure_embedding",
        export_path="evaluation_results_embedding.csv"
    )
    
    print(f"\n評価結果サマリー:")
    print(f"  - MRR: {metrics.mrr:.4f}")
    print(f"  - Recall@5: {metrics.recall_at_k.get(5, 0):.4f}")
    print(f"  - Precision@5: {metrics.precision_at_k.get(5, 0):.4f}")
    print(f"  - nDCG@5: {metrics.ndcg_at_k.get(5, 0):.4f}")
    
    # Comprehensive evaluation with multiple methods
    print("\n2. 複数手法での包括的評価")
    all_metrics = await rag_system.run_comprehensive_evaluation(
        test_questions=test_questions,
        methods=["azure_embedding", "text_overlap"],  # Reduced methods for faster demo
        export_path="evaluation_results_comprehensive.csv"
    )
    
    print("\n" + "=" * 80)
    print("評価比較結果")
    print("=" * 80)
    
    # Create comparison table
    comparison_data = []
    for method, metrics in all_metrics.items():
        comparison_data.append({
            'Method': method,
            'MRR': f"{metrics.mrr:.4f}",
            'Recall@5': f"{metrics.recall_at_k.get(5, 0):.4f}",
            'Precision@5': f"{metrics.precision_at_k.get(5, 0):.4f}",
            'nDCG@5': f"{metrics.ndcg_at_k.get(5, 0):.4f}",
            'Hit Rate@5': f"{metrics.hit_rate_at_k.get(5, 0):.4f}"
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    print(df_comparison.to_string(index=False))


async def evaluate_from_csv(rag_system: RAGSystem, csv_path: str):
    """
    Evaluate RAG system using CSV file
    """
    print("=" * 80)
    print(f"CSVファイルからの評価: {csv_path}")
    print("=" * 80)
    
    results = await rag_system.evaluate_from_csv(
        csv_path=csv_path,
        similarity_method="azure_embedding",
        export_path="evaluation_results_from_csv.csv"
    )
    
    if results:
        print(f"\n評価完了: {len(results)}件の質問を処理しました")
        
        # Calculate average metrics
        avg_mrr = sum(r.mrr for r in results) / len(results)
        avg_recall_5 = sum(r.recall_at_k.get(5, 0) for r in results) / len(results)
        avg_precision_5 = sum(r.precision_at_k.get(5, 0) for r in results) / len(results)
        
        print(f"\n平均評価指標:")
        print(f"  - 平均MRR: {avg_mrr:.4f}")
        print(f"  - 平均Recall@5: {avg_recall_5:.4f}")
        print(f"  - 平均Precision@5: {avg_precision_5:.4f}")


async def main():
    """
    Main function to run the evaluation
    """
    # Initialize configuration and RAG system
    load_dotenv()  # Ensure environment variables are loaded
    config = Config()
    rag_system = RAGSystem(config)
    
    print("\n" + "=" * 80)
    print("RAGシステム評価デモンストレーション")
    print("=" * 80)
    
    # Check for CSV evaluation data first, then fallback to sample questions
    csv_path = "evaluation_data.csv"
    if Path(csv_path).exists():
        print(f"\nCSVファイルからの評価: {csv_path}")
        await evaluate_from_csv(rag_system, csv_path)
    else:
        print("\nサンプル質問での評価（CSVファイルが見つかりません）")
        test_questions = create_sample_test_questions()
        await evaluate_rag_system(rag_system, test_questions)
        
        print(f"\nCSVファイル '{csv_path}' を作成すると、より詳細な評価が可能です。")
        print("CSVファイルを作成する場合は、以下の形式で保存してください:")
        print("  - 列: 質問, 想定の引用元1, 想定の引用元2, ...")
    
    print("\n" + "=" * 80)
    print("評価完了")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())