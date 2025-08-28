"""
RAG Evaluation Test Scenarios
=============================
This file contains various test scenarios for evaluating different aspects of the RAG system.
"""

import asyncio
from typing import List, Dict, Any
from dotenv import load_dotenv
from rag.config import Config
from rag_system_enhanced import RAGSystem

# Load environment variables
load_dotenv()

# Basic RAG concepts test scenarios
BASIC_RAG_SCENARIOS = [
    {
        'question': 'RAGシステムの基本的な仕組みを説明してください',
        'expected_sources': [
            'RAGはRetrieval-Augmented Generationの略で、検索と生成を組み合わせた手法',
            '外部知識ベースから関連文書を検索し、コンテキストとして提供する',
            '大規模言語モデルの知識を動的に拡張する技術'
        ]
    },
    {
        'question': 'ベクトル検索とキーワード検索の違いは何ですか？',
        'expected_sources': [
            'ベクトル検索は意味的類似性を計算する手法',
            'キーワード検索は単語の出現頻度に基づく手法',
            'ベクトル検索は文脈や同義語を考慮できる'
        ]
    },
    {
        'question': 'ハイブリッド検索の利点を教えてください',
        'expected_sources': [
            'ベクトル検索とキーワード検索の両方の利点を活用',
            'RRFアルゴリズムで結果を統合',
            '検索精度と網羅性の向上'
        ]
    }
]

# Technical implementation scenarios
TECHNICAL_SCENARIOS = [
    {
        'question': 'PostgreSQLでpgvectorを最適化する方法は？',
        'expected_sources': [
            'HNSWインデックスの使用',
            'インデックスパラメータの調整',
            '定期的なVACUUMとANALYZE',
            'メモリパラメータの最適化'
        ]
    },
    {
        'question': 'LangChainでカスタムチェーンを作成する手順は？',
        'expected_sources': [
            'RunnablePassthroughとRunnableLambdaの使用',
            '入出力型の明確な定義',
            'エラーハンドリングの実装',
            'テストケースの作成'
        ]
    },
    {
        'question': 'Streamlitでの効率的な状態管理方法は？',
        'expected_sources': [
            'st.session_stateの活用',
            '状態の初期化と存在確認',
            '再実行の最小化',
            '構造化されたデータクラスの使用'
        ]
    }
]

# Advanced topics scenarios
ADVANCED_SCENARIOS = [
    {
        'question': '大規模データセットでのRAG性能最適化戦略は？',
        'expected_sources': [
            '階層化インデックス構造の使用',
            'ベクトル圧縮技術の適用',
            'キャッシュ戦略の実装',
            '分散処理システムの構築'
        ]
    },
    {
        'question': 'セキュリティを考慮したRAGシステム設計のポイントは？',
        'expected_sources': [
            'アクセス制御リストの実装',
            'データの匿名化処理',
            'HTTPS暗号化とトークン認証',
            'ログ記録と監査機能'
        ]
    },
    {
        'question': 'マルチモーダルRAGシステムの実装方法は？',
        'expected_sources': [
            '異なるモダリティの統一処理',
            '専用エンベディングモデルの使用',
            'クロスモーダル検索機能',
            '統合ベクトル空間での類似性計算'
        ]
    }
]

# Edge cases and challenging scenarios
EDGE_CASE_SCENARIOS = [
    {
        'question': 'あいまいな質問に対する処理方法',
        'expected_sources': [
            '質問の意図を明確化するプロンプト',
            '複数の解釈の提示',
            'ユーザーからの追加情報の要求'
        ]
    },
    {
        'question': '関連情報が見つからない場合の対応',
        'expected_sources': [
            '検索クエリの拡張',
            'ゼロショット回答の提供',
            '代替的な検索戦略の実行'
        ]
    },
    {
        'question': '矛盾する情報が複数ある場合の処理',
        'expected_sources': [
            '情報源の信頼性評価',
            '時系列での情報の整理',
            '矛盾点の明示的な説明'
        ]
    }
]

# Evaluation quality scenarios
EVALUATION_SCENARIOS = [
    {
        'question': 'RAGシステムの評価指標について説明してください',
        'expected_sources': [
            'Recall: 関連文書の再現率',
            'Precision: 検索結果の精度',
            'MRR: 平均逆順位',
            'nDCG: 正規化減損累積利得'
        ]
    },
    {
        'question': '類似度計算の手法にはどのようなものがありますか？',
        'expected_sources': [
            'コサイン類似度による計算',
            'LLMベースの関連性判定',
            'テキスト重複度の計算',
            'ハイブリッド手法の組み合わせ'
        ]
    }
]

# Domain-specific scenarios (Japanese language processing)
JAPANESE_NLP_SCENARIOS = [
    {
        'question': '日本語処理の特有の課題は何ですか？',
        'expected_sources': [
            '語間にスペースがない連続文字列',
            '漢字、ひらがな、カタカナの混在',
            '同音異義語による曖昧性',
            '敬語や方言への対応'
        ]
    },
    {
        'question': 'SudachiPyの利点を教えてください',
        'expected_sources': [
            '高精度な日本語形態素解析',
            '正規化機能による表記揺れ対応',
            '複数の辞書モードの提供',
            '豊富な品詞情報の出力'
        ]
    }
]

# Performance and scalability scenarios
PERFORMANCE_SCENARIOS = [
    {
        'question': 'システムのレスポンス時間を改善する方法は？',
        'expected_sources': [
            'インデックスの最適化',
            'キャッシュの効果的な活用',
            '並列処理の実装',
            'クエリの効率化'
        ]
    },
    {
        'question': '大量のドキュメントを効率的に処理するには？',
        'expected_sources': [
            'バッチ処理の実装',
            'データの前処理と正規化',
            '増分インデックス更新',
            'リソース使用量の監視'
        ]
    }
]

ALL_SCENARIOS = {
    'basic_rag': BASIC_RAG_SCENARIOS,
    'technical': TECHNICAL_SCENARIOS,
    'advanced': ADVANCED_SCENARIOS,
    'edge_cases': EDGE_CASE_SCENARIOS,
    'evaluation': EVALUATION_SCENARIOS,
    'japanese_nlp': JAPANESE_NLP_SCENARIOS,
    'performance': PERFORMANCE_SCENARIOS
}

async def run_scenario_evaluation(rag_system: RAGSystem, scenario_name: str, scenarios: List[Dict[str, Any]]):
    """
    Run evaluation for a specific scenario set
    """
    print(f"\n{'='*80}")
    print(f"評価シナリオ: {scenario_name}")
    print(f"{'='*80}")
    
    # Run evaluation
    metrics = await rag_system.evaluate_system(
        test_questions=scenarios,
        similarity_method="azure_embedding",
        export_path=f"evaluation_{scenario_name}.csv"
    )
    
    # Print summary
    print(f"\n{scenario_name} 評価結果:")
    print(f"  - 質問数: {metrics.num_questions}")
    print(f"  - 平均MRR: {metrics.mrr:.4f}")
    print(f"  - 平均Recall@5: {metrics.recall_at_k.get(5, 0):.4f}")
    print(f"  - 平均Precision@5: {metrics.precision_at_k.get(5, 0):.4f}")
    print(f"  - 平均nDCG@5: {metrics.ndcg_at_k.get(5, 0):.4f}")
    
    return metrics

async def run_comprehensive_scenario_evaluation():
    """
    Run evaluation for all scenario sets
    """
    # Initialize RAG system
    config = Config()
    rag_system = RAGSystem(config)
    
    print("RAG評価システム - 多様なシナリオテスト")
    print("="*80)
    
    all_metrics = {}
    
    # Run evaluation for each scenario set
    for scenario_name, scenarios in ALL_SCENARIOS.items():
        try:
            metrics = await run_scenario_evaluation(rag_system, scenario_name, scenarios)
            all_metrics[scenario_name] = metrics
        except Exception as e:
            print(f"エラー - {scenario_name}: {e}")
    
    # Create comparison report
    print(f"\n{'='*80}")
    print("総合評価比較")
    print(f"{'='*80}")
    
    comparison_data = []
    for scenario_name, metrics in all_metrics.items():
        comparison_data.append({
            'Scenario': scenario_name,
            'Questions': metrics.num_questions,
            'MRR': f"{metrics.mrr:.4f}",
            'Recall@5': f"{metrics.recall_at_k.get(5, 0):.4f}",
            'Precision@5': f"{metrics.precision_at_k.get(5, 0):.4f}",
            'nDCG@5': f"{metrics.ndcg_at_k.get(5, 0):.4f}"
        })
    
    # Print comparison table
    if comparison_data:
        import pandas as pd
        df_comparison = pd.DataFrame(comparison_data)
        print(df_comparison.to_string(index=False))
        
        # Save to CSV
        df_comparison.to_csv("scenario_comparison.csv", index=False, encoding='utf-8')
        print(f"\n比較結果を scenario_comparison.csv に保存しました")

def create_custom_scenario(questions_and_sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Create a custom evaluation scenario
    
    Args:
        questions_and_sources: List of dictionaries with 'question' and 'expected_sources' keys
    
    Returns:
        List of scenario dictionaries
    """
    return questions_and_sources

async def evaluate_custom_scenario(rag_system: RAGSystem, 
                                 scenario_name: str,
                                 scenarios: List[Dict[str, Any]],
                                 similarity_method: str = "azure_embedding"):
    """
    Evaluate a custom scenario
    """
    return await run_scenario_evaluation(rag_system, scenario_name, scenarios)

if __name__ == "__main__":
    # Run comprehensive evaluation
    asyncio.run(run_comprehensive_scenario_evaluation())