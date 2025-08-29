# ハイブリッド検索の実装ガイド

## ハイブリッド検索とは

ハイブリッド検索は、キーワードベースの全文検索（BM25、TF-IDFなど）とベクトルベースのセマンティック検索を組み合わせた高度な検索手法です。この手法により、統計的手法による精密な語彙一致と、埋め込みによる意味的類似性の両方を活用して検索精度を大幅に向上させることができます。

## ハイブリッド検索の必要性

### 各検索手法の特徴と限界

#### キーワード検索（BM25）の特徴
**利点：**
- 精密な語彙一致による高い適合率
- 実装が比較的簡単
- 計算コストが低い
- 結果の解釈可能性が高い

**限界：**
- 同義語や類義語への対応が困難
- 文脈や意図の理解が不十分
- 表記揺れや省略語への対応が弱い

#### ベクトル検索の特徴
**利点：**
- 意味的類似性の高精度な計算
- 同義語や関連概念の自動発見
- 多言語対応が可能
- 文脈を考慮した検索

**限界：**
- 固有名詞や専門用語の精密一致が困難
- 計算コストが高い
- ブラックボックス的で解釈が困難

### ハイブリッド検索の効果

両手法を組み合わせることで：
1. **補完効果**: 各手法の弱点を相互に補完
2. **精度向上**: 単独手法より高い検索精度を実現
3. **ロバスト性**: 多様なクエリタイプに対応
4. **バランス**: 精密性と再現性のバランス改善

## Reciprocal Rank Fusion (RRF)アルゴリズム

RRFは異なる検索手法の結果を効果的に統合するアルゴリズムで、各手法の順位を考慮して最終的なランキングを決定します。

### RRFの計算式

```
RRF(d) = Σ(1 / (k + rank_i(d)))
```

- `d`: 文書
- `k`: 定数（通常60）
- `rank_i(d)`: 検索手法iにおける文書dの順位

### RRFの実装例

```python
def reciprocal_rank_fusion(results_list, k=60):
    """
    複数の検索結果をRRFアルゴリズムで統合
    
    Args:
        results_list: 各検索手法の結果リスト
        k: RRF定数
    
    Returns:
        統合された検索結果
    """
    doc_scores = {}
    
    for results in results_list:
        for rank, (doc_id, score) in enumerate(results, 1):
            if doc_id not in doc_scores:
                doc_scores[doc_id] = 0
            doc_scores[doc_id] += 1 / (k + rank)
    
    # スコア順にソート
    return sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

# 使用例
vector_results = [(doc1, 0.9), (doc2, 0.8), (doc3, 0.7)]
keyword_results = [(doc3, 15.5), (doc1, 12.3), (doc4, 10.1)]

fused_results = reciprocal_rank_fusion([vector_results, keyword_results])
```

## ハイブリッド検索の実装パターン

### パターン1: スコア加重平均

```python
class WeightedHybridSearch:
    def __init__(self, vector_weight=0.7, keyword_weight=0.3):
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
    
    def search(self, query, k=10):
        # ベクトル検索実行
        vector_results = self.vector_search(query, k*2)
        
        # キーワード検索実行
        keyword_results = self.keyword_search(query, k*2)
        
        # スコア正規化
        vector_normalized = self.normalize_scores(vector_results)
        keyword_normalized = self.normalize_scores(keyword_results)
        
        # 加重融合
        combined_scores = {}
        for doc_id, score in vector_normalized.items():
            combined_scores[doc_id] = self.vector_weight * score
        
        for doc_id, score in keyword_normalized.items():
            if doc_id in combined_scores:
                combined_scores[doc_id] += self.keyword_weight * score
            else:
                combined_scores[doc_id] = self.keyword_weight * score
        
        return sorted(combined_scores.items(), 
                     key=lambda x: x[1], reverse=True)[:k]
    
    def normalize_scores(self, results):
        """スコアを0-1の範囲に正規化"""
        if not results:
            return {}
        
        max_score = max(score for _, score in results)
        min_score = min(score for _, score in results)
        score_range = max_score - min_score
        
        if score_range == 0:
            return {doc_id: 1.0 for doc_id, _ in results}
        
        return {
            doc_id: (score - min_score) / score_range 
            for doc_id, score in results
        }
```

### パターン2: RRFベース統合

```python
class RRFHybridSearch:
    def __init__(self, rrf_k=60):
        self.rrf_k = rrf_k
    
    def search(self, query, k=10):
        # 複数の検索手法を実行
        vector_results = self.vector_search(query, k*3)
        bm25_results = self.bm25_search(query, k*3)
        tfidf_results = self.tfidf_search(query, k*3)
        
        # RRFで統合
        all_results = [vector_results, bm25_results, tfidf_results]
        return self.reciprocal_rank_fusion(all_results)[:k]
    
    def reciprocal_rank_fusion(self, results_list):
        doc_scores = {}
        
        for results in results_list:
            for rank, (doc_id, _) in enumerate(results, 1):
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0
                doc_scores[doc_id] += 1 / (self.rrf_k + rank)
        
        return sorted(doc_scores.items(), 
                     key=lambda x: x[1], reverse=True)
```

### パターン3: 段階的フィルタリング

```python
class TieredHybridSearch:
    def __init__(self, 
                 stage1_k=1000, 
                 stage2_k=100, 
                 final_k=10):
        self.stage1_k = stage1_k
        self.stage2_k = stage2_k
        self.final_k = final_k
    
    def search(self, query):
        # Stage 1: 高速な粗検索
        candidates = self.fast_keyword_search(query, self.stage1_k)
        
        # Stage 2: 候補に対するベクトル検索
        vector_filtered = self.vector_rerank(query, candidates, self.stage2_k)
        
        # Stage 3: 詳細なリランキング
        final_results = self.detailed_rerank(query, vector_filtered, self.final_k)
        
        return final_results
    
    def vector_rerank(self, query, candidates, k):
        """候補文書をベクトル類似度で再ランキング"""
        query_vector = self.encode_query(query)
        scored_candidates = []
        
        for doc_id in candidates:
            doc_vector = self.get_document_vector(doc_id)
            similarity = self.cosine_similarity(query_vector, doc_vector)
            scored_candidates.append((doc_id, similarity))
        
        return sorted(scored_candidates, 
                     key=lambda x: x[1], reverse=True)[:k]
```

## 実用的なハイブリッド検索システム

### LangChain Retrieverとしての実装

```python
from langchain.schema import BaseRetriever, Document
from typing import List
import numpy as np

class HybridRetriever(BaseRetriever):
    def __init__(self, 
                 vector_store,
                 bm25_retriever,
                 fusion_method="rrf",
                 vector_weight=0.7,
                 keyword_weight=0.3,
                 rrf_k=60):
        
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.fusion_method = fusion_method
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.rrf_k = rrf_k
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        # ベクトル検索
        vector_docs = self.vector_store.similarity_search_with_score(query, k=20)
        
        # BM25検索
        bm25_docs = self.bm25_retriever.get_relevant_documents(query)
        
        if self.fusion_method == "rrf":
            return self._rrf_fusion(query, vector_docs, bm25_docs)
        elif self.fusion_method == "weighted":
            return self._weighted_fusion(vector_docs, bm25_docs)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
    
    def _rrf_fusion(self, query, vector_docs, bm25_docs):
        doc_scores = {}
        
        # ベクトル検索結果の処理
        for rank, (doc, score) in enumerate(vector_docs, 1):
            doc_id = doc.metadata.get('id', str(hash(doc.page_content)))
            doc_scores[doc_id] = {
                'score': 1 / (self.rrf_k + rank),
                'doc': doc
            }
        
        # BM25検索結果の処理
        for rank, doc in enumerate(bm25_docs, 1):
            doc_id = doc.metadata.get('id', str(hash(doc.page_content)))
            if doc_id in doc_scores:
                doc_scores[doc_id]['score'] += 1 / (self.rrf_k + rank)
            else:
                doc_scores[doc_id] = {
                    'score': 1 / (self.rrf_k + rank),
                    'doc': doc
                }
        
        # スコア順にソート
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        return [item['doc'] for item in sorted_docs[:10]]
    
    def _weighted_fusion(self, vector_docs, bm25_docs):
        # 実装省略（前述の例を参考）
        pass
```

### Elasticsearch + ベクトル検索の統合

```python
from elasticsearch import Elasticsearch
import numpy as np

class ElasticsearchHybridSearch:
    def __init__(self, es_client, index_name, vector_field="embedding"):
        self.es = es_client
        self.index_name = index_name
        self.vector_field = vector_field
    
    def search(self, query, query_vector, k=10):
        # ハイブリッドクエリの構築
        hybrid_query = {
            "query": {
                "bool": {
                    "should": [
                        # キーワード検索
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["title^2", "content"],
                                "type": "best_fields"
                            }
                        },
                        # ベクトル検索
                        {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": "cosineSimilarity(params.query_vector, doc[params.vector_field]) + 1.0",
                                    "params": {
                                        "query_vector": query_vector.tolist(),
                                        "vector_field": self.vector_field
                                    }
                                }
                            }
                        }
                    ]
                }
            },
            "size": k
        }
        
        response = self.es.search(
            index=self.index_name,
            body=hybrid_query
        )
        
        return self._process_results(response)
    
    def _process_results(self, response):
        results = []
        for hit in response['hits']['hits']:
            results.append({
                'id': hit['_id'],
                'content': hit['_source'],
                'score': hit['_score']
            })
        return results
```

## ハイブリッド検索の最適化

### パラメータチューニング

```python
class HybridSearchOptimizer:
    def __init__(self, search_engine, evaluation_dataset):
        self.search_engine = search_engine
        self.eval_data = evaluation_dataset
    
    def optimize_weights(self, weight_range=(0.1, 0.9), step=0.1):
        best_score = 0
        best_weights = None
        
        for vector_weight in np.arange(weight_range[0], weight_range[1], step):
            keyword_weight = 1 - vector_weight
            
            # 重みを設定
            self.search_engine.set_weights(vector_weight, keyword_weight)
            
            # 評価実行
            score = self.evaluate()
            
            if score > best_score:
                best_score = score
                best_weights = (vector_weight, keyword_weight)
        
        return best_weights, best_score
    
    def evaluate(self):
        total_score = 0
        for query_data in self.eval_data:
            results = self.search_engine.search(query_data['query'])
            score = self.calculate_ndcg(results, query_data['relevant_docs'])
            total_score += score
        
        return total_score / len(self.eval_data)
```

### 動的重み調整

```python
class AdaptiveHybridSearch:
    def __init__(self):
        self.query_classifier = self.load_query_classifier()
    
    def search(self, query, k=10):
        # クエリタイプを分類
        query_type = self.classify_query(query)
        
        # タイプに応じて重みを調整
        if query_type == "factual":
            vector_weight, keyword_weight = 0.3, 0.7  # キーワード重視
        elif query_type == "conceptual":
            vector_weight, keyword_weight = 0.8, 0.2  # ベクトル重視
        else:
            vector_weight, keyword_weight = 0.6, 0.4  # バランス
        
        return self.weighted_search(query, vector_weight, keyword_weight, k)
    
    def classify_query(self, query):
        # クエリタイプ分類のロジック
        if any(word in query.lower() for word in ['what', 'who', 'when', 'where']):
            return "factual"
        elif any(word in query.lower() for word in ['why', 'how', 'explain', 'concept']):
            return "conceptual"
        else:
            return "mixed"
```

## ハイブリッド検索の評価

### 包括的評価フレームワーク

```python
class HybridSearchEvaluator:
    def __init__(self):
        self.metrics = ["precision", "recall", "f1", "ndcg", "map"]
    
    def comprehensive_evaluation(self, search_systems, test_queries):
        results = {}
        
        for system_name, search_system in search_systems.items():
            system_results = {}
            
            for metric in self.metrics:
                scores = []
                
                for query_data in test_queries:
                    search_results = search_system.search(query_data['query'])
                    score = self.calculate_metric(
                        metric, search_results, query_data['relevant_docs']
                    )
                    scores.append(score)
                
                system_results[metric] = np.mean(scores)
            
            results[system_name] = system_results
        
        return results
    
    def calculate_metric(self, metric, results, relevant_docs):
        if metric == "precision":
            return self.precision_at_k(results, relevant_docs, k=10)
        elif metric == "recall":
            return self.recall_at_k(results, relevant_docs, k=10)
        elif metric == "ndcg":
            return self.ndcg_at_k(results, relevant_docs, k=10)
        # 他のメトリクスの実装...
```

## まとめ

ハイブリッド検索は現代の情報検索システムにおいて重要な技術であり、適切に実装することで単独の検索手法を大幅に上回る性能を実現できます。RRFアルゴリズムを中心とした融合技術と、継続的な最適化により、ユーザーニーズに最適化された高性能な検索システムの構築が可能です。