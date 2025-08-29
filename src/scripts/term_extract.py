"""golden_retriever_rag.py
~~~~~~~~~~~~~~~~~~~~~~~
Golden-Retriever方式を実装した専門用語対応RAGシステム
既存のrag_system.pyを拡張

主な追加機能:
1. 専門用語辞書（Jargon Dictionary）の管理
2. クエリからの専門用語抽出
3. 専門用語を用いたクエリ補強
4. 文書の要約・メタデータ付与による検索精度向上
"""
from __future__ import annotations

import os
import json
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from operator import itemgetter
import pandas as pd

from dotenv import load_dotenv
from sqlalchemy import create_engine, text, inspect as sqlalchemy_inspect

# LangChain imports
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredFileLoader,
    Docx2txtLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI, AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_community.vectorstores.pgvector import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, RunnablePassthrough, RunnableLambda, RunnableParallel, RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks.manager import get_openai_callback

# Load .env early
load_dotenv()

# Import existing RAG system as base
from src.core.rag_system import Config as BaseConfig, RAGSystem as BaseRAGSystem
from src.rag.retriever import JapaneseHybridRetriever as HybridRetriever

###############################################################################
# Enhanced Config with Golden-Retriever settings                              #
###############################################################################

@dataclass
class GoldenRetrieverConfig(BaseConfig):
    """拡張設定：Golden-Retriever固有の設定を追加"""
    # Jargon Dictionary settings
    enable_jargon_extraction: bool = True
    jargon_table_name: str = "jargon_dictionary"
    max_jargon_terms_per_query: int = 5
    
    # Document preprocessing settings
    enable_doc_summarization: bool = True
    enable_metadata_enrichment: bool = True
    
    # Query augmentation settings
    confidence_threshold: float = 0.7  # 専門用語の信頼度閾値

###############################################################################
# Jargon Dictionary Manager                                                   #
###############################################################################

class JargonDictionaryManager:
    """専門用語辞書を管理するクラス"""
    
    def __init__(self, connection_string: str, table_name: str = "jargon_dictionary"):
        self.connection_string = connection_string
        self.table_name = table_name
        self._init_jargon_table()
    
    def _init_jargon_table(self):
        """専門用語辞書テーブルを初期化"""
        engine = create_engine(self.connection_string)
        with engine.connect() as conn:
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id SERIAL PRIMARY KEY,
                    term TEXT UNIQUE NOT NULL,
                    definition TEXT NOT NULL,
                    domain TEXT,
                    aliases TEXT[],
                    related_terms TEXT[],
                    confidence_score FLOAT DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            # インデックスを作成
            conn.execute(text(f"""
                CREATE INDEX IF NOT EXISTS idx_jargon_term 
                ON {self.table_name} (LOWER(term))
            """))
            conn.execute(text(f"""
                CREATE INDEX IF NOT EXISTS idx_jargon_aliases 
                ON {self.table_name} USING GIN(aliases)
            """))
            conn.commit()
    
    def add_term(self, term: str, definition: str, domain: Optional[str] = None,
                 aliases: Optional[List[str]] = None, related_terms: Optional[List[str]] = None,
                 confidence_score: float = 1.0) -> bool:
        """専門用語を辞書に追加"""
        try:
            engine = create_engine(self.connection_string)
            with engine.connect() as conn:
                conn.execute(text(f"""
                    INSERT INTO {self.table_name} 
                    (term, definition, domain, aliases, related_terms, confidence_score)
                    VALUES (:term, :definition, :domain, :aliases, :related_terms, :confidence_score)
                    ON CONFLICT (term) DO UPDATE SET
                        definition = EXCLUDED.definition,
                        domain = EXCLUDED.domain,
                        aliases = EXCLUDED.aliases,
                        related_terms = EXCLUDED.related_terms,
                        confidence_score = EXCLUDED.confidence_score,
                        updated_at = CURRENT_TIMESTAMP
                """), {
                    "term": term,
                    "definition": definition,
                    "domain": domain,
                    "aliases": aliases or [],
                    "related_terms": related_terms or [],
                    "confidence_score": confidence_score
                })
                conn.commit()
            return True
        except Exception as e:
            print(f"Error adding term to jargon dictionary: {e}")
            return False
    
    def lookup_terms(self, terms: List[str]) -> Dict[str, Dict[str, Any]]:
        """複数の専門用語を一度に検索"""
        if not terms:
            return {}
        
        engine = create_engine(self.connection_string)
        results = {}
        
        try:
            with engine.connect() as conn:
                # 大文字小文字を区別しない検索
                placeholders = ', '.join([f':term_{i}' for i in range(len(terms))])
                query = text(f"""
                    SELECT term, definition, domain, aliases, related_terms, confidence_score
                    FROM {self.table_name}
                    WHERE LOWER(term) IN ({placeholders})
                    OR term = ANY(:aliases_check)
                """)
                
                params = {f"term_{i}": term.lower() for i, term in enumerate(terms)}
                params["aliases_check"] = terms
                
                rows = conn.execute(query, params).fetchall()
                
                for row in rows:
                    results[row.term] = {
                        "definition": row.definition,
                        "domain": row.domain,
                        "aliases": row.aliases or [],
                        "related_terms": row.related_terms or [],
                        "confidence_score": row.confidence_score
                    }
        except Exception as e:
            print(f"Error looking up terms: {e}")
        
        return results
    
    def bulk_import_from_csv(self, csv_path: str) -> Tuple[int, int]:
        """CSVファイルから専門用語を一括インポート"""
        try:
            df = pd.read_csv(csv_path)
            success_count = 0
            error_count = 0
            
            required_columns = ["term", "definition"]
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"CSV must contain columns: {required_columns}")
            
            for _, row in df.iterrows():
                success = self.add_term(
                    term=row["term"],
                    definition=row["definition"],
                    domain=row.get("domain"),
                    aliases=row.get("aliases", "").split("|") if "aliases" in row and pd.notna(row["aliases"]) else None,
                    related_terms=row.get("related_terms", "").split("|") if "related_terms" in row and pd.notna(row["related_terms"]) else None,
                    confidence_score=row.get("confidence_score", 1.0)
                )
                if success:
                    success_count += 1
                else:
                    error_count += 1
            
            return success_count, error_count
        except Exception as e:
            print(f"Error importing from CSV: {e}")
            return 0, -1

###############################################################################
# Golden-Retriever Enhanced RAG System                                        #
###############################################################################

class GoldenRetrieverRAG(BaseRAGSystem):
    """Golden-Retriever方式を実装した拡張RAGシステム"""
    
    def __init__(self, cfg: GoldenRetrieverConfig):
        super().__init__(cfg)
        self.config: GoldenRetrieverConfig = cfg
        self.jargon_manager = JargonDictionaryManager(
            self.connection_string, 
            cfg.jargon_table_name
        )
        
        # 専門用語抽出用のプロンプト
        self.jargon_extraction_prompt = ChatPromptTemplate.from_template(
            """あなたは専門用語の抽出エキスパートです。以下の質問から、専門用語、略語、固有名詞、技術用語を抽出してください。
一般的な単語は除外し、ドメイン固有の用語のみを抽出してください。

質問: {question}

抽出された専門用語（改行で区切って、最大{max_terms}個まで）:"""
        )
        self._jargon_extraction_chain = self.jargon_extraction_prompt | self.llm | StrOutputParser()
        
        # クエリ補強用のプロンプト
        self.query_augmentation_prompt = ChatPromptTemplate.from_template(
            """以下の質問と専門用語の定義を考慮して、より明確で検索しやすい質問に書き換えてください。
専門用語の定義を質問に自然に組み込み、曖昧さを排除してください。

元の質問: {original_question}

専門用語と定義:
{jargon_definitions}

補強された質問:"""
        )
        self._query_augmentation_chain = self.query_augmentation_prompt | self.llm | StrOutputParser()
        
        # 文書要約用のプロンプト（オフラインプロセス用）
        self.document_summary_prompt = ChatPromptTemplate.from_template(
            """以下の文書の内容を要約し、検索に適したメタデータを生成してください。

文書の内容:
{content}

以下の形式で出力してください:
要約: （200文字以内の要約）
キーワード: （カンマ区切りの主要キーワード、最大10個）
専門分野: （該当する専門分野）
重要度: （1-5の数値、5が最も重要）"""
        )
        self._doc_summary_chain = self.document_summary_prompt | self.llm | StrOutputParser()
    
    def extract_jargon_terms(self, question: str, config: Optional[RunnableConfig] = None) -> List[str]:
        """質問から専門用語を抽出"""
        try:
            response = self._jargon_extraction_chain.invoke({
                "question": question,
                "max_terms": self.config.max_jargon_terms_per_query
            }, config=config)
            
            # 抽出された用語をリストに変換
            terms = [term.strip() for term in response.split('\n') if term.strip()]
            return terms[:self.config.max_jargon_terms_per_query]
        except Exception as e:
            print(f"Error extracting jargon terms: {e}")
            return []
    
    def augment_query_with_jargon(self, question: str, jargon_terms: List[str], 
                                  config: Optional[RunnableConfig] = None) -> str:
        """専門用語の定義を使ってクエリを補強"""
        if not jargon_terms:
            return question
        
        # 専門用語辞書から定義を取得
        jargon_info = self.jargon_manager.lookup_terms(jargon_terms)
        
        if not jargon_info:
            return question
        
        # 定義情報を整形
        definitions_text = []
        for term, info in jargon_info.items():
            definition = info["definition"]
            domain = info.get("domain", "一般")
            definitions_text.append(f"- {term}: {definition} (分野: {domain})")
        
        if not definitions_text:
            return question
        
        try:
            augmented_query = self._query_augmentation_chain.invoke({
                "original_question": question,
                "jargon_definitions": "\n".join(definitions_text)
            }, config=config)
            return augmented_query
        except Exception as e:
            print(f"Error augmenting query: {e}")
            return question
    
    def process_document_with_metadata(self, doc: Document, config: Optional[RunnableConfig] = None) -> Document:
        """文書にメタデータと要約を追加（オフラインプロセス）"""
        if not self.config.enable_metadata_enrichment:
            return doc
        
        try:
            # 文書の要約とメタデータを生成
            response = self._doc_summary_chain.invoke({
                "content": doc.page_content[:2000]  # 最初の2000文字を使用
            }, config=config)
            
            # レスポンスをパース
            lines = response.split('\n')
            summary = ""
            keywords = []
            domain = "一般"
            importance = 3
            
            for line in lines:
                if line.startswith("要約:"):
                    summary = line.replace("要約:", "").strip()
                elif line.startswith("キーワード:"):
                    keywords = [k.strip() for k in line.replace("キーワード:", "").split(",")]
                elif line.startswith("専門分野:"):
                    domain = line.replace("専門分野:", "").strip()
                elif line.startswith("重要度:"):
                    try:
                        importance = int(line.replace("重要度:", "").strip())
                    except:
                        importance = 3
            
            # メタデータを更新
            enhanced_metadata = doc.metadata.copy() if doc.metadata else {}
            enhanced_metadata.update({
                "summary": summary,
                "keywords": keywords,
                "domain": domain,
                "importance": importance,
                "processed_with_golden_retriever": True
            })
            
            doc.metadata = enhanced_metadata
            return doc
        except Exception as e:
            print(f"Error processing document metadata: {e}")
            return doc
    
    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        """文書をチャンクに分割し、メタデータを付与"""
        # 基底クラスのチャンク化を実行
        chunks = super().chunk_documents(docs)
        
        if self.config.enable_metadata_enrichment:
            # 各チャンクにメタデータを付与
            enhanced_chunks = []
            for chunk in chunks:
                enhanced_chunk = self.process_document_with_metadata(chunk)
                enhanced_chunks.append(enhanced_chunk)
            return enhanced_chunks
        
        return chunks
    
    def query_golden_retriever(self, question: str, *, 
                              use_query_expansion: bool = False,
                              use_rag_fusion: bool = False,
                              config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """Golden-Retriever方式でクエリを実行"""
        openai_cb_result = {}
        golden_retriever_info = {
            "enabled": self.config.enable_jargon_extraction,
            "extracted_terms": [],
            "augmented_query": question,
            "jargon_definitions": {}
        }
        
        # Golden-Retriever処理
        if self.config.enable_jargon_extraction:
            # 1. 専門用語を抽出
            jargon_terms = self.extract_jargon_terms(question, config=config)
            golden_retriever_info["extracted_terms"] = jargon_terms
            
            if jargon_terms:
                # 2. 専門用語の定義を取得
                jargon_definitions = self.jargon_manager.lookup_terms(jargon_terms)
                golden_retriever_info["jargon_definitions"] = jargon_definitions
                
                # 3. クエリを補強
                augmented_query = self.augment_query_with_jargon(
                    question, jargon_terms, config=config
                )
                golden_retriever_info["augmented_query"] = augmented_query
                
                # 補強されたクエリを使用
                question = augmented_query
        
        # 基底クラスのquery メソッドを呼び出し
        with get_openai_callback() as cb:
            result = super().query(
                question,
                use_query_expansion=use_query_expansion,
                use_rag_fusion=use_rag_fusion,
                config=config
            )
            openai_cb_result = {"total_tokens": cb.total_tokens, "cost": cb.total_cost}
        
        # Golden-Retriever情報を結果に追加
        result["golden_retriever"] = golden_retriever_info
        result["usage"] = openai_cb_result
        
        return result
    
    def add_jargon_terms_from_documents(self, threshold: float = 0.8) -> int:
        """文書から専門用語を自動抽出して辞書に追加"""
        # 実装例：文書中の頻出する専門的な用語を抽出
        # ここでは簡単な実装を示す
        engine = create_engine(self.connection_string)
        added_count = 0
        
        try:
            with engine.connect() as conn:
                # キーワードメタデータから専門用語候補を取得
                result = conn.execute(text("""
                    SELECT metadata->>'keywords' as keywords
                    FROM document_chunks
                    WHERE collection_name = :collection
                    AND metadata->>'processed_with_golden_retriever' = 'true'
                    AND metadata->>'keywords' IS NOT NULL
                """), {"collection": self.config.collection_name})
                
                keyword_freq = {}
                for row in result:
                    if row.keywords:
                        keywords = json.loads(row.keywords) if isinstance(row.keywords, str) else row.keywords
                        for keyword in keywords:
                            keyword_freq[keyword] = keyword_freq.get(keyword, 0) + 1
                
                # 頻出キーワードを専門用語として追加
                for term, freq in keyword_freq.items():
                    if freq >= 3:  # 3回以上出現
                        # 簡単な定義を生成（実際にはもっと高度な処理が必要）
                        success = self.jargon_manager.add_term(
                            term=term,
                            definition=f"{term}に関する専門用語",
                            confidence_score=min(freq / 10, 1.0)
                        )
                        if success:
                            added_count += 1
                
        except Exception as e:
            print(f"Error extracting jargon from documents: {e}")
        
        return added_count

###############################################################################
# Helper Functions                                                            #
###############################################################################

def initialize_sample_jargon_dictionary(rag_system: GoldenRetrieverRAG):
    """サンプルの専門用語辞書を初期化"""
    sample_terms = [
        {
            "term": "RAG",
            "definition": "Retrieval-Augmented Generation - 検索により取得した情報を用いて回答を生成する手法",
            "domain": "AI/NLP",
            "aliases": ["Retrieval-Augmented Generation"],
            "related_terms": ["Vector Search", "Embeddings", "LLM"]
        },
        {
            "term": "Golden-Retriever",
            "definition": "専門用語辞書を用いてクエリを補強し、検索精度を向上させるRAG最適化手法",
            "domain": "AI/NLP",
            "related_terms": ["RAG", "Query Augmentation", "Jargon Dictionary"]
        },
        {
            "term": "Embeddings",
            "definition": "テキストや画像などのデータを固定長の数値ベクトルに変換したもの",
            "domain": "機械学習",
            "aliases": ["埋め込み", "エンベディング"],
            "related_terms": ["Vector", "Semantic Search"]
        },
        {
            "term": "LLM",
            "definition": "Large Language Model - 大規模言語モデル。大量のテキストデータで学習した自然言語処理モデル",
            "domain": "AI/NLP",
            "aliases": ["Large Language Model", "大規模言語モデル"],
            "related_terms": ["GPT", "Transformer", "Fine-tuning"]
        },
        {
            "term": "Vector Search",
            "definition": "ベクトル空間における類似度を基に検索を行う手法",
            "domain": "情報検索",
            "aliases": ["ベクトル検索", "Semantic Search"],
            "related_terms": ["Embeddings", "Cosine Similarity", "ANN"]
        }
    ]
    
    added = 0
    for term_data in sample_terms:
        success = rag_system.jargon_manager.add_term(**term_data)
        if success:
            added += 1
    
    print(f"サンプル専門用語を{added}個追加しました。")
    return added

# 使用例
if __name__ == "__main__":
    # 設定
    config = GoldenRetrieverConfig(
        enable_jargon_extraction=True,
        enable_doc_summarization=True,
        enable_metadata_enrichment=True
    )
    
    # システム初期化
    rag = GoldenRetrieverRAG(config)
    
    # サンプル専門用語を追加
    initialize_sample_jargon_dictionary(rag)
    
    # クエリ実行例
    question = "RAGシステムでEmbeddingsを使ってVector Searchを行う方法は？"
    result = rag.query_golden_retriever(question, use_rag_fusion=True)
    
    print("=== Golden-Retriever RAG Results ===")
    print(f"元の質問: {question}")
    print(f"抽出された専門用語: {result['golden_retriever']['extracted_terms']}")
    print(f"補強されたクエリ: {result['golden_retriever']['augmented_query']}")
    print(f"回答: {result['answer']}")