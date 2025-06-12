"""rag_system_japanese_enhanced.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Enhanced RAG System with Japanese Morphological Analysis support
Optimized for Amazon RDS PostgreSQL with Janome tokenizer

主な改善点:
- Janomeを使用した日本語形態素解析
- トークナイズされた日本語テキストの全文検索
- Amazon RDS互換のPostgreSQL設定
- 日本語・英語の自動判定とハイブリッド検索
"""
from __future__ import annotations

import os
import asyncio
import json
import re
import unicodedata
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from operator import itemgetter
import pandas as pd

from dotenv import load_dotenv
from sqlalchemy import create_engine, text, inspect as sqlalchemy_inspect

# Janome for Japanese morphological analysis
try:
    from janome.tokenizer import Tokenizer
    JANOME_AVAILABLE = True
except ImportError:
    print("Warning: Janome not installed. Japanese tokenization will be limited.")
    JANOME_AVAILABLE = False

# LangChain imports
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredFileLoader,
    Docx2txtLoader,
)

try:
    from langchain_community.document_loaders import TextractLoader
except ImportError:
    TextractLoader = None

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

try:
    import psycopg
    _PG_DIALECT = "psycopg"
except ModuleNotFoundError:
    try:
        import psycopg2
        _PG_DIALECT = "psycopg2"
    except ModuleNotFoundError:
        _PG_DIALECT = None

###############################################################################
# Japanese Text Processing Utilities                                          #
###############################################################################

class JapaneseTextProcessor:
    """日本語テキスト処理のユーティリティクラス"""
    
    def __init__(self):
        self.tokenizer = Tokenizer() if JANOME_AVAILABLE else None
        # 日本語ストップワード（必要に応じて拡張）
        self.stop_words = {
            'の', 'に', 'は', 'を', 'た', 'が', 'で', 'て', 'と', 'し', 'れ', 'さ',
            'ある', 'いる', 'も', 'する', 'から', 'な', 'こと', 'として', 'い', 'や',
            'れる', 'など', 'なっ', 'ない', 'この', 'ため', 'その', 'あっ', 'よう',
            'また', 'もの', 'という', 'あり', 'まで', 'られ', 'なる', 'へ', 'か',
            'だ', 'これ', 'によって', 'により', 'おり', 'より', 'による', 'ず', 'なり',
            'られる', 'において', 'ば', 'なかっ', 'なく', 'しかし', 'について', 'せ', 'だっ',
            'その後', 'できる', 'それ', 'う', 'ので', 'なお', 'のみ', 'でき', 'き',
            'つ', 'における', 'および', 'いう', 'さらに', 'でも', 'ら', 'たり', 'その他',
            'に関する', 'たち', 'ます', 'ん', 'なら', 'に対して', '特に', 'せる', '及び',
            'これら', 'とき', 'では', 'にて', 'ほか', 'ながら', 'うち', 'そして', 'とも',
            'ただし', 'かつて', 'それぞれ', 'または', 'お', 'ほど', 'ものの', 'に対する',
            'ほとんど', 'と共に', 'といった', 'です', 'ました', 'ません'
        }
    
    def is_japanese(self, text: str) -> bool:
        """テキストが日本語を含むかどうかを判定"""
        for char in text:
            name = unicodedata.name(char, '')
            if 'CJK' in name or 'HIRAGANA' in name or 'KATAKANA' in name:
                return True
        return False
    
    def tokenize(self, text: str, remove_stop_words: bool = True) -> List[str]:
        """日本語テキストをトークナイズ"""
        if not self.tokenizer or not self.is_japanese(text):
            # 日本語でない場合は空白で分割
            return text.split()
        
        tokens = []
        for token in self.tokenizer.tokenize(text):
            # 名詞、動詞、形容詞、形容動詞を抽出（必要に応じて調整）
            if token.part_of_speech.split(',')[0] in ['名詞', '動詞', '形容詞', '形容動詞']:
                base_form = token.base_form
                if base_form == '*':
                    base_form = token.surface
                
                if remove_stop_words and base_form in self.stop_words:
                    continue
                    
                tokens.append(base_form)
        
        return tokens
    
    def normalize_text(self, text: str) -> str:
        """テキストの正規化（全角・半角統一など）"""
        # NFKC正規化（全角英数字を半角に変換など）
        text = unicodedata.normalize('NFKC', text)
        # 連続する空白を単一の空白に
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

###############################################################################
# Config dataclass with Japanese support                                      #
###############################################################################

@dataclass
class Config:
    # Database settings (Amazon RDS compatible)
    db_host: str = os.getenv("DB_HOST", "localhost")
    db_port: str = os.getenv("DB_PORT", "5432")
    db_name: str = os.getenv("DB_NAME", "postgres")
    db_user: str = os.getenv("DB_USER", "postgres")
    db_password: str = os.getenv("DB_PASSWORD", "your-password")

    # OpenAI API settings
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    embedding_model_identifier: str = os.getenv("EMBEDDING_MODEL_IDENTIFIER", "text-embedding-3-small")
    llm_model_identifier: str = os.getenv("LLM_MODEL_IDENTIFIER", "gpt-4o-mini")

    # Azure OpenAI Service settings
    azure_openai_api_key: Optional[str] = os.getenv("AZURE_OPENAI_API_KEY")
    azure_openai_endpoint: Optional[str] = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_openai_api_version: Optional[str] = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    azure_openai_chat_deployment_name: Optional[str] = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
    azure_openai_embedding_deployment_name: Optional[str] = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

    # RAG and Search settings
    chunk_size: int = int(os.getenv("CHUNK_SIZE", 1000))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 200))
    vector_search_k: int = int(os.getenv("VECTOR_SEARCH_K", 10))
    keyword_search_k: int = int(os.getenv("KEYWORD_SEARCH_K", 10))
    final_k: int = int(os.getenv("FINAL_K", 5))
    collection_name: str = os.getenv("COLLECTION_NAME", "documents")
    
    # 日本語検索設定
    enable_japanese_search: bool = os.getenv("ENABLE_JAPANESE_SEARCH", "true").lower() == "true"
    japanese_min_token_length: int = int(os.getenv("JAPANESE_MIN_TOKEN_LENGTH", 2))
    
    # 言語設定（英語と日本語の両方をサポート）
    fts_language: str = os.getenv("FTS_LANGUAGE", "english")
    rrf_k_for_fusion: int = int(os.getenv("RRF_K_FOR_FUSION", 60))

    # Text-to-SQL settings
    enable_text_to_sql: bool = True 
    max_sql_results: int = int(os.getenv("MAX_SQL_RESULTS", 1000))
    max_sql_preview_rows_for_llm: int = int(os.getenv("MAX_SQL_PREVIEW_ROWS_FOR_LLM", 20))
    user_table_prefix: str = os.getenv("USER_TABLE_PREFIX", "data_")

###############################################################################
# Enhanced Hybrid Retriever with Japanese Support                             #
###############################################################################

class JapaneseHybridRetriever(BaseRetriever):
    vector_store: PGVector
    connection_string: str
    config_params: Config
    text_processor: JapaneseTextProcessor
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.text_processor = JapaneseTextProcessor()

    def _vector_search(self, q: str, config: Optional[RunnableConfig] = None) -> List[Tuple[Document, float]]:
        if not self.vector_store: 
            return []
        try: 
            return self.vector_store.similarity_search_with_score(q, k=self.config_params.vector_search_k)
        except Exception as exc: 
            print(f"[HybridRetriever] vector search error: {exc}")
            return []

    def _keyword_search(self, q: str, config: Optional[RunnableConfig] = None) -> List[Tuple[Document, float]]:
        """改良されたキーワード検索（日本語対応）"""
        engine = create_engine(self.connection_string)
        res: List[Tuple[Document, float]] = []
        
        # クエリの正規化とトークナイズ
        normalized_query = self.text_processor.normalize_text(q)
        is_japanese = self.text_processor.is_japanese(normalized_query)
        
        try:
            with engine.connect() as conn:
                if is_japanese and self.config_params.enable_japanese_search:
                    # 日本語検索の場合
                    tokens = self.text_processor.tokenize(normalized_query)
                    
                    if not tokens:
                        return []
                    
                    # トークナイズされたコンテンツから検索
                    # 各トークンに対してLIKE検索を実行（より柔軟な検索）
                    conditions = []
                    params = {}
                    for i, token in enumerate(tokens[:5]):  # 最大5トークンまで
                        if len(token) >= self.config_params.japanese_min_token_length:
                            conditions.append(f"(content LIKE :token{i} OR tokenized_content LIKE :token{i})")
                            params[f"token{i}"] = f"%{token}%"
                    
                    if not conditions:
                        return []
                    
                    where_clause = " AND ".join(conditions)
                    sql = f"""
                        SELECT chunk_id, content, metadata, 
                               (LENGTH(content) - LENGTH(REPLACE(LOWER(content), LOWER(:original_query), ''))) / LENGTH(:original_query) AS score
                        FROM document_chunks 
                        WHERE {where_clause}
                        AND collection_name = :collection_name
                        ORDER BY score DESC 
                        LIMIT :k;
                    """
                    
                    params.update({
                        "original_query": normalized_query,
                        "collection_name": self.config_params.collection_name,
                        "k": self.config_params.keyword_search_k
                    })
                    
                    db_result = conn.execute(text(sql), params)
                else:
                    # 英語または通常の全文検索
                    sql = f"""
                        SELECT chunk_id, content, metadata, 
                               ts_rank(to_tsvector('{self.config_params.fts_language}', content), 
                                      plainto_tsquery('{self.config_params.fts_language}', :q)) AS score 
                        FROM document_chunks 
                        WHERE to_tsvector('{self.config_params.fts_language}', content) @@ 
                              plainto_tsquery('{self.config_params.fts_language}', :q) 
                        AND collection_name = :collection_name 
                        ORDER BY score DESC 
                        LIMIT :k;
                    """
                    
                    db_result = conn.execute(text(sql), {
                        "q": normalized_query, 
                        "k": self.config_params.keyword_search_k, 
                        "collection_name": self.config_params.collection_name
                    })
                
                for row in db_result:
                    md = row.metadata if isinstance(row.metadata, dict) else json.loads(row.metadata or "{}")
                    res.append((Document(page_content=row.content, metadata=md), float(row.score)))
                    
        except Exception as exc:
            print(f"[HybridRetriever] keyword search error: {exc}")
            
        return res

    @staticmethod
    def _rrf_hybrid(rank: int, k: int = 60) -> float:
        return 1.0 / (k + rank)

    def _reciprocal_rank_fusion_hybrid(self, vres: List[Tuple[Document, float]], kres: List[Tuple[Document, float]]) -> List[Document]:
        score_map: Dict[str, Dict[str, Any]] = {}
        _id = lambda d: d.metadata.get("chunk_id", d.page_content[:100])
        
        for r, (d, _) in enumerate(vres, 1):
            doc_id_val = _id(d)
            score_map.setdefault(doc_id_val, {"doc": d, "s": 0.0})["s"] += self._rrf_hybrid(r)
            
        for r, (d, _) in enumerate(kres, 1):
            doc_id_val = _id(d)
            score_map.setdefault(doc_id_val, {"doc": d, "s": 0.0})["s"] += self._rrf_hybrid(r)
            
        ranked = sorted(score_map.values(), key=lambda x: x["s"], reverse=True)
        return [x["doc"] for x in ranked[:self.config_params.final_k]]

    def _get_relevant_documents(self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None, **kwargs: Any) -> List[Document]:
        config = kwargs.get("config")
        vres = self._vector_search(query, config=config)
        kres = self._keyword_search(query, config=config)
        return self._reciprocal_rank_fusion_hybrid(vres, kres)

    async def _aget_relevant_documents(self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None, **kwargs: Any) -> List[Document]:
        config = kwargs.get("config")
        vres = self._vector_search(query, config=config)
        kres = self._keyword_search(query, config=config)
        return self._reciprocal_rank_fusion_hybrid(vres, kres)

###############################################################################
# Enhanced RAG System with Japanese Support                                   #
###############################################################################

class RAGSystem:
    def __init__(self, cfg: Config):
        if _PG_DIALECT is None:
            raise RuntimeError("PostgreSQL driver not installed.")
        self.config = cfg
        self.text_processor = JapaneseTextProcessor()

        # Initialize LLM and Embeddings
        if cfg.azure_openai_api_key and cfg.azure_openai_endpoint and \
           cfg.azure_openai_chat_deployment_name and cfg.azure_openai_embedding_deployment_name:
            self.llm = AzureChatOpenAI(
                azure_endpoint=cfg.azure_openai_endpoint,
                api_key=cfg.azure_openai_api_key,
                api_version=cfg.azure_openai_api_version,
                azure_deployment=cfg.azure_openai_chat_deployment_name,
                temperature=0.7
            )
            self.embeddings = AzureOpenAIEmbeddings(
                azure_endpoint=cfg.azure_openai_endpoint,
                api_key=cfg.azure_openai_api_key,
                api_version=cfg.azure_openai_api_version,
                azure_deployment=cfg.azure_openai_embedding_deployment_name
            )
            print("RAGSystem initialized with Azure OpenAI.")
        elif cfg.openai_api_key:
            if not cfg.llm_model_identifier or not cfg.embedding_model_identifier:
                raise ValueError("OpenAI model identifiers are missing in Config.")
            self.llm = ChatOpenAI(
                openai_api_key=cfg.openai_api_key,
                model_name=cfg.llm_model_identifier,
                temperature=0.7
            )
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=cfg.openai_api_key,
                model=cfg.embedding_model_identifier
            )
            print("RAGSystem initialized with OpenAI.")
        else:
            raise ValueError("Azure OpenAI or OpenAI API credentials are not configured.")

        self.connection_string = self._conn_str()
        self._init_db()
        
        self.vector_store = PGVector(
            connection_string=self.connection_string,
            collection_name=cfg.collection_name,
            embedding_function=self.embeddings,
            use_jsonb=True,
            distance_strategy=DistanceStrategy.COSINE
        )
        
        self.retriever = JapaneseHybridRetriever(
            vector_store=self.vector_store,
            connection_string=self.connection_string,
            config_params=cfg,
            text_processor=self.text_processor
        )

        # プロンプトテンプレート
        self.base_rag_prompt = ChatPromptTemplate.from_template(
            """あなたは親切で知識豊富なアシスタントです。以下のコンテキストを参考に質問に答えてください。

コンテキスト:
{context}

質問: {question}

回答:"""
        )
        
        self.query_expansion_prompt = ChatPromptTemplate.from_template(
            """以下の質問に対して、より良い検索結果を得るために、関連する追加の検索クエリを3つ生成してください。
元の質問の意図を保ちながら、異なる表現や関連する概念を含めてください。

元の質問: {question}

追加クエリ（改行で区切って3つ）:"""
        )
        
        self._query_expansion_llm_chain = self.query_expansion_prompt | self.llm | StrOutputParser()

        # SQL関連のプロンプト（既存のものを継承）
        self.multi_table_text_to_sql_prompt = ChatPromptTemplate.from_template(
            """あなたはPostgreSQLエキスパートです。以下に提示される複数のテーブルスキーマの中から、ユーザーの質問に答えるために最も適切と思われるテーブルを選択し、必要であればそれらのテーブル間でJOINを適切に使用して、SQLクエリを生成してください。
SQLはPostgreSQL構文に準拠し、テーブル名やカラム名が日本語の場合はダブルクォーテーションで囲んでください。
最終的な結果セットが過度に大きくならないよう、適切にLIMIT句を使用してください（例: LIMIT {max_sql_results}）。

利用可能なテーブルのスキーマ情報一覧:
{schemas_info}

ユーザーの質問: {question}

SQLクエリのみを返してください:
```sql
SELECT ...
```
"""
        )
        self._multi_table_sql_chain = self.multi_table_text_to_sql_prompt | self.llm | StrOutputParser()

        self.single_table_text_to_sql_prompt = ChatPromptTemplate.from_template(
            """あなたはPostgreSQLエキスパートです。以下のテーブル情報を参考に、質問をSQLに変換してください。
SQLはPostgreSQL構文に準拠し、テーブル名やカラム名が日本語の場合はダブルクォーテーションで囲んでください。
最終的な結果セットが過度に大きくならないよう、適切にLIMIT句を使用してください（例: LIMIT {max_sql_results}）。

テーブル情報:
{schema_info}

質問: {question}

SQLクエリのみを返してください:
```sql
SELECT ...
```
"""
        )
        self._single_table_sql_chain = self.single_table_text_to_sql_prompt | self.llm | StrOutputParser()

        self.query_detection_prompt = ChatPromptTemplate.from_template(
            """この質問はSQL分析とRAG検索のどちらが適切ですか？

利用可能なデータテーブルの概要:
{tables_info}

ユーザーの質問: {question}

判断基準:
- SQLが適している場合: 具体的な数値データに基づく分析、集計、ランキング、フィルタリング、特定レコードの抽出など
- RAGが適している場合: ドキュメントの内容に関する要約、説明、概念理解、自由形式の質問

回答は「SQL」または「RAG」のいずれか一つのみを返してください。"""
        )
        self._detection_chain = self.query_detection_prompt | self.llm | StrOutputParser()

        self.sql_answer_generation_prompt = ChatPromptTemplate.from_template(
            """与えられた元の質問と、それに基づいて実行されたSQLクエリ、およびその実行結果を考慮して、ユーザーにとって分かりやすい言葉で回答を生成してください。

元の質問: {original_question}

実行されたSQLクエリ:
```sql
{sql_query}
```

SQL実行結果のプレビュー (最大 {max_preview_rows} 件表示):
{sql_results_preview_str}
(このプレビューは全 {total_row_count} 件中の一部です)

上記の情報を踏まえた、質問に対する回答:"""
        )
        self._sql_answer_generation_chain = self.sql_answer_generation_prompt | self.llm | StrOutputParser()

    def _conn_str(self) -> str:
        c = self.config
        return f"postgresql+{_PG_DIALECT}://{c.db_user}:{c.db_password}@{c.db_host}:{c.db_port}/{c.db_name}"

    def _init_db(self):
        """データベースの初期化（日本語検索用カラムを追加）"""
        engine = create_engine(self.connection_string)
        with engine.connect() as conn:
            # 既存のコード...
            
            # term_dictionary テーブルの作成
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS term_dictionary (
                    id SERIAL PRIMARY KEY,
                    term VARCHAR(255) UNIQUE NOT NULL,
                    synonyms TEXT[] DEFAULT '{}',
                    definition TEXT,
                    sources TEXT[] DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """))
            
            # term_dictionary のインデックス作成
            try:
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_term_dictionary_term ON term_dictionary(term);
                """))
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_term_dictionary_synonyms ON term_dictionary USING GIN(synonyms);
                """))
            except Exception as e:
                print(f"Note: Could not create term_dictionary indexes: {e}")
            
            # 更新時刻を自動更新するトリガー関数
            try:
                conn.execute(text("""
                    CREATE OR REPLACE FUNCTION update_updated_at_column()
                    RETURNS TRIGGER AS $$
                    BEGIN
                        NEW.updated_at = CURRENT_TIMESTAMP;
                        RETURN NEW;
                    END;
                    $$ language 'plpgsql';
                """))
                
                # トリガーの作成
                conn.execute(text("""
                    DROP TRIGGER IF EXISTS update_term_dictionary_updated_at ON term_dictionary;
                    CREATE TRIGGER update_term_dictionary_updated_at 
                        BEFORE UPDATE ON term_dictionary 
                        FOR EACH ROW 
                        EXECUTE FUNCTION update_updated_at_column();
                """))
            except Exception as e:
                print(f"Note: Could not create term_dictionary trigger: {e}")
            
            conn.commit()

    def _tokenize_content(self, content: str) -> str:
        """コンテンツをトークナイズして検索用の文字列を生成"""
        if not self.text_processor.tokenizer:
            return ""
        
        tokens = self.text_processor.tokenize(content)
        return " ".join(tokens)

    def _store_chunks_for_keyword_search(self, chunks: List[Document]):
        """チャンクをデータベースに保存（日本語トークナイズ付き）"""
        eng = create_engine(self.connection_string)
        sql = text("""
            INSERT INTO document_chunks(collection_name, document_id, chunk_id, content, tokenized_content, metadata, created_at) 
            VALUES(:coll_name, :doc_id, :cid, :cont, :tok_cont, :meta, CURRENT_TIMESTAMP) 
            ON CONFLICT(chunk_id) DO UPDATE SET 
                content = EXCLUDED.content,
                tokenized_content = EXCLUDED.tokenized_content,
                metadata = EXCLUDED.metadata,
                document_id = EXCLUDED.document_id,
                collection_name = EXCLUDED.collection_name,
                created_at = CURRENT_TIMESTAMP;
        """)
        
        try:
            with eng.connect() as conn, conn.begin():
                for c in chunks:
                    if not (isinstance(c.metadata, dict) and 
                            'chunk_id' in c.metadata and 
                            'document_id' in c.metadata):
                        print(f"Skipping chunk due to missing metadata: {c.page_content[:50]}...")
                        continue
                    
                    # コンテンツの正規化とトークナイズ
                    normalized_content = self.text_processor.normalize_text(c.page_content)
                    tokenized_content = self._tokenize_content(normalized_content) if self.config.enable_japanese_search else ""
                    
                    meta_json = None
                    try:
                        meta_json = json.dumps(c.metadata or {})
                    except TypeError as te:
                        print(f"Could not serialize metadata for chunk_id {c.metadata.get('chunk_id')}: {te}")
                        meta_json = json.dumps({})

                    conn.execute(sql, {
                        "coll_name": self.config.collection_name,
                        "doc_id": c.metadata["document_id"],
                        "cid": c.metadata["chunk_id"],
                        "cont": normalized_content,
                        "tok_cont": tokenized_content,
                        "meta": meta_json
                    })
        except Exception as e:
            print(f"Error storing chunks for keyword search: {type(e).__name__} - {e}")

    def _generate_expanded_queries(self, original_query: str, config: Optional[RunnableConfig] = None) -> List[str]:
        """クエリ拡張（日本語対応）"""
        try:
            # 元のクエリを正規化
            normalized_query = self.text_processor.normalize_text(original_query)
            
            # LLMでクエリ拡張
            expanded_str = self._query_expansion_llm_chain.invoke({"question": normalized_query}, config=config)
            additional_queries = [q.strip() for q in expanded_str.split('\n') if q.strip()][:3]
            
            # 日本語の場合、同義語や類義語も考慮
            if self.text_processor.is_japanese(normalized_query):
                tokens = self.text_processor.tokenize(normalized_query)
                # トークンの組み合わせで追加クエリを生成
                if len(tokens) > 1:
                    additional_queries.append(" ".join(tokens))
            
            return [normalized_query] + additional_queries
        except Exception as e:
            print(f"[Query Expansion LLM] Error: {e}")
            return [original_query]

    def _retrieve_for_one_query(self, query: str, config: Optional[RunnableConfig] = None) -> List[Document]:
        """単一クエリでの検索"""
        if not isinstance(query, str):
            print(f"[RetrieverOneQuery] Expected string query, got {type(query)}: {query}")
            if isinstance(query, dict) and "question" in query and isinstance(query["question"], str):
                query = query["question"]
            elif isinstance(query, dict) and any(isinstance(v, str) for v in query.values()):
                query = next(v for v in query.values() if isinstance(v, str))
            else:
                return []
        return self.retriever.invoke(query, config=config)

    def _retrieve_for_multiple_queries(self, queries: List[str], config: Optional[RunnableConfig] = None) -> List[List[Document]]:
        """複数クエリでの検索"""
        if not queries:
            return []
        tasks = {f"docs_for_query_{i}": RunnableLambda(self._retrieve_for_one_query) for i in range(len(queries))}
        if not tasks:
            return [[] for _ in queries]
        
        parallel_retriever = RunnableParallel(**tasks)
        input_dict_for_parallel = {f"docs_for_query_{i}": q_str for i, q_str in enumerate(queries)}
        results_dict = parallel_retriever.invoke(input_dict_for_parallel, config=config)
        
        ordered_results: List[List[Document]] = []
        for i in range(len(queries)):
            task_key = f"docs_for_query_{i}"
            ordered_results.append(results_dict.get(task_key, []))
        return ordered_results

    def _reciprocal_rank_fusion(self, list_of_document_lists: List[List[Document]]) -> List[Document]:
        """Reciprocal Rank Fusion"""
        fused_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}
        k_rrf = self.config.rrf_k_for_fusion
        
        for doc_list in list_of_document_lists:
            for rank, doc in enumerate(doc_list):
                chunk_id = doc.metadata.get("chunk_id")
                if not chunk_id:
                    continue
                if chunk_id not in doc_map:
                    doc_map[chunk_id] = doc
                rrf_score = 1.0 / (k_rrf + rank + 1)
                fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + rrf_score
        
        sorted_chunk_ids = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)
        return [doc_map[cid] for cid in sorted_chunk_ids][:self.config.final_k]

    def _combine_documents_simple(self, list_of_document_lists: List[List[Document]]) -> List[Document]:
        """単純なドキュメント結合"""
        all_docs: List[Document] = []
        seen_chunk_ids = set()
        
        for doc_list in list_of_document_lists:
            for doc in doc_list:
                chunk_id = doc.metadata.get('chunk_id', '')
                if chunk_id and chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk_id)
                    all_docs.append(doc)
        return all_docs[:self.config.final_k]

    def _get_answer_generation_chain(self) -> RunnableSequence:
        return self.base_rag_prompt | self.llm | StrOutputParser()

    def _build_rag_pipeline(
        self,
        retrieval_chain: RunnableSequence,
        expanded_info_updater: Optional[RunnableLambda] = None
    ) -> RunnableSequence:
        context_preparation = RunnablePassthrough.assign(
            context=itemgetter("final_sources") | RunnableLambda(format_docs)
        )
        answer_logic = {
            "answer": self._get_answer_generation_chain(),
            "sources": itemgetter("final_sources"),
            "expanded_info": itemgetter("expanded_info")
        }
        pipeline = retrieval_chain | context_preparation
        if expanded_info_updater:
            pipeline = pipeline | expanded_info_updater
        pipeline = pipeline | answer_logic
        return pipeline

    def query(self, question: str, *, use_query_expansion: bool = False, use_rag_fusion: bool = False, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """メインのクエリメソッド"""
        openai_cb_result = {}
        chain_input = {
            "question": question,
            "expanded_info": {"used": False, "queries": [question], "strategy": "Standard RAG"}
        }

        if use_rag_fusion:
            retrieval_chain_for_fusion = (
                RunnablePassthrough.assign(expanded_queries=itemgetter("question") | RunnableLambda(self._generate_expanded_queries))
                | RunnablePassthrough.assign(doc_lists=itemgetter("expanded_queries") | RunnableLambda(self._retrieve_for_multiple_queries))
                | RunnablePassthrough.assign(final_sources=itemgetter("doc_lists") | RunnableLambda(self._reciprocal_rank_fusion))
            )
            fusion_info_updater = RunnablePassthrough.assign(
                expanded_info=lambda x: {**x["expanded_info"], "queries": x["expanded_queries"], "strategy": "RAG-Fusion (RRF)", "used": True}
            )
            active_chain = self._build_rag_pipeline(retrieval_chain_for_fusion, fusion_info_updater)
        elif use_query_expansion:
            retrieval_chain_for_qe = (
                RunnablePassthrough.assign(expanded_queries=itemgetter("question") | RunnableLambda(self._generate_expanded_queries))
                | RunnablePassthrough.assign(doc_lists=itemgetter("expanded_queries") | RunnableLambda(self._retrieve_for_multiple_queries))
                | RunnablePassthrough.assign(final_sources=itemgetter("doc_lists") | RunnableLambda(self._combine_documents_simple))
            )
            qe_info_updater = RunnablePassthrough.assign(
                expanded_info=lambda x: {**x["expanded_info"], "queries": x["expanded_queries"], "strategy": "Query Expansion (Simple Combination)", "used": True}
            )
            active_chain = self._build_rag_pipeline(retrieval_chain_for_qe, qe_info_updater)
        else:
            retrieval_chain_standard = RunnablePassthrough.assign(
                final_sources=itemgetter("question") | self.retriever
            )
            active_chain = self._build_rag_pipeline(retrieval_chain_standard)

        with get_openai_callback() as cb:
            result = active_chain.invoke(chain_input, config=config)
            openai_cb_result = {"total_tokens": cb.total_tokens, "cost": cb.total_cost}

        answer_text = result.get("answer", "エラー: 回答を生成できませんでした。")
        final_sources = result.get("sources", [])
        final_expanded_info = result.get("expanded_info", chain_input["expanded_info"])

        sources_data = [{
            "excerpt": str(s.page_content)[:200] + ("…" if len(s.page_content) > 200 else ""),
            "full_content": str(s.page_content),
            "metadata": s.metadata or {}
        } for s in final_sources]
        
        return {
            "question": question,
            "answer": answer_text,
            "sources": sources_data,
            "usage": openai_cb_result,
            "query_expansion": final_expanded_info
        }

    # 以下、既存のSQL関連メソッドをそのまま継承
    def create_table_from_file(self, file_path: str, table_name: Optional[str] = None) -> tuple[bool, str, str]:
        """ファイルからテーブルを作成"""
        try:
            path = Path(file_path)
            if not path.exists():
                return False, f"ファイルが見つかりません: {file_path}", ""

            if not table_name:
                clean_stem = re.sub(r'[^a-zA-Z0-9_]', '_', path.stem.lower())
                table_name = f"{self.config.user_table_prefix}{clean_stem}"

            df: Optional[pd.DataFrame] = None
            if path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif path.suffix.lower() == '.csv':
                encodings = ['utf-8', 'shift_jis', 'cp932', 'latin1']
                for enc in encodings:
                    try:
                        # ヘッダー行の自動検出を改善
                        df = pd.read_csv(file_path, encoding=enc, header=0, skip_blank_lines=True)
                        print(f"CSVファイルを {enc} エンコーディングで読み込みました")
                        print(f"元のカラム名: {list(df.columns)}")
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        print(f"CSV読み込みエラー ({enc}): {e}")
                        continue
                if df is None:
                    return False, "CSVファイルのエンコーディングエラーまたは読み込みエラー", ""
            else:
                return False, f"サポートされていないファイル形式: {path.suffix}", ""

            if df is None or df.empty:
                return False, "ファイルが空です", ""

            # データ型の最適化
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_numeric(df[col], errors='raise')
                    except (ValueError, TypeError):
                        try:
                            df[col] = pd.to_datetime(df[col], errors='raise')
                        except (ValueError, TypeError):
                            pass
            
            # カラム名の正規化（日本語対応）
            normalized_columns = []
            for i, col in enumerate(df.columns):
                # 文字列に変換（NaNやNoneの場合に対応）
                col_str = str(col) if col is not None else ""
                
                # 空文字列や'Unnamed:'で始まる場合は汎用名を使用
                if not col_str or col_str.startswith('Unnamed:') or col_str.strip() == '':
                    normalized_columns.append(f'col_{i}')
                    continue
                
                # 空白を アンダースコアに変換
                normalized_col = re.sub(r'\s+', '_', col_str.strip())
                
                # PostgreSQLで問題となる文字のみを処理（日本語文字は保持）
                # ダブルクォートで囲むため、基本的にはそのまま使用可能
                # ただし、先頭が数字の場合は接頭辞を追加
                if normalized_col and normalized_col[0].isdigit():
                    normalized_col = f'col_{normalized_col}'
                
                # 空になった場合は汎用名を使用
                if not normalized_col:
                    normalized_col = f'col_{i}'
                
                normalized_columns.append(normalized_col)
            
            df.columns = normalized_columns

            engine = create_engine(self.connection_string)
            with engine.connect() as conn:
                conn.execute(text(f'DROP TABLE IF EXISTS public."{table_name}" CASCADE'))
                df.to_sql(table_name, conn, if_exists='replace', index=False, schema='public')
                conn.commit()

            print(f"正規化後のカラム名: {list(df.columns)}")
            schema_info = self._get_table_schema(table_name)
            return True, f"テーブル '{table_name}' が {len(df)} 行で作成されました。カラム名: {list(df.columns)}", schema_info
        except Exception as e:
            return False, f"テーブル作成エラー: {type(e).__name__} - {str(e)}", ""

    def _get_table_schema(self, table_name: str) -> str:
        """テーブルスキーマ取得"""
        try:
            engine = create_engine(self.connection_string)
            with engine.connect() as conn:
                stmt_cols = text("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = :table_name AND table_schema = 'public'
                    ORDER BY ordinal_position
                """)
                result_cols = conn.execute(stmt_cols, {"table_name": table_name})
                columns_info = result_cols.fetchall()

                if not columns_info:
                    return f"テーブル名: \"{table_name}\"\nカラム情報: (見つかりませんでした)"

                schema = f"テーブル名: \"{table_name}\" (スキーマ: public)\nカラム情報:\n"
                for col_name, col_type in columns_info:
                    schema += f"  - \"{col_name}\": {col_type}\n"

                try:
                    stmt_sample = text(f'SELECT * FROM public."{table_name}" LIMIT 3')
                    sample_result = conn.execute(stmt_sample)
                    rows = sample_result.fetchall()

                    if rows:
                        schema += f"\nサンプルデータ (上位{len(rows)}行):\n"
                        sample_column_names = list(sample_result.keys())
                        df_sample = pd.DataFrame(rows, columns=sample_column_names)
                        schema += df_sample.to_string(index=False, max_colwidth=50)
                    else:
                        schema += "\nサンプルデータ: (空のテーブル)\n"
                
                except Exception as e_sample:
                    schema += f"\nサンプルデータ取得エラー: {str(e_sample)}\n"
                
                return schema
        except Exception as e:
            return f"スキーマ情報取得エラー: {str(e)}"

    def _execute_and_summarize_sql(self, original_question: str, generated_sql_query: str, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """SQL実行と結果の要約"""
        sql_llm_usage = {"total_tokens": 0, "cost": 0.0}
        try:
            if not generated_sql_query or not generated_sql_query.strip():
                return {
                    "success": False,
                    "error": "SQLクエリが提供されませんでした",
                    "natural_language_answer": "SQLクエリが提供されませんでした",
                    "usage": sql_llm_usage,
                    "generated_sql": "",
                    "columns": [],
                    "row_count_fetched": 0,
                    "full_results_sample": [],
                    "results_preview": []
                }

            engine = create_engine(self.connection_string)
            with engine.connect() as conn:
                db_result = conn.execute(text(generated_sql_query))
                columns = list(db_result.keys())
                rows = db_result.fetchmany(self.config.max_sql_results)
            
            full_results_sample = [dict(zip(columns, row)) for row in rows]
            row_count_fetched = len(full_results_sample)
            preview_rows_for_llm = full_results_sample[:self.config.max_sql_preview_rows_for_llm]

            if preview_rows_for_llm:
                df_preview = pd.DataFrame(preview_rows_for_llm)
                sql_results_preview_str = df_preview.to_string(index=False, max_rows=self.config.max_sql_preview_rows_for_llm)
                if row_count_fetched > len(preview_rows_for_llm):
                    sql_results_preview_str += f"\n...他 {row_count_fetched - len(preview_rows_for_llm)} 件の結果があります"
            else:
                sql_results_preview_str = "結果はありませんでした"
            
            with get_openai_callback() as cb:
                answer_generation_payload = {
                    "original_question": original_question,
                    "sql_query": generated_sql_query,
                    "sql_results_preview_str": sql_results_preview_str,
                    "max_preview_rows": self.config.max_sql_preview_rows_for_llm,
                    "total_row_count": row_count_fetched
                }
                natural_language_answer = self._sql_answer_generation_chain.invoke(answer_generation_payload, config=config)
                sql_llm_usage = {"total_tokens": cb.total_tokens, "cost": cb.total_cost}

            return {
                "success": True,
                "question": original_question,
                "generated_sql": generated_sql_query,
                "natural_language_answer": natural_language_answer,
                "results_preview": preview_rows_for_llm,
                "full_results_sample": full_results_sample,
                "row_count_fetched": row_count_fetched,
                "columns": columns,
                "usage": sql_llm_usage
            }
                
        except Exception as e:
            error_message = f"SQL実行エラー: {type(e).__name__} - {str(e)}"
            return {
                "success": False,
                "error": str(e),
                "results_preview": [],
                "natural_language_answer": error_message,
                "usage": sql_llm_usage,
                "generated_sql": generated_sql_query,
                "columns": [],
                "row_count_fetched": 0,
                "full_results_sample": []
            }

    def execute_sql_query(self, question: str, table_name: str, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """単一テーブルに対するSQL実行"""
        try:
            schema_info = self._get_table_schema(table_name)
            if "エラー" in schema_info or "見つかりませんでした" in schema_info:
                return {
                    "success": False,
                    "error": f"テーブル {table_name} のスキーマ取得失敗",
                    "natural_language_answer": f"テーブル「{table_name}」のスキーマ情報を取得できませんでした",
                    "generated_sql": "",
                    "columns": [],
                    "row_count_fetched": 0,
                    "full_results_sample": [],
                    "results_preview": [],
                    "usage": {"total_tokens": 0, "cost": 0.0}
                }
            
            sql_generation_payload = {
                "question": question,
                "schema_info": schema_info,
                "max_sql_results": self.config.max_sql_results
            }
            sql_response_str = self._single_table_sql_chain.invoke(sql_generation_payload, config=config)
            generated_sql_for_single_table = self._extract_sql(sql_response_str)

            if not generated_sql_for_single_table:
                return {
                    "success": False,
                    "error": "SQL生成失敗",
                    "results_preview": [],
                    "natural_language_answer": "SQLクエリを生成できませんでした",
                    "generated_sql": "",
                    "columns": [],
                    "row_count_fetched": 0,
                    "full_results_sample": [],
                    "usage": {"total_tokens": 0, "cost": 0.0}
                }
            
            return self._execute_and_summarize_sql(
                original_question=question,
                generated_sql_query=generated_sql_for_single_table,
                config=config
            )
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "results_preview": [],
                "natural_language_answer": f"SQL分析エラー: {str(e)}",
                "generated_sql": "",
                "columns": [],
                "row_count_fetched": 0,
                "full_results_sample": [],
                "usage": {"total_tokens": 0, "cost": 0.0}
            }

    def _extract_sql(self, llm_output: str) -> str:
        """LLM出力からSQL抽出"""
        match = re.search(r"```sql\s*(.*?)\s*```", llm_output, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        lines = llm_output.strip().split('\n')
        for line in lines:
            if line.strip().upper().startswith("SELECT"):
                sql_candidate = "\n".join(lines[lines.index(line):])
                return sql_candidate.strip()
        return llm_output.strip()

    def query_unified(self, question: str, **kwargs) -> Dict[str, Any]:
        """統合クエリ処理"""
        # プレフィックスによる明示的なルーティング
        actual_question = question
        forced_mode = None
        
        if question.startswith("#SQL"):
            forced_mode = "sql"
            actual_question = question[4:].strip()  # "#SQL"を除去
        elif question.startswith("#RAG"):
            forced_mode = "rag"
            actual_question = question[4:].strip()  # "#RAG"を除去
        
        tables = self.get_data_tables()
        run_config = kwargs.get("config")
        overall_usage = {"total_tokens": 0, "cost": 0.0}

        # 強制的にRAGモードが指定された場合
        if forced_mode == "rag":
            rag_result = self.query(actual_question, **kwargs)
            rag_result["query_type"] = "rag"
            rag_result["info"] = "Forced RAG mode by #RAG prefix."
            return rag_result

        # 強制的にSQLモードが指定された場合、またはSQL判定の場合
        if forced_mode == "sql":
            # SQLモードが強制指定されているが、テーブルがない場合はエラー
            if not tables:
                return {
                    "query_type": "sql_error",
                    "question": actual_question,
                    "answer": "SQLモードが指定されましたが、利用可能なデータテーブルがありません。",
                    "sql_details": {"success": False, "error": "データテーブルなし"},
                    "sources": [],
                    "usage": overall_usage,
                    "query_expansion": {}
                }

        # SQLモードが強制指定されていない場合の従来の条件チェック
        if not forced_mode and (not self.config.enable_text_to_sql or not tables):
            rag_result = self.query(actual_question, **kwargs)
            rag_result["query_type"] = "rag"
            if not self.config.enable_text_to_sql:
                rag_result["info"] = "Text-to-SQL is disabled."
            elif not tables:
                rag_result["info"] = "No data tables found."
            return rag_result
        
        try:
            # SQLモードが強制指定されていない場合は自動判定を行う
            if not forced_mode:
                tables_info_for_detection = []
                for t in tables:
                    schema_summary = t.get('schema', '')
                    col_lines = [line.strip() for line in schema_summary.split('\n') if line.strip().startswith('- "')]
                    if col_lines:
                        extracted_cols = [cl.split(':')[0].replace('- "', '').replace('"', '').strip() for cl in col_lines]
                        schema_summary_for_detection = ", ".join(extracted_cols)
                    else:
                        schema_summary_for_detection = schema_summary[:150] + "..." if len(schema_summary) > 150 else schema_summary
                    tables_info_for_detection.append(f"- テーブル名 \"{t['table_name']}\" ({t.get('row_count', 'N/A')}行, カラム例: {schema_summary_for_detection})")

                tables_info_str_for_detection = "\n".join(tables_info_for_detection)
                if not tables_info_str_for_detection.strip():
                    rag_result = self.query(actual_question, **kwargs)
                    rag_result["query_type"] = "rag"
                    rag_result["info"] = "Could not generate table summary."
                    return rag_result

                detection_payload = {"question": actual_question, "tables_info": tables_info_str_for_detection}
                decision_chain_config = RunnableConfig(run_name="QueryTypeDetection", tags=["sql_rag_detection"])
                if run_config and run_config.get("callbacks"):
                    decision_chain_config["callbacks"] = run_config.get("callbacks")

                decision = self._detection_chain.invoke(detection_payload, config=decision_chain_config)
                
                # 自動判定でRAGが選択された場合
                if "SQL" not in decision.upper():
                    rag_result = self.query(actual_question, **kwargs)
                    rag_result["query_type"] = "rag"
                    return rag_result
            
            # SQL処理を実行（強制指定または自動判定でSQL選択）
            if forced_mode == "sql" or "SQL" in decision.upper():
                all_full_schemas_str = "\n\n---\n\n".join(
                    [t['schema'] for t in tables if t.get('schema')]
                )
                if not all_full_schemas_str.strip():
                    return {
                        "query_type": "sql_error",
                        "question": question,
                        "answer": "スキーマ情報を取得できませんでした",
                        "sql_details": {"success": False, "error": "スキーマ情報なし"},
                        "sources": [],
                        "usage": overall_usage,
                        "query_expansion": {}
                    }

                sql_generation_payload = {
                    "question": actual_question,
                    "schemas_info": all_full_schemas_str,
                    "max_sql_results": self.config.max_sql_results
                }
                sql_gen_config = RunnableConfig(run_name="MultiTableSQLGeneration", tags=["text_to_sql"])
                if run_config and run_config.get("callbacks"):
                    sql_gen_config["callbacks"] = run_config.get("callbacks")
                
                generated_sql_from_llm = self._multi_table_sql_chain.invoke(sql_generation_payload, config=sql_gen_config)
                actual_generated_sql = self._extract_sql(generated_sql_from_llm)

                if not actual_generated_sql:
                    return {
                        "query_type": "sql_error",
                        "question": actual_question,
                        "answer": "SQLクエリを生成できませんでした",
                        "sql_details": {"success": False, "error": "SQL生成失敗", "generated_sql": generated_sql_from_llm},
                        "sources": [],
                        "usage": overall_usage,
                        "query_expansion": {}
                    }
                
                sql_execution_details = self._execute_and_summarize_sql(
                    original_question=actual_question,
                    generated_sql_query=actual_generated_sql,
                    config=run_config
                )
                
                if sql_execution_details.get("usage"):
                    overall_usage["total_tokens"] += sql_execution_details["usage"]["total_tokens"]
                    overall_usage["cost"] += sql_execution_details["usage"]["cost"]

                return {
                    "query_type": "sql",
                    "question": question,
                    "answer": sql_execution_details.get("natural_language_answer", "回答生成失敗"),
                    "sql_details": sql_execution_details,
                    "sources": [],
                    "usage": overall_usage,
                    "query_expansion": {}
                }
            else:
                rag_result = self.query(question, **kwargs)
                rag_result["query_type"] = "rag"
                return rag_result
                
        except Exception as e:
            print(f"[Query Unified] Error: {type(e).__name__} - {e}")
            rag_result = self.query(question, **kwargs)
            rag_result["query_type"] = "rag_fallback_error"
            rag_result["error_info_fallback"] = f"エラー: {str(e)}"
            return rag_result

    def get_data_tables(self) -> List[Dict[str, Any]]:
        """データテーブル一覧取得"""
        tables_data: List[Dict[str, Any]] = []
        engine = create_engine(self.connection_string)
        try:
            with engine.connect() as conn:
                stmt_tables = text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                """)
                result_tables = conn.execute(stmt_tables)
                
                all_public_tables = [row[0] for row in result_tables if row and row[0]]
                
                user_tables = [
                    tbl_name for tbl_name in all_public_tables 
                    if tbl_name.startswith(self.config.user_table_prefix)
                ]

                for table_name in user_tables:
                    try:
                        stmt_count = text(f'SELECT COUNT(*) FROM public."{table_name}"')
                        count_result = conn.execute(stmt_count).scalar_one_or_none()
                        row_count = count_result if count_result is not None else 0

                        schema_info = self._get_table_schema(table_name)
                        
                        tables_data.append({
                            "table_name": table_name,
                            "row_count": row_count,
                            "schema": schema_info
                        })
                    except Exception as e:
                        print(f"テーブル '{table_name}' の詳細取得エラー: {e}")
                        continue
            return tables_data
        except Exception as e:
            print(f"データテーブル一覧取得エラー: {e}")
            return []

    def delete_data_table(self, table_name: str) -> tuple[bool, str]:
        """データテーブル削除"""
        if not table_name or not table_name.startswith(self.config.user_table_prefix):
            return False, f"無効なテーブル名: {table_name}"

        engine = create_engine(self.connection_string)
        try:
            with engine.connect() as conn:
                conn.execute(text(f'DROP TABLE IF EXISTS public."{table_name}" CASCADE'))
                conn.commit()
            return True, f"テーブル '{table_name}' は正常に削除されました。"
        except Exception as e:
            print(f"テーブル '{table_name}' の削除中にエラー: {type(e).__name__} - {str(e)}")
            return False, f"テーブル削除エラー: {str(e)}"

    def load_documents(self, paths: List[str]) -> List[Document]:
        """ドキュメントのロード"""
        docs: List[Document] = []
        for p_str in paths:
            path = Path(p_str)
            if not path.exists():
                print(f"File not found: {p_str}")
                continue
            suf = path.suffix.lower()
            try:
                if suf == ".pdf":
                    docs.extend(PyPDFLoader(str(path)).load())
                elif suf in {".txt", ".md"}:
                    docs.extend(TextLoader(str(path), encoding="utf-8").load())
                elif suf == ".docx":
                    try:
                        docs.extend(UnstructuredFileLoader(str(path), mode="single", strategy="fast").load())
                    except Exception:
                        docs.extend(Docx2txtLoader(str(path)).load())
                elif suf == ".doc" and TextractLoader:
                    try:
                        docs.extend(TextractLoader(str(path)).load())
                    except Exception as te:
                        print(f"TextractLoader error for {p_str}: {te}")
            except Exception as e:
                print(f"Error loading {p_str}: {type(e).__name__} - {e}")
        return docs

    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        """ドキュメントのチャンク分割（日本語対応）"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        out: List[Document] = []
        
        for i, d in enumerate(docs):
            src = d.metadata.get("source", f"doc_source_{i}")
            doc_id = Path(src).name
            
            try:
                # テキストの正規化
                normalized_content = self.text_processor.normalize_text(d.page_content)
                d.page_content = normalized_content
                
                doc_splits = text_splitter.split_documents([d])
                for j, c in enumerate(doc_splits):
                    new_metadata = d.metadata.copy() if d.metadata else {}
                    new_metadata.update(c.metadata if c.metadata else {})
                    new_metadata.update({
                        "chunk_id": f"{doc_id}_{i}_{j}",
                        "document_id": doc_id,
                        "original_document_source": src,
                        "collection_name": self.config.collection_name
                    })
                    c.metadata = new_metadata
                    out.append(c)
            except Exception as e:
                print(f"Error splitting document {src}: {type(e).__name__} - {e}")
        return out

    def ingest_documents(self, paths: List[str]):
        """ドキュメントの取り込み"""
        docs = self.load_documents(paths)
        if not docs:
            print("No documents loaded for ingestion.")
            return
        
        chunks = self.chunk_documents(docs)
        if not chunks:
            print("No chunks created from documents.")
            return
        
        valid_chunks = [
            c for c in chunks 
            if isinstance(c, Document) and 
               isinstance(c.metadata, dict) and 
               'chunk_id' in c.metadata and 
               c.page_content and 
               c.page_content.strip()
        ]
        
        if not valid_chunks:
            print("No valid chunks to ingest after validation.")
            return
        
        chunk_ids_for_vectorstore = [c.metadata['chunk_id'] for c in valid_chunks]
        
        try:
            self.vector_store.add_documents(valid_chunks, ids=chunk_ids_for_vectorstore)
            self._store_chunks_for_keyword_search(valid_chunks)
            print(f"Successfully ingested {len(valid_chunks)} chunks from {len(paths)} file(s).")
        except Exception as e:
            print(f"Error during ingestion: {type(e).__name__} - {e}")

    def delete_document_by_id(self, document_id_to_delete: str) -> tuple[bool, str]:
        """ドキュメントIDによる削除"""
        if not document_id_to_delete:
            return False, "ドキュメントIDは空にできません。"
        
        engine = create_engine(self.connection_string)
        chunk_ids_to_delete: List[str] = []
        deleted_rows_table = 0
        
        try:
            with engine.connect() as conn, conn.begin():
                res_proxy = conn.execute(
                    text("SELECT chunk_id FROM document_chunks WHERE document_id = :doc_id AND collection_name = :coll"),
                    {"doc_id": document_id_to_delete, "coll": self.config.collection_name}
                )
                chunk_ids_to_delete = [row[0] for row in res_proxy if row and row[0]]

                if not chunk_ids_to_delete:
                    return True, f"ドキュメントID '{document_id_to_delete}' に該当するチャンクは見つかりませんでした。"
                
                del_res = conn.execute(
                    text("DELETE FROM document_chunks WHERE document_id = :doc_id AND collection_name = :coll"),
                    {"doc_id": document_id_to_delete, "coll": self.config.collection_name}
                )
                deleted_rows_table = del_res.rowcount
                
                if self.vector_store and hasattr(self.vector_store, 'delete') and chunk_ids_to_delete:
                    try:
                        self.vector_store.delete(ids=chunk_ids_to_delete)
                    except Exception as e_vec_del:
                        print(f"ベクトルストア削除エラー: {e_vec_del}")
                        return False, f"データベースから削除しましたが、ベクトルストア削除中にエラー: {e_vec_del}"

            return True, f"ドキュメントID '{document_id_to_delete}' の {deleted_rows_table} 個のチャンクを削除しました。"
        except Exception as e:
            return False, f"削除中にエラー: {type(e).__name__} - {e}"


    # ---------------------------------------------------------------------
    #  Term-Extractor Embedding 連携
    # ---------------------------------------------------------------------
    def extract_terms(self, input_dir: str | Path, output_json: str | Path) -> None:
        """指定フォルダを走査して専門用語辞書を生成し、PostgreSQL+pgvector に保存
        Parameters
        ----------
        input_dir : str | Path
            抽出対象ドキュメントフォルダ
        output_json : str | Path
            生成した辞書を保存する JSON のパス
        """
        from term_extractor_embeding import run_pipeline as _term_run_pipeline
        input_path = Path(input_dir)
        output_path = Path(output_json)
        # term_extractor_embeding は asyncio ベース
        asyncio.run(_term_run_pipeline(input_path, output_path))
        print(f"[TermExtractor] 用語抽出完了 → {output_path}")
def format_docs(docs: List[Document]) -> str:
    """ドキュメントのフォーマット"""
    if not docs:
        return "(コンテキスト無し)"
    return "\n\n".join([f"[ソース {i+1} ChunkID: {d.metadata.get('chunk_id', 'N/A')}]\n{d.page_content}" for i, d in enumerate(docs)])


__all__ = ["Config", "RAGSystem", "JapaneseTextProcessor"]
