"""
rag_system_enhanced.py
======================
This file acts as a facade for the RAG system, assembling components
from the `rag` subdirectory into a cohesive `RAGSystem` class.
"""
from __future__ import annotations

import os
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


try:
    import psycopg
    _PG_DIALECT = "psycopg"
except ModuleNotFoundError:
    _PG_DIALECT = "psycopg2"

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import PGVector, DistanceStrategy
from langchain_core.runnables import RunnableConfig
from langchain_community.callbacks.manager import get_openai_callback

# --- Refactored Module Imports ---
from rag.config import Config
from rag.text_processor import JapaneseTextProcessor
from rag.jargon import JargonDictionaryManager
from rag.retriever import JapaneseHybridRetriever
from rag.ingestion import IngestionHandler
from rag.sql_handler import SQLHandler
from rag.chains import create_chains, create_retrieval_chain, create_full_rag_chain

load_dotenv()

def format_docs(docs: List[Any]) -> str:
    """Helper function to format documents for context."""
    if not docs:
        return "(コンテキスト無し)"
    return "\n\n".join([f"[ソース {i+1} ChunkID: {d.metadata.get('chunk_id', 'N/A')}]\n{d.page_content}" for i, d in enumerate(docs)])

class RAGSystem:
    def __init__(self, cfg: Config):
        self.config = cfg
        self.text_processor = JapaneseTextProcessor()
        self.connection_string = f"postgresql+{_PG_DIALECT}://{cfg.db_user}:{cfg.db_password}@{cfg.db_host}:{cfg.db_port}/{cfg.db_name}"
        
        self._init_llms_and_embeddings()
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

        self.jargon_manager = JargonDictionaryManager(self.connection_string, cfg.jargon_table_name)
        self.ingestion_handler = IngestionHandler(cfg, self.vector_store, self.text_processor, self.connection_string)
        self.sql_handler = SQLHandler(cfg, self.llm, self.connection_string)

        # Create the modular chains
        self.retrieval_chain = create_retrieval_chain(self.llm, self.retriever, self.jargon_manager, self.config)
        self.rag_chain = create_full_rag_chain(self.retrieval_chain, self.llm)

        # Create the remaining chains (mostly for SQL and synthesis)
        self.chains = create_chains(self.llm, cfg.max_sql_results)
        self.sql_handler.multi_table_sql_chain = self.chains["multi_table_sql"]
        self.sql_handler.sql_answer_generation_chain = self.chains["sql_answer_generation"]

    def _init_llms_and_embeddings(self):
        cfg = self.config
        if not all([cfg.azure_openai_api_key, cfg.azure_openai_endpoint, cfg.azure_openai_chat_deployment_name, cfg.azure_openai_embedding_deployment_name]):
            raise ValueError("Azure OpenAI API credentials are not fully configured.")
        
        self.llm = AzureChatOpenAI(
            azure_endpoint=cfg.azure_openai_endpoint, api_key=cfg.azure_openai_api_key, 
            api_version=cfg.azure_openai_api_version, azure_deployment=cfg.azure_openai_chat_deployment_name, temperature=0.7
        )
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=cfg.azure_openai_endpoint, api_key=cfg.azure_openai_api_key, 
            api_version=cfg.azure_openai_api_version, azure_deployment=cfg.azure_openai_embedding_deployment_name
        )
        print("RAGSystem initialized with Azure OpenAI.")

    def _init_db(self):
        engine = create_engine(self.connection_string)
        with engine.connect() as conn:
            conn.execute(text("CREATE TABLE IF NOT EXISTS document_chunks (chunk_id TEXT PRIMARY KEY, collection_name TEXT, document_id TEXT, content TEXT, tokenized_content TEXT, metadata JSONB, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_doc_chunks_coll_doc ON document_chunks(collection_name, document_id);"))
            conn.commit()

    # --- Method Delegation ---
    def ingest_documents(self, paths: List[str]):
        return self.ingestion_handler.ingest_documents(paths)

    def delete_document_by_id(self, doc_id: str) -> tuple[bool, str]:
        return self.ingestion_handler.delete_document_by_id(doc_id)

    def create_table_from_file(self, file_path: str, table_name: Optional[str] = None) -> tuple[bool, str, str]:
        return self.sql_handler.create_table_from_file(file_path, table_name)

    def get_data_tables(self) -> List[Dict[str, Any]]:
        return self.sql_handler.get_data_tables()

    def delete_data_table(self, table_name: str) -> tuple[bool, str]:
        return self.sql_handler.delete_data_table(table_name)

    def get_chunks_by_document_id(self, document_id: str):
        return self.sql_handler.get_chunks_by_document_id(document_id)

    # --- Core Query Logic ---
    def query(self, question: str, *, use_query_expansion: bool = False, use_rag_fusion: bool = False, use_jargon_augmentation: bool = True, use_reranking: bool = True, search_type: str = "ハイブリッド検索", config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """Executes the main RAG chain for a standard RAG query."""
        with get_openai_callback() as cb:
            chain_input = {
                "question": question, "use_query_expansion": use_query_expansion,
                "use_rag_fusion": use_rag_fusion, "use_jargon_augmentation": use_jargon_augmentation,
                "use_reranking": use_reranking, "config": config, "search_type": search_type
            }
            result = self.rag_chain.invoke(chain_input, config=config)
            usage = {"total_tokens": cb.total_tokens, "cost": cb.total_cost}
        return {
            "answer": result.get("answer", "回答を生成できませんでした。"), "sources": result.get("documents", []),
            "question": question, "usage": usage, "query_expansion": {}, "reranking": {}, "golden_retriever": {}
        }

    def query_unified(self, question: str, **kwargs) -> Dict[str, Any]:
        """Executes RAG retrieval and SQL search, then synthesizes the results."""
        config = kwargs.get("config")
        
        # 1. Execute RAG Document Retrieval
        chain_input = {
            "question": question, "use_query_expansion": kwargs.get("use_query_expansion"),
            "use_rag_fusion": kwargs.get("use_rag_fusion"), "use_jargon_augmentation": kwargs.get("use_jargon_augmentation"),
            "use_reranking": kwargs.get("use_reranking"), "config": config, "search_type": kwargs.get("search_type")
        }
        with get_openai_callback() as cb_rag:
            # We call the full chain here to get a complete trace, even if we only use parts of the output.
            rag_results = self.rag_chain.invoke(chain_input, config=config)
        
        rag_context = format_docs(rag_results.get("documents", []))
        
        # 2. Execute Text-to-SQL Search
        sql_details = None
        sql_data_summary = "（利用可能なデータベース情報はありません）"
        tables = self.get_data_tables()
        cb_sql_total = 0
        cb_sql_cost = 0.0

        if self.config.enable_text_to_sql and tables:
            try:
                with get_openai_callback() as cb_sql:
                    schemas_info = "\n\n---\n\n".join([t['schema'] for t in tables if t.get('schema')])
                    generated_sql = self.chains["multi_table_sql"].invoke({"question": question, "schemas_info": schemas_info, "max_sql_results": self.config.max_sql_results}, config=config)
                    sql_details = self.sql_handler._execute_and_summarize_sql(question, self.sql_handler._extract_sql(generated_sql), config=config)
                    sql_data_summary = sql_details.get("natural_language_answer", "（SQLクエリは実行されましたが、要約を生成できませんでした）")
                cb_sql_total = cb_sql.total_tokens
                cb_sql_cost = cb_sql.total_cost
            except Exception as e:
                print(f"Text-to-SQL process failed: {e}")
                sql_data_summary = f"（SQL処理中にエラーが発生しました: {e}）"
        
        # 3. Synthesize Final Answer
        # If SQL search was not productive, just return the original RAG result.
        if not sql_details or "error" in sql_details:
             return {
                "answer": rag_results.get("answer", "回答が見つかりませんでした。"),
                "sources": rag_results.get("documents", []),
                "question": question,
                "usage": {"total_tokens": cb_rag.total_tokens, "cost": cb_rag.total_cost}
            }

        # Otherwise, synthesize both results
        with get_openai_callback() as cb_synth:
            final_answer = self.chains["synthesis"].invoke({"question": question, "rag_context": rag_context, "sql_data": sql_data_summary}, config=config)
        
        total_tokens = cb_rag.total_tokens + cb_sql_total + cb_synth.total_tokens
        total_cost = cb_rag.total_cost + cb_sql_cost + cb_synth.total_cost
        
        return {
            "answer": final_answer, "sources": rag_results.get("documents", []), "question": question,
            "query_type": "hybrid", "sql_details": sql_details, "usage": {"total_tokens": total_tokens, "cost": total_cost},
            "query_expansion": {}, "reranking": {}, "golden_retriever": {}
        }

    def extract_terms(self, input_dir: str | Path, output_json: str | Path) -> None:
        from scripts.term_extractor_embeding import run_pipeline as term_pipeline
        asyncio.run(term_pipeline(Path(input_dir), Path(output_json)))
        print(f"[TermExtractor] Extraction complete -> {output_json}")

__all__ = ["Config", "RAGSystem"]
