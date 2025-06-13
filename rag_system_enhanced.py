"""
rag_system_enhanced.py
======================
This file acts as a facade for the RAG system, assembling components
from the `rag` subdirectory into a cohesive `RAGSystem` class.
"""
from __future__ import annotations

import os
import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from operator import itemgetter

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

try:
    import psycopg
    _PG_DIALECT = "psycopg"
except ModuleNotFoundError:
    _PG_DIALECT = "psycopg2"

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import PGVector, DistanceStrategy
from langchain_core.runnables import RunnableConfig, RunnablePassthrough, RunnableLambda
from langchain_community.callbacks.manager import get_openai_callback

# --- Refactored Module Imports ---
from rag.config import Config
from rag.text_processor import JapaneseTextProcessor
from rag.jargon import JargonDictionaryManager
from rag.retriever import JapaneseHybridRetriever
from rag.ingestion import IngestionHandler
from rag.sql_handler import SQLHandler
from rag.chains import create_chains

load_dotenv()

def format_docs(docs: List[Document]) -> str:
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

        self.chains = create_chains(self.llm, cfg.max_sql_results)
        self.sql_handler.multi_table_sql_chain = self.chains["multi_table_sql"]
        self.sql_handler.sql_answer_generation_chain = self.chains["sql_answer_generation"]

    def _init_llms_and_embeddings(self):
        cfg = self.config
        if not all([cfg.azure_openai_api_key, cfg.azure_openai_endpoint, cfg.azure_openai_chat_deployment_name, cfg.azure_openai_embedding_deployment_name]):
            raise ValueError("Azure OpenAI API credentials are not fully configured. Please provide all required Azure settings.")
        
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

    def _init_db(self):
        engine = create_engine(self.connection_string)
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    chunk_id TEXT PRIMARY KEY, collection_name TEXT, document_id TEXT,
                    content TEXT, tokenized_content TEXT, metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """))
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
    def query(self, question: str, *, use_query_expansion: bool = False, use_rag_fusion: bool = False, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        original_question = question
        retrieval_query = question
        golden_retriever_info = {"enabled": self.config.enable_jargon_extraction, "augmented_query": question}

        # 1. Augment query if jargon extraction is enabled
        if self.config.enable_jargon_extraction:
            jargon_terms = self.chains["jargon_extraction"].invoke({"question": original_question, "max_terms": self.config.max_jargon_terms_per_query}, config=config).split('\n')
            jargon_terms = [t.strip() for t in jargon_terms if t.strip()]
            if jargon_terms:
                jargon_defs = self.jargon_manager.lookup_terms(jargon_terms)
                if jargon_defs:
                    defs_text = "\n".join([f"- {term}: {info['definition']}" for term, info in jargon_defs.items()])
                    retrieval_query = self.chains["query_augmentation"].invoke({"original_question": original_question, "jargon_definitions": defs_text}, config=config)
                golden_retriever_info.update({"extracted_terms": jargon_terms, "jargon_definitions": jargon_defs, "augmented_query": retrieval_query})
        
        # 2. Retrieve documents based on the mode
        expanded_info = {"used": False, "queries": [retrieval_query], "strategy": "Standard"}
        final_sources = []

        if use_rag_fusion or use_query_expansion:
            expanded_queries = self.chains["query_expansion"].invoke({"question": retrieval_query}, config=config).split('\n')
            expanded_queries = [q.strip() for q in expanded_queries if q.strip()]
            
            # Use a map to retrieve for all queries
            doc_lists = self.retriever.batch(expanded_queries, config)

            if use_rag_fusion:
                final_sources = self._reciprocal_rank_fusion(doc_lists)
                expanded_info.update({"used": True, "queries": expanded_queries, "strategy": "RAG-Fusion"})
            else: # Query Expansion
                final_sources = self._combine_documents_simple(doc_lists)
                expanded_info.update({"used": True, "queries": expanded_queries, "strategy": "Query Expansion"})
        else: # Standard retrieval
            final_sources = self.retriever.invoke(retrieval_query, config=config)

        # 3. Rerank documents if enabled
        reranked_info = {"used": False, "original_order": [d.metadata.get('chunk_id') for d in final_sources]}
        if self.config.enable_reranking and final_sources:
            docs_for_rerank = [f"ドキュメント {i}:\n{doc.page_content}" for i, doc in enumerate(final_sources)]
            rerank_input = {
                "question": original_question,
                "documents": "\n\n---\n\n".join(docs_for_rerank)
            }
            try:
                reranked_indices_str = self.chains["reranking"].invoke(rerank_input, config=config)
                reranked_indices = [int(i.strip()) for i in reranked_indices_str.split(',') if i.strip().isdigit()]
                
                # 順序を入れ替え
                final_sources = [final_sources[i] for i in reranked_indices if i < len(final_sources)]
                reranked_info.update({"used": True, "new_order": [d.metadata.get('chunk_id') for d in final_sources]})
            except Exception as e:
                print(f"Reranking failed: {e}")
                reranked_info.update({"used": False, "error": str(e)})
        
        # 4. Generate answer based on retrieved documents
        with get_openai_callback() as cb:
            chain_output = self.chains["answer_generation"].invoke({
                "context": final_sources,
                "question": original_question
            }, config=config)
            usage = {"total_tokens": cb.total_tokens, "cost": cb.total_cost}
        
        answer = chain_output

        # 5. Assemble final response
        return {
            "answer": answer,
            "sources": final_sources,
            "query_expansion": expanded_info,
            "reranking": reranked_info,
            "question": original_question,
            "usage": usage,
            "golden_retriever": golden_retriever_info
        }

    def query_unified(self, question: str, **kwargs) -> Dict[str, Any]:
        forced_mode, actual_question = (("sql", question[4:].strip()) if question.startswith("#SQL") else
                                        ("rag", question[4:].strip()) if question.startswith("#RAG") else
                                        (None, question))
        
        if forced_mode == "rag":
            return {**self.query(actual_question, **kwargs), "query_type": "rag"}

        tables = self.get_data_tables()
        if not self.config.enable_text_to_sql or not tables:
            return {**self.query(actual_question, **kwargs), "query_type": "rag"}

        if forced_mode == "sql":
            decision = "SQL"
        else:
            tables_info = "\n".join([f"- {t['table_name']}: {t.get('row_count', 'N/A')} rows" for t in tables])
            decision = self.chains["query_detection"].invoke({"question": actual_question, "tables_info": tables_info}, config=kwargs.get("config"))

        if "SQL" in decision.upper():
            schemas_info = "\n\n---\n\n".join([t['schema'] for t in tables if t.get('schema')])
            generated_sql = self.chains["multi_table_sql"].invoke({"question": actual_question, "schemas_info": schemas_info, "max_sql_results": self.config.max_sql_results}, config=kwargs.get("config"))
            sql_details = self.sql_handler._execute_and_summarize_sql(actual_question, self.sql_handler._extract_sql(generated_sql), config=kwargs.get("config"))
            return {"query_type": "sql", "question": question, "answer": sql_details.get("natural_language_answer", "Error"), "sql_details": sql_details, "sources": []}
        else:
            return {**self.query(actual_question, **kwargs), "query_type": "rag"}

    def _combine_documents_simple(self, list_of_document_lists: List[List[Any]]) -> List[Document]:
        """A simple combination of documents, preserving order and removing duplicates."""
        all_docs: List[Document] = []
        seen_chunk_ids = set()
        for doc_list in list_of_document_lists:
            for doc in doc_list:
                chunk_id = doc.metadata.get('chunk_id', '')
                if chunk_id and chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk_id)
                    all_docs.append(doc)
        return all_docs[:self.config.final_k]

    def _reciprocal_rank_fusion(self, list_of_document_lists: List[List[Any]]) -> List[Document]:
        fused_scores, doc_map = {}, {}
        for doc_list in list_of_document_lists:
            for rank, doc in enumerate(doc_list):
                doc_id = doc.metadata.get("chunk_id")
                if not doc_id: continue
                doc_map[doc_id] = doc
                fused_scores[doc_id] = fused_scores.get(doc_id, 0) + (1.0 / (self.config.rrf_k_for_fusion + rank + 1))
        return [doc_map[cid] for cid in sorted(fused_scores, key=fused_scores.get, reverse=True)][:self.config.final_k]

    def extract_terms(self, input_dir: str | Path, output_json: str | Path) -> None:
        from scripts.term_extractor_embeding import run_pipeline as term_pipeline
        asyncio.run(term_pipeline(Path(input_dir), Path(output_json)))
        print(f"[TermExtractor] Extraction complete -> {output_json}")

__all__ = ["Config", "RAGSystem"]
