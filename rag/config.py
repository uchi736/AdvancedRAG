import os
from dataclasses import dataclass, field
from typing import Optional

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
    enable_parent_child_chunking: bool = os.getenv("ENABLE_PARENT_CHILD_CHUNKING", "false").lower() == "true"
    parent_chunk_size: int = int(os.getenv("PARENT_CHUNK_SIZE", 2000))
    parent_chunk_overlap: int = int(os.getenv("PARENT_CHUNK_OVERLAP", 400))
    child_chunk_size: int = int(os.getenv("CHILD_CHUNK_SIZE", 400))
    child_chunk_overlap: int = int(os.getenv("CHILD_CHUNK_OVERLAP", 100))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", 1000)) # Kept for fallback
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 200)) # Kept for fallback
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

    # Golden-Retriever settings
    enable_jargon_extraction: bool = os.getenv("ENABLE_JARGON_EXTRACTION", "true").lower() == "true"
    jargon_table_name: str = os.getenv("JARGON_TABLE_NAME", "jargon_dictionary")
    max_jargon_terms_per_query: int = int(os.getenv("MAX_JARGON_TERMS_PER_QUERY", 5))
    enable_doc_summarization: bool = os.getenv("ENABLE_DOC_SUMMARIZATION", "true").lower() == "true"
    enable_metadata_enrichment: bool = os.getenv("ENABLE_METADATA_ENRICHMENT", "true").lower() == "true"
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", 0.7))
