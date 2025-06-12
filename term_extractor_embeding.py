#!/usr/bin/env python3
"""term_extractor_embeding.py
LCEL記法版 専門用語・類義語辞書生成（SudachiPy + RAG統合版）
------------------------------------------------
* LangChain Expression Language (LCEL) でチェイン構築
* SudachiPyとN-gramによる候補語生成の前処理を追加
* Google Embedding APIとPGVectorによるRAG実装
* LangSmithによる処理トレース対応
* 構造化出力 (Pydantic) で型安全性確保
* `.env` から `GOOGLE_API_KEY` と `PG_URL` 読み込み
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from sudachipy import tokenizer, dictionary
from sqlalchemy import create_engine, text

# ── ENV ───────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
PG_URL = os.getenv("PG_URL")
JARGON_TABLE_NAME = os.getenv("JARGON_TABLE_NAME", "jargon_dictionary")

if not API_KEY:
    sys.exit("[ERROR] .env に GOOGLE_API_KEY を設定してください")
if not PG_URL:
    sys.exit("[ERROR] .env に PG_URL を設定してください")

# ── LangChain imports ─────────────────────────────
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredFileLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_community.vectorstores import PGVector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# LangSmith設定の確認
if os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true":
    if not os.getenv("LANGCHAIN_API_KEY"):
        logger.warning("LANGCHAIN_API_KEY not set. LangSmith tracing disabled.")
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
    else:
        logger.info(f"LangSmith tracing enabled - Project: {os.getenv('LANGCHAIN_PROJECT', 'default')}")

# ── SudachiPy Setup ───────────────────────────────
sudachi_mode = tokenizer.Tokenizer.SplitMode.A

# ── Embeddings Setup ──────────────────────────────
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=API_KEY,
    task_type="RETRIEVAL_DOCUMENT"
)

# ── Vector Store Components ──────────────────────
class VectorStore:
    """PGVectorを使用したベクトルストア"""
    
    def __init__(self, connection_string: str, collection_name: str):
        self.connection_string = connection_string
        self.collection_name = collection_name
        self.vector_store: Optional[PGVector] = None

    def _sync_initialize(self, chunks: List[str], chunk_ids: List[str]):
        """同期的にPGVectorを初期化"""
        logger.info(f"Initializing PGVector with {len(chunks)} chunks...")
        # 既存のコレクションをクリアする代わりに、新しいデータで上書きする
        self.vector_store = PGVector.from_texts(
            texts=chunks,
            embedding=embeddings,
            collection_name=self.collection_name,
            ids=chunk_ids,
            connection_string=self.connection_string,
            pre_delete_collection=True, # コレクションを再作成
        )
        logger.info("PGVector initialized successfully.")

    async def initialize(self, chunks: List[str], chunk_ids: List[str]):
        """チャンクをエンベディング化してベクトルストアに保存"""
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as pool:
            await loop.run_in_executor(pool, self._sync_initialize, chunks, chunk_ids)
    
    async def search_similar_chunks(self, query_text: str, current_chunk_id: str, n_results: int = 3) -> str:
        """類似チャンクを検索して関連文脈として返す"""
        if not self.vector_store:
            return "関連情報なし"
        
        try:
            # 類似チャンクを検索
            results_with_scores = await self.vector_store.asimilarity_search_with_score(
                query=query_text,
                k=n_results + 1 # 自分自身が含まれる可能性があるため+1
            )
            
            related_contexts = []
            for doc, score in results_with_scores:
                # 自分自身のチャンクはスキップ
                if doc.metadata.get("id") == current_chunk_id:
                    continue
                
                # 類似度が閾値以上のもののみ使用（スコアが高いほど類似）
                if score > 0.7:  # 閾値は調整可能
                    related_contexts.append(f"[関連文脈 (類似度: {score:.2f})]\n{doc.page_content}")
                
                if len(related_contexts) >= n_results:
                    break
            
            return "\n\n".join(related_contexts) if related_contexts else "関連情報なし"
            
        except Exception as e:
            logger.error(f"Error searching similar chunks: {e}")
            return "関連情報なし"

# グローバルなベクトルストアインスタンス
vector_store = VectorStore(PG_URL, "term_extraction_chunks")

# ── Pydantic Models for Structured Output ────────
class Term(BaseModel):
    """専門用語の構造"""
    headword: str = Field(description="専門用語の見出し語")
    synonyms: List[str] = Field(default_factory=list, description="類義語・別名のリスト")
    definition: str = Field(description="30-50字程度の簡潔な定義")
    category: Optional[str] = Field(default=None, description="カテゴリ名")

class TermList(BaseModel):
    """用語リストの構造"""
    terms: List[Term] = Field(default_factory=list, description="専門用語のリスト")

# ── LLM Setup ─────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.1,
    google_api_key=API_KEY,
)

# ── Output Parser ─────────────────────────────────
json_parser = JsonOutputParser(pydantic_object=TermList)

# ── Prompts ───────────────────────────────────────
validation_prompt = ChatPromptTemplate.from_messages([
    ("system", """あなたは、提供されたテキストの文脈と候補リストを基に、専門用語を厳密に検証する専門家です。
候補リストの中から、与えられたテキストの文脈において専門用語として成立するものだけを選び出してください。
選んだ用語について、定義、類義語、カテゴリをJSON形式で返してください。
**重要：候補リストに存在しない用語を新たに追加してはいけません。**

関連する文脈情報も提供される場合は、それを参考にして：
- より正確な定義を作成
- 類義語を発見
- 適切なカテゴリ分類
を行ってください。

一般的すぎる単語（例：システム、データ、情報、処理、管理）は除外し、その分野特有の専門用語のみを選択してください。

{format_instructions}"""),
    ("user", """以下のテキストと候補リストを参考に、専門用語をJSON形式で返してください。

## テキスト本文:
{text}

## 関連する文脈情報:
{related_contexts}

## 候補リスト:
{candidates}
"""),
]).partial(format_instructions=json_parser.get_format_instructions())

consolidate_prompt = ChatPromptTemplate.from_messages([
    ("system", """用語一覧の重複を統合してください。
同じ意味の用語は1つにまとめ、類義語はsynonymsに含めてください。
必ず以下の形式の有効なJSONのみを返してください：

{format_instructions}"""),
    ("user", "{terms_json}"),
]).partial(format_instructions=json_parser.get_format_instructions())

# ── Document Processing Components ────────────────
SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    keep_separator=True,
    separators=["\n\n", "。", "\n", " "],
)

LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
    ".txt": TextLoader,
    ".md": UnstructuredFileLoader,
    ".html": UnstructuredFileLoader,
    ".htm": UnstructuredFileLoader,
}

# ── Helper Functions ──────────────────────────────

def load_document(file_path: Path) -> List[Document]:
    """ファイルパスからドキュメントをロード"""
    try:
        loader_cls = LOADER_MAP.get(file_path.suffix.lower(), TextLoader)
        logger.info(f"Loading {file_path.name} with {loader_cls.__name__}")
        return loader_cls(str(file_path)).load()
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return []

def split_documents(docs: List[Document]) -> List[str]:
    """ドキュメントリストをテキストチャンクに分割"""
    if not docs:
        return []
    full_text = "\n".join(doc.page_content for doc in docs)
    chunks = SPLITTER.split_text(full_text)
    logger.info(f"Split into {len(chunks)} chunks")
    return chunks

# ── Candidate Generation Function ─────────────────
def generate_candidates_from_chunk(text: str) -> List[str]:
    """SudachiPyとN-gramでチャンクから候補語を生成する"""
    if not text.strip():
        return []
    
    try:
        local_tokenizer = dictionary.Dictionary().create()
        tokens = local_tokenizer.tokenize(text, sudachi_mode)
        
        noun_tokens = [{'surface': t.surface(), 'position': i} for i, t in enumerate(tokens) if t.part_of_speech()[0] == '名詞']
        if not noun_tokens: return []
        
        candidates = {t['surface'] for t in noun_tokens if len(t['surface']) > 1}
        
        noun_groups = []
        current_group = []
        for token in noun_tokens:
            if not current_group or token['position'] == current_group[-1]['position'] + 1:
                current_group.append(token)
            else:
                if len(current_group) >= 2: noun_groups.append(current_group)
                current_group = [token]
        if len(current_group) >= 2: noun_groups.append(current_group)
        
        for group in noun_groups:
            surfaces = [t['surface'] for t in group]
            for n in range(2, min(5, len(surfaces) + 1)):
                for i in range(len(surfaces) - n + 1):
                    candidates.add("".join(surfaces[i:i+n]))
        
        return sorted(list(candidates), key=len, reverse=True)[:100]
        
    except Exception as e:
        logger.error(f"Error in candidate generation: {e}")
        return []

# ── Database Saving Function ──────────────────────
def _save_terms_to_db(terms: List[Dict[str, Any]]):
    """抽出した用語をPostgreSQLに保存"""
    engine = create_engine(PG_URL)
    sql = text(
        f"""
        INSERT INTO {JARGON_TABLE_NAME} (term, definition, domain, aliases)
        VALUES (:term, :definition, :domain, :aliases)
        ON CONFLICT (term) DO UPDATE
        SET definition = EXCLUDED.definition,
            domain = EXCLUDED.domain,
            aliases = EXCLUDED.aliases,
            updated_at = CURRENT_TIMESTAMP;
        """
    )
    with engine.begin() as conn:
        for t in terms:
            conn.execute(
                sql,
                {
                    "term": t.get("headword"),
                    "definition": t.get("definition", ""),
                    "domain": t.get("category"),
                    "aliases": t.get("synonyms", []),
                },
            )
    logger.info(f"Upserted {len(terms)} terms into PostgreSQL table '{JARGON_TABLE_NAME}'")

# ── LCEL Chains with Tracing ─────────────────────

file_processing_pipeline = (
    RunnableLambda(load_document, name="load_document")
    | RunnableLambda(split_documents, name="split_documents")
).with_config({"run_name": "file_processing"})

candidate_generation_chain = RunnableLambda(generate_candidates_from_chunk, name="generate_candidates")

async def extract_with_context(chunk_data: Dict[str, str]) -> Dict:
    """RAGを含む用語抽出"""
    chunk_text, chunk_id = chunk_data["text"], chunk_data["chunk_id"]
    candidates = await candidate_generation_chain.ainvoke(chunk_text)
    if not candidates: return {"terms": []}
    
    related_contexts = await vector_store.search_similar_chunks(chunk_text[:1000], chunk_id, n_results=3)
    
    prompt_data = {
        "text": chunk_text[:3000],
        "candidates": "\n".join(candidates),
        "related_contexts": related_contexts
    }
    
    extraction_chain = (validation_prompt | llm | json_parser).with_config({"run_name": "term_validation"})
    return await extraction_chain.ainvoke(prompt_data)

extract_with_context_chain = RunnableLambda(extract_with_context, name="extract_with_rag")

term_consolidation_chain = (
    RunnablePassthrough.assign(terms_json=lambda x: json.dumps({"terms": x["terms"]}, ensure_ascii=False))
    | consolidate_prompt | llm | json_parser
).with_config({"run_name": "term_consolidation"})

async def consolidate_in_batches(all_terms: List[Dict]) -> List[Dict]:
    """大量の用語をバッチ処理で統合"""
    if not all_terms: return []
    if len(all_terms) <= 50:
        result = await term_consolidation_chain.ainvoke({"terms": all_terms})
        return result.get("terms", [])
    
    batch_size = 30
    consolidated = []
    for i in range(0, len(all_terms), batch_size):
        batch = all_terms[i:i+batch_size]
        result = await term_consolidation_chain.ainvoke({"terms": batch})
        consolidated.extend(result.get("terms", []))
        if i + batch_size < len(all_terms): await asyncio.sleep(7)
    return consolidated

async def extract_terms_with_rate_limit(chunks_with_ids: List[Dict[str, str]]) -> List[TermList]:
    """レート制限を考慮した用語抽出"""
    batch_size, delay = 3, 7
    results = []
    for i in range(0, len(chunks_with_ids), batch_size):
        batch = chunks_with_ids[i:i+batch_size]
        batch_results = await asyncio.gather(*(extract_with_context_chain.ainvoke(c) for c in batch))
        results.extend(batch_results)
        if i + batch_size < len(chunks_with_ids):
            logger.info(f"Processed {i + len(batch)}/{len(chunks_with_ids)} chunks. Waiting {delay}s...")
            await asyncio.sleep(delay)
    return results

def merge_duplicate_terms(term_lists: List[TermList]) -> List[Term]:
    """重複する用語をマージ"""
    merged: Dict[str, Dict] = {}
    for term_list in term_lists:
        for term in term_list.get("terms", []):
            if not isinstance(term, dict) or not term.get("headword"): continue
            key = term["headword"].lower().strip()
            if key not in merged:
                merged[key] = term
            else:
                merged[key]["synonyms"] = list(set(merged[key].get("synonyms", []) + term.get("synonyms", [])))
                if not merged[key].get("definition"): merged[key]["definition"] = term.get("definition")
                if not merged[key].get("category"): merged[key]["category"] = term.get("category")
    logger.info(f"Merged {len(merged)} unique terms")
    return list(merged.values())

# メインパイプライン
async def run_pipeline(input_dir: Path, output_json: Path):
    """メインの処理パイプライン"""
    files = [p for ext in LOADER_MAP for p in input_dir.glob(f"**/*{ext}")]
    if not files:
        logger.error(f"No supported files found in {input_dir}"); return

    logger.info(f"Found {len(files)} files to process")
    file_chunks = await asyncio.gather(*(file_processing_pipeline.ainvoke(f) for f in files))
    all_chunks = [c for chunks in file_chunks for c in chunks if c.strip()]
    if not all_chunks:
        logger.error("No text chunks generated"); return

    logger.info(f"Total chunks to process: {len(all_chunks)}")
    chunk_ids = [f"chunk_{i}" for i in range(len(all_chunks))]
    
    await vector_store.initialize(all_chunks, chunk_ids)
    
    chunks_with_ids = [{"text": c, "chunk_id": cid} for c, cid in zip(all_chunks, chunk_ids)]
    term_lists = await extract_terms_with_rate_limit(chunks_with_ids)
    
    valid_term_lists = [tl for tl in term_lists if tl and tl.get("terms")]
    if not valid_term_lists:
        logger.error("No terms extracted"); final_terms = []
    else:
        unique_terms = merge_duplicate_terms(valid_term_lists)
        final_terms = await consolidate_in_batches(unique_terms)
    
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"terms": final_terms}, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(final_terms)} terms to {output_json}")

    if final_terms:
        _save_terms_to_db(final_terms)

# ── Entry Point ───────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python term_extractor_embeding.py <input_dir> <output_json>")
    
    asyncio.run(run_pipeline(Path(sys.argv[1]), Path(sys.argv[2])))
