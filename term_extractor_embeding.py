"""
Term Extractor with Embedding + PostgreSQL Dictionary Upsert
===========================================================

* 2025-06-12 修正版 *

主な変更
--------
1. GoogleGenerativeAIEmbeddings を model / google_api_key 付きで明示生成
2. .env に GOOGLE_EMBED_MODEL を持たせればモデル名を上書き可能
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredFileLoader,
    Docx2txtLoader,
)
from langchain_community.vectorstores import PGVector
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from sqlalchemy import create_engine, text

# ---------------------------------------------------------------------------
# 環境変数
# ---------------------------------------------------------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PG_URL = os.getenv("PG_URL")
# 新規: モデル名を環境変数で上書き可
EMBED_MODEL = os.getenv("GOOGLE_EMBED_MODEL", "models/text-embedding-004")

if not GOOGLE_API_KEY:
    sys.exit("[ERROR] GOOGLE_API_KEY が .env に設定されていません")

# ---------------------------------------------------------------------------
# ロガー設定
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("TermExtractor")
logger.info("Using GOOGLE_API_KEY = %s...", GOOGLE_API_KEY[:8])
logger.info("Using GOOGLE_EMBED_MODEL = %s", EMBED_MODEL)
if PG_URL:
    logger.info("Using PG_URL = %s", PG_URL.split("@")[-1])
else:
    logger.warning("PG_URL が設定されていないため、辞書は DB に保存されません")

# ---------------------------------------------------------------------------
# SudachiPy 辞書確認
# ---------------------------------------------------------------------------
try:
    from sudachipy import dictionary as _sudachi_dict

    _sudachi_dict.Dictionary().create()
except (ImportError, FileNotFoundError) as e:
    sys.exit(
        "[ERROR] SudachiPy か辞書 (sudachidict_core) が見つかりません:\n"
        "  $ pip install sudachipy sudachidict_core\n"
        f"  詳細: {e}"
    )

# ---------------------------------------------------------------------------
# Loader map & helper
# ---------------------------------------------------------------------------

def _load_table_as_text(path: Path):
    """CSV/Excel を DataFrame → CSV 文字列にして読み込む"""
    import pandas as pd

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    text_csv = df.to_csv(index=False)
    from langchain_core.documents import Document

    return [Document(page_content=text_csv, metadata={"source": str(path)})]


LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
    ".txt": TextLoader,
    ".md": UnstructuredFileLoader,
    ".html": UnstructuredFileLoader,
    ".htm": UnstructuredFileLoader,
    ".csv": _load_table_as_text,
    ".xlsx": _load_table_as_text,
    ".xls": _load_table_as_text,
}

# ---------------------------------------------------------------------------
# VectorStore wrapper
# ---------------------------------------------------------------------------


class VectorStore:
    """PGVector ラッパ（非同期初期化のみ実装）"""

    def __init__(self):
        # ★ 修正点: model / google_api_key を明示
        self.embedder = GoogleGenerativeAIEmbeddings(
            model=EMBED_MODEL,
            google_api_key=GOOGLE_API_KEY,
        )
        self.vector_store: PGVector | None = None

    def _sync_initialize(self, texts: List[str], metadatas: List[dict], ids: List[str]):
        self.vector_store = PGVector.from_texts(
            texts=texts,
            embedding=self.embedder,
            collection_name="chunks",
            metadatas=metadatas,
            ids=ids,
            connection_string=PG_URL,
        )

    async def initialize(self, texts: List[str], ids: List[str]):
        metadatas = [{"id": cid} for cid in ids]
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as pool:
            await loop.run_in_executor(pool, self._sync_initialize, texts, metadatas, ids)


# ---------------------------------------------------------------------------
# PostgreSQL 用語 upsert
# ---------------------------------------------------------------------------


def _save_terms_to_db(engine, terms: List[Dict[str, Any]]):
    sql = text(
        """
        INSERT INTO term_dictionary (term, synonyms, definition, sources)
        VALUES (:term, :synonyms, :definition, :sources)
        ON CONFLICT (term) DO UPDATE
        SET synonyms=EXCLUDED.synonyms,
            definition=EXCLUDED.definition,
            sources=EXCLUDED.sources;
        """
    )
    with engine.begin() as conn:
        for t in terms:
            conn.execute(
                sql,
                {
                    "term": t["term"],
                    "synonyms": t.get("synonyms", []),
                    "definition": t.get("definition", ""),
                    "sources": t.get("sources", []),
                },
            )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


async def run_pipeline(input_path: Path, output_path: Path):
    """用語抽出 → ベクトルストア登録 → 辞書 JSON + DB 保存"""

    # -------- ファイル収集 --------
    if input_path.is_file():
        files = [input_path] if input_path.suffix.lower() in LOADER_MAP else []
    else:
        files: List[Path] = [
            p for p in input_path.rglob("*") if p.suffix.lower() in LOADER_MAP
        ]
    if not files:
        logger.error("No supported files found in %s", input_path)
        return
    logger.info("Found %d files to process", len(files))

    # -------- ドキュメント読み込み --------
    docs = []
    for f in files:
        suf = f.suffix.lower()
        loader_cls = LOADER_MAP[suf]
        logger.info("Loading %s with %s", f.name, loader_cls.__name__)
        try:
            if suf in {".txt", ".md"} and loader_cls is TextLoader:
                # UTF-8 で失敗したら自動判定
                try:
                    docs.extend(loader_cls(str(f), encoding="utf-8").load())
                except UnicodeDecodeError:
                    import chardet

                    raw = f.read_bytes()
                    enc = chardet.detect(raw)["encoding"] or "utf-8"
                    docs.extend(loader_cls(str(f), encoding=enc, errors="ignore").load())
            else:
                docs.extend(loader_cls(str(f)).load())
        except Exception as e:
            logger.error("Failed to load %s: %s", f, e)

    if not docs:
        logger.error("Document loading failed for all files – aborting")
        return

    # -------- チャンク分割 (簡易) --------
    chunks = [d.page_content for d in docs]
    chunk_ids = [f"chunk-{i}" for i in range(len(chunks))]
    logger.info("Split into %d chunks", len(chunks))

    # -------- VectorStore 初期化 --------
    vector_store = VectorStore()
    logger.info("Initializing vector store...")
    try:
        await vector_store.initialize(chunks, chunk_ids)
    except Exception as e:
        logger.error("Failed to initialize vector store: %s", e)
        return
    logger.info("Vector store initialized successfully")

    # -------- 用語抽出 (ダミー実装) --------
    final_terms = [
        {
            "term": "テスト用語",
            "synonyms": ["試験用語"],
            "definition": "これはテストレコードです。",
            "sources": [str(p) for p in files],
        }
    ]

    # -------- JSON 保存 --------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(final_terms, f, ensure_ascii=False, indent=2)
    logger.info("Saved %d terms to %s", len(final_terms), output_path)

    # -------- DB 保存 --------
    if PG_URL:
        try:
            engine = create_engine(PG_URL, echo=False, future=True)
            _save_terms_to_db(engine, final_terms)
            logger.info("Upserted %d terms into PostgreSQL", len(final_terms))
        except Exception as e:
            logger.error("Failed to upsert terms into DB: %s", e)


# ---------------------------------------------------------------------------
# CLI エントリポイント
# ---------------------------------------------------------------------------


def _main():
    import argparse

    parser = argparse.ArgumentParser(description="Run term extraction pipeline")
    parser.add_argument("input", type=Path, help="Input file or directory")
    parser.add_argument("output", type=Path, help="Output JSON path")
    args = parser.parse_args()

    asyncio.run(run_pipeline(args.input, args.output))


if __name__ == "__main__":
    _main()
