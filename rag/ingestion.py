import json
from pathlib import Path
from sqlalchemy import create_engine, text
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredFileLoader, Docx2txtLoader
)
try:
    from langchain_community.document_loaders import TextractLoader
except ImportError:
    TextractLoader = None
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class IngestionHandler:
    def __init__(self, config, vector_store, text_processor, connection_string):
        self.config = config
        self.vector_store = vector_store
        self.text_processor = text_processor
        self.connection_string = connection_string

    def load_documents(self, paths: List[str]) -> List[Document]:
        docs: List[Document] = []
        for p_str in paths:
            path = Path(p_str)
            if not path.exists():
                print(f"File not found: {p_str}")
                continue
            suf = path.suffix.lower()
            try:
                if suf == ".pdf":
                    docs.extend(UnstructuredFileLoader(str(path), mode="single", strategy="fast").load())
                elif suf in {".txt", ".md"}:
                    docs.extend(TextLoader(str(path), encoding="utf-8").load())
                elif suf == ".docx":
                    docs.extend(UnstructuredFileLoader(str(path), mode="single", strategy="fast").load())
                elif suf == ".doc" and TextractLoader:
                    docs.extend(TextractLoader(str(path)).load())
            except Exception as e:
                print(f"Error loading {p_str}: {type(e).__name__} - {e}")
        return docs

    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        out: List[Document] = []
        for i, d in enumerate(docs):
            src = d.metadata.get("source", f"doc_source_{i}")
            doc_id = Path(src).name
            try:
                normalized_content = self.text_processor.normalize_text(d.page_content)
                d.page_content = normalized_content
                doc_splits = text_splitter.split_documents([d])
                for j, c in enumerate(doc_splits):
                    new_metadata = {
                        **d.metadata, **c.metadata,
                        "chunk_id": f"{doc_id}_{i}_{j}",
                        "document_id": doc_id,
                        "original_document_source": src,
                        "collection_name": self.config.collection_name
                    }
                    c.metadata = new_metadata
                    out.append(c)
            except Exception as e:
                print(f"Error splitting document {src}: {type(e).__name__} - {e}")
        return out

    def _store_chunks_for_keyword_search(self, chunks: List[Document]):
        eng = create_engine(self.connection_string)
        sql = text("""
            INSERT INTO document_chunks(collection_name, document_id, chunk_id, content, tokenized_content, metadata, created_at) 
            VALUES(:coll_name, :doc_id, :cid, :cont, :tok_cont, :meta, CURRENT_TIMESTAMP) 
            ON CONFLICT(chunk_id) DO UPDATE SET 
                content = EXCLUDED.content, tokenized_content = EXCLUDED.tokenized_content,
                metadata = EXCLUDED.metadata, document_id = EXCLUDED.document_id,
                collection_name = EXCLUDED.collection_name, created_at = CURRENT_TIMESTAMP;
        """)
        try:
            with eng.connect() as conn, conn.begin():
                for c in chunks:
                    normalized_content = self.text_processor.normalize_text(c.page_content)
                    tokenized_content = self.text_processor.tokenize(normalized_content) if self.config.enable_japanese_search else ""
                    conn.execute(sql, {
                        "coll_name": self.config.collection_name,
                        "doc_id": c.metadata["document_id"],
                        "cid": c.metadata["chunk_id"],
                        "cont": normalized_content,
                        "tok_cont": " ".join(tokenized_content),
                        "meta": json.dumps(c.metadata or {})
                    })
        except Exception as e:
            print(f"Error storing chunks for keyword search: {type(e).__name__} - {e}")

    def ingest_documents(self, paths: List[str]):
        print("Loading documents...")
        all_docs = self.load_documents(paths)
        if not all_docs:
            print("No documents loaded.")
            return

        print(f"Chunking {len(all_docs)} documents...")
        chunks = self.chunk_documents(all_docs)
        valid_chunks = [c for c in chunks if c.page_content and c.page_content.strip()]
        if not valid_chunks:
            print("No valid chunks to ingest.")
            return
        
        print(f"Ingesting {len(valid_chunks)} chunks...")
        chunk_ids = [c.metadata['chunk_id'] for c in valid_chunks]
        try:
            self.vector_store.add_documents(valid_chunks, ids=chunk_ids)
            self._store_chunks_for_keyword_search(valid_chunks)
            print(f"Successfully ingested {len(valid_chunks)} chunks.")
        except Exception as e:
            print(f"Error during ingestion: {type(e).__name__} - {e}")

    def delete_document_by_id(self, doc_id: str) -> tuple[bool, str]:
        if not doc_id: return False, "Document ID cannot be empty."
        
        engine = create_engine(self.connection_string)
        try:
            with engine.connect() as conn, conn.begin():
                res = conn.execute(
                    text("SELECT chunk_id FROM document_chunks WHERE document_id = :doc_id AND collection_name = :coll"),
                    {"doc_id": doc_id, "coll": self.config.collection_name}
                )
                chunk_ids = [row[0] for row in res]

                if not chunk_ids:
                    return True, f"No chunks found for document ID '{doc_id}'."
                
                del_res = conn.execute(
                    text("DELETE FROM document_chunks WHERE document_id = :doc_id AND collection_name = :coll"),
                    {"doc_id": doc_id, "coll": self.config.collection_name}
                )
                
                if self.vector_store:
                    self.vector_store.delete(ids=chunk_ids)

            return True, f"Deleted {del_res.rowcount} chunks for document ID '{doc_id}'."
        except Exception as e:
            return False, f"Deletion error: {type(e).__name__} - {e}"
