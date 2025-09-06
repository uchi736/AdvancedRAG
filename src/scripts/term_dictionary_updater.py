#!/usr/bin/env python3
"""term_dictionary_updater.py
既存の専門用語辞書の関連語を更新するツール
------------------------------------------------
既存の専門用語をクエリとしてPGVectorで類似文脈を検索し、
新たな関連語を発見して辞書を更新する
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from datetime import datetime

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Project-specific imports
sys.path.append(str(Path(__file__).resolve().parents[1]))
from rag.config import Config

# term_extractor_embedingからコンポーネントをインポート
from src.scripts.term_extractor_embeding import (
    VectorStore,
    SynonymDetector,
    embeddings
)

# ── ENV ───────────────────────────────────────────
load_dotenv()
cfg = Config()

PG_URL = f"postgresql+psycopg://{cfg.db_user}:{cfg.db_password}@{cfg.db_host}:{cfg.db_port}/{cfg.db_name}"
JARGON_TABLE_NAME = cfg.jargon_table_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Term Dictionary Updater ──────────────────────
class TermDictionaryUpdater:
    """既存用語の関連語を更新するクラス"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.vector_store = VectorStore(connection_string, "term_extraction_chunks")
        self.synonym_detector = SynonymDetector()
        self.update_log = []
        
    def get_existing_terms(self) -> List[Dict[str, Any]]:
        """jargon_dictionaryから既存の用語を取得"""
        engine = create_engine(self.connection_string)
        query = text(f"""
            SELECT term, definition, domain, aliases, created_at, updated_at
            FROM {JARGON_TABLE_NAME}
            ORDER BY term
        """)
        
        terms = []
        try:
            with engine.connect() as conn:
                result = conn.execute(query)
                for row in result:
                    terms.append({
                        'term': row.term,
                        'definition': row.definition,
                        'domain': row.domain,
                        'aliases': row.aliases or [],
                        'created_at': row.created_at,
                        'updated_at': row.updated_at
                    })
            logger.info(f"Retrieved {len(terms)} existing terms from database")
        except Exception as e:
            logger.error(f"Error retrieving terms from database: {e}")
            
        return terms
    
    async def search_related_contexts(self, term: str, definition: str) -> List[str]:
        """用語と定義をクエリとして類似文脈を検索"""
        # ベクトルストアが初期化されているか確認
        if not self.vector_store.vector_store:
            logger.warning("Vector store not initialized. Skipping context search.")
            return []
        
        # クエリを構築（用語と定義の最初の部分を使用）
        query = f"{term} {definition[:100]}"
        
        try:
            # 類似文脈を検索
            results = await self.vector_store.vector_store.asimilarity_search_with_score(
                query=query,
                k=10  # 10件取得
            )
            
            # テキストのみを抽出
            contexts = []
            for doc, score in results:
                if score > 0.5:  # 類似度が0.5以上のものだけ
                    contexts.append(doc.page_content)
            
            logger.debug(f"Found {len(contexts)} related contexts for term: {term}")
            return contexts
            
        except Exception as e:
            logger.error(f"Error searching contexts for {term}: {e}")
            return []
    
    def extract_new_synonyms(self, term: str, contexts: List[str], existing_aliases: List[str]) -> List[str]:
        """文脈から新しい関連語を抽出"""
        if not contexts:
            return []
        
        # 全文脈を結合
        combined_text = "\n".join(contexts)
        
        # 文脈から候補語を抽出（簡易版）
        import re
        from collections import Counter
        
        # 用語の周辺に出現する語を収集
        pattern = rf'(.{{0,50}}){re.escape(term)}(.{{0,50}})'
        matches = re.findall(pattern, combined_text, re.IGNORECASE)
        
        nearby_words = []
        for before, after in matches:
            # 前後のテキストから名詞っぽいものを抽出
            words = re.findall(r'[ァ-ヶー]+|[a-zA-Z0-9]+', before + after)
            nearby_words.extend(words)
        
        # 頻出語を候補とする
        word_counts = Counter(nearby_words)
        candidates = [word for word, count in word_counts.items() if count >= 2 and len(word) >= 2]
        
        # 既存の関連語と重複しないものを選択
        existing_set = set(existing_aliases + [term])
        new_synonyms = []
        
        for candidate in candidates:
            if candidate not in existing_set:
                # 編集距離で類似性をチェック
                from difflib import SequenceMatcher
                similarity = SequenceMatcher(None, term.lower(), candidate.lower()).ratio()
                
                # 類似度が高い場合（70-95%）は関連語として採用
                if 0.7 < similarity < 0.95:
                    new_synonyms.append(candidate)
                # または用語を含む複合語
                elif term in candidate or candidate in term:
                    if len(candidate) > 2:  # 短すぎる語は除外
                        new_synonyms.append(candidate)
        
        # 重複を除去
        new_synonyms = list(set(new_synonyms))
        
        if new_synonyms:
            logger.info(f"Found {len(new_synonyms)} new synonyms for {term}: {new_synonyms}")
        
        return new_synonyms
    
    def update_term_in_db(self, term: str, new_aliases: List[str]) -> bool:
        """データベースの用語を更新"""
        if not new_aliases:
            return False
        
        engine = create_engine(self.connection_string)
        
        try:
            with engine.begin() as conn:
                # 既存の関連語を取得
                get_query = text(f"""
                    SELECT aliases FROM {JARGON_TABLE_NAME}
                    WHERE term = :term
                """)
                result = conn.execute(get_query, {"term": term}).fetchone()
                
                if result:
                    existing_aliases = result.aliases or []
                    # 新しい関連語を既存のものと統合
                    all_aliases = list(set(existing_aliases + new_aliases))
                    
                    # 更新
                    update_query = text(f"""
                        UPDATE {JARGON_TABLE_NAME}
                        SET aliases = :aliases,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE term = :term
                    """)
                    conn.execute(update_query, {
                        "term": term,
                        "aliases": all_aliases
                    })
                    
                    logger.info(f"Updated term '{term}' with {len(new_aliases)} new aliases")
                    return True
                    
        except Exception as e:
            logger.error(f"Error updating term {term}: {e}")
            
        return False
    
    async def update_all_terms(self) -> Dict[str, Any]:
        """全ての既存用語の関連語を更新"""
        logger.info("Starting term dictionary update...")
        
        # 1. 既存用語を取得
        existing_terms = self.get_existing_terms()
        if not existing_terms:
            logger.warning("No existing terms found in database")
            return {"updated_count": 0, "changes": []}
        
        # 2. ベクトルストアを初期化（既存のチャンクを使用）
        try:
            # 既存のチャンクがあるか確認
            engine = create_engine(self.connection_string)
            with engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT COUNT(*) as count 
                    FROM langchain_pg_collection 
                    WHERE name = 'term_extraction_chunks'
                """)).fetchone()
                
                if result and result.count > 0:
                    logger.info("Using existing vector store")
                    # 既存のベクトルストアを使用
                    from langchain_community.vectorstores import PGVector
                    self.vector_store.vector_store = PGVector(
                        collection_name="term_extraction_chunks",
                        connection_string=self.connection_string,
                        embedding_function=embeddings
                    )
                else:
                    logger.warning("No existing vector store found. Please run term extraction first.")
                    return {"updated_count": 0, "changes": []}
                    
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            return {"updated_count": 0, "changes": []}
        
        # 3. 各用語を処理
        updated_terms = []
        
        for term_data in existing_terms:
            term = term_data['term']
            definition = term_data['definition']
            existing_aliases = term_data['aliases']
            
            logger.info(f"Processing term: {term}")
            
            # 類似文脈を検索
            contexts = await self.search_related_contexts(term, definition)
            
            if contexts:
                # 新しい関連語を抽出
                new_synonyms = self.extract_new_synonyms(term, contexts, existing_aliases)
                
                if new_synonyms:
                    # DBを更新
                    if self.update_term_in_db(term, new_synonyms):
                        updated_terms.append({
                            'term': term,
                            'new_synonyms': new_synonyms,
                            'old_synonyms': existing_aliases,
                            'total_contexts': len(contexts)
                        })
        
        # 4. 更新ログを作成
        result = {
            'updated_count': len(updated_terms),
            'total_terms': len(existing_terms),
            'changes': updated_terms,
            'timestamp': datetime.now().isoformat()
        }
        
        # ログをファイルに保存
        output_path = Path("output/term_update_log.json")
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Update complete. Updated {len(updated_terms)}/{len(existing_terms)} terms")
        logger.info(f"Update log saved to {output_path}")
        
        return result

# ── Main Execution ────────────────────────────────
async def main():
    """メイン実行関数"""
    updater = TermDictionaryUpdater(PG_URL)
    result = await updater.update_all_terms()
    
    # 結果を表示
    print("\n" + "="*50)
    print("Term Dictionary Update Results")
    print("="*50)
    print(f"Total terms processed: {result['total_terms']}")
    print(f"Terms updated: {result['updated_count']}")
    
    if result['changes']:
        print("\nUpdated terms:")
        for change in result['changes'][:10]:  # 最初の10件だけ表示
            print(f"\n📝 {change['term']}")
            print(f"   Old synonyms: {change['old_synonyms']}")
            print(f"   New synonyms: {change['new_synonyms']}")
            print(f"   Contexts found: {change['total_contexts']}")
    
    print(f"\nFull log saved to: output/term_update_log.json")

if __name__ == "__main__":
    asyncio.run(main())