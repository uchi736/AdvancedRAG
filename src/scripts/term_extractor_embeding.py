#!/usr/bin/env python3
"""term_extractor_embeding.py
LCEL記法版 専門用語・類義語辞書生成（SudachiPy + RAG統合版）
------------------------------------------------
* LangChain Expression Language (LCEL) でチェイン構築
* SudachiPyとN-gramによる候補語生成の前処理を追加
* Azure OpenAI APIとPGVectorによるRAG実装
* LangSmithによる処理トレース対応
* 構造化出力 (Pydantic) で型安全性確保
* `.env` から Azure OpenAI設定 と `PG_URL` 読み込み
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

# --- Project-specific imports ---
# 親ディレクトリをパスに追加してragモジュールをインポート
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.rag.config import Config

# ── ENV ───────────────────────────────────────────
load_dotenv()
cfg = Config()

PG_URL = f"postgresql+psycopg://{cfg.db_user}:{cfg.db_password}@{cfg.db_host}:{cfg.db_port}/{cfg.db_name}"
JARGON_TABLE_NAME = cfg.jargon_table_name

if not all([cfg.azure_openai_api_key, cfg.azure_openai_endpoint, cfg.azure_openai_chat_deployment_name, cfg.azure_openai_embedding_deployment_name]):
    sys.exit("[ERROR] .env にAzure OpenAIの関連設定が不足しています。")

# ── LangChain imports ─────────────────────────────
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
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
sudachi_mode = tokenizer.Tokenizer.SplitMode.C  # Mode.Cで詳細な形態素解析

# ── Embeddings Setup ──────────────────────────────
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=cfg.azure_openai_endpoint,
    api_key=cfg.azure_openai_api_key,
    api_version=cfg.azure_openai_api_version,
    azure_deployment=cfg.azure_openai_embedding_deployment_name
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
        self.vector_store = PGVector.from_texts(
            texts=chunks,
            embedding=embeddings,
            collection_name=self.collection_name,
            ids=chunk_ids,
            connection_string=self.connection_string,
            pre_delete_collection=True,
        )
        logger.info("PGVector initialized successfully.")

    async def initialize(self, chunks: List[str], chunk_ids: List[str]):
        """チャンクをエンベディング化してベクトルストアに保存"""
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as pool:
            await loop.run_in_executor(pool, self._sync_initialize, chunks, chunk_ids)
    
    async def search_similar_chunks(self, query_text: str, current_chunk_id: str, n_results: int = 3) -> str:
        """改良版：より関連性の高い文脈を取得"""
        if not self.vector_store:
            return "関連情報なし"
        
        try:
            # クエリを重要な部分に絞る（最初と最後の文を使用）
            sentences = query_text.split('。')
            if len(sentences) > 3:
                # 最初の2文と最後の1文を結合してクエリとする
                query = '。'.join(sentences[:2] + [sentences[-1]])
            else:
                query = query_text[:1000]  # 最大1000文字に制限
            
            # 類似度検索（より多く取得して精選）
            results_with_scores = await self.vector_store.asimilarity_search_with_score(
                query=query,
                k=n_results * 2  # 2倍取得してフィルタリング
            )
            
            # スコアと内容の両方で選別
            related_contexts = []
            seen_contents = set()
            
            # 動的閾値の計算（上位スコアの平均値）
            if results_with_scores:
                top_scores = [score for _, score in results_with_scores[:3]]
                if top_scores:
                    dynamic_threshold = sum(top_scores) / len(top_scores) * 0.7  # 上位平均の70%
                else:
                    dynamic_threshold = 0.7
            else:
                dynamic_threshold = 0.7
            
            for doc, score in results_with_scores:
                # 自身のチャンクは除外
                if doc.metadata.get("id") == current_chunk_id:
                    continue
                
                # 動的閾値でフィルタリング
                if score < dynamic_threshold:
                    continue
                
                # 内容の重複チェック（最初の200文字でハッシュ）
                content_hash = hash(doc.page_content[:200])
                if content_hash in seen_contents:
                    continue
                seen_contents.add(content_hash)
                
                related_contexts.append((doc.page_content, score))
                
                if len(related_contexts) >= n_results:
                    break
            
            # スコア順に整形して返す
            if related_contexts:
                return "\n\n".join([
                    f"[関連文脈 {i+1} (類似度: {score:.2f})]\n{content[:500]}"  # 500文字に制限
                    for i, (content, score) in enumerate(related_contexts)
                ])
            else:
                return "関連情報なし"
            
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
llm = AzureChatOpenAI(
    azure_endpoint=cfg.azure_openai_endpoint,
    api_key=cfg.azure_openai_api_key,
    api_version=cfg.azure_openai_api_version,
    azure_deployment=cfg.azure_openai_chat_deployment_name,
    temperature=0.1,
)

# ── Output Parser ─────────────────────────────────
json_parser = JsonOutputParser(pydantic_object=TermList)

# ── Prompts ───────────────────────────────────────
validation_prompt = ChatPromptTemplate.from_messages([
    ("system", """あなたは専門分野の用語抽出専門家です。

【専門用語の判定基準】
1. ドメイン固有性：その分野でのみ、または特別な意味で使われる
2. 概念の複合性：複数の概念が結合して新しい意味を形成している
3. 定義の必要性：一般の人には説明が必要な概念である
4. 文脈での重要性：文書の主題理解に不可欠である

【類義語・関連語の判定基準】
1. 表記違い：同じ概念の異なる表現（例：医薬品/薬品、品質管理/QC）
2. 略語と正式名称：（例：GMP/適正製造規範、API/原薬）
3. 上位・下位概念：（例：製造設備/製造装置、試験/検査）
4. 同じカテゴリの関連語：（例：原薬/添加剤/賦形剤）

【必ず除外すべき用語】
- 一般的すぎる単語：システム、データ、情報、処理、管理、方法、状態、結果、目的、対象、内容
- 単純な動作や状態：実施、確認、作成、使用、記録、設定、表示、入力、出力
- 文脈で専門的意味を持たない一般名詞
- 単なる数量表現や時間表現

【抽出ルール】
- 候補リストにない用語は絶対に追加しない
- 類義語は正確に識別し、最も一般的な表記を見出し語とする
- 検出された関連語候補を参考に、synonymsフィールドに適切に設定する
- 定義は30-50字で、その分野の初学者にも理解できる表現にする
- カテゴリは以下から選択：「規制」「製造」「品質」「技術」「管理」「安全」「分析」「その他」

【重要度評価】
各候補について以下の観点で評価：
- 文書内での出現頻度と分布
- 他の専門用語との共起関係
- 文書の主題との関連性

{format_instructions}"""),
    ("user", """以下の情報から専門用語を厳密に選定してください。

## 分析対象テキスト:
{text}

## 関連文脈（参考情報）:
{related_contexts}

## 候補リスト（この中からのみ選択）:
{candidates}

## 関連語候補（自動検出）:
{synonym_hints}

各候補について、上記の判定基準に照らして専門用語かどうかを判断し、
関連語候補も参考にしながら、該当するもののみをJSON形式で出力してください。
特に、関連語候補に含まれる語は積極的にsynonymsフィールドに含めてください。
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

# ── Domain Knowledge & Scoring ─────────────────
import math
from collections import defaultdict, Counter
from typing import Tuple
from difflib import SequenceMatcher

# ドメイン特化型キーワード辞書
DOMAIN_KEYWORDS = {
    "医薬": ["品", "部外品", "製剤", "原薬", "添加剤", "成分", "薬効", "薬理", "薬物"],
    "製造": ["管理", "工程", "バリデーション", "設備", "施設", "製法", "品質", "検証"],
    "品質": ["管理", "保証", "試験", "規格", "基準", "標準", "適合", "検査"],
    "規制": ["要件", "申請", "承認", "届出", "査察", "法令", "通知", "ガイドライン"],
    "安全": ["性", "評価", "リスク", "毒性", "副作用", "有害", "事象"],
    "技術": ["分析", "方法", "手法", "システム", "プロセス", "機器", "装置"],
}

# 除外すべき一般語
STOPWORDS = {
    "こと", "もの", "ため", "場合", "とき", "ところ", "方法", 
    "状態", "結果", "目的", "対象", "内容", "情報", "データ",
    "システム", "プロセス", "サービス", "ソフトウェア",
    "確認", "実施", "作成", "使用", "管理", "処理", "記録"
}

class TermScorer:
    """高度な専門用語スコアリング"""
    
    @staticmethod
    def calculate_c_value(candidates_freq: Dict[str, int]) -> Dict[str, float]:
        """
        C値を計算
        C-value(a) = log2|a| × (freq(a) - (1/|Ta|) × Σb∈Ta freq(b))
        """
        c_values = {}
        
        for candidate, freq in candidates_freq.items():
            # 候補語の長さ（文字数を使用）
            length = len(candidate)
            
            # より長い候補語を探す
            longer_terms = []
            for other_candidate in candidates_freq:
                if other_candidate != candidate and candidate in other_candidate:
                    longer_terms.append(other_candidate)
            
            # C値を計算
            if not longer_terms:
                c_value = math.log2(length) * freq if length > 1 else freq
            else:
                sum_freq = sum(candidates_freq[term] for term in longer_terms)
                t_a = len(longer_terms)
                c_value = math.log2(length) * (freq - sum_freq / t_a) if length > 1 else (freq - sum_freq / t_a)
            
            c_values[candidate] = max(c_value, 0)  # 負の値を0にクリップ
        
        return c_values
    
    @staticmethod
    def calculate_distribution_score(positions: List[int], doc_length: int) -> float:
        """文書内分布スコア（均等分布ほど高スコア）"""
        if len(positions) <= 1:
            return 0.5
        
        # 位置の標準偏差を計算
        mean_pos = sum(positions) / len(positions)
        variance = sum((p - mean_pos) ** 2 for p in positions) / len(positions)
        std_dev = math.sqrt(variance)
        
        # 理想的な均等分布との差を評価
        ideal_gap = doc_length / (len(positions) + 1)
        distribution_score = 1.0 / (1.0 + std_dev / ideal_gap)
        
        return distribution_score
    
    @staticmethod
    def calculate_position_score(first_pos: int, doc_length: int) -> float:
        """初出位置スコア（文書前半ほど高スコア）"""
        return 1.0 - (first_pos / doc_length) * 0.5
    
    @staticmethod
    def calculate_cooccurrence_score(candidate: str, noun_phrases: List[List[str]]) -> float:
        """共起関係スコア"""
        score = 0.0
        
        for phrase in noun_phrases:
            phrase_str = ''.join(phrase)
            if candidate in phrase_str:
                # ドメインキーワードとの共起をチェック
                for domain, keywords in DOMAIN_KEYWORDS.items():
                    for keyword in keywords:
                        if keyword in phrase_str and keyword != candidate:
                            score += 1.5
                            break
        
        return min(score, 10.0)  # 最大10点

class SynonymDetector:
    """類義語・関連語の高度な検出"""
    
    @staticmethod
    def find_synonyms(candidates: List[str], noun_phrases: List[List[Dict]]) -> Dict[str, List[str]]:
        """候補語から類義語・関連語を検出"""
        synonyms = defaultdict(set)
        
        # 1. 部分文字列関係の検出（包含関係）
        for i, cand1 in enumerate(candidates):
            for cand2 in candidates[i+1:]:
                # 片方が他方を含む場合（完全一致は除く）
                if cand1 != cand2:
                    if cand1 in cand2:
                        # 短い方が長い方の中核概念の可能性
                        synonyms[cand2].add(cand1)
                    elif cand2 in cand1:
                        synonyms[cand1].add(cand2)
        
        # 2. 同じ文脈での共起関係
        cooccurrence_map = defaultdict(set)
        window_size = 5  # 前後5語以内を文脈とする
        
        for phrase in noun_phrases:
            surfaces = [t.get('surface', '') if isinstance(t, dict) else str(t) for t in phrase]
            phrase_str = ''.join(surfaces)
            
            # フレーズ内で共起する候補語を記録
            occurring_cands = [c for c in candidates if c in phrase_str]
            for cand1 in occurring_cands:
                for cand2 in occurring_cands:
                    if cand1 != cand2:
                        cooccurrence_map[cand1].add(cand2)
        
        # 共起頻度が高い語を関連語として追加
        for cand, related in cooccurrence_map.items():
            if len(related) >= 2:  # 2回以上共起
                synonyms[cand].update(related)
        
        # 3. 編集距離による類似語検出
        for i, cand1 in enumerate(candidates):
            for cand2 in candidates[i+1:]:
                if cand1 != cand2:
                    similarity = SequenceMatcher(None, cand1, cand2).ratio()
                    
                    # 高い類似度（70-95%）の語を関連語とする
                    if 0.7 < similarity < 0.95:
                        synonyms[cand1].add(cand2)
                        synonyms[cand2].add(cand1)
        
        # 4. 語幹・語尾パターンによる関連語検出
        stem_groups = defaultdict(list)
        suffix_patterns = ['管理', 'システム', '装置', '機器', '設備', '工程', '方法', '技術']
        
        for cand in candidates:
            # 語幹グループ（最初の2-3文字）
            if len(cand) >= 3:
                stem = cand[:3]
                stem_groups[stem].append(cand)
            
            # 語尾パターンマッチング
            for suffix in suffix_patterns:
                if cand.endswith(suffix) and len(cand) > len(suffix):
                    base = cand[:-len(suffix)]
                    # 同じベースを持つ他の候補を探す
                    for other_cand in candidates:
                        if other_cand != cand and other_cand.startswith(base):
                            synonyms[cand].add(other_cand)
        
        # 語幹が同じグループを関連語とする
        for stem, group in stem_groups.items():
            if len(group) > 1:
                for word in group:
                    synonyms[word].update(w for w in group if w != word)
        
        # 5. 略語と正式名称のパターン検出
        abbreviation_patterns = {
            'GMP': '適正製造規範',
            'GQP': '品質保証',
            'GVP': '製造販売後安全管理',
            'QC': '品質管理',
            'QA': '品質保証',
            'SOP': '標準作業手順',
            'ICH': '医薬品規制調和国際会議',
        }
        
        for abbr, full_name in abbreviation_patterns.items():
            if abbr in candidates and full_name in candidates:
                synonyms[abbr].add(full_name)
                synonyms[full_name].add(abbr)
        
        # 6. ドメイン固有の関連語
        domain_relations = {
            '原薬': ['原料', '主成分', 'API'],
            '添加剤': ['賦形剤', '添加物'],
            '製剤': ['医薬品', '薬剤'],
            '試験': ['検査', 'テスト', '評価'],
            '規格': ['基準', '標準', 'スペック'],
            '工程': ['プロセス', '過程', '段階'],
        }
        
        for main_term, related_terms in domain_relations.items():
            if main_term in candidates:
                for related in related_terms:
                    if related in candidates:
                        synonyms[main_term].add(related)
                        synonyms[related].add(main_term)
        
        # setをlistに変換
        return {k: list(v) for k, v in synonyms.items() if v}
    
    @staticmethod
    def create_synonym_hints(synonyms: Dict[str, List[str]]) -> str:
        """LLMプロンプト用の関連語ヒントを生成"""
        if not synonyms:
            return "関連語候補は検出されませんでした。"
        
        hints = []
        for term, related in list(synonyms.items())[:20]:  # 上位20件まで
            if related:
                hints.append(f"- {term}: {', '.join(related[:5])}")  # 各用語につき最大5個の関連語
        
        return "検出された関連語候補:\n" + "\n".join(hints)

# ── Candidate Generation Function with Advanced Scoring ─────────────────
def generate_candidates_from_chunk(text: str) -> Tuple[List[str], Dict[str, List[str]]]:
    """改良版：高度なスコアリングによる候補語生成と関連語検出"""
    if not text.strip():
        return [], {}
    
    try:
        # 各呼び出しごとに新しいtokenizerインスタンスを作成
        local_tokenizer = dictionary.Dictionary().create()
        
        # Mode.Cで詳細な分かち書き
        tokens = local_tokenizer.tokenize(text, sudachi_mode)
        doc_length = len(tokens)
        
        # 詳細な品詞情報を持つ名詞句を抽出
        noun_phrases = []
        current_phrase = []
        token_positions = []  # 各トークンの文書内位置
        current_pos = 0
        
        for i, token in enumerate(tokens):
            pos_info = token.part_of_speech()
            pos_main = pos_info[0]
            pos_sub = pos_info[1] if len(pos_info) > 1 else None
            
            # 位置情報を記録
            token_positions.append(current_pos)
            current_pos += len(token.surface())
            
            # 品詞詳細による判定
            if pos_main == '名詞':
                # 除外する品詞細分類
                if pos_sub in ['非自立', '代名詞', '数詞', '接尾']:
                    if len(current_phrase) >= 2:
                        noun_phrases.append(current_phrase[:])
                    current_phrase = []
                # 重要な品詞細分類を優先
                elif pos_sub in ['サ変接続', '一般', '固有名詞', '複合']:
                    current_phrase.append({
                        'surface': token.normalized_form(),
                        'pos_sub': pos_sub,
                        'position': i
                    })
                else:
                    # その他の名詞も文脈に応じて含める
                    if current_phrase:
                        current_phrase.append({
                            'surface': token.normalized_form(),
                            'pos_sub': pos_sub,
                            'position': i
                        })
            else:
                if len(current_phrase) >= 1:
                    noun_phrases.append(current_phrase[:])
                current_phrase = []
        
        if current_phrase:
            noun_phrases.append(current_phrase)
        
        # 候補語の詳細情報を収集
        candidates_info = defaultdict(lambda: {
            'freq': 0,
            'positions': [],
            'first_pos': float('inf'),
            'pos_patterns': []
        })
        
        # 単体名詞と複合名詞を候補に追加
        for phrase in noun_phrases:
            surfaces = [t['surface'] for t in phrase]
            pos_patterns = [t['pos_sub'] for t in phrase]
            
            # 単体名詞（2文字以上、ストップワード除外）
            for i, word in enumerate(surfaces):
                if len(word) >= 2 and word not in STOPWORDS:
                    pos_idx = phrase[i]['position']
                    candidates_info[word]['freq'] += 1
                    candidates_info[word]['positions'].append(token_positions[pos_idx] if pos_idx < len(token_positions) else 0)
                    candidates_info[word]['first_pos'] = min(
                        candidates_info[word]['first_pos'],
                        token_positions[pos_idx] if pos_idx < len(token_positions) else 0
                    )
                    candidates_info[word]['pos_patterns'].append([pos_patterns[i]])
            
            # 複合名詞（2語以上の組み合わせ）
            if len(phrase) >= 2:
                for length in range(2, min(len(phrase) + 1, 6)):  # 最大5語まで
                    for i in range(len(phrase) - length + 1):
                        compound = ''.join(surfaces[i:i+length])
                        
                        # ストップワードのみの複合語は除外
                        if compound in STOPWORDS:
                            continue
                        
                        if len(compound) <= 20:  # 20文字以内
                            pos_idx = phrase[i]['position']
                            candidates_info[compound]['freq'] += 1
                            candidates_info[compound]['positions'].append(
                                token_positions[pos_idx] if pos_idx < len(token_positions) else 0
                            )
                            candidates_info[compound]['first_pos'] = min(
                                candidates_info[compound]['first_pos'],
                                token_positions[pos_idx] if pos_idx < len(token_positions) else 0
                            )
                            candidates_info[compound]['pos_patterns'].append(pos_patterns[i:i+length])
        
        # 総合スコアを計算
        scorer = TermScorer()
        scored_candidates = []
        
        # 頻度辞書を作成（C値計算用）
        candidates_freq = {cand: info['freq'] for cand, info in candidates_info.items()}
        c_values = scorer.calculate_c_value(candidates_freq)
        
        for candidate, info in candidates_info.items():
            # 基本スコア（C値）
            c_score = c_values.get(candidate, 0)
            
            # 文書内分布スコア
            dist_score = scorer.calculate_distribution_score(info['positions'], doc_length)
            
            # 初出位置スコア
            pos_score = scorer.calculate_position_score(info['first_pos'], current_pos)
            
            # 共起関係スコア
            cooc_score = scorer.calculate_cooccurrence_score(candidate, [
                [t['surface'] for t in phrase] for phrase in noun_phrases
            ])
            
            # 品詞パターンボーナス（サ変接続や固有名詞を含む複合語を優遇）
            pos_bonus = 0
            for pattern in info['pos_patterns']:
                if 'サ変接続' in pattern:
                    pos_bonus += 0.5
                if '固有名詞' in pattern:
                    pos_bonus += 0.3
            
            # 総合スコア
            total_score = (
                c_score * 1.0 +
                dist_score * 0.3 +
                pos_score * 0.2 +
                cooc_score * 0.4 +
                pos_bonus
            )
            
            # 最低頻度フィルタ（共起スコアが高い場合は例外）
            if info['freq'] >= 2 or cooc_score >= 3.0:
                scored_candidates.append((candidate, total_score))
        
        # スコアでソートして上位を選択
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 階層的フィルタリング（部分文字列の除外）
        final_candidates = []
        seen_substrings = set()
        
        for candidate, score in scored_candidates[:150]:
            # 既に選ばれた長い用語の部分文字列は除外
            is_substring = False
            for seen in seen_substrings:
                if candidate in seen and len(candidate) < len(seen):
                    is_substring = True
                    break
            
            if not is_substring and score > 0:
                final_candidates.append(candidate)
                seen_substrings.add(candidate)
                if len(final_candidates) >= 100:
                    break
        
        # 関連語検出
        synonym_detector = SynonymDetector()
        detected_synonyms = synonym_detector.find_synonyms(
            final_candidates, 
            noun_phrases
        )
        
        logger.debug(f"候補語生成完了: {len(final_candidates)}件, 関連語: {len(detected_synonyms)}グループ")
        return final_candidates, detected_synonyms
        
    except Exception as e:
        logger.error(f"Error in candidate generation: {e}")
        return [], {}

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
    """RAGと関連語検出を含む用語抽出"""
    chunk_text, chunk_id = chunk_data["text"], chunk_data["chunk_id"]
    
    # 候補語と関連語を生成
    candidates, detected_synonyms = await candidate_generation_chain.ainvoke(chunk_text)
    if not candidates: 
        return {"terms": []}
    
    # 類似文脈を検索
    related_contexts = await vector_store.search_similar_chunks(chunk_text[:1000], chunk_id, n_results=3)
    
    # 関連語ヒントを生成
    synonym_detector = SynonymDetector()
    synonym_hints = synonym_detector.create_synonym_hints(detected_synonyms)
    
    # プロンプトデータを構築
    prompt_data = {
        "text": chunk_text[:3000],
        "candidates": "\n".join(candidates),
        "related_contexts": related_contexts,
        "synonym_hints": synonym_hints
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
    """改良版：編集距離を使った高度な重複マージ"""
    from difflib import SequenceMatcher
    
    def similarity_ratio(a: str, b: str) -> float:
        """文字列の類似度を計算（0-1のスコア）"""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()
    
    merged: Dict[str, Dict] = {}
    
    for term_list in term_lists:
        for term in term_list.get("terms", []):
            if not isinstance(term, dict) or not term.get("headword"):
                continue
            
            headword = term["headword"].strip()
            
            # 既存の用語との類似度チェック
            best_match = None
            best_score = 0
            
            for existing_key, existing_term in merged.items():
                existing_headword = existing_term.get("headword", "")
                
                # 編集距離による類似度
                score = similarity_ratio(headword, existing_headword)
                
                # 部分文字列チェック（どちらかが他方を含む場合）
                if headword in existing_headword or existing_headword in headword:
                    score = max(score, 0.9)  # 部分一致は高スコア
                
                if score > 0.85 and score > best_score:  # 85%以上の類似度
                    best_match = existing_key
                    best_score = score
            
            if best_match:
                # 類似語として統合
                existing = merged[best_match]
                
                # より長い見出し語を採用（情報量が多い）
                if len(headword) > len(existing["headword"]):
                    # 短い方を類義語に追加
                    existing["synonyms"] = list(set(
                        existing.get("synonyms", []) + [existing["headword"]]
                    ))
                    existing["headword"] = headword
                else:
                    # 現在の見出し語を類義語に追加
                    if headword != existing["headword"]:
                        existing["synonyms"] = list(set(
                            existing.get("synonyms", []) + [headword]
                        ))
                
                # 類義語をマージ
                existing["synonyms"] = list(set(
                    existing.get("synonyms", []) + term.get("synonyms", [])
                ))
                
                # より良い定義を選択（長い方が詳細）
                if len(term.get("definition", "")) > len(existing.get("definition", "")):
                    existing["definition"] = term.get("definition")
                
                # カテゴリが未設定なら更新
                if not existing.get("category") and term.get("category"):
                    existing["category"] = term.get("category")
            else:
                # 新規用語として追加
                key = headword.lower()
                merged[key] = {
                    "headword": headword,
                    "synonyms": term.get("synonyms", []),
                    "definition": term.get("definition", ""),
                    "category": term.get("category")
                }
    
    # 類義語リストから重複と見出し語自身を除去
    for key, term in merged.items():
        if "synonyms" in term:
            # 見出し語と同じものを除去
            term["synonyms"] = [
                syn for syn in term["synonyms"] 
                if syn and syn != term["headword"]
            ]
            # 重複除去
            term["synonyms"] = list(set(term["synonyms"]))
    
    logger.info(f"Merged to {len(merged)} unique terms")
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
