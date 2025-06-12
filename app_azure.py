"""streamlit_rag_ui_hybrid.py – Hybrid Modern RAG System with Text-to-SQL
=======================================================
チャット画面はChatGPT風、その他は洗練されたモダンデザイン
用語辞書タブを追加 (Added Term Dictionary Tab)
Golden-Retriever機能を追加 (Added Golden-Retriever Feature)

起動: streamlit run streamlit_rag_ui_hybrid.py
(Launch: streamlit run streamlit_rag_ui_hybrid.py)
"""
from __future__ import annotations

import streamlit as st

# ── Page Configuration (最優先で呼び出し) ─────────────────────────────────
st.set_page_config(
    page_title="RAG System • Document Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# その他のインポート（set_page_configの後）
import os
import json
import tempfile
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import uuid

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np

from langchain_core.runnables import RunnableConfig

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ── Environment & Configuration ────────────────────────────────────────────
load_dotenv()
ENV_DEFAULTS = {
    # Azure OpenAI Settings
    "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY", ""),
    "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT", ""),
    "AZURE_OPENAI_API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME": os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", ""),
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", ""),
    # Model names
    "EMBEDDING_MODEL_IDENTIFIER": os.getenv("EMBEDDING_MODEL_IDENTIFIER", "text-embedding-ada-002"),
    "LLM_MODEL_IDENTIFIER": os.getenv("LLM_MODEL_IDENTIFIER", "gpt-4o"),
    "COLLECTION_NAME": os.getenv("COLLECTION_NAME", "documents"),
    "FINAL_K": int(os.getenv("FINAL_K", 5)),
    "ENABLE_JARGON_EXTRACTION": os.getenv("ENABLE_JARGON_EXTRACTION", "true").lower() == "true",
}

# PostgreSQL URL for term dictionary
PG_URL = os.getenv("PG_URL", "")
JARGON_TABLE_NAME = os.getenv("JARGON_TABLE_NAME", "jargon_dictionary")

# ── RAG System Import ─────────────────────────────────────────────────────
try:
    from rag_system_enhanced import Config, RAGSystem
except ModuleNotFoundError:
    st.error("❌ rag_system_enhanced.py が見つかりません。アプリケーションを起動できません。")
    st.stop()
except ImportError as e:
    st.error(f"❌ RAGシステムのインポート中にエラーが発生しました: {e}")
    st.stop()

# ── Hybrid CSS Design ─────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    :root {
        /* ダークテーマカラー (Dark theme colors) */
        --bg-primary: #0a0a0a; --bg-secondary: #141414; --bg-tertiary: #1a1a1a;
        --surface: #242424; --surface-hover: #2a2a2a; --border: #333333;
        /* ChatGPT風カラー（チャット部分用） (ChatGPT-style colors (for chat part)) */
        --chat-bg: #343541; --sidebar-bg: #202123; --user-msg-bg: #343541;
        --ai-msg-bg: #444654; --chat-border: #4e4f60;
        /* テキストカラー (Text colors) */
        --text-primary: #ffffff; --text-secondary: #b3b3b3; --text-tertiary: #808080;
        /* アクセントカラー (Accent colors) */
        --accent: #7c3aed; --accent-hover: #8b5cf6; --accent-light: rgba(124, 58, 237, 0.15);
        --accent-green: #10a37f;
        /* ステータスカラー (Status colors) */
        --success: #10b981; --error: #ef4444; --warning: #f59e0b; --info: #3b82f6;
    }
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    .stApp { background: var(--bg-primary); color: var(--text-primary); }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: var(--bg-secondary); }
    ::-webkit-scrollbar-thumb { background: #555; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #666; }

    /* ヘッダーの修正 (Header correction) */
    .main-header {
        background: linear-gradient(135deg, var(--accent) 0%, #a855f7 100%);
        padding: 0.1rem 1rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(124, 58, 237, 0.3);
        max-width: 100%;
        margin-left: auto;
        margin-right: auto;
    }
    .header-title { font-size: 2.5rem; font-weight: 700; margin: 0; letter-spacing: -1px; }
    .header-subtitle { font-size: 1.1rem; opacity: 0.9; margin-top: 0.5rem; }
    .card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem; }
    .chat-welcome { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 300px; text-align: center; margin-top: -50px; }
    .chat-welcome h2 { color: var(--text-primary); font-size: 2rem; margin-bottom: 1rem; }
    .initial-input-container { margin-top: -100px; width: 100%; max-width: 700px; margin-left: auto; margin-right: auto; }
    .messages-area { padding: 20px 0; min-height: 400px; max-height: calc(100vh - 400px); overflow-y: auto; }
    .message-row { display: flex; padding: 16px 20px; gap: 16px; margin-bottom: 8px; } .user-message-row { background-color: var(--user-msg-bg); } .ai-message-row { background-color: var(--ai-msg-bg); }
    .avatar { width: 36px; height: 36px; border-radius: 4px; display: flex; align-items: center; justify-content: center; font-size: 14px; font-weight: 600; flex-shrink: 0; }
    .user-avatar { background-color: #5436DA; color: white; } .ai-avatar { background-color: var(--accent-green); color: white; }
    .message-content { color: var(--text-primary); line-height: 1.6; flex: 1; } .message-content p { margin: 0; }
    .chat-input-area { border-top: 1px solid var(--chat-border); padding: 20px; background-color: var(--chat-bg); border-radius: 0 0 12px 12px; }
    .source-container { background: var(--bg-secondary); border-radius: 12px; padding: 1.5rem; border: 1px solid var(--border); margin-top: 1rem; }
    .source-item { background: var(--surface); border-radius: 8px; padding: 1rem; margin-bottom: 0.75rem; border-left: 3px solid var(--accent); cursor: pointer; transition: all 0.2s ease; }
    .source-item:hover { transform: translateX(4px); background: var(--surface-hover); } .source-title { font-weight: 600; color: var(--text-primary); margin-bottom: 0.5rem; }
    .source-excerpt { font-size: 0.875rem; color: var(--text-secondary); line-height: 1.5; }
    .full-text-container { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1rem; max-height: 300px; overflow-y: auto; margin-top: 0.5rem; font-size: 0.875rem; line-height: 1.6; color: var(--text-primary); }
    .stat-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; text-align: center; transition: transform 0.2s ease; } .stat-card:hover { transform: translateY(-2px); }
    .stat-number { font-size: 2rem; font-weight: 700; color: var(--accent); } .stat-label { color: var(--text-secondary); font-size: 0.875rem; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 0.5rem; }
    .stButton > button { background: var(--accent); color: white; border: none; border-radius: 8px; padding: 0.75rem 1.5rem; font-weight: 500; transition: all 0.2s ease; } .stButton > button:hover { background: var(--accent-hover); transform: translateY(-1px); }
    .stTextInput > div > div > input, .stTextArea > div > div > textarea { background: var(--surface); border: 1px solid var(--border); color: var(--text-primary); border-radius: 8px; }
    .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus { border-color: var(--accent); box-shadow: 0 0 0 2px var(--accent-light); }
    .stFormLabel { color: var(--text-primary) !important; font-weight: 500; }
    .stTextInput input::placeholder, .stTextArea textarea::placeholder { color: var(--text-secondary) !important; }
    .stTextInput input, .stTextArea textarea, .stSelectbox > div > div > div[data-baseweb="select"] > div { color: var(--text-primary) !important; font-size: 1rem !important; }
    .stSelectbox > div > div > div { background: var(--surface); border: 1px solid var(--border); color: var(--text-primary); }
    .stTabs [data-baseweb="tab-list"] { background: transparent; gap: 0.5rem; }
    .stTabs [data-baseweb="tab"] { background: var(--surface); color: var(--text-secondary); border: 1px solid var(--border); border-radius: 8px; padding: 0.75rem 1.5rem; font-weight: 500; }
    .stTabs [aria-selected="true"] { background: var(--accent); color: white; border-color: var(--accent); }
    .stFileUploader > div { background: var(--surface); border: 2px dashed var(--border); border-radius: 12px; } .stFileUploader > div:hover { border-color: var(--accent); background: var(--surface-hover); }
    .stProgress > div > div > div > div { background: var(--accent); } div[data-testid="metric-container"] { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1rem; }
    .css-1d391kg { background: var(--bg-secondary); } .stAlert { background: var(--surface); color: var(--text-primary); border: 1px solid var(--border); }
    
    /* 用語辞書テーブル用のスタイル */
    .term-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 0.8rem;
        transition: all 0.2s ease;
    }
    .term-card:hover {
        transform: translateX(2px);
        border-left: 3px solid var(--accent);
    }
    .term-headword {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--accent);
        margin-bottom: 0.5rem;
    }
    .term-definition {
        color: var(--text-primary);
        line-height: 1.5;
        margin-bottom: 0.5rem;
    }
    .term-meta {
        color: var(--text-secondary);
        font-size: 0.875rem;
    }
    .term-sources {
        color: var(--text-tertiary);
        font-size: 0.8rem;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Helper Functions ──────────────────────────────────────────────────────
def _persist_uploaded_file(uploaded_file) -> Path:
    if uploaded_file is None:
        raise ValueError("Uploaded file cannot be None")
    tmp_dir = Path(tempfile.gettempdir()) / "rag_uploads"
    tmp_dir.mkdir(exist_ok=True)
    tmp_path = tmp_dir / uploaded_file.name
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return tmp_path

@st.cache_resource(show_spinner=False)
def initialize_rag_system(config_obj: Config) -> RAGSystem:
    return RAGSystem(config_obj)

def get_collection_statistics(rag: RAGSystem) -> Dict[str, Any]:
    if not rag:
        return {"documents": 0, "chunks": 0, "collection_name": "N/A"}
    try:
        engine = create_engine(rag.connection_string)
        with engine.connect() as conn:
            query = text("SELECT COUNT(DISTINCT document_id) AS num_documents, COUNT(*) AS num_chunks FROM document_chunks WHERE collection_name = :collection")
            result = conn.execute(query, {"collection": rag.config.collection_name}).first()
        return {
            "documents": result.num_documents if result else 0,
            "chunks": result.num_chunks if result else 0,
            "collection_name": rag.config.collection_name
        }
    except Exception as e:
        st.error(f"統計情報の取得に失敗: {e}")
        return {"documents": 0, "chunks": 0, "collection_name": rag.config.collection_name if rag else "N/A"}

def get_documents_dataframe(rag: RAGSystem) -> pd.DataFrame:
    if not rag:
        return pd.DataFrame()
    try:
        engine = create_engine(rag.connection_string)
        with engine.connect() as conn:
            query = text("SELECT document_id, COUNT(*) as chunk_count, MAX(created_at) as last_updated FROM document_chunks WHERE collection_name = :collection GROUP BY document_id ORDER BY last_updated DESC")
            result = conn.execute(query, {"collection": rag.config.collection_name})
            df = pd.DataFrame(result.fetchall(), columns=["Document ID", "Chunks", "Last Updated"])
        if not df.empty and "Last Updated" in df.columns:
            df["Last Updated"] = pd.to_datetime(df["Last Updated"]).dt.strftime("%Y-%m-%d %H:%M")
        return df
    except Exception as e:
        st.error(f"登録済みドキュメントリストの取得に失敗: {e}")
        return pd.DataFrame()

def get_query_history_data(days: int = 30) -> pd.DataFrame:
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    queries = [20 + int(10 * abs(np.sin(i / 5.0))) + np.random.randint(-3, 4) for i in range(days)]
    queries = [max(0, q) for q in queries]
    return pd.DataFrame({'Date': dates, 'Queries': queries})

def render_simple_chart(df: pd.DataFrame):
    """簡単なチャート描画"""
    try:
        if df.empty:
            st.info("チャートを描画するデータがありません。")
            return

        if not PLOTLY_AVAILABLE:
            st.warning("Plotlyライブラリがインストールされていないため、チャートを表示できません。`pip install plotly plotly-express`でインストールしてください。")
            return

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
            st.info("数値型の列がないため、チャートを描画できません。")
            return

        categorical_cols = df.select_dtypes(include=['object', 'category', 'datetime64']).columns.tolist()

        chart_type_options = ["なし"]
        if len(df.columns) >= 2 and categorical_cols and numeric_cols:
            chart_type_options.append("棒グラフ")
        if numeric_cols:
            chart_type_options.append("折れ線グラフ")
        if len(numeric_cols) >= 2:
            chart_type_options.append("散布図")

        if len(chart_type_options) == 1:
            st.info("適切なデータ形式ではないため、チャートタイプを選択できません。")
            return

        chart_type = st.selectbox("可視化タイプを選択:", chart_type_options, key=f"sql_chart_type_selector_{df.shape[0]}_{df.shape[1]}")

        if chart_type == "棒グラフ":
            if categorical_cols and numeric_cols:
                x_col_bar = st.selectbox("X軸 (カテゴリ/日付)", categorical_cols, key=f"bar_x_sql_{df.shape[0]}")
                y_col_bar = st.selectbox("Y軸 (数値)", numeric_cols, key=f"bar_y_sql_{df.shape[0]}")
                if x_col_bar and y_col_bar:
                    fig = px.bar(df.head(25), x=x_col_bar, y=y_col_bar, title=f"{y_col_bar} by {x_col_bar}")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("棒グラフにはカテゴリ列と数値列が必要です。")

        elif chart_type == "折れ線グラフ":
            y_cols_line = st.multiselect("Y軸 (数値 - 複数選択可)", numeric_cols, default=numeric_cols[0] if numeric_cols else None, key=f"line_y_sql_{df.shape[0]}")
            date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            x_col_line_options = ["(インデックス)"] + categorical_cols
            
            chosen_x_col = None
            if date_cols:
                x_col_line_options = ["(インデックス)"] + date_cols + [c for c in categorical_cols if c not in date_cols]
                chosen_x_col = date_cols[0]
            elif categorical_cols:
                chosen_x_col = categorical_cols[0]

            x_col_line = st.selectbox("X軸", x_col_line_options, index=x_col_line_options.index(chosen_x_col) if chosen_x_col and chosen_x_col in x_col_line_options else 0, key=f"line_x_sql_{df.shape[0]}")

            if y_cols_line:
                title_ys = ", ".join(y_cols_line)
                if x_col_line and x_col_line != "(インデックス)":
                    fig = px.line(df.head(100), x=x_col_line, y=y_cols_line, title=f"{title_ys} over {x_col_line}", markers=True)
                else:
                    fig = px.line(df.head(100), y=y_cols_line, title=f"{title_ys} Trend", markers=True)
                st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "散布図":
            if len(numeric_cols) >= 2:
                x_col_scatter = st.selectbox("X軸 (数値)", numeric_cols, key=f"scatter_x_sql_{df.shape[0]}")
                y_col_scatter = st.selectbox("Y軸 (数値)", [nc for nc in numeric_cols if nc != x_col_scatter], key=f"scatter_y_sql_{df.shape[0]}")
                color_col_scatter_options = ["なし"] + categorical_cols + [nc for nc in numeric_cols if nc != x_col_scatter and nc != y_col_scatter]
                color_col_scatter = st.selectbox("色分け (任意)", color_col_scatter_options, key=f"scatter_color_sql_{df.shape[0]}")

                if x_col_scatter and y_col_scatter:
                    fig = px.scatter(
                        df.head(500),
                        x=x_col_scatter, 
                        y=y_col_scatter, 
                        color=color_col_scatter if color_col_scatter != "なし" else None,
                        title=f"{y_col_scatter} vs {x_col_scatter}" + (f" by {color_col_scatter}" if color_col_scatter != "なし" else "")
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("散布図には少なくとも2つの数値列が必要です。")

    except Exception as e:
        st.error(f"チャート描画エラー: {type(e).__name__} - {e}")

def render_sql_result_in_chat(sql_details_dict: Dict[str, Any]):
    """チャット内でのSQL関連情報表示"""
    if not sql_details_dict or not isinstance(sql_details_dict, dict):
        st.warning("チャット表示用のSQL詳細情報がありません。")
        return

    with st.expander("🔍 実行されたSQL (チャット内)", expanded=False):
        st.code(sql_details_dict.get("generated_sql", "SQLが生成されませんでした。"), language="sql")

    results_data_preview = sql_details_dict.get("results_preview")
    if results_data_preview and isinstance(results_data_preview, list) and len(results_data_preview) > 0:
        with st.expander("📊 SQL実行結果プレビュー (チャット内)", expanded=False):
            try:
                df_chat_preview = pd.DataFrame(results_data_preview)
                st.dataframe(df_chat_preview, use_container_width=True, height = min(300, (len(df_chat_preview) + 1) * 35 + 3))
                
                total_fetched = sql_details_dict.get("row_count_fetched", 0)
                preview_count = len(results_data_preview)
                if total_fetched > preview_count:
                    st.caption(f"結果の最初の{preview_count}件を表示（全{total_fetched}件取得）。")
                elif total_fetched > 0:
                    st.caption(f"全{total_fetched}件の結果を表示。")
            except Exception as e:
                st.error(f"チャット内でのSQL結果プレビュー表示エラー: {e}")
    elif sql_details_dict.get("success"):
        with st.expander("📊 SQL実行結果プレビュー (チャット内)", expanded=False):
            st.info("SQLクエリは成功しましたが、該当するデータはありませんでした。")

# ── 用語辞書関連の関数 ──────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def load_terms_from_db(keyword: str = "") -> pd.DataFrame:
    """PostgreSQLから用語辞書を読み込む"""
    if not PG_URL:
        return pd.DataFrame()
    
    try:
        engine = create_engine(PG_URL)
        
        # テーブルの存在確認
        with engine.connect() as conn:
            check_table = text(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = '{JARGON_TABLE_NAME}'
                );
            """)
            table_exists = conn.execute(check_table).scalar()
            
            if not table_exists:
                return pd.DataFrame()
        
        # 用語データの取得
        if keyword:
            query = f"""
                SELECT term, definition, domain, aliases, related_terms, confidence_score, updated_at
                FROM {JARGON_TABLE_NAME}
                WHERE term ILIKE :keyword 
                   OR definition ILIKE :keyword
                   OR EXISTS (
                       SELECT 1 FROM unnest(aliases) AS s 
                       WHERE s ILIKE :keyword
                   )
                ORDER BY term
            """
            params = {"keyword": f"%{keyword}%"}
        else:
            query = f"""
                SELECT term, definition, domain, aliases, related_terms, confidence_score, updated_at
                FROM {JARGON_TABLE_NAME}
                ORDER BY term
            """
            params = {}
        
        df = pd.read_sql(query, engine, params=params)
        
        if not df.empty and "updated_at" in df.columns:
            df["updated_at"] = pd.to_datetime(df["updated_at"]).dt.strftime("%Y-%m-%d %H:%M")
        
        return df
        
    except Exception as e:
        st.error(f"用語辞書の読み込みエラー: {e}")
        return pd.DataFrame()

def render_term_card(term_data: pd.Series):
    """用語カードのレンダリング"""
    st.markdown(f"""
    <div class="term-card">
        <div class="term-headword">{term_data['term']}</div>
        <div class="term-definition">{term_data['definition']}</div>
        <div class="term-meta">
            <strong>分野:</strong> {term_data.get('domain', 'N/A')} | 
            <strong>信頼度:</strong> {term_data.get('confidence_score', 1.0):.2f}
        </div>
        <div class="term-meta">
            <strong>類義語:</strong> {', '.join(term_data['aliases']) if term_data['aliases'] else 'なし'}
        </div>
        <div class="term-meta">
            <strong>関連語:</strong> {', '.join(term_data['related_terms']) if term_data['related_terms'] else 'なし'}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Initialize Session State ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_sources" not in st.session_state:
    st.session_state.current_sources = []
if "last_query_expansion" not in st.session_state:
    st.session_state.last_query_expansion = {}
if "last_golden_retriever" not in st.session_state:
    st.session_state.last_golden_retriever = {}
if "use_query_expansion" not in st.session_state:
    st.session_state.use_query_expansion = False
if "use_rag_fusion" not in st.session_state:
    st.session_state.use_rag_fusion = False
if "use_jargon_augmentation" not in st.session_state:
    st.session_state.use_jargon_augmentation = ENV_DEFAULTS["ENABLE_JARGON_EXTRACTION"]
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "rag_system" not in st.session_state:
    if ENV_DEFAULTS["AZURE_OPENAI_API_KEY"] and ENV_DEFAULTS["AZURE_OPENAI_ENDPOINT"] and \
       ENV_DEFAULTS["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] and ENV_DEFAULTS["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"]:
        try:
            app_config = Config(
                azure_openai_api_key=ENV_DEFAULTS["AZURE_OPENAI_API_KEY"],
                azure_openai_endpoint=ENV_DEFAULTS["AZURE_OPENAI_ENDPOINT"],
                azure_openai_api_version=ENV_DEFAULTS["AZURE_OPENAI_API_VERSION"],
                azure_openai_chat_deployment_name=ENV_DEFAULTS["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
                azure_openai_embedding_deployment_name=ENV_DEFAULTS["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"],
                db_password=os.getenv("DB_PASSWORD", "postgres"),
                llm_model_identifier=ENV_DEFAULTS["LLM_MODEL_IDENTIFIER"],
                embedding_model_identifier=ENV_DEFAULTS["EMBEDDING_MODEL_IDENTIFIER"],
                collection_name=ENV_DEFAULTS["COLLECTION_NAME"],
                final_k=ENV_DEFAULTS["FINAL_K"],
                enable_jargon_extraction=st.session_state.use_jargon_augmentation
            )
            st.session_state.rag_system = initialize_rag_system(app_config)
            st.toast("✅ RAGシステムがAzure OpenAIで正常に初期化されました", icon="🎉")
        except Exception as e:
            st.error(f"Azure RAGシステムの初期化中にエラーが発生しました: {type(e).__name__} - {e}")
            st.warning("""
### 🔧 Azure OpenAI 接続エラーの解決方法 (一般的な例)

1.  **.envファイルまたは環境変数を確認してください**:
    `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION`,
    `AZURE_OPENAI_CHAT_DEPLOYMENT_NAME`, `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME`
    が正しく設定されているか確認してください。
2.  **デプロイメント名**: Azureポータルで設定したチャットモデルと埋め込みモデルのデプロイメント名が正確であることを確認してください。
3.  **エンドポイントとAPIバージョン**: エンドポイントのURLとAPIバージョンが正しいか確認してください。
4.  **ネットワーク接続**: Azure OpenAIサービスへのネットワークアクセスが可能であることを確認してください（ファイアウォール、プロキシ設定など）。
            """)
            st.session_state.rag_system = None
    else:
        st.warning("Azure OpenAIのAPIキーと関連設定がされていません。チャット機能を利用できません。サイドバーから設定してください。")
        st.session_state.rag_system = None

# ── Main Header & Langsmith Info ───────────────────────────────────────────
st.markdown("""<div class="main-header"><h1 class="header-title">iRAG</h1><p class="header-subtitle">IHI's Smart Knowledge Base with SQL Analytics</p></div>""", unsafe_allow_html=True)

# LangSmith Tracing Info (Optional)
langsmith_api_key = os.getenv("LANGCHAIN_API_KEY")
langsmith_project = os.getenv("LANGCHAIN_PROJECT")
if langsmith_api_key:
    st.sidebar.success(f"ιχ LangSmith Tracing: ENABLED{' (Project: ' + langsmith_project + ')' if langsmith_project else ''}")
else:
    st.sidebar.info("ιχ LangSmith Tracing: DISABLED (環境変数を設定してください)")

rag: RAGSystem | None = st.session_state.get("rag_system")

# ── Sidebar Configuration ─────────────────────────────────────────────────
# --- Term Dictionary Extraction ---
with st.sidebar.expander("📚 用語辞書生成", expanded=False):
    st.markdown("専門用語・類義語辞書を PostgreSQL + pgvector に保存します。")
    input_dir = st.text_input("入力フォルダ", value="./docs", key="term_input_dir")
    output_json = st.text_input("出力 JSON パス", value="./output/terms.json", key="term_output_json")
    if st.button("🚀 抽出実行", key="run_term_dict"):
        if rag is None:
            st.error("RAGシステムが初期化されていません。")
        else:
            with st.spinner("用語抽出中..."):
                try:
                    rag.extract_terms(input_dir, output_json)
                    st.success(f"辞書を生成しました ✔️ → {output_json}")
                except Exception as e:
                    st.error(f"用語抽出エラー: {e}")

with st.sidebar:
    st.markdown("<h2 style='color: var(--text-primary);'>⚙️ Configuration</h2>", unsafe_allow_html=True)
    if rag:
        service_type = "Azure OpenAI"
        st.success(f"✅ System Online ({service_type}) - Collection: **{rag.config.collection_name}**")
    else:
        st.warning("⚠️ System Offline - Azure OpenAI APIキーまたはDB設定を確認してください。")

    with st.form("config_form"):
        st.markdown("### 🔑 Azure OpenAI API設定")
        azure_api_key_input = st.text_input(
            "Azure OpenAI API Key",
            value=ENV_DEFAULTS["AZURE_OPENAI_API_KEY"] or "",
            type="password",
            help="Azure OpenAI APIキーを入力してください。"
        )
        azure_endpoint_input = st.text_input(
            "Azure OpenAI Endpoint",
            value=ENV_DEFAULTS["AZURE_OPENAI_ENDPOINT"] or "",
            help="Azure OpenAIのエンドポイントURLを入力してください。"
        )
        azure_api_version_input = st.text_input(
            "Azure OpenAI API Version",
            value=ENV_DEFAULTS["AZURE_OPENAI_API_VERSION"] or "2024-02-01",
            help="Azure OpenAIのAPIバージョンを入力してください (例: 2024-02-01)。"
        )
        azure_chat_deployment_input = st.text_input(
            "Azure Chat Deployment Name",
            value=ENV_DEFAULTS["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] or "",
            help="Azure OpenAIのチャットモデルのデプロイメント名。"
        )
        azure_embedding_deployment_input = st.text_input(
            "Azure Embedding Deployment Name",
            value=ENV_DEFAULTS["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"] or "",
            help="Azure OpenAIの埋め込みモデルのデプロイメント名。"
        )
        
        st.markdown("### 🤖 Model Identifiers (UI用)")
        embedding_model_options = ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"]
        llm_model_options = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]

        current_emb_model_id = rag.config.embedding_model_identifier if rag and hasattr(rag.config, 'embedding_model_identifier') else ENV_DEFAULTS["EMBEDDING_MODEL_IDENTIFIER"]
        current_llm_model_id = rag.config.llm_model_identifier if rag and hasattr(rag.config, 'llm_model_identifier') else ENV_DEFAULTS["LLM_MODEL_IDENTIFIER"]
        current_collection_name = rag.config.collection_name if rag and hasattr(rag.config, 'collection_name') else ENV_DEFAULTS["COLLECTION_NAME"]
        current_final_k = rag.config.final_k if rag and hasattr(rag.config, 'final_k') else ENV_DEFAULTS["FINAL_K"]
        
        embedding_model_idx = embedding_model_options.index(current_emb_model_id) if current_emb_model_id in embedding_model_options else 0
        llm_model_idx = llm_model_options.index(current_llm_model_id) if current_llm_model_id in llm_model_options else 0

        embedding_model_id_sb = st.selectbox("Embedding Model Identifier", embedding_model_options, index=embedding_model_idx)
        llm_model_id_sb = st.selectbox("Language Model Identifier", llm_model_options, index=llm_model_idx)

        st.markdown("### 🔍 Search Settings")
        collection_name_ti = st.text_input("Collection Name", value=current_collection_name)
        final_k_sl = st.slider("検索結果数 (Final K)", 1, 20, current_final_k, help="LLMに渡す最終的なチャンク数")

        apply_button = st.form_submit_button("Apply Settings", use_container_width=True)

if apply_button:
    base_config_params = {}
    if rag and hasattr(rag, 'config'):
        base_config_params = rag.config.__dict__.copy()
    else:
        try:
            base_config_params = Config().__dict__.copy()
        except Exception:
            pass

    updated_params_from_form = {
        "azure_openai_api_key": azure_api_key_input or base_config_params.get("azure_openai_api_key", ENV_DEFAULTS["AZURE_OPENAI_API_KEY"]),
        "azure_openai_endpoint": azure_endpoint_input or base_config_params.get("azure_openai_endpoint", ENV_DEFAULTS["AZURE_OPENAI_ENDPOINT"]),
        "azure_openai_api_version": azure_api_version_input or base_config_params.get("azure_openai_api_version", ENV_DEFAULTS["AZURE_OPENAI_API_VERSION"]),
        "azure_openai_chat_deployment_name": azure_chat_deployment_input or base_config_params.get("azure_openai_chat_deployment_name", ENV_DEFAULTS["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"]),
        "azure_openai_embedding_deployment_name": azure_embedding_deployment_input or base_config_params.get("azure_openai_embedding_deployment_name", ENV_DEFAULTS["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"]),
        "embedding_model_identifier": embedding_model_id_sb,
        "llm_model_identifier": llm_model_id_sb,
        "collection_name": collection_name_ti,
        "final_k": int(final_k_sl),
        "openai_api_key": None,
    }
    
    final_config_params = {**base_config_params, **updated_params_from_form}
    final_config_params["openai_api_key"] = None

    try:
        cfg_for_update = Config(**final_config_params)

        with st.spinner("設定を適用し、システムを再初期化しています..."):
            if "rag_system" in st.session_state:
                del st.session_state["rag_system"]
                st.cache_resource.clear()
            
            st.session_state.rag_system = initialize_rag_system(cfg_for_update)
            rag = st.session_state.rag_system
        st.success("✅ 設定が正常に適用され、システムが再初期化されました。")
        time.sleep(1)
        st.rerun()
    except Exception as e:
        st.error(f"設定適用エラー: {type(e).__name__} - {e}")

# ── Main Tabs（用語辞書タブを追加） ─────────────────────────────────────────
tab_titles = ["💬 Chat", "📖 Dictionary", "🗃️ Data", "📁 Documents", "⚙️ Settings"]
tabs = st.tabs(tab_titles)
tab_chat, tab_dictionary, tab_data, tab_documents, tab_settings = tabs

# ── Tab 1: Chat Interface (ChatGPT Style) ────────────────────────────────
with tab_chat:
    if not rag:
        st.info("🔧 RAGシステムが初期化されていません。サイドバーでAzure OpenAI APIキーを設定し、「Apply Settings」をクリックするか、データベース設定を確認してください。")
    else:
        has_messages = len(st.session_state.messages) > 0
        if not has_messages:
            st.markdown("""
            <div class="chat-welcome">
                <h2>Chat with your data</h2>
                <p style="color: var(--text-secondary);">
                    アップロードされたドキュメントから関連情報を検索し、AIが回答します<br>
                    (Searches for relevant information from uploaded documents and AI answers)
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="initial-input-container">', unsafe_allow_html=True)

            st.markdown("<h6>高度なRAG設定:</h6>", unsafe_allow_html=True)
            opt_cols_initial = st.columns(3)
            with opt_cols_initial[0]:
                use_qe_initial = st.checkbox("クエリ拡張", value=st.session_state.use_query_expansion, key="use_qe_initial_v7_tab_chat", help="質問を自動的に拡張して検索 (RRFなし)")
            with opt_cols_initial[1]:
                use_rf_initial = st.checkbox("RAG-Fusion", value=st.session_state.use_rag_fusion, key="use_rf_initial_v7_tab_chat", help="クエリ拡張とRRFで結果を統合")
            with opt_cols_initial[2]:
                use_ja_initial = st.checkbox("専門用語で補強", value=st.session_state.use_jargon_augmentation, key="use_ja_initial_v7_tab_chat", help="専門用語辞書を使って質問を補強")

            user_input_initial = st.text_area("質問を入力:", placeholder="例：このドキュメントの要約を教えてください / 売上上位10件を表示して", height=100, key="initial_input_textarea_v7_tab_chat", label_visibility="collapsed")

            if st.button("送信", type="primary", use_container_width=True, key="initial_send_button_v7_tab_chat"):
                if user_input_initial:
                    st.session_state.messages.append({"role": "user", "content": user_input_initial})
                    st.session_state.use_query_expansion = use_qe_initial
                    st.session_state.use_rag_fusion = use_rf_initial
                    st.session_state.use_jargon_augmentation = use_ja_initial
                    rag.config.enable_jargon_extraction = use_ja_initial

                    with st.spinner("考え中..."):
                        try:
                            trace_config = RunnableConfig(
                                run_name="RAG Initial Query Unified",
                                tags=["streamlit", "rag", "initial_query", st.session_state.session_id],
                                metadata={
                                    "session_id": st.session_state.session_id,
                                    "user_query": user_input_initial,
                                    "use_query_expansion": st.session_state.use_query_expansion,
                                    "use_rag_fusion": st.session_state.use_rag_fusion,
                                    "use_jargon_augmentation": st.session_state.use_jargon_augmentation,
                                    "query_source": "initial_input"
                                }
                            )
                            if hasattr(rag, 'query_unified'):
                                response = rag.query_unified(
                                    user_input_initial,
                                    use_query_expansion=st.session_state.use_query_expansion,
                                    use_rag_fusion=st.session_state.use_rag_fusion,
                                    config=trace_config
                                )
                            else:
                                st.warning("警告: `query_unified` メソッドが見つかりません。標準の `query` メソッドを使用します。SQL自動判別は機能しません。")
                                response = rag.query(
                                    user_input_initial,
                                    use_query_expansion=st.session_state.use_query_expansion,
                                    use_rag_fusion=st.session_state.use_rag_fusion,
                                    config=trace_config
                                )

                            answer = response.get("answer", "申し訳ございません。回答を生成できませんでした。")
                            message_data: Dict[str, Any] = {"role": "assistant", "content": answer}

                            if response.get("query_type") == "sql" and response.get("sql_details"):
                                message_data["sql_details"] = response["sql_details"]
                            elif response.get("sql_details"):
                                message_data["sql_details"] = response["sql_details"]

                            st.session_state.messages.append(message_data)
                            st.session_state.current_sources = response.get("sources", [])
                            st.session_state.last_query_expansion = response.get("query_expansion", {})
                            st.session_state.last_golden_retriever = response.get("golden_retriever", {})
                        except Exception as e:
                            st.error(f"チャット処理中にエラーが発生しました: {type(e).__name__} - {e}")
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            chat_col, source_col = st.columns([2, 1])
            with chat_col:
                message_container_height = 600
                with st.container(height=message_container_height):
                    for idx, message in enumerate(st.session_state.messages):
                        avatar_char = "👤" if message['role'] == 'user' else "🤖"
                        avatar_class = 'user-avatar' if message['role'] == 'user' else 'ai-avatar'
                        avatar_html = f"<div class='avatar {avatar_class}'>{avatar_char}</div>"
                        
                        st.markdown(f"<div class='message-row {'user-message-row' if message['role'] == 'user' else 'ai-message-row'}'>{avatar_html}<div class='message-content'>{message['content']}</div></div>", unsafe_allow_html=True)

                        if message['role'] == 'assistant' and message.get("sql_details"):
                            render_sql_result_in_chat(message["sql_details"])

                st.markdown("---")

                opt_cols_chat = st.columns(3)
                with opt_cols_chat[0]:
                    use_qe_chat = st.checkbox("クエリ拡張", value=st.session_state.use_query_expansion, key="use_qe_chat_continued_v7_tab_chat", help="クエリ拡張 (RRFなし)")
                with opt_cols_chat[1]:
                    use_rf_chat = st.checkbox("RAG-Fusion", value=st.session_state.use_rag_fusion, key="use_rf_chat_continued_v7_tab_chat", help="RAG-Fusion (拡張+RRF)")
                with opt_cols_chat[2]:
                    use_ja_chat = st.checkbox("専門用語で補強", value=st.session_state.use_jargon_augmentation, key="use_ja_chat_continued_v7_tab_chat", help="専門用語辞書を使って質問を補強")

                user_input_continued = st.text_area(
                    "メッセージを入力:",
                    placeholder="続けて質問してください...",
                    label_visibility="collapsed",
                    key=f"chat_input_continued_text_v7_tab_chat_{len(st.session_state.messages)}"
                )

                if st.button("送信", type="primary", key=f"chat_send_button_continued_v7_tab_chat_{len(st.session_state.messages)}", use_container_width=True):
                    if user_input_continued:
                        st.session_state.messages.append({"role": "user", "content": user_input_continued})
                        st.session_state.use_query_expansion = use_qe_chat
                        st.session_state.use_rag_fusion = use_rf_chat
                        st.session_state.use_jargon_augmentation = use_ja_chat
                        rag.config.enable_jargon_extraction = use_ja_chat

                        with st.spinner("考え中..."):
                            try:
                                trace_config_cont = RunnableConfig(
                                    run_name="RAG Chat Query Unified",
                                    tags=["streamlit", "rag", "chat_query", st.session_state.session_id],
                                    metadata={
                                        "session_id": st.session_state.session_id,
                                        "user_query": user_input_continued,
                                        "use_query_expansion": st.session_state.use_query_expansion,
                                        "use_rag_fusion": st.session_state.use_rag_fusion,
                                        "use_jargon_augmentation": st.session_state.use_jargon_augmentation,
                                        "query_source": "continued_chat"
                                    }
                                )
                                if hasattr(rag, 'query_unified'):
                                    response = rag.query_unified(
                                        user_input_continued,
                                        use_query_expansion=st.session_state.use_query_expansion,
                                        use_rag_fusion=st.session_state.use_rag_fusion,
                                        config=trace_config_cont
                                    )
                                else:
                                    st.warning("警告: `query_unified` メソッドが見つかりません。標準の `query` メソッドを使用します。SQL自動判別は機能しません。")
                                    response = rag.query(
                                        user_input_continued,
                                        use_query_expansion=st.session_state.use_query_expansion,
                                        use_rag_fusion=st.session_state.use_rag_fusion,
                                        config=trace_config_cont
                                    )

                                answer = response.get("answer", "申し訳ございません。回答を生成できませんでした。")
                                message_data_cont: Dict[str, Any] = {"role": "assistant", "content": answer}

                                if response.get("query_type") == "sql" and response.get("sql_details"):
                                    message_data_cont["sql_details"] = response["sql_details"]
                                elif response.get("sql_details"):
                                    message_data_cont["sql_details"] = response["sql_details"]

                                st.session_state.messages.append(message_data_cont)
                                st.session_state.current_sources = response.get("sources", [])
                                st.session_state.last_query_expansion = response.get("query_expansion", {})
                                st.session_state.last_golden_retriever = response.get("golden_retriever", {})
                            except Exception as e:
                                st.error(f"チャット処理中にエラーが発生しました: {type(e).__name__} - {e}")
                        st.rerun()

                button_col, info_col = st.columns([1, 3])
                with button_col:
                    if st.button("🗑️ 会話をクリア", use_container_width=True, key="clear_chat_button_v7_tab_chat"):
                        st.session_state.messages = []
                        st.session_state.current_sources = []
                        st.session_state.last_query_expansion = {}
                        st.session_state.last_golden_retriever = {}
                        st.rerun()
                with info_col:
                    last_expansion = st.session_state.get("last_query_expansion", {})
                    last_golden = st.session_state.get("last_golden_retriever", {})
                    
                    if last_golden and last_golden.get("enabled"):
                        with st.expander("⚜️ Golden-Retriever 詳細", expanded=False):
                            st.write(f"**補強されたクエリ:** `{last_golden.get('augmented_query')}`")
                            st.write(f"**抽出された専門用語:** `{', '.join(last_golden.get('extracted_terms', [])) or 'なし'}`")
                    elif last_expansion and last_expansion.get("used", False):
                        with st.expander(f"📋 拡張クエリ詳細 ({last_expansion.get('strategy', 'N/A')})", expanded=False):
                            queries = last_expansion.get("queries", [])
                            st.caption("以下のクエリで検索しました（該当する場合）：")
                            for i, q_text in enumerate(queries):
                                st.write(f"• {'**' if i == 0 else ''}{q_text}{'** (元の質問)' if i == 0 else ''}")
                    elif any(msg.get("sql_details") for msg in st.session_state.messages if msg["role"] == "assistant"):
                        st.caption("SQL分析が実行されました。詳細はメッセージ内の実行結果をご確認ください。")

            with source_col:
                st.markdown("""<div style="position: sticky; top: 1rem;"><h4 style="color: var(--text-primary); margin-bottom: 1rem;">📚 参照ソース (RAG)</h4></div>""", unsafe_allow_html=True)
                if st.session_state.current_sources:
                    for i, source in enumerate(st.session_state.current_sources):
                        doc_id = source.get('metadata', {}).get('document_id', 'Unknown Document')
                        chunk_id_val = source.get('metadata', {}).get('chunk_id', f'N/A_{i}')
                        excerpt = source.get('excerpt', '抜粋なし')
                        
                        expander_key = f"source_expander_chat_{st.session_state.session_id}_{chunk_id_val}_tab_chat"
                        
                        with st.expander(f"ソース {i+1}: {doc_id} (Chunk: {chunk_id_val})", expanded=False):
                            st.markdown(f"""<div class="source-excerpt" style="margin-bottom: 1rem;">{excerpt}</div>""", unsafe_allow_html=True)
                            
                            button_key = f"full_text_btn_chat_{st.session_state.session_id}_{chunk_id_val}_tab_chat"
                            show_full_text_key = f"show_full_chat_{st.session_state.session_id}_{chunk_id_val}_tab_chat"

                            if st.button(f"全文を表示##{chunk_id_val}", key=button_key):
                                st.session_state[show_full_text_key] = not st.session_state.get(show_full_text_key, False)
                            
                            if st.session_state.get(show_full_text_key, False):
                                full_text = source.get('full_content', 'コンテンツなし')
                                st.markdown(f"""<div class="full-text-container">{full_text}</div>""", unsafe_allow_html=True)
                else:
                    st.info("RAG検索が実行されると、参照したソースがここに表示されます。")

# ── Tab 2: Dictionary (用語辞書) ──────────────────────────────────────────
with tab_dictionary:
    st.markdown("### 📖 専門用語辞書")
    st.caption("登録された専門用語・類義語を検索・確認できます。")
    
    if not PG_URL:
        st.warning("⚠️ PG_URLが設定されていないため、用語辞書機能を利用できません。.envファイルでPG_URLを設定してください。")
    else:
        # 検索ボックス
        col1, col2 = st.columns([3, 1])
        with col1:
            search_keyword = st.text_input(
                "🔍 用語検索",
                placeholder="検索したい用語を入力してください...",
                key="term_search_input"
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # スペーサー
            if st.button("🔄 更新", key="refresh_terms", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        
        # 用語データの読み込み
        with st.spinner("用語辞書を読み込み中..."):
            terms_df = load_terms_from_db(search_keyword)
        
        if terms_df.empty:
            if search_keyword:
                st.info(f"「{search_keyword}」に該当する用語が見つかりませんでした。")
            else:
                st.info("まだ用語が登録されていません。サイドバーの「📚 用語辞書生成」から用語を抽出してください。")
        else:
            # 統計情報
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("登録用語数", f"{len(terms_df):,}")
            with col2:
                total_synonyms = sum(len(syn_list) if syn_list else 0 for syn_list in terms_df['aliases'])
                st.metric("類義語総数", f"{total_synonyms:,}")
            with col3:
                avg_confidence = terms_df['confidence_score'].mean()
                st.metric("平均信頼度", f"{avg_confidence:.2f}")
            
            st.markdown("---")
            
            # 表示形式の選択
            view_mode = st.radio(
                "表示形式",
                ["カード形式", "テーブル形式"],
                horizontal=True,
                key="dict_view_mode"
            )
            
            if view_mode == "カード形式":
                # カード形式での表示
                for idx, row in terms_df.iterrows():
                    render_term_card(row)
            else:
                # テーブル形式での表示
                display_df = terms_df.copy()
                # 配列を文字列に変換
                display_df['aliases'] = display_df['aliases'].apply(
                    lambda x: ', '.join(x) if x else ''
                )
                display_df['related_terms'] = display_df['related_terms'].apply(
                    lambda x: ', '.join(x) if x else ''
                )
                
                # カラム名を日本語に
                display_df.columns = ['用語', '定義', '分野', '類義語', '関連語', '信頼度', '更新日時']
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                    height=min(600, (len(display_df) + 1) * 35 + 3)
                )
            
            # CSVダウンロード
            st.markdown("---")
            csv = terms_df.to_csv(index=False)
            st.download_button(
                label="📥 用語辞書をCSVでダウンロード",
                data=csv,
                file_name=f"jargon_dictionary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="csv_download_button"
            )

# ── Tab 3: Data Management (SQL用テーブル) ───────────────────────────────────
with tab_data:
    if not rag:
        st.info("RAGシステムが初期化されていません。サイドバーで設定を確認してください。")
    elif not all(hasattr(rag, attr) for attr in ['create_table_from_file', 'get_data_tables', 'delete_data_table']):
        st.warning("RAGシステムがデータテーブル管理機能をサポートしていません。")
    else:
        st.markdown("### 📊 データファイル管理 (SQL分析用)")
        st.caption("Excel/CSVファイルをアップロードして、SQLで分析可能なテーブルを作成・管理します。")

        uploaded_sql_data_files_list = st.file_uploader(
            "Excel/CSVファイルを選択 (.xlsx, .xls, .csv)",
            accept_multiple_files=True,
            type=["xlsx", "xls", "csv"],
            key="sql_data_file_uploader_v7_tab_data"
        )

        if uploaded_sql_data_files_list:
            if st.button("🚀 選択したファイルからテーブルを作成/更新", type="primary", key="create_table_button_v7_tab_data"):
                progress_bar_sql_data_create = st.progress(0, text="処理開始...")
                status_text_sql_data_create = st.empty()

                for i, file_item_sql in enumerate(uploaded_sql_data_files_list):
                    status_text_sql_data_create.info(f"処理中: {file_item_sql.name}")
                    try:
                        temp_dir_for_sql_data_path = Path(tempfile.gettempdir()) / "rag_sql_data_uploads"
                        temp_dir_for_sql_data_path.mkdir(parents=True, exist_ok=True)
                        temp_file_path_sql = temp_dir_for_sql_data_path / file_item_sql.name
                        with open(temp_file_path_sql, "wb") as f:
                            f.write(file_item_sql.getbuffer())

                        success_create, message_create, schema_info_create = rag.create_table_from_file(str(temp_file_path_sql))
                        if success_create:
                            st.success(f"✅ {file_item_sql.name}: {message_create}")
                            if schema_info_create:
                                st.text("作成/更新されたテーブルスキーマ:")
                                st.code(schema_info_create, language='text')
                        else:
                            st.error(f"❌ {file_item_sql.name}: {message_create}")
                    except Exception as e_upload_sql:
                        st.error(f"❌ {file_item_sql.name} の処理中にエラー: {type(e_upload_sql).__name__} - {e_upload_sql}")
                    finally:
                        progress_bar_sql_data_create.progress((i + 1) / len(uploaded_sql_data_files_list), text=f"完了: {file_item_sql.name}")

                if 'progress_bar_sql_data_create' in locals(): progress_bar_sql_data_create.empty()
                if 'status_text_sql_data_create' in locals(): status_text_sql_data_create.empty()
                st.rerun()

        st.markdown("---")
        st.markdown("### 📋 登録済みデータテーブル")
        tables_list_display = rag.get_data_tables()
        if tables_list_display:
            for table_info_item in tables_list_display:
                table_name_display = table_info_item.get('table_name', '不明なテーブル')
                row_count_display = table_info_item.get('row_count', 'N/A')
                schema_display_text = table_info_item.get('schema', 'スキーマ情報なし')
                with st.expander(f"📊 {table_name_display} ({row_count_display:,}行)"):
                    st.code(schema_display_text, language='text')
                    st.warning(f"**注意:** テーブル '{table_name_display}' を削除すると元に戻せません。")
                    if st.button(f"🗑️ テーブル '{table_name_display}' を削除", key=f"delete_table_{table_name_display}_v7_tab_data", type="secondary"):
                        with st.spinner(f"テーブル '{table_name_display}' を削除中..."):
                            del_success_flag, del_msg_text = rag.delete_data_table(table_name_display)
                        if del_success_flag:
                            st.success(del_msg_text)
                            st.rerun()
                        else:
                            st.error(del_msg_text)
        else:
            st.info("分析可能なデータテーブルはまだありません。上記からファイルをアップロードしてください。")

# ── Tab 4: Document Management ────────────────────────────────────────────
with tab_documents:
    if rag:
        st.markdown("### 📤 ドキュメントアップロード")
        uploaded_docs_list = st.file_uploader(
            "ファイルを選択またはドラッグ&ドロップ (.pdf, .txt, .md, .docx, .doc)",
            accept_multiple_files=True,
            type=["pdf", "txt", "md", "docx", "doc"],
            label_visibility="collapsed",
            key=f"doc_uploader_v7_tab_documents_{rag.config.collection_name if rag else 'default'}"
        )

        if uploaded_docs_list:
            st.markdown(f"#### 選択されたファイル ({len(uploaded_docs_list)})")
            file_info_display_list = []
            for file_item in uploaded_docs_list:
                file_info_display_list.append({
                    "ファイル名": file_item.name,
                    "サイズ": f"{file_item.size / 1024:.1f} KB",
                    "タイプ": file_item.type or "不明"
                })
            st.dataframe(pd.DataFrame(file_info_display_list), use_container_width=True, hide_index=True)

            if st.button("🚀 ドキュメントを処理 (インジェスト)", type="primary", use_container_width=True, key="process_docs_button_v7_tab_documents"):
                progress_bar_docs_ingest = st.progress(0, text="処理開始...")
                status_text_docs_ingest = st.empty()
                try:
                    paths_to_ingest_list = []
                    for i, file_item_to_ingest in enumerate(uploaded_docs_list):
                        status_text_docs_ingest.info(f"一時保存中: {file_item_to_ingest.name}")
                        paths_to_ingest_list.append(str(_persist_uploaded_file(file_item_to_ingest)))
                        progress_bar_docs_ingest.progress((i + 1) / (len(uploaded_docs_list) * 2), text=f"一時保存完了: {file_item_to_ingest.name}")

                    status_text_docs_ingest.info(f"インデックスを構築中... ({len(paths_to_ingest_list)}件のファイル)")
                    rag.ingest_documents(paths_to_ingest_list)
                    progress_bar_docs_ingest.progress(1.0, text="インジェスト完了！")
                    st.success(f"✅ {len(uploaded_docs_list)}個のファイルが正常に処理されました！")
                    time.sleep(1)
                    st.balloons()
                    st.rerun()
                except Exception as e_ingest:
                    st.error(f"ドキュメント処理中にエラーが発生しました: {type(e_ingest).__name__} - {e_ingest}")
                finally:
                    if 'progress_bar_docs_ingest' in locals(): progress_bar_docs_ingest.empty()
                    if 'status_text_docs_ingest' in locals(): status_text_docs_ingest.empty()

        st.markdown("### 📚 登録済みドキュメント")
        docs_df_display = get_documents_dataframe(rag)
        if not docs_df_display.empty:
            st.dataframe(docs_df_display, use_container_width=True, hide_index=True)

            st.markdown("### 🗑️ ドキュメント削除")
            doc_ids_for_deletion_options = ["選択してください..."] + docs_df_display["Document ID"].tolist()
            doc_to_delete_selected = st.selectbox(
                "削除するドキュメントIDを選択:",
                doc_ids_for_deletion_options,
                label_visibility="collapsed",
                key=f"doc_delete_selectbox_v7_tab_documents_{rag.config.collection_name if rag else 'default'}"
            )
            if doc_to_delete_selected != "選択してください...":
                st.warning(f"**警告:** ドキュメント '{doc_to_delete_selected}' を削除すると、関連する全てのチャンクがデータベースとベクトルストアから削除されます。この操作は元に戻せません。")
                if st.button(f"'{doc_to_delete_selected}' を削除実行", type="secondary", key="doc_delete_button_v7_tab_documents"):
                    try:
                        with st.spinner(f"削除中: {doc_to_delete_selected}"):
                            success, message = rag.delete_document_by_id(doc_to_delete_selected)
                        if success:
                            st.success(message)
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(message)
                    except Exception as e_delete:
                        st.error(f"ドキュメント削除中にエラーが発生しました: {type(e_delete).__name__} - {e_delete}")
        else:
            st.info("まだドキュメントが登録されていません。上のセクションからアップロードしてください。")
    else:
        st.info("RAGシステムが初期化されていません。サイドバーで設定を確認してください。")

# ── Tab 5: Settings ───────────────────────────────────────────────────────
with tab_settings:
    st.markdown("### ⚙️ システム詳細設定")
    st.caption("RAGシステムの詳細な設定を行います。変更後は「設定を適用」ボタンをクリックしてください。システムの再初期化が必要な場合があります。")

    temp_default_cfg = Config()

    current_values_dict: Dict[str, Any] = {}
    if rag and hasattr(rag, 'config'):
        current_values_dict = rag.config.__dict__.copy()
    else:
        current_values_dict = temp_default_cfg.__dict__.copy()
        for key, value in ENV_DEFAULTS.items():
            if key.lower() in current_values_dict:
                 current_values_dict[key.lower()] = value
            # Ensure openai_api_key is not carried over if it was in ENV_DEFAULTS
            if key.lower() == "openai_api_key":
                current_values_dict[key.lower()] = None

    for key_from_class in temp_default_cfg.__dict__:
        if key_from_class not in current_values_dict:
            current_values_dict[key_from_class] = getattr(temp_default_cfg, key_from_class)
        # Ensure openai_api_key is explicitly None if not already handled
        if key_from_class == "openai_api_key":
            current_values_dict[key_from_class] = None

    with st.form("detailed_settings_form_v7_tab_settings"):
        col1_settings, col2_settings = st.columns(2)
        with col1_settings:
            st.markdown("#### 🔑 Azure OpenAI 設定")
            form_azure_openai_api_key = st.text_input(
                "Azure OpenAI APIキー",
                value=current_values_dict.get("azure_openai_api_key", "") or "",
                type="password", key="setting_azure_key_v7",
                help="Azure OpenAI APIキー。"
            )
            form_azure_openai_endpoint = st.text_input(
                "Azure OpenAI エンドポイント",
                value=current_values_dict.get("azure_openai_endpoint", "") or "",
                key="setting_azure_endpoint_v7",
                help="Azure OpenAI エンドポイント URL。"
            )
            form_azure_openai_api_version = st.text_input(
                "Azure OpenAI APIバージョン",
                value=current_values_dict.get("azure_openai_api_version", "") or "",
                key="setting_azure_version_v7",
                help="Azure OpenAI APIバージョン (例: 2024-02-01)。"
            )
            form_azure_chat_deployment = st.text_input(
                "Azure チャットデプロイメント名",
                value=current_values_dict.get("azure_openai_chat_deployment_name", "") or "",
                key="setting_azure_chat_deploy_v7",
                help="Azure OpenAI チャットモデルのデプロイメント名。"
            )
            form_azure_embedding_deployment = st.text_input(
                "Azure 埋め込みデプロイメント名",
                value=current_values_dict.get("azure_openai_embedding_deployment_name", "") or "",
                key="setting_azure_embed_deploy_v7",
                help="Azure OpenAI 埋め込みモデルのデプロイメント名。"
            )
            
            st.markdown("#### 🤖 AIモデル識別子 (UI用)")
            emb_opts_form = ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"]
            current_emb_model_id_form = current_values_dict.get("embedding_model_identifier", temp_default_cfg.embedding_model_identifier)
            emb_idx_form = emb_opts_form.index(current_emb_model_id_form) if current_emb_model_id_form in emb_opts_form else 0
            embedding_model_id_form_val = st.selectbox("埋め込みモデル識別子", emb_opts_form, index=emb_idx_form, help="埋め込みモデルのUI表示用識別子", key="setting_emb_model_id_v7")

            llm_opts_form = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]
            current_llm_model_id_form = current_values_dict.get("llm_model_identifier", temp_default_cfg.llm_model_identifier)
            llm_idx_form = llm_opts_form.index(current_llm_model_id_form) if current_llm_model_id_form in llm_opts_form else 0
            llm_model_id_form_val = st.selectbox("言語モデル識別子", llm_opts_form, index=llm_idx_form, help="言語モデルのUI表示用識別子", key="setting_llm_model_id_v7")

            st.markdown("#### 📄 チャンク設定")
            chunk_size_form_val = st.number_input("チャンクサイズ", 100, 5000, int(current_values_dict.get("chunk_size", temp_default_cfg.chunk_size)), 100, help="1つのチャンクの最大文字数", key="setting_chunk_size_v7_tab_settings")
            chunk_overlap_form_val = st.number_input("チャンクオーバーラップ", 0, 1000, int(current_values_dict.get("chunk_overlap", temp_default_cfg.chunk_overlap)), 50, help="隣接するチャンク間で重複する文字数", key="setting_chunk_overlap_v7_tab_settings")

        with col2_settings:
            st.markdown("#### 🔍 検索・RAG設定")
            collection_name_form_val = st.text_input("コレクション名", current_values_dict.get("collection_name", temp_default_cfg.collection_name), help="ドキュメントを格納するコレクションの名前", key="setting_collection_name_v7_tab_settings")
            final_k_form_val = st.slider("最終検索結果数 (Final K)", 1, 20, int(current_values_dict.get("final_k", temp_default_cfg.final_k)), help="LLMに渡す最終的なチャンク数", key="setting_final_k_v7_tab_settings")
            vector_search_k_form_val = st.number_input("ベクトル検索数 (Vector K)", 1, 50, int(current_values_dict.get("vector_search_k", temp_default_cfg.vector_search_k)), help="ベクトル検索で取得する候補数", key="setting_vector_k_v7_tab_settings")
            keyword_search_k_form_val = st.number_input("キーワード検索数 (Keyword K)", 1, 50, int(current_values_dict.get("keyword_search_k", temp_default_cfg.keyword_search_k)), help="キーワード検索で取得する候補数", key="setting_keyword_k_v7_tab_settings")
            rrf_k_for_fusion_form_val = st.number_input("RAG-Fusion用RRF係数 (k)", 1, 100, int(current_values_dict.get("rrf_k_for_fusion", temp_default_cfg.rrf_k_for_fusion)), help="RAG-Fusion時のRRFで使用するk値 (通常60程度)", key="setting_rrf_k_v7_tab_settings")
            
        st.markdown("---")
        st.markdown("#### 🗄️ データベース設定 (変更には注意が必要です)")
        db_col1_settings, db_col2_settings = st.columns(2)
        with db_col1_settings:
            db_host_form_val = st.text_input("DBホスト", current_values_dict.get("db_host", temp_default_cfg.db_host), key="setting_db_host_v7_tab_settings")
            db_name_form_val = st.text_input("DB名", current_values_dict.get("db_name", temp_default_cfg.db_name), key="setting_db_name_v7_tab_settings")
            db_user_form_val = st.text_input("DBユーザー", current_values_dict.get("db_user", temp_default_cfg.db_user), key="setting_db_user_v7_tab_settings")
        with db_col2_settings:
            db_port_form_val = st.text_input("DBポート", str(current_values_dict.get("db_port", temp_default_cfg.db_port)), key="setting_db_port_v7_tab_settings")
            db_password_form_val = st.text_input("DBパスワード", current_values_dict.get("db_password", temp_default_cfg.db_password), type="password", key="setting_db_pass_v7_tab_settings")
            fts_language_options = ["english", "japanese", "simple", "german", "french"]
            current_fts_lang = current_values_dict.get("fts_language", temp_default_cfg.fts_language)
            fts_lang_idx = fts_language_options.index(current_fts_lang) if current_fts_lang in fts_language_options else 0
            fts_language_form_val = st.selectbox("FTS言語", fts_language_options, index=fts_lang_idx, key="setting_fts_lang_v7_tab_settings", help="全文検索インデックスで使用する言語")

        st.markdown("#### 📈 SQL分析設定")
        max_sql_results_form_val = st.number_input(
            "SQL最大取得行数", 10, 10000,
            int(current_values_dict.get("max_sql_results", temp_default_cfg.max_sql_results)), 10,
            help="SQLクエリでデータベースから取得する最大行数。",
            key="setting_max_sql_results_v7_tab_settings"
        )
        max_sql_preview_llm_form_val = st.number_input(
            "SQL結果LLMプレビュー行数", 1, 100,
            int(current_values_dict.get("max_sql_preview_rows_for_llm", temp_default_cfg.max_sql_preview_rows_for_llm)), 1,
            help="SQL実行結果をLLMに渡して要約させる際の最大プレビュー行数。",
            key="setting_max_sql_preview_llm_v7_tab_settings"
        )

        s_col_form, r_col_form = st.columns([3,1])
        apply_settings_button_form = s_col_form.form_submit_button("🔄 設定を適用", type="primary", use_container_width=True)
        reset_settings_button_form = r_col_form.form_submit_button("↩️ デフォルトにリセット", use_container_width=True)

    if apply_settings_button_form:
        try:
            form_values = {
                "azure_openai_api_key": form_azure_openai_api_key,
                "azure_openai_endpoint": form_azure_openai_endpoint,
                "azure_openai_api_version": form_azure_openai_api_version,
                "azure_openai_chat_deployment_name": form_azure_chat_deployment,
                "azure_openai_embedding_deployment_name": form_azure_embedding_deployment,
                "openai_api_key": None, # Explicitly set to None
                "embedding_model_identifier": embedding_model_id_form_val,
                "llm_model_identifier": llm_model_id_form_val,
                "collection_name": collection_name_form_val,
                "final_k": int(final_k_form_val),
                "chunk_size": int(chunk_size_form_val),
                "chunk_overlap": int(chunk_overlap_form_val),
                "vector_search_k": int(vector_search_k_form_val),
                "keyword_search_k": int(keyword_search_k_form_val),
                "db_host": db_host_form_val,
                "db_port": str(db_port_form_val),
                "db_name": db_name_form_val,
                "db_user": db_user_form_val,
                "db_password": db_password_form_val,
                "fts_language": fts_language_form_val,
                "rrf_k_for_fusion": int(rrf_k_for_fusion_form_val),
                "max_sql_results": int(max_sql_results_form_val),
                "max_sql_preview_rows_for_llm": int(max_sql_preview_llm_form_val)
            }
            
            new_app_config_obj = Config(**form_values)

            cfg_changed_flag = False
            if rag and hasattr(rag, 'config'):
                for field_name in new_app_config_obj.__dict__:
                    old_value = getattr(rag.config, field_name, None)
                    new_value = getattr(new_app_config_obj, field_name)
                    if new_value != old_value:
                        cfg_changed_flag = True
                        break
            else:
                cfg_changed_flag = True

            if cfg_changed_flag:
                st.info("設定が変更されました。システムを再初期化します...")

            with st.spinner("設定を適用し、システムを初期化しています..."):
                if "rag_system" in st.session_state:
                    del st.session_state["rag_system"]
                    st.cache_resource.clear()
                
                st.session_state.rag_system = initialize_rag_system(new_app_config_obj)
                rag = st.session_state.rag_system
            st.success("✅ 設定が正常に適用され、システムが初期化されました！")
            time.sleep(1)
            st.rerun()
        except Exception as e_apply_settings:
            st.error(f"❌ 設定の適用中にエラーが発生しました: {type(e_apply_settings).__name__} - {e_apply_settings}")

    if reset_settings_button_form:
        st.info("設定をデフォルト値にリセットし、システムを再初期化します...")
        
        default_config_for_reset_obj = Config()
        
        default_config_for_reset_obj.azure_openai_api_key = ENV_DEFAULTS["AZURE_OPENAI_API_KEY"]
        default_config_for_reset_obj.azure_openai_endpoint = ENV_DEFAULTS["AZURE_OPENAI_ENDPOINT"]
        default_config_for_reset_obj.azure_openai_api_version = ENV_DEFAULTS["AZURE_OPENAI_API_VERSION"]
        default_config_for_reset_obj.azure_openai_chat_deployment_name = ENV_DEFAULTS["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"]
        default_config_for_reset_obj.azure_openai_embedding_deployment_name = ENV_DEFAULTS["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"]
        default_config_for_reset_obj.openai_api_key = None
        default_config_for_reset_obj.embedding_model_identifier = ENV_DEFAULTS["EMBEDDING_MODEL_IDENTIFIER"]
        default_config_for_reset_obj.llm_model_identifier = ENV_DEFAULTS["LLM_MODEL_IDENTIFIER"]
        default_config_for_reset_obj.collection_name = ENV_DEFAULTS["COLLECTION_NAME"]
        default_config_for_reset_obj.final_k = ENV_DEFAULTS["FINAL_K"]

        with st.spinner("デフォルト設定でシステムを初期化しています..."):
            if "rag_system" in st.session_state:
                del st.session_state["rag_system"]
                st.cache_resource.clear()
            st.session_state.rag_system = initialize_rag_system(default_config_for_reset_obj)
            rag = st.session_state.rag_system
        st.success("✅ 設定がデフォルトにリセットされ、システムが初期化されました！")
        time.sleep(1)
        st.rerun()

    st.markdown("---")
    st.markdown("### 📋 現在の有効な設定")
    if rag and hasattr(rag, 'config'):
        config_display_dict = rag.config.__dict__.copy()
        
        sensitive_keys = ["db_password", "openai_api_key", "azure_openai_api_key"]
        for key in sensitive_keys:
            if key in config_display_dict and config_display_dict[key]:
                value = str(config_display_dict[key])
                config_display_dict[key] = f"***{value[-4:]}" if len(value) > 7 else "********"
            elif key == "openai_api_key" and key in config_display_dict :
                 config_display_dict[key] = "None (Fallback Disabled)"

        
        col1_disp_settings, col2_disp_settings = st.columns(2)
        
        items_to_display_list = list(config_display_dict.items())
        mid_point_display = (len(items_to_display_list) + 1) // 2

        with col1_disp_settings:
            for key_disp, value_disp in items_to_display_list[:mid_point_display]:
                st.markdown(f"**{key_disp.replace('_', ' ').capitalize()}:** `{str(value_disp)}`")
        with col2_disp_settings:
            for key_disp, value_disp in items_to_display_list[mid_point_display:]:
                st.markdown(f"**{key_disp.replace('_', ' ').capitalize()}:** `{str(value_disp)}`")
    else:
        st.info("システムが初期化されていません。上記フォームから設定を適用してください。")
