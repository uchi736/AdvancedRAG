import streamlit as st
import os
import time
from src.core.rag_system import Config

def render_sidebar(rag_system, env_defaults):
    """Renders the sidebar and handles configuration updates."""
    # Term Dictionary Extraction
    with st.sidebar.expander("📚 用語辞書生成", expanded=False):
        st.markdown("専門用語・類義語辞書を PostgreSQL + pgvector に保存します。")
        input_dir = st.text_input("入力フォルダ", value="./docs", key="term_input_dir")
        output_json = st.text_input("出力 JSON パス", value="./output/terms.json", key="term_output_json")
        if st.button("🚀 抽出実行", key="run_term_dict"):
            if rag_system is None:
                st.error("RAGシステムが初期化されていません。")
            else:
                with st.spinner("用語抽出中..."):
                    try:
                        rag_system.extract_terms(input_dir, output_json)
                        st.success(f"辞書を生成しました ✔️ → {output_json}")
                    except Exception as e:
                        st.error(f"用語抽出エラー: {e}")

    with st.sidebar:
        st.markdown("<h2 style='color: var(--text-primary);'>⚙️ Configuration</h2>", unsafe_allow_html=True)
        if rag_system:
            st.success(f"✅ System Online (Azure) - Collection: **{rag_system.config.collection_name}**")
        else:
            st.warning("⚠️ System Offline")
        
        st.info("すべての設定は「詳細設定」タブで行えます。")

        # Add search type selection
        st.session_state.search_type = st.radio(
            "検索タイプを選択",
            ('ハイブリッド検索', 'ベクトル検索'),
            index=0 if st.session_state.get('search_type', 'ハイブリッド検索') == 'ハイブリッド検索' else 1,
            key='search_type_radio'
        )


def render_langsmith_info():
    """Renders LangSmith tracing info in the sidebar."""
    langsmith_api_key = os.getenv("LANGCHAIN_API_KEY")
    langsmith_project = os.getenv("LANGCHAIN_PROJECT")
    if langsmith_api_key:
        st.sidebar.success(f"ιχ LangSmith Tracing: ENABLED{' (Project: ' + langsmith_project + ')' if langsmith_project else ''}")
    else:
        st.sidebar.info("ιχ LangSmith Tracing: DISABLED (環境変数を設定してください)")
