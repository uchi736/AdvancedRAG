import streamlit as st
import os
import time
from rag_system_enhanced import Config

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
            st.warning("⚠️ System Offline - 設定を確認してください。")

        with st.form("config_form"):
            st.markdown("### 🔑 Azure OpenAI API設定")
            azure_api_key = st.text_input("Azure OpenAI API Key", value=env_defaults["AZURE_OPENAI_API_KEY"], type="password")
            azure_endpoint = st.text_input("Azure OpenAI Endpoint", value=env_defaults["AZURE_OPENAI_ENDPOINT"])
            azure_api_version = st.text_input("Azure OpenAI API Version", value=env_defaults["AZURE_OPENAI_API_VERSION"])
            azure_chat_deployment = st.text_input("Azure Chat Deployment Name", value=env_defaults["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"])
            azure_embedding_deployment = st.text_input("Azure Embedding Deployment Name", value=env_defaults["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"])
            
            st.markdown("### 🤖 Model Identifiers (UI用)")
            emb_opts = ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"]
            llm_opts = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]
            
            current_emb = rag_system.config.embedding_model_identifier if rag_system else env_defaults["EMBEDDING_MODEL_IDENTIFIER"]
            current_llm = rag_system.config.llm_model_identifier if rag_system else env_defaults["LLM_MODEL_IDENTIFIER"]
            
            emb_idx = emb_opts.index(current_emb) if current_emb in emb_opts else 0
            llm_idx = llm_opts.index(current_llm) if current_llm in llm_opts else 0

            embedding_model_id = st.selectbox("Embedding Model Identifier", emb_opts, index=emb_idx)
            llm_model_id = st.selectbox("Language Model Identifier", llm_opts, index=llm_idx)

            st.markdown("### 🔍 Search Settings")
            collection_name = st.text_input("Collection Name", value=rag_system.config.collection_name if rag_system else env_defaults["COLLECTION_NAME"])
            final_k = st.slider("検索結果数 (Final K)", 1, 20, value=rag_system.config.final_k if rag_system else env_defaults["FINAL_K"])

            apply_button = st.form_submit_button("Apply Settings", use_container_width=True)

    if apply_button:
        from state import initialize_rag_system
        base_params = rag_system.config.__dict__.copy() if rag_system else Config().__dict__.copy()
        
        updated_params = {
            "azure_openai_api_key": azure_api_key,
            "azure_openai_endpoint": azure_endpoint,
            "azure_openai_api_version": azure_api_version,
            "azure_openai_chat_deployment_name": azure_chat_deployment,
            "azure_openai_embedding_deployment_name": azure_embedding_deployment,
            "embedding_model_identifier": embedding_model_id,
            "llm_model_identifier": llm_model_id,
            "collection_name": collection_name,
            "final_k": int(final_k),
            "openai_api_key": None,
        }
        
        final_params = {**base_params, **updated_params}

        try:
            new_config = Config(**final_params)
            with st.spinner("設定を適用し、システムを再初期化しています..."):
                if "rag_system" in st.session_state:
                    del st.session_state["rag_system"]
                st.cache_resource.clear()
                st.session_state.rag_system = initialize_rag_system(new_config)
            st.success("✅ 設定が正常に適用され、システムが再初期化されました。")
            time.sleep(1)
            st.rerun()
        except Exception as e:
            st.error(f"設定適用エラー: {type(e).__name__} - {e}")

def render_langsmith_info():
    """Renders LangSmith tracing info in the sidebar."""
    langsmith_api_key = os.getenv("LANGCHAIN_API_KEY")
    langsmith_project = os.getenv("LANGCHAIN_PROJECT")
    if langsmith_api_key:
        st.sidebar.success(f"ιχ LangSmith Tracing: ENABLED{' (Project: ' + langsmith_project + ')' if langsmith_project else ''}")
    else:
        st.sidebar.info("ιχ LangSmith Tracing: DISABLED (環境変数を設定してください)")
