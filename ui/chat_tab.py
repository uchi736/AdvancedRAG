import streamlit as st
import os
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any
from utils.helpers import render_sql_result_in_chat

def render_chat_tab(rag_system):
    """Renders the chat tab."""
    if not rag_system:
        st.info("🔧 RAGシステムが初期化されていません。サイドバーでAzure OpenAI APIキーを設定し、「Apply Settings」をクリックするか、データベース設定を確認してください。")
        return

    has_messages = len(st.session_state.messages) > 0
    if not has_messages:
        _render_initial_chat_view(rag_system)
    else:
        _render_continued_chat_view(rag_system)

def _render_initial_chat_view(rag):
    """Renders the view for a new chat session."""
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
    opt_cols_initial = st.columns(4)
    with opt_cols_initial[0]:
        use_qe_initial = st.checkbox("クエリ拡張", value=st.session_state.use_query_expansion, key="use_qe_initial_v7_tab_chat", help="質問を自動的に拡張して検索 (RRFなし)")
    with opt_cols_initial[1]:
        use_rf_initial = st.checkbox("RAG-Fusion", value=st.session_state.use_rag_fusion, key="use_rf_initial_v7_tab_chat", help="クエリ拡張とRRFで結果を統合")
    with opt_cols_initial[2]:
        use_ja_initial = st.checkbox("専門用語で補強", value=st.session_state.use_jargon_augmentation, key="use_ja_initial_v7_tab_chat", help="専門用語辞書を使って質問を補強")
    with opt_cols_initial[3]:
        use_rr_initial = st.checkbox("LLMリランク", value=st.session_state.use_reranking, key="use_rr_initial_v7_tab_chat", help="LLMで検索結果を並べ替え")

    user_input_initial = st.text_area("質問を入力:", placeholder="例：このドキュメントの要約を教えてください / 売上上位10件を表示して", height=100, key="initial_input_textarea_v7_tab_chat", label_visibility="collapsed")

    if st.button("送信", type="primary", use_container_width=True, key="initial_send_button_v7_tab_chat"):
        if user_input_initial:
            st.session_state.messages.append({"role": "user", "content": user_input_initial})
            st.session_state.use_query_expansion = use_qe_initial
            st.session_state.use_rag_fusion = use_rf_initial
            st.session_state.use_jargon_augmentation = use_ja_initial
            st.session_state.use_reranking = use_rr_initial
            rag.config.enable_jargon_extraction = use_ja_initial
            rag.config.enable_reranking = use_rr_initial
            _handle_query(rag, user_input_initial, "initial_input")
            st.rerun()
            
    st.markdown('</div>', unsafe_allow_html=True)

def _render_continued_chat_view(rag):
    """Renders the view for an ongoing chat session."""
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

        opt_cols_chat = st.columns(4)
        with opt_cols_chat[0]:
            use_qe_chat = st.checkbox("クエリ拡張", value=st.session_state.use_query_expansion, key="use_qe_chat_continued_v7_tab_chat", help="クエリ拡張 (RRFなし)")
        with opt_cols_chat[1]:
            use_rf_chat = st.checkbox("RAG-Fusion", value=st.session_state.use_rag_fusion, key="use_rf_chat_continued_v7_tab_chat", help="RAG-Fusion (拡張+RRF)")
        with opt_cols_chat[2]:
            use_ja_chat = st.checkbox("専門用語で補強", value=st.session_state.use_jargon_augmentation, key="use_ja_chat_continued_v7_tab_chat", help="専門用語辞書を使って質問を補強")
        with opt_cols_chat[3]:
            use_rr_chat = st.checkbox("LLMリランク", value=st.session_state.use_reranking, key="use_rr_chat_continued_v7_tab_chat", help="LLMで検索結果を並べ替え")

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
                st.session_state.use_reranking = use_rr_chat
                rag.config.enable_jargon_extraction = use_ja_chat
                rag.config.enable_reranking = use_rr_chat
                _handle_query(rag, user_input_continued, "continued_chat")
                st.rerun()

        button_col, info_col = st.columns([1, 3])
        with button_col:
            if st.button("🗑️ 会話をクリア", use_container_width=True, key="clear_chat_button_v7_tab_chat"):
                st.session_state.messages = []
                st.session_state.current_sources = []
                st.session_state.last_query_expansion = {}
                st.session_state.last_golden_retriever = {}
                st.session_state.last_reranking = {}
                st.rerun()
        with info_col:
            _render_query_info()

    with source_col:
        _render_sources()

def _handle_query(rag, user_input, query_source):
    """Handles the query logic and updates session state."""
    with st.spinner("考え中..."):
        try:
            trace_config = RunnableConfig(
                run_name=f"RAG Query Unified ({query_source})",
                tags=["streamlit", "rag", query_source, st.session_state.session_id],
                metadata={
                    "session_id": st.session_state.session_id,
                    "user_query": user_input,
                    "use_query_expansion": st.session_state.use_query_expansion,
                    "use_rag_fusion": st.session_state.use_rag_fusion,
                    "use_jargon_augmentation": st.session_state.use_jargon_augmentation,
                    "use_reranking": st.session_state.use_reranking,
                    "query_source": query_source
                }
            )
            if hasattr(rag, 'query_unified'):
                response = rag.query_unified(
                    user_input,
                    use_query_expansion=st.session_state.use_query_expansion,
                    use_rag_fusion=st.session_state.use_rag_fusion,
                    config=trace_config
                )
            else:
                st.warning("警告: `query_unified` メソッドが見つかりません。標準の `query` メソッドを使用します。SQL自動判別は機能しません。")
                response = rag.query(
                    user_input,
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
            st.session_state.last_reranking = response.get("reranking", {})
        except Exception as e:
            st.error(f"チャット処理中にエラーが発生しました: {type(e).__name__} - {e}")

def _render_query_info():
    """Renders information about the last query execution."""
    last_expansion = st.session_state.get("last_query_expansion", {})
    last_golden = st.session_state.get("last_golden_retriever", {})
    last_reranking = st.session_state.get("last_reranking", {})

    if last_reranking and last_reranking.get("used"):
        with st.expander("🔃 LLMリランカー詳細", expanded=False):
            st.write("**元の順序:**")
            st.write(last_reranking.get("original_order", []))
            st.write("**新しい順序:**")
            st.write(last_reranking.get("new_order", []))

    if last_golden and last_golden.get("enabled"):
        with st.expander("⚜️ Golden-Retriever 詳細", expanded=False):
            st.write(f"**補強されたクエリ:** `{last_golden.get('augmented_query')}`")
            st.write(f"**抽出された専門用語:** `{', '.join(last_golden.get('extracted_terms', [])) or 'なし'}`")
    
    if last_expansion and last_expansion.get("used", False):
        with st.expander(f"📋 拡張クエリ詳細 ({last_expansion.get('strategy', 'N/A')})", expanded=False):
            queries = last_expansion.get("queries", [])
            st.caption("以下のクエリで検索しました（該当する場合）：")
            for i, q_text in enumerate(queries):
                st.write(f"• {'**' if i == 0 else ''}{q_text}{'** (元の質問)' if i == 0 else ''}")

    if any(msg.get("sql_details") for msg in st.session_state.messages if msg["role"] == "assistant"):
        st.caption("SQL分析が実行されました。詳細はメッセージ内の実行結果をご確認ください。")

def _render_sources():
    """Renders the source documents for the last response."""
    st.markdown("""<div style="position: sticky; top: 1rem;"><h4 style="color: var(--text-primary); margin-bottom: 1rem;">📚 参照ソース (RAG)</h4></div>""", unsafe_allow_html=True)
    if st.session_state.current_sources:
        for i, source in enumerate(st.session_state.current_sources):
            metadata = source.metadata
            doc_id = metadata.get('document_id', 'Unknown Document')
            chunk_id_val = metadata.get('chunk_id', f'N/A_{i}')
            source_type = metadata.get('type', 'text')

            header_text = f"ソース {i+1}: {doc_id}"
            if source_type == 'image_summary':
                header_text += " (画像)"
            else:
                header_text += f" (Chunk: {chunk_id_val})"

            with st.expander(header_text, expanded=False):
                if source_type == 'image_summary':
                    st.markdown("**画像要約:**")
                    st.markdown(f"<div class='source-excerpt'>{source.page_content}</div>", unsafe_allow_html=True)
                    image_path = metadata.get("original_image_path")
                    if image_path and os.path.exists(image_path):
                        st.image(image_path, caption=f"元画像: {os.path.basename(image_path)}")
                    else:
                        st.warning("画像ファイルが見つかりませんでした。")
                else:
                    # Standard text source
                    excerpt = source.page_content[:200] + "..." if len(source.page_content) > 200 else source.page_content
                    st.markdown(f"""<div class="source-excerpt" style="margin-bottom: 1rem;">{excerpt}</div>""", unsafe_allow_html=True)
                    
                    button_key = f"full_text_btn_chat_{st.session_state.session_id}_{chunk_id_val}_tab_chat"
                    show_full_text_key = f"show_full_chat_{st.session_state.session_id}_{chunk_id_val}_tab_chat"

                    if st.button(f"全文を表示##{chunk_id_val}", key=button_key):
                        st.session_state[show_full_text_key] = not st.session_state.get(show_full_text_key, False)
                    
                    if st.session_state.get(show_full_text_key, False):
                        full_text = source.page_content
                        st.markdown(f"""<div class="full-text-container">{full_text}</div>""", unsafe_allow_html=True)
    else:
        st.info("RAG検索が実行されると、参照したソースがここに表示されます。")
