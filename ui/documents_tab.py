import streamlit as st
import pandas as pd
import time
from utils.helpers import _persist_uploaded_file, get_documents_dataframe

def render_documents_tab(rag_system):
    """Renders the document management tab."""
    if not rag_system:
        st.info("RAGシステムが初期化されていません。サイドバーで設定を確認してください。")
        return

    st.markdown("### 📤 ドキュメントアップロード")
    uploaded_docs = st.file_uploader(
        "ファイルを選択またはドラッグ&ドロップ (.pdf, .txt, .md, .docx, .doc)",
        accept_multiple_files=True,
        type=["pdf", "txt", "md", "docx", "doc"],
        label_visibility="collapsed",
        key=f"doc_uploader_v7_tab_documents_{rag_system.config.collection_name}"
    )

    if uploaded_docs:
        st.markdown(f"#### 選択されたファイル ({len(uploaded_docs)})")
        file_info = [{"ファイル名": f.name, "サイズ": f"{f.size / 1024:.1f} KB", "タイプ": f.type or "不明"} for f in uploaded_docs]
        st.dataframe(pd.DataFrame(file_info), use_container_width=True, hide_index=True)

        if st.button("🚀 ドキュメントを処理 (インジェスト)", type="primary", use_container_width=True, key="process_docs_button_v7_tab_documents"):
            progress_bar = st.progress(0, text="処理開始...")
            status_text = st.empty()
            try:
                paths_to_ingest = []
                for i, file in enumerate(uploaded_docs):
                    status_text.info(f"一時保存中: {file.name}")
                    paths_to_ingest.append(str(_persist_uploaded_file(file)))
                    progress_bar.progress((i + 1) / (len(uploaded_docs) * 2), text=f"一時保存完了: {file.name}")

                status_text.info(f"インデックスを構築中... ({len(paths_to_ingest)}件のファイル)")
                rag_system.ingest_documents(paths_to_ingest)
                progress_bar.progress(1.0, text="インジェスト完了！")
                st.success(f"✅ {len(uploaded_docs)}個のファイルが正常に処理されました！")
                time.sleep(1)
                st.balloons()
                st.rerun()
            except Exception as e:
                st.error(f"ドキュメント処理中にエラーが発生しました: {type(e).__name__} - {e}")
            finally:
                progress_bar.empty()
                status_text.empty()

    st.markdown("### 📚 登録済みドキュメント")
    docs_df = get_documents_dataframe(rag_system)
    if not docs_df.empty:
        st.dataframe(docs_df, use_container_width=True, hide_index=True)

        st.markdown("### 🗑️ ドキュメント削除")
        doc_ids_for_deletion = ["選択してください..."] + docs_df["Document ID"].tolist()
        doc_to_delete = st.selectbox(
            "削除するドキュメントIDを選択:",
            doc_ids_for_deletion,
            label_visibility="collapsed",
            key=f"doc_delete_selectbox_v7_tab_documents_{rag_system.config.collection_name}"
        )
        if doc_to_delete != "選択してください...":
            st.warning(f"**警告:** ドキュメント '{doc_to_delete}' を削除すると、関連する全てのチャンクがデータベースとベクトルストアから削除されます。この操作は元に戻せません。")
            if st.button(f"'{doc_to_delete}' を削除実行", type="secondary", key="doc_delete_button_v7_tab_documents"):
                try:
                    with st.spinner(f"削除中: {doc_to_delete}"):
                        success, message = rag_system.delete_document_by_id(doc_to_delete)
                    if success:
                        st.success(message)
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(message)
                except Exception as e:
                    st.error(f"ドキュメント削除中にエラーが発生しました: {type(e).__name__} - {e}")
    else:
        st.info("まだドキュメントが登録されていません。上のセクションからアップロードしてください。")
