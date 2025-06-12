import streamlit as st
import os
from datetime import datetime
from utils.helpers import load_terms_from_db, render_term_card

def render_dictionary_tab():
    """Renders the dictionary tab."""
    st.markdown("### 📖 専門用語辞書")
    st.caption("登録された専門用語・類義語を検索・確認できます。")
    
    pg_url = os.getenv("PG_URL", "")
    jargon_table_name = os.getenv("JARGON_TABLE_NAME", "jargon_dictionary")

    if not pg_url:
        st.warning("⚠️ PG_URLが設定されていないため、用語辞書機能を利用できません。.envファイルでPG_URLを設定してください。")
        return

    # Search and refresh buttons
    col1, col2 = st.columns([3, 1])
    with col1:
        search_keyword = st.text_input(
            "🔍 用語検索",
            placeholder="検索したい用語を入力してください...",
            key="term_search_input"
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 更新", key="refresh_terms", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # Load term data
    with st.spinner("用語辞書を読み込み中..."):
        terms_df = load_terms_from_db(pg_url, jargon_table_name, search_keyword)
    
    if terms_df.empty:
        if search_keyword:
            st.info(f"「{search_keyword}」に該当する用語が見つかりませんでした。")
        else:
            st.info("まだ用語が登録されていません。サイドバーの「📚 用語辞書生成」から用語を抽出してください。")
        return

    # Statistics
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
    
    # View mode selection
    view_mode = st.radio(
        "表示形式",
        ["カード形式", "テーブル形式"],
        horizontal=True,
        key="dict_view_mode"
    )
    
    if view_mode == "カード形式":
        for _, row in terms_df.iterrows():
            render_term_card(row)
    else:
        display_df = terms_df.copy()
        display_df['aliases'] = display_df['aliases'].apply(lambda x: ', '.join(x) if x else '')
        display_df['related_terms'] = display_df['related_terms'].apply(lambda x: ', '.join(x) if x else '')
        display_df.columns = ['用語', '定義', '分野', '類義語', '関連語', '信頼度', '更新日時']
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=min(600, (len(display_df) + 1) * 35 + 3)
        )
    
    # CSV download
    st.markdown("---")
    csv = terms_df.to_csv(index=False)
    st.download_button(
        label="📥 用語辞書をCSVでダウンロード",
        data=csv,
        file_name=f"jargon_dictionary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        key="csv_download_button"
    )
