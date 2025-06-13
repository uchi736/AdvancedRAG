from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def create_chains(llm, max_sql_results: int) -> dict:
    """Creates and returns a dictionary of all LangChain runnables."""
    
    # --- RAG Chains ---
    base_rag_prompt = ChatPromptTemplate.from_template(
        """あなたは親切で知識豊富なアシスタントです。以下のコンテキストを参考に質問に答えてください。
コンテキストには、テキストの抜粋だけでなく、図や表を説明する要約が含まれている場合があります。

コンテキスト:
{context}

質問: {question}

回答:"""
    )
    
    # This chain returns the final answer string.
    answer_generation_chain = (
        RunnablePassthrough.assign(
            context=(lambda x: "\n\n".join(doc.page_content for doc in x["context"]))
        )
        | base_rag_prompt
        | llm
        | StrOutputParser()
    )

    query_expansion_prompt = ChatPromptTemplate.from_template(
        """以下の質問に対して、より良い検索結果を得るために、関連する追加の検索クエリを3つ生成してください。
元の質問の意図を保ちながら、異なる表現や関連する概念を含めてください。

元の質問: {question}

追加クエリ（改行で区切って3つ）:"""
    )
    query_expansion_chain = query_expansion_prompt | llm | StrOutputParser()

    # --- Golden-Retriever (Jargon) Chains ---
    jargon_extraction_prompt = ChatPromptTemplate.from_template(
        """あなたは専門用語の抽出エキスパートです。以下の質問から、専門用語、略語、固有名詞、技術用語を抽出してください。
一般的な単語は除外し、ドメイン固有の用語のみを抽出してください。

質問: {question}

抽出された専門用語（改行で区切って、最大{max_terms}個まで）:"""
    )
    jargon_extraction_chain = jargon_extraction_prompt | llm | StrOutputParser()
    
    query_augmentation_prompt = ChatPromptTemplate.from_template(
        """以下の質問と専門用語の定義を考慮して、より明確で検索しやすい質問に書き換えてください。
専門用語の定義を質問に自然に組み込み、曖昧さを排除してください。

元の質問: {original_question}

専門用語と定義:
{jargon_definitions}

補強された質問:"""
    )
    query_augmentation_chain = query_augmentation_prompt | llm | StrOutputParser()

    # --- Text-to-SQL Chains ---
    query_detection_prompt = ChatPromptTemplate.from_template(
        """この質問はSQL分析とRAG検索のどちらが適切ですか？

利用可能なデータテーブルの概要:
{tables_info}

ユーザーの質問: {question}

判断基準:
- SQLが適している場合: 具体的な数値データに基づく分析、集計、ランキング、フィルタリング、特定レコードの抽出など
- RAGが適している場合: ドキュメントの内容に関する要約、説明、概念理解、自由形式の質問

回答は「SQL」または「RAG」のいずれか一つのみを返してください。"""
    )
    detection_chain = query_detection_prompt | llm | StrOutputParser()

    multi_table_text_to_sql_prompt = ChatPromptTemplate.from_template(
        f"""あなたはPostgreSQLエキスパートです。以下に提示される複数のテーブルスキーマの中から、ユーザーの質問に答えるために最も適切と思われるテーブルを選択し、必要であればそれらのテーブル間でJOINを適切に使用して、SQLクエリを生成してください。
SQLはPostgreSQL構文に準拠し、テーブル名やカラム名が日本語の場合はダブルクォーテーションで囲んでください。
最終的な結果セットが過度に大きくならないよう、適切にLIMIT句を使用してください（例: LIMIT {max_sql_results}）。

利用可能なテーブルのスキーマ情報一覧:
{{schemas_info}}

ユーザーの質問: {{question}}

SQLクエリのみを返してください:
```sql
SELECT ...
```
"""
    )
    multi_table_sql_chain = multi_table_text_to_sql_prompt | llm | StrOutputParser()

    sql_answer_generation_prompt = ChatPromptTemplate.from_template(
        """与えられた元の質問と、それに基づいて実行されたSQLクエリ、およびその実行結果を考慮して、ユーザーにとって分かりやすい言葉で回答を生成してください。

元の質問: {original_question}

実行されたSQLクエリ:
```sql
{sql_query}
```

SQL実行結果のプレビュー (最大 {max_preview_rows} 件表示):
{sql_results_preview_str}
(このプレビューは全 {total_row_count} 件中の一部です)

上記の情報を踏まえた、質問に対する回答:"""
    )
    sql_answer_generation_chain = sql_answer_generation_prompt | llm | StrOutputParser()

    return {
        "answer_generation": answer_generation_chain,
        "query_expansion": query_expansion_chain,
        "jargon_extraction": jargon_extraction_chain,
        "query_augmentation": query_augmentation_chain,
        "query_detection": detection_chain,
        "multi_table_sql": multi_table_sql_chain,
        "sql_answer_generation": sql_answer_generation_chain,
    }
