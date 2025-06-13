from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

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

    # --- Reranking Chain ---
    reranking_prompt = ChatPromptTemplate.from_template(
        """あなたは検索結果を評価し、並べ替えるエキスパートです。
以下のユーザーの質問と、検索によって取得されたドキュメントのリストが与えられます。
あなたのタスクは、質問に最も関連性の高いドキュメントから順に、ドキュメントのインデックス（0から始まる番号）を並べ替えることです。

ユーザーの質問: {question}

ドキュメントリスト:
{documents}

関連性の高い順に並べ替えたドキュメントのインデックスを、カンマ区切りで返してください。
例: 2,0,1,3

並べ替えたインデックス:"""
    )
    reranking_chain = reranking_prompt | llm | StrOutputParser()

    # --- Semantic Router Chain ---
    semantic_router_prompt = ChatPromptTemplate.from_template(
        """あなたはユーザーの質問の意図を分析し、最適な処理ルートを判断するエキスパートです。
以下の情報に基づいて、質問を「SQL」か「RAG」のどちらにルーティングすべきかを決定してください。

利用可能なデータテーブルの概要:
{tables_info}

ユーザーの質問: {question}

判断基準:
- 「SQL」ルート: 質問が、上記テーブル内のデータに対する具体的な集計、計算、フィルタリング、ランキング、または個々のレコードの検索を要求している場合。例：「売上トップ5の製品は？」「昨年の平均注文額は？」
- 「RAG」ルート: 質問が、一般的な知識、ドキュメントの内容に関する説明、要約、概念の理解、または自由形式の対話を求めている場合。例：「このレポートの要点を教えて」「弊社のコンプライアンス方針について説明して」

思考プロセスをステップバイステップで記述し、最終的な判断をJSON形式で出力してください。

思考プロセス:
1. ユーザーの質問の主要なキーワードと意図を分析します。
2. 質問が利用可能なデータテーブルの情報を活用して解決できるか評価します。
3. 判断基準と照らし合わせ、最も適切なルートを選択します。

出力形式:
{{
  "route": "SQL" or "RAG",
  "reason": "判断理由を簡潔に記述"
}}
"""
    )
    semantic_router_chain = semantic_router_prompt | llm | JsonOutputParser()

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

    # --- Final Answer Synthesis Chain ---
    synthesis_prompt = ChatPromptTemplate.from_template(
        """あなたは高度なAIアシスタントです。ユーザーの質問に対して、以下の2種類の検索結果が提供されました。
1. **RAG検索結果**: ドキュメントから抽出された、関連性の高いテキスト情報。
2. **SQL検索結果**: データベースから取得された、具体的なデータや集計結果。

これらの情報を包括的に分析し、両方の結果を適切に組み合わせて、ユーザーに一つのまとまりのある、分かりやすい回答を生成してください。

ユーザーの質問: {question}

RAG検索結果 (ドキュメントからの抜粋):
---
{rag_context}
---

SQL検索結果 (データベースからのデータ):
---
{sql_data}
---

上記の情報を統合した最終的な回答:"""
    )
    synthesis_chain = synthesis_prompt | llm | StrOutputParser()

    return {
        "answer_generation": answer_generation_chain,
        "query_expansion": query_expansion_chain,
        "jargon_extraction": jargon_extraction_chain,
        "query_augmentation": query_augmentation_chain,
        "reranking": reranking_chain,
        "semantic_router": semantic_router_chain,
        "multi_table_sql": multi_table_sql_chain,
        "sql_answer_generation": sql_answer_generation_chain,
        "synthesis": synthesis_chain,
    }
