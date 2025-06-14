from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableLambda, RunnableBranch, Runnable, ConfigurableField
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from operator import itemgetter
from typing import List, Dict, Any

# Forward declaration to avoid circular import
class JapaneseHybridRetriever:
    pass
class JargonDictionaryManager:
    pass

def _format_docs(docs: List[Any]) -> str:
    """Helper function to format documents for context."""
    return "\n\n".join([doc.page_content for doc in docs])

def _reciprocal_rank_fusion(doc_lists: List[List[Any]], k=60) -> List[Any]:
    """Performs Reciprocal Rank Fusion on a list of document lists."""
    if not doc_lists:
        return []
    
    fused_scores: Dict[str, float] = {}
    doc_map: Dict[str, Any] = {}

    for docs in doc_lists:
        for rank, doc in enumerate(docs):
            doc_id = doc.metadata.get("chunk_id")
            if not doc_id:
                continue
            
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
            
            previous_score = fused_scores.get(doc_id, 0.0)
            fused_scores[doc_id] = previous_score + 1.0 / (k + rank + 1)

    sorted_docs = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
    return [doc_map[doc_id] for doc_id in sorted_docs]

def _combine_documents_simple(list_of_document_lists: List[List[Any]]) -> List[Any]:
    """A simple combination of documents, preserving order and removing duplicates."""
    all_docs: List[Any] = []
    seen_chunk_ids = set()
    for doc_list in list_of_document_lists:
        for doc in doc_list:
            chunk_id = doc.metadata.get('chunk_id', '')
            if chunk_id and chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                all_docs.append(doc)
    return all_docs

def create_retrieval_chain(
    llm: Runnable, 
    retriever: JapaneseHybridRetriever, 
    jargon_manager: JargonDictionaryManager, 
    config_obj: Any
) -> Runnable:
    """
    Creates a traceable chain that handles up to the document retrieval and reranking steps.
    """
    jargon_extraction_prompt = ChatPromptTemplate.from_template("質問から専門用語を抽出してください: {question}")
    jargon_extraction_chain = jargon_extraction_prompt | llm | StrOutputParser()

    query_augmentation_prompt = ChatPromptTemplate.from_template("質問: {original_question}\n専門用語定義: {jargon_definitions}\n\n上記を元に質問を補強してください:")
    query_augmentation_chain = query_augmentation_prompt | llm | StrOutputParser()

    def augment_query_with_jargon(input_dict: dict) -> dict:
        original_question = input_dict["question"]
        jargon_terms = jargon_extraction_chain.invoke({"question": original_question, "max_terms": config_obj.max_jargon_terms_per_query}).split('\n')
        jargon_terms = [t.strip() for t in jargon_terms if t.strip()]
        augmented_query = original_question
        if jargon_terms:
            jargon_defs = jargon_manager.lookup_terms(jargon_terms)
            if jargon_defs:
                defs_text = "\n".join([f"- {term}: {info['definition']}" for term, info in jargon_defs.items()])
                augmented_query = query_augmentation_chain.invoke({"original_question": original_question, "jargon_definitions": defs_text})
        return {**input_dict, "retrieval_query": augmented_query}

    query_expansion_prompt = ChatPromptTemplate.from_template("質問を拡張してください: {question}")
    query_expansion_chain = query_expansion_prompt | llm | StrOutputParser() | (lambda x: x.split('\n'))

    def get_retriever_with_search_type(input_or_config: Any):
        search_type = "ハイブリッド検索"
        if isinstance(input_or_config, dict) and "configurable" in input_or_config:
             search_type = input_or_config.get("configurable", {}).get("search_type", "ハイブリッド検索")
        return retriever.with_config(configurable={"search_type": search_type})

    expansion_retrieval_chain = RunnablePassthrough.assign(expanded_queries=query_expansion_chain).assign(
        doc_lists=itemgetter("expanded_queries") | RunnableLambda(lambda x: retriever.with_config(configurable={"search_type": x.get("search_type")}).batch(x["expanded_queries"], x["config"]))
    ).assign(
        documents=RunnableBranch(
            (lambda x: x.get("use_rag_fusion"), itemgetter("doc_lists") | RunnableLambda(_reciprocal_rank_fusion)),
            itemgetter("doc_lists") | RunnableLambda(_combine_documents_simple)
        )
    )
    
    def get_docs_for_batch(x):
        retriever_with_config = retriever.with_config(configurable={"search_type": x.get("search_type")})
        return retriever_with_config.batch(x["expanded_queries"], x.get("config"))

    expansion_retrieval_chain = RunnablePassthrough.assign(
        expanded_queries=query_expansion_chain
    ).assign(
        doc_lists=get_docs_for_batch
    ).assign(
        documents=RunnableBranch(
            (lambda x: x.get("use_rag_fusion"), itemgetter("doc_lists") | RunnableLambda(_reciprocal_rank_fusion)),
            itemgetter("doc_lists") | RunnableLambda(_combine_documents_simple)
        )
    )

    standard_retrieval_chain = RunnablePassthrough.assign(
        documents=lambda x: retriever.with_config(configurable={"search_type": x.get("search_type")}).invoke(x["retrieval_query"], x.get("config"))
    )

    reranking_prompt = ChatPromptTemplate.from_template("質問: {question}\n\nドキュメント:\n{documents}\n\n最も関連性の高い順にドキュメントのインデックスをカンマ区切りで返してください:")
    reranking_chain = reranking_prompt | llm | StrOutputParser()

    def rerank_documents(input_dict: dict) -> List[Any]:
        docs = input_dict["documents"]
        if not docs: return []
        docs_for_rerank = [f"ドキュメント {i}:\n{doc.page_content}" for i, doc in enumerate(docs)]
        rerank_input = {"question": input_dict["question"], "documents": "\n\n---\n\n".join(docs_for_rerank)}
        try:
            reranked_indices_str = reranking_chain.invoke(rerank_input)
            reranked_indices = [int(i.strip()) for i in reranked_indices_str.split(',') if i.strip().isdigit()]
            return [docs[i] for i in reranked_indices if i < len(docs)]
        except Exception:
            return docs

    retrieval_and_rerank_chain = (
        RunnableBranch(
            (lambda x: x.get("use_jargon_augmentation"), RunnableLambda(augment_query_with_jargon)),
            RunnablePassthrough.assign(retrieval_query=itemgetter("question"))
        )
        | RunnableBranch(
            (lambda x: x.get("use_rag_fusion") or x.get("use_query_expansion"), expansion_retrieval_chain),
            standard_retrieval_chain
        )
        | RunnablePassthrough.assign(
            documents=RunnableBranch(
                (lambda x: x.get("use_reranking"), RunnableLambda(rerank_documents)),
                itemgetter("documents")
            )
        )
    )
    return retrieval_and_rerank_chain

def create_full_rag_chain(retrieval_chain: Runnable, llm: Runnable) -> Runnable:
    """Creates the final answer generation part of the RAG chain."""
    answer_generation_prompt = ChatPromptTemplate.from_template("コンテキスト:\n{context}\n\n質問: {question}\n\n回答:")
    
    full_chain = (
        retrieval_chain
        | RunnablePassthrough.assign(
            answer=(
                RunnablePassthrough.assign(context=itemgetter("documents") | RunnableLambda(_format_docs))
                | answer_generation_prompt
                | llm
                | StrOutputParser()
            )
        )
    )
    return full_chain

def create_chains(llm, max_sql_results: int) -> dict:
    """Creates and returns a dictionary of all LangChain runnables for SQL and Synthesis."""
    semantic_router_prompt = ChatPromptTemplate.from_template(
        """あなたはユーザーの質問の意図を分析し、最適な処理ルートを判断するエキスパートです。
... (rest of the prompt is unchanged) ...
"""
    )
    semantic_router_chain = semantic_router_prompt | llm | JsonOutputParser()

    multi_table_text_to_sql_prompt = ChatPromptTemplate.from_template(
        f"""あなたはPostgreSQLエキスパートです。
... (rest of the prompt is unchanged) ...
"""
    )
    multi_table_sql_chain = multi_table_text_to_sql_prompt | llm | StrOutputParser()

    sql_answer_generation_prompt = ChatPromptTemplate.from_template(
        """与えられた元の質問と、それに基づいて実行されたSQLクエリ、およびその実行結果を考慮して、ユーザーにとって分かりやすい言葉で回答を生成してください。
... (rest of the prompt is unchanged) ...
"""
    )
    sql_answer_generation_chain = sql_answer_generation_prompt | llm | StrOutputParser()

    synthesis_prompt = ChatPromptTemplate.from_template(
        """あなたは高度なAIアシスタントです。ユーザーの質問に対して、以下の2種類の検索結果が提供されました。
... (rest of the prompt is unchanged) ...
"""
    )
    synthesis_chain = synthesis_prompt | llm | StrOutputParser()

    return {
        "semantic_router": semantic_router_chain,
        "multi_table_sql": multi_table_sql_chain,
        "sql_answer_generation": sql_answer_generation_chain,
        "synthesis": synthesis_chain,
    }
