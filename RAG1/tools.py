from .indexing import vector_store
from langchain.tools import tool

last_retrieved_context = []

@tool
def collecting_info(query: str) -> str:
    """
    Search the PDF Knowledge base and return relevant information
    for the user's question
    """
    global last_retrieved_context

    docs = vector_store.similarity_search(query, k=3)

    # save the real retrieved chunks; a question can trigger several tool calls,
    # so keep every chunk seen (without repeats) until ask() resets the list
    for doc in docs:
        if doc.page_content not in last_retrieved_context:
            last_retrieved_context.append(doc.page_content)

    return "\n\n".join(doc.page_content for doc in docs)


