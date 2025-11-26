
from typing import Iterable, List

from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.common.custom_exception import CustomException
from app.common.logger import get_logger
from app.config.settings import settings

logger = get_logger(__name__)


def _normalize_query(query: str | Iterable[str]) -> List[str]:
    if query is None:
        return []
    if isinstance(query, str):
        return [query]
    return [str(item) for item in query if str(item).strip()]


def _build_search_note(user_query: str) -> str:
    tavily = TavilySearchResults(max_results=2)
    try:
        search_response = tavily.invoke({"query": user_query})
    except Exception as error:
        logger.warning("Tavily search failed: %s", error)
        return ""

    results = search_response[0] if isinstance(search_response, tuple) else search_response

    if not results:
        return ""

    snippets = []
    for idx, result in enumerate(results, start=1):
        try:
            title = result.get("title") or result.get("url") or f"Result {idx}"
            content = result.get("content") or result.get("snippet") or ""
            snippets.append(f"{idx}. {title}\n{content}".strip())
        except AttributeError:
            snippets.append(f"{idx}. {result}")

    return "\n\n".join(snippets)


def get_response_from_ai_agents(llm_id, query, allow_search, system_prompt):
    llm = ChatGroq(model=llm_id)
    user_messages = _normalize_query(query)

    if not user_messages:
        raise CustomException("No user messages provided to agent")

    messages_state: List[SystemMessage | HumanMessage] = []

    if system_prompt:
        messages_state.append(SystemMessage(content=system_prompt))

    if allow_search:
        search_note = _build_search_note(user_messages[-1])
        if search_note:
            messages_state.append(
                SystemMessage(
                    content=(
                        "Use the following web search results when responding:\n"
                        f"{search_note}"
                    )
                )
            )

    for user_message in user_messages:
        messages_state.append(HumanMessage(content=user_message))

    response = llm.invoke(messages_state)

    if isinstance(response, AIMessage):
        return response.content

    if hasattr(response, "content"):
        return response.content

    return str(response)