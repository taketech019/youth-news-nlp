import re
from typing import Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters
from llama_index.llms.openai_like import OpenAILike

from rag_chatbot.config import NIM_API_KEY, NIM_BASE_URL, NIM_MODEL, TOP_K, SYSTEM_PROMPT

_YEAR_PATTERN = re.compile(r"(?<!\d)(20[12]\d)(?!\d)")


def extract_year(query: str) -> Optional[str]:
    """쿼리 문자열에서 첫 번째 4자리 연도 반환. 없으면 None."""
    m = _YEAR_PATTERN.search(query)
    return m.group(1) if m else None


def extract_newspaper(selected: list[str]) -> Optional[str]:
    """선택된 신문사가 정확히 1개면 반환, 그 외엔 None (필터 미적용).

    2개 선택 시 OR 필터를 지원하지 않아 전체 검색으로 fallback.
    """
    if len(selected) == 1:
        return selected[0]
    return None


def build_metadata_filters(
    year: Optional[str],
    newspaper: Optional[str],
) -> Optional[MetadataFilters]:
    """year, newspaper 제약을 MetadataFilters로 변환. 둘 다 None이면 None 반환."""
    filters = []
    if year:
        filters.append(MetadataFilter(key="year", value=year))
    if newspaper:
        filters.append(MetadataFilter(key="newspaper", value=newspaper))
    return MetadataFilters(filters=filters) if filters else None


def build_query_engine(
    index: VectorStoreIndex,
    year: Optional[str] = None,
    newspaper: Optional[str] = None,
):
    """메타데이터 필터가 적용된 LlamaIndex 쿼리 엔진 반환."""
    llm = OpenAILike(
        model=NIM_MODEL,
        api_base=NIM_BASE_URL,
        api_key=NIM_API_KEY,
        is_chat_model=True,
        context_window=32768,
        system_prompt=SYSTEM_PROMPT,
    )
    metadata_filters = build_metadata_filters(year=year, newspaper=newspaper)
    kwargs: dict = {"similarity_top_k": TOP_K, "llm": llm}
    if metadata_filters:
        kwargs["filters"] = metadata_filters

    return index.as_query_engine(**kwargs)


def query(
    index: VectorStoreIndex,
    user_input: str,
    year_override: Optional[str] = None,
    selected_newspapers: Optional[list[str]] = None,
) -> tuple[str, list]:
    """쿼리 실행 → (답변 텍스트, source_nodes 리스트) 반환.

    year_override가 있으면 쿼리 텍스트에서 연도를 추출하지 않고 그 값을 사용.
    """
    year = year_override or extract_year(user_input)
    newspaper = extract_newspaper(selected_newspapers) if selected_newspapers else None
    engine = build_query_engine(index, year=year, newspaper=newspaper)
    response = engine.query(user_input)
    return str(response), list(response.source_nodes)
