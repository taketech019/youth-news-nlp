import math
import re
from typing import Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters
from llama_index.llms.openai_like import OpenAILike

from rag_chatbot.config import NIM_API_KEY, NIM_BASE_URL, NIM_MODEL, TOP_K, SYSTEM_PROMPT

_YEAR_PATTERN = re.compile(r"(?<!\d)(20[12]\d)(?!\d)")


def extract_year(query: str) -> Optional[str]:
    """쿼리에서 연도 추출. 정확히 1개일 때만 반환, 2개 이상(범위 질문)이면 None."""
    matches = _YEAR_PATTERN.findall(query)
    return matches[0] if len(matches) == 1 else None


def extract_newspaper(selected: list[str]) -> Optional[str]:
    """선택된 신문사가 정확히 1개면 반환, 그 외엔 None."""
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


def _make_llm() -> OpenAILike:
    return OpenAILike(
        model=NIM_MODEL,
        api_base=NIM_BASE_URL,
        api_key=NIM_API_KEY,
        is_chat_model=True,
        context_window=32768,
        system_prompt=SYSTEM_PROMPT,
    )


def build_query_engine(
    index: VectorStoreIndex,
    year: Optional[str] = None,
    newspaper: Optional[str] = None,
):
    """메타데이터 필터가 적용된 LlamaIndex 쿼리 엔진 반환."""
    metadata_filters = build_metadata_filters(year=year, newspaper=newspaper)
    kwargs: dict = {"similarity_top_k": TOP_K, "llm": _make_llm()}
    if metadata_filters:
        kwargs["filters"] = metadata_filters
    return index.as_query_engine(**kwargs)


def _retrieve_balanced(
    index: VectorStoreIndex,
    query_text: str,
    year: Optional[str],
    newspapers: list[str],
) -> list:
    """신문사별로 ceil(TOP_K / n)개씩 검색 후 합산, 유사도 기준 재정렬하여 TOP_K 반환.

    OR 필터 미지원 문제를 우회하면서 결과가 특정 신문사에 편중되지 않도록 보장.
    예) TOP_K=5, 신문사 3개 → 신문사당 2개 검색(총 6개) → 상위 5개 반환
    """
    per_k = math.ceil(TOP_K / len(newspapers))
    all_nodes: list = []
    for newspaper in newspapers:
        filters = build_metadata_filters(year=year, newspaper=newspaper)
        retriever_kwargs: dict = {"similarity_top_k": per_k}
        if filters:
            retriever_kwargs["filters"] = filters
        retriever = index.as_retriever(**retriever_kwargs)
        all_nodes.extend(retriever.retrieve(query_text))
    all_nodes.sort(key=lambda n: n.score or 0.0, reverse=True)
    return all_nodes[:TOP_K]


def query(
    index: VectorStoreIndex,
    user_input: str,
    year_override: Optional[str] = None,
    selected_newspapers: Optional[list[str]] = None,
) -> tuple[str, list]:
    """쿼리 실행 → (답변 텍스트, source_nodes 리스트) 반환."""
    year = year_override or extract_year(user_input)

    if selected_newspapers and len(selected_newspapers) > 1:
        # 2개 이상 신문사: 신문사별 균등 배분 검색 후 합산
        source_nodes = _retrieve_balanced(index, user_input, year, selected_newspapers)
        response = get_response_synthesizer(llm=_make_llm()).synthesize(
            user_input, nodes=source_nodes
        )
        return str(response), source_nodes

    # 0~1개 신문사: 기존 단일 쿼리 엔진
    newspaper = extract_newspaper(selected_newspapers) if selected_newspapers else None
    engine = build_query_engine(index, year=year, newspaper=newspaper)
    response = engine.query(user_input)
    return str(response), list(response.source_nodes)
