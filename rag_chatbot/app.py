import streamlit as st

from rag_chatbot.indexer import load_or_build_index
from rag_chatbot.retriever import query

st.set_page_config(page_title="청년 뉴스 RAG 챗봇", layout="wide")


@st.cache_resource(show_spinner="인덱스 로딩 중... (최초 실행 시 수십 분 소요)")
def get_index():
    return load_or_build_index()


# --- 사이드바 ---
with st.sidebar:
    st.header("검색 필터")

    year_option = st.selectbox(
        "연도",
        options=["전체"] + [str(y) for y in range(2016, 2026)],
        index=0,
    )
    selected_newspapers = st.multiselect(
        "신문사",
        options=["동아일보", "한겨레", "한국일보"],
        default=["동아일보", "한겨레", "한국일보"],
    )

    if len(selected_newspapers) == 0:
        st.warning("신문사를 하나 이상 선택하세요.")
    elif len(selected_newspapers) == 2:
        st.caption("2개 선택 시 신문사 필터가 해제됩니다 (3개 전체 검색)")

    st.divider()
    st.caption("granite-embedding-97m · kimi-k2.6 via Nvidia NIM")

# --- 메인 ---
st.title("청년 이슈 뉴스 챗봇")
st.caption("2016–2025년 국내 일간지 기사 기반 RAG 챗봇")

index = get_index()

if "messages" not in st.session_state:
    st.session_state.messages = []

# 대화 히스토리 렌더링
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("참고 기사"):
                for src in msg["sources"]:
                    m = src.node.metadata
                    st.markdown(
                        f"- **{m.get('newspaper','')}** ({m.get('year','')}) "
                        f"— {m.get('title','')[:60]}"
                    )

# 입력 처리
user_input = st.chat_input("청년 관련 질문을 입력하세요...")

if user_input:
    if not selected_newspapers:
        st.error("신문사를 하나 이상 선택한 후 질문하세요.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 사이드바 연도 선택이 "전체"면 query()가 쿼리 텍스트에서 추출, 아니면 사이드바 값 우선
    year_override = year_option if year_option != "전체" else None

    with st.chat_message("assistant"):
        with st.spinner("검색 중..."):
            try:
                answer, sources = query(
                    index,
                    user_input,
                    year_override=year_override,
                    selected_newspapers=selected_newspapers,
                )
            except Exception as e:
                answer = f"오류가 발생했습니다: {e}"
                sources = []

        st.markdown(answer)
        if sources:
            with st.expander("참고 기사"):
                for src in sources:
                    m = src.node.metadata
                    st.markdown(
                        f"- **{m.get('newspaper','')}** ({m.get('year','')}) "
                        f"— {m.get('title','')[:60]}"
                    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
