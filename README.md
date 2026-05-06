# 청년 이슈 뉴스 분석 프로젝트

동아일보·한겨레·한국일보 (2016–2025, 35,299건) 청년 관련 기사 분석

## 구조

| 디렉토리 | 목적 |
|----------|------|
| `01_crawl/` | 네이버 뉴스 크롤링 |
| `02_network_analysis/` | 텍스트 전처리 및 의미연결망 분석 |
| `rag_chatbot/` | RAG 기반 뉴스 검색 챗봇 |
| `tests/` | RAG 챗봇 단위 테스트 |

> 수집 데이터(`news_list/`, `news_main/`), 전처리 결과(`preprocessed_data/`), 벡터 인덱스(`output/`)는 로컬 전용이며 저장소에 포함되지 않습니다.

## RAG 챗봇 실행

```bash
cp .env.template .env   # NVIDIA_NIM_API_KEY 입력
pip install -r requirements_rag.txt
streamlit run rag_chatbot/app.py
```

## 스택

- 임베딩: `ibm-granite/granite-embedding-97m-multilingual-r2` (HuggingFace 로컬)
- LLM: Nvidia NIM — `moonshotai/kimi-k2.6`
- 벡터 DB: ChromaDB (SQLite 로컬 저장)
- UI: Streamlit
