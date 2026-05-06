# Step 4. 공출현 행렬 구축 — 상세 구현 계획

## Context

35,299건의 신문 기사(동아일보 13,796 + 한겨레 10,690 + 한국일보 10,813)를 대상으로 "청년" 담론의 의미연결망을 분석한다.  
불용어 처리 및 명사(NNG/NNP) 토큰 추출까지 완료된 상태이며(`preprocessed_data/*.csv`의 `tokens` 컬럼),  
이제 **문장 단위 공출현 행렬**을 구축해 RQ2(의미연결망 분석) 및 RQ3(연도별 비교)의 입력 데이터를 만든다.

---

## 설계 결정사항

| 항목 | 결정값 | 근거 |
|------|--------|------|
| 분석 단위 | **문장** | 의미적 밀착도 확보, 과제정의서 권장 방식 |
| 어휘 크기 | **상위 150개** | 35K 기사 규모 대응, 가독성·정보량 균형 |
| 엣지 가중치 | 공출현 빈도(정수) | 빈도 기반 연결 강도 반영 |
| 최소 엣지 임계값 | 전체: ≥5회, 연도별: ≥2회 | 노이즈 엣지 제거 |
| 내보내기 형식 | GEXF | Gephi 직접 로드 지원 |

---

## 현재 파일 구조

```
PR_융합기초프로그래밍_청년/
├── preprocessed_data/
│   ├── 청년_동아일보_정제완료_EA13796.csv   # tokens 컬럼 포함
│   ├── 청년_한겨레_정제완료_EA10690.csv
│   └── 청년_한국일보_정제완료_EA10813.csv
├── text_preprocessing.ipynb    # synonym_stopword 함수 정의됨 (재사용)
├── text_mining.ipynb           # 이 파일에 Step 4 코드를 추가
└── output/                     # (신규 생성) GEXF, matrix CSV 저장
```

---

## 구현 계획 (text_mining.ipynb에 셀 추가)

### 셀 A — 의존성 로드 및 Kiwi 초기화

```python
import re, ast, pickle
from pathlib import Path
from collections import Counter
from itertools import combinations

import pandas as pd
from tqdm.notebook import tqdm
from kiwipiepy import Kiwi

Path("output").mkdir(exist_ok=True)

# text_preprocessing.ipynb에서 정의한 상수 재정의
# (SYNONYM_1ST, SYNONYM_2ND, STOPWORDS, USER_DICT를 동일하게 붙여넣기)
# — synonym_stopword 모듈이 별도 .py로 추출되어 있으면 import로 대체
from synonym_stopword import (
    SYNONYM_1ST, SYNONYM_2ND, STOPWORDS, USER_DICT,
    preprocess_1st, preprocess_2nd
)

kiwi = Kiwi()
for word, tag in USER_DICT:
    kiwi.add_user_word(word, tag)
```

> **주의**: `synonym_stopword`가 노트북 셀에만 정의되어 있으면,  
> `text_preprocessing.ipynb`의 해당 셀을 `.py` 파일로 저장 후 import한다.

---

### 셀 B — 데이터 로드 및 병합

```python
dfs = []
for path in Path("preprocessed_data").glob("*.csv"):
    df = pd.read_csv(path, usecols=["id", "date", "year", "text_cleaned"])
    df["media"] = path.stem.split("_")[1]   # 동아일보 / 한겨레 / 한국일보
    dfs.append(df)

corpus = pd.concat(dfs, ignore_index=True).dropna(subset=["text_cleaned"])
print(f"총 기사: {len(corpus):,}건 / 연도 범위: {corpus['year'].min()}–{corpus['year'].max()}")
```

> `tokens` 컬럼 대신 `text_cleaned`를 사용해 문장 단위 재토큰화한다.  
> 메모리 절약을 위해 필요한 컬럼만 로드.

---

### 셀 C — 문장 단위 토큰 추출 함수

```python
def sent_tokenize(text_cleaned: str) -> list[list[str]]:
    """기사 text_cleaned → 문장별 명사 토큰 리스트"""
    # 1차 동의어 정규화 (복합 표현 통일)
    text = preprocess_1st(text_cleaned)
    # 문장 분리
    sents = kiwi.split_into_sents(text, return_tokens=False)
    result = []
    for sent in sents:
        raw_tokens = [
            t.form for t in kiwi.tokenize(sent.text)
            if t.tag in ("NNG", "NNP") and len(t.form) > 1
        ]
        # 2차 동의어 + 불용어 처리
        cleaned = preprocess_2nd(raw_tokens)
        if len(cleaned) >= 2:   # 단독 토큰 문장은 공출현 불가 → 제외
            result.append(cleaned)
    return result
```

---

### 셀 D — 전체 말뭉치 문장 토큰화 (캐시 저장)

```python
CACHE_PATH = Path("output/sent_tokens_cache.pkl")

if CACHE_PATH.exists():
    with open(CACHE_PATH, "rb") as f:
        corpus["sent_tokens"] = pickle.load(f)
    print("캐시에서 로드 완료")
else:
    corpus["sent_tokens"] = [
        sent_tokenize(text) for text in tqdm(corpus["text_cleaned"], desc="문장 토큰화")
    ]
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(corpus["sent_tokens"].tolist(), f)
    print("토큰화 완료 및 캐시 저장")
```

> 첫 실행 후 캐시를 저장해 재실행 시 약 20~40분 소요를 건너뜀.

---

### 셀 E — 전역 빈도 계산 및 Top 150 어휘 선정

```python
global_freq = Counter()
for sent_list in corpus["sent_tokens"]:
    for sent in sent_list:
        global_freq.update(sent)

top_n = 150
vocab = {word for word, _ in global_freq.most_common(top_n)}

# 확인용 출력
print("Top 20 단어:")
for word, cnt in global_freq.most_common(20):
    print(f"  {word}: {cnt:,}")
```

---

### 셀 F — 공출현 카운터 빌더 (재사용 함수)

```python
def build_cooc(sent_token_series, vocab: set) -> Counter:
    """문장별 토큰 리스트 시리즈 → 어휘 내 단어 쌍 공출현 Counter"""
    cooc = Counter()
    for sent_list in sent_token_series:
        for sent in sent_list:
            in_vocab = sorted({w for w in sent if w in vocab})
            for w1, w2 in combinations(in_vocab, 2):
                cooc[(w1, w2)] += 1
    return cooc
```

---

### 셀 G — 전체 기간 공출현 행렬

```python
global_cooc = build_cooc(corpus["sent_tokens"], vocab)

# DataFrame으로 변환 (Gephi 외 Python 분석용)
cooc_df = pd.DataFrame(
    [(w1, w2, cnt) for (w1, w2), cnt in global_cooc.items()],
    columns=["word1", "word2", "weight"]
).sort_values("weight", ascending=False)

cooc_df.to_csv("output/cooc_global.csv", index=False, encoding="utf-8-sig")
print(f"전체 엣지 수: {len(cooc_df):,} / 상위 5개:")
print(cooc_df.head())
```

---

### 셀 H — 연도별 공출현 행렬

```python
yearly_cooc = {}
for year, grp in corpus.groupby("year"):
    yearly_cooc[year] = build_cooc(grp["sent_tokens"], vocab)
    df_y = pd.DataFrame(
        [(w1, w2, cnt) for (w1, w2), cnt in yearly_cooc[year].items()],
        columns=["word1", "word2", "weight"]
    ).sort_values("weight", ascending=False)
    df_y.to_csv(f"output/cooc_{year}.csv", index=False, encoding="utf-8-sig")

print(f"연도별 처리 완료: {sorted(yearly_cooc.keys())}")
```

---

### 셀 I — NetworkX 그래프 생성 및 GEXF 내보내기

```python
import networkx as nx

def make_graph(cooc: Counter, min_weight: int = 5) -> nx.Graph:
    G = nx.Graph()
    for (w1, w2), cnt in cooc.items():
        if cnt >= min_weight:
            G.add_edge(w1, w2, weight=cnt)
    # 고립 노드 제거
    G.remove_nodes_from(list(nx.isolates(G)))
    return G

# 전체 기간 네트워크
G_global = make_graph(global_cooc, min_weight=5)
nx.write_gexf(G_global, "output/network_global.gexf")

# 연도별 네트워크
for year, cooc in yearly_cooc.items():
    G_year = make_graph(cooc, min_weight=2)
    nx.write_gexf(G_year, f"output/network_{year}.gexf")

# 기본 지표 출력
print(f"[전체] 노드: {G_global.number_of_nodes()}, 엣지: {G_global.number_of_edges()}, 밀도: {nx.density(G_global):.4f}")
```

---

## 성능 최적화 포인트

| 문제 | 대응 |
|------|------|
| 35K 기사 × kiwi 재호출 → 20~40분 | 셀 D의 pickle 캐시로 1회만 수행 |
| 메모리 부족 | CSV 로드 시 불필요한 컬럼 제외 (셀 B) |
| 연도별 vocab 차이 | 전역 Top 150을 고정 어휘로 사용 (연도별 비교 가능성 확보) |

---

## 검증 방법

1. **`cooc_global.csv` 상위 10개 확인** — (정부, 일자리), (경제, 정책) 등 예상 가능한 쌍이 상위인지 검토
2. **그래프 노드 수 확인** — 150개 이하여야 함 (고립 노드 제거 후)
3. **연도별 GEXF 파일 Gephi 로드 테스트** — 2016.gexf, 2025.gexf 비교 로드
4. **밀도 참고 논문 비교** — 논문(밀도 4.86) 대비 본 연구 수치 기록

---

## 완료 후 산출물

```
output/
├── sent_tokens_cache.pkl      # 재실행 캐시
├── cooc_global.csv            # 전체 기간 공출현 엣지 리스트
├── cooc_2016.csv ~ cooc_2025.csv
├── network_global.gexf        # Gephi용 전체 네트워크
└── network_2016.gexf ~ network_2025.gexf
```
