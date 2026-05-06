# Step 5~7. 분석 결과 검증 및 시각화·네트워크 분석 계획

## Context

Step 4(공출현 행렬 구축) 완료 후 `output/` 산출물을 검토한 결과,  
연구목적과 부합하는 의미 있는 쌍들이 다수 확인됐으나 **즉시 수정해야 할 데이터 품질 문제**가 발견됐다.  
이를 먼저 해결하고, RQ1·RQ2·RQ3를 순서대로 해결하는 분석·시각화 단계를 진행한다.

---

## 🚨 즉시 수정: STOPWORDS 노이즈 문제

### 발견된 문제

`cooc_global.csv` 1위 쌍: **`금지, 동아일보` (6,991회)** — 의미연결망 분석에 무관한 저작권 문구 잔류

**원인:** 기사 말미 저작권 고지 `[동아일보 무단전재 및 재배포 금지]`가 text_cleaned에 남아 있고,  
`synonym_stopword.py`의 STOPWORDS에 신문사명·저작권 단어가 누락됨.

**GEXF 노드 중 추가 경량어 발견:**
- `지난해`, `이후`, `동안`, `이상`, `사실`, `자신`, `시작` 등 — 내용어로 보기 어려움

### 수정할 파일

**`synonym_stopword.py`** — STOPWORDS 집합에 아래 단어 추가:

```python
# 신문사명 (기사 저작권 문구에 등장)
"동아일보", "한겨레", "한국일보",

# 저작권 문구 토큰
"금지", "재배포", "무단전재", "저작권",

# 경량 시간·지시 명사 (GEXF 노드 발견)
"지난해", "이후", "동안", "이상", "사실", "자신", "시작",
"이전", "현재", "지금", "그동안",
```

### 수정 후 재생성 절차

`sent_tokens_cache.pkl`이 이미 존재하므로 **재토큰화 없이** 빠르게 재생성 가능.

`text_mining.ipynb` — setup 셀(2a46b4fe) 재실행 후 아래 셀들만 순서대로 재실행:
1. sent_tokenize 셀 (캐시 로드, 즉시)
2. global_freq → vocab 셀
3. build_cooc → cooc_global.csv 덮어쓰기
4. yearly_cooc → cooc_2016~2025.csv 덮어쓰기
5. GEXF 재생성

---

## Step 5. 빈도 분석 시각화 (RQ1)

**목표:** "청년과 관련해 주로 언급되는 단어는 무엇인가?"

### 구현 (text_mining.ipynb에 셀 추가)

**셀 V1 — 전체 기간 Top 20 막대그래프**
```python
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
# 한글 폰트 설정 (Windows: 맑은 고딕)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

top20 = global_freq.most_common(20)
words, counts = zip(*top20)
plt.figure(figsize=(12, 6))
plt.bar(words, counts)
plt.title('청년 담론 전체 기간 빈출 단어 Top 20 (2016–2025)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(output_dir / 'fig_top20_bar.png', dpi=150)
plt.show()
```

**셀 V2 — 연도별 Top 10 Heatmap (RQ3 시각화)**
```python
import pandas as pd, seaborn as sns

# 연도별 빈도를 pivot table로 변환
yearly_freq = {}
for year, grp in df.groupby("year"):
    freq = Counter()
    for sent_list in grp["sent_tokens"]:
        for sent in sent_list:
            freq.update(sent)
    yearly_freq[year] = freq

# Top 30 단어 × 연도 매트릭스
top30_words = [w for w, _ in global_freq.most_common(30)]
heat_df = pd.DataFrame(
    {year: [yearly_freq[year].get(w, 0) for w in top30_words] for year in sorted(yearly_freq)},
    index=top30_words
)
# 연도별 정규화 (연도 간 기사 수 차이 보정)
heat_norm = heat_df.div(heat_df.sum(axis=0), axis=1) * 1000

plt.figure(figsize=(14, 10))
sns.heatmap(heat_norm, cmap='YlOrRd', annot=False, fmt='.1f', linewidths=0.3)
plt.title('연도별 주요 단어 상대빈도 Heatmap (‰)')
plt.tight_layout()
plt.savefig(output_dir / 'fig_yearly_heatmap.png', dpi=150)
plt.show()
```

**셀 V3 — 워드클라우드**
```python
from wordcloud import WordCloud

wc = WordCloud(font_path='C:/Windows/Fonts/malgun.ttf',
               background_color='white', width=1200, height=600,
               max_words=100)
wc.generate_from_frequencies(dict(global_freq))
plt.figure(figsize=(15, 7))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title('청년 담론 워드클라우드 (2016–2025)')
plt.savefig(output_dir / 'fig_wordcloud.png', dpi=150)
plt.show()
```

---

## Step 6. 의미연결망 분석 (RQ2)

**목표:** 중심성 3종 계산 + Gephi 시각화 준비

### 6-1. Python 중심성 분석 (text_mining.ipynb)

**셀 N1 — 중심성 3종 계산**
```python
import networkx as nx

deg = nx.degree_centrality(G_global)
btw = nx.betweenness_centrality(G_global, weight='weight')
eig = nx.eigenvector_centrality(G_global, weight='weight', max_iter=1000)

# DataFrame으로 정리
centrality_df = pd.DataFrame({'degree': deg, 'betweenness': btw, 'eigenvector': eig})
centrality_df = centrality_df.sort_values('degree', ascending=False)
centrality_df.to_csv(output_dir / 'centrality_global.csv', encoding='utf-8-sig')

print("=== 연결정도 중심성 Top 10 ===")
print(centrality_df['degree'].nlargest(10))
print("\n=== 매개 중심성 Top 10 ===")
print(centrality_df['betweenness'].nlargest(10))
```

**셀 N2 — 기본 네트워크 지표**
```python
print(f"노드 수: {G_global.number_of_nodes()}")
print(f"엣지 수: {G_global.number_of_edges()}")
print(f"밀도: {nx.density(G_global):.4f}")
print(f"포괄성: {nx.number_of_nodes(G_global) / 150:.3f}")
# 평균 연결거리 (연결된 성분만)
giant = G_global.subgraph(max(nx.connected_components(G_global), key=len))
print(f"평균 연결거리: {nx.average_shortest_path_length(giant):.3f}")
```

### 6-2. Gephi 시각화 절차 (수동 작업)

1. Gephi 열기 → `network_global.gexf` 로드
2. *Statistics* → **Modularity** 실행 (Resolution: 1.0)
3. *Appearance* → 노드 색상: Modularity Class / 크기: Degree
4. *Layout* → **ForceAtlas 2** (Prevent Overlap 체크, Scaling: 20)
5. Labels 표시 → PNG 고해상도 export (`fig_network_global.png`)

---

## Step 7. 연도별 비교 분석 (RQ3)

**목표:** 청년 담론의 시계열 변화 실증

### 구현 (text_mining.ipynb)

**셀 Y1 — 특정 키워드 연도별 빈도 추이**
```python
# 분석 관심 단어 (연구에서 의미 있는 변화가 예상되는 단어)
keywords = ['일자리', 'MZ세대', '공정', '주거', '저출산', '코로나', '세대갈등']

trend_df = pd.DataFrame(
    {w: [yearly_freq[y].get(w, 0) for y in sorted(yearly_freq)] for w in keywords},
    index=sorted(yearly_freq)
)
# 정규화
trend_norm = trend_df.div(
    [sum(yearly_freq[y].values()) for y in sorted(yearly_freq)], axis=0
) * 10000

trend_norm.plot(figsize=(14, 6), marker='o')
plt.title('주요 단어 연도별 빈도 추이 (만분율)')
plt.xlabel('연도'); plt.ylabel('빈도 (‰‰)')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig(output_dir / 'fig_keyword_trend.png', dpi=150)
plt.show()
```

**셀 Y2 — 연도별 중심성 변화 (매개 중심성 Top 10 Bump Chart)**
```python
# 연도별 G에서 betweenness 계산
yearly_btw = {}
for year, cooc in yearly_cooc.items():
    G_y = make_graph(cooc, min_weight=2)
    yearly_btw[year] = nx.betweenness_centrality(G_y, weight='weight')

top10_words = centrality_df['betweenness'].nlargest(10).index.tolist()
rank_df = pd.DataFrame(
    {year: pd.Series(yearly_btw[year]).rank(ascending=False) for year in sorted(yearly_btw)}
).loc[top10_words]

rank_df.T.plot(figsize=(14, 6), marker='o')
plt.title('주요 단어 매개 중심성 순위 변화 (Bump Chart)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(output_dir / 'fig_bump_chart.png', dpi=150)
plt.show()
```

---

## 실행 순서 요약

```
[즉시] synonym_stopword.py STOPWORDS 보완
      → cooc/GEXF 재생성 (캐시 덕분에 ~5분)

[Step 5 / RQ1]
  셀 V1: 전체 Top 20 막대그래프
  셀 V2: 연도별 Heatmap
  셀 V3: 워드클라우드

[Step 6 / RQ2]
  셀 N1: 중심성 3종 CSV 저장
  셀 N2: 네트워크 기본 지표
  (Gephi 수동) network_global.gexf 시각화

[Step 7 / RQ3]
  셀 Y1: 키워드 연도별 추이 선 그래프
  셀 Y2: 매개 중심성 Bump Chart
  (Gephi 수동) network_2016 vs 2025 비교 레이아웃
```

---

## 수정 대상 파일

| 파일 | 변경 내용 |
|------|----------|
| `synonym_stopword.py` | STOPWORDS에 신문사명·저작권어·경량어 추가 |
| `text_mining.ipynb` | 셀 V1~V3, N1~N2, Y1~Y2 추가 |
| `output/*.csv, *.gexf` | 재생성 (덮어쓰기) |

---

## 검증 포인트

1. **cooc_global.csv 1위 쌍** — `금지,동아일보`가 사라지고 `정부,정책` 등 의미 있는 쌍이 1위 탈환 여부
2. **GEXF 노드 목록** — 신문사명·경량어 미포함 확인
3. **시각화 한글 폰트** — 맑은 고딕 렌더링 확인
4. **참고 논문 수치 비교** — 밀도(논문 4.86), 노드 수(논문 110) 대비 본 연구 수치 기록
