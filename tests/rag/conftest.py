import pytest
import pandas as pd

MOCK_ROWS = [
    {
        "title": "청년 취업 현황",
        "text_cleaned": "청년 취업률이 올해 크게 하락했다. 정부는 대책 마련에 나섰다. 전문가들은 구조적 문제라고 지적한다.",
        "year": "2020",
        "date": "202003",
    },
    {
        "title": "청년 주거 문제",
        "text_cleaned": "청년층의 주거 불안이 심화되고 있다. 월세 부담이 급증하고 있다.",
        "year": "2021",
        "date": "202106",
    },
    {
        "title": "청년 정책 발표",
        "text_cleaned": "정부가 청년 정책을 발표했다. 일자리 지원이 핵심이다.",
        "year": "2022",
        "date": "202201",
    },
]


@pytest.fixture
def mock_csv(tmp_path) -> dict:
    files = {}
    for newspaper in ["동아일보", "한겨레", "한국일보"]:
        path = tmp_path / f"청년_{newspaper}_test.csv"
        pd.DataFrame(MOCK_ROWS).to_csv(path, index=False, encoding="utf-8-sig")
        files[newspaper] = str(path)
    return files
