from rag_chatbot.retriever import (
    extract_year,
    extract_newspaper,
    build_metadata_filters,
)


def test_extract_year_with_marker():
    assert extract_year("2022년 청년 취업 현황은?") == "2022"


def test_extract_year_without_marker():
    assert extract_year("2019 청년 정책 변화") == "2019"


def test_extract_year_returns_none():
    assert extract_year("청년 주거 문제가 뭐야?") is None


def test_extract_newspaper_single():
    assert extract_newspaper(["동아일보"]) == "동아일보"


def test_extract_newspaper_multiple_returns_none():
    assert extract_newspaper(["동아일보", "한겨레"]) is None


def test_extract_newspaper_all_returns_none():
    assert extract_newspaper(["동아일보", "한겨레", "한국일보"]) is None


def test_build_metadata_filters_year_only():
    f = build_metadata_filters(year="2022", newspaper=None)
    assert f is not None
    assert len(f.filters) == 1
    assert f.filters[0].key == "year"
    assert f.filters[0].value == "2022"


def test_build_metadata_filters_newspaper_only():
    f = build_metadata_filters(year=None, newspaper="한겨레")
    assert f is not None
    assert f.filters[0].key == "newspaper"
    assert f.filters[0].value == "한겨레"


def test_build_metadata_filters_combined():
    f = build_metadata_filters(year="2020", newspaper="동아일보")
    assert len(f.filters) == 2


def test_build_metadata_filters_none_when_no_constraints():
    assert build_metadata_filters(year=None, newspaper=None) is None
