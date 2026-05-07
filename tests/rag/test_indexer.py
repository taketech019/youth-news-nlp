import pandas as pd
from rag_chatbot.indexer import load_documents


def test_load_documents_count(mock_csv):
    docs = load_documents(mock_csv)
    assert len(docs) == 9  # 3 신문사 × 3 행


def test_load_documents_metadata(mock_csv):
    docs = load_documents(mock_csv)
    newspapers = {d.metadata["newspaper"] for d in docs}
    years = {d.metadata["year"] for d in docs}
    assert newspapers == {"동아일보", "한겨레", "한국일보"}
    assert "2020" in years and "2021" in years and "2022" in years
    assert all("title" in d.metadata and "date" in d.metadata for d in docs)


def test_load_documents_skips_null_text(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame([
        {"title": "test", "text_cleaned": None, "year": "2020", "date": "202001"}
    ]).to_csv(bad_csv, index=False, encoding="utf-8-sig")
    docs = load_documents({"동아일보": str(bad_csv)})
    assert len(docs) == 0


def test_load_documents_splits_sentences(mock_csv):
    docs = load_documents(mock_csv)
    # "청년 취업 현황" 기사는 3문장 → 문장 분리 후 "\n" 2개 포함
    target = next(d for d in docs if d.metadata["title"] == "청년 취업 현황")
    assert target.text.count("\n") == 2
