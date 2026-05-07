import re
import chromadb
import pandas as pd
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from rag_chatbot.config import (
    CHROMA_DB_PATH, CHROMA_COLLECTION, EMBED_MODEL_NAME, DATA_FILES
)

_SENT_RE = re.compile(r'(?<=[.!?])\s+')


def load_documents(data_files: dict | None = None) -> list[Document]:
    """CSV에서 기사 로드 → 문장 분리 → LlamaIndex Document 목록 반환.

    tokens 컬럼은 사용하지 않음. text_cleaned를 granite 임베딩 모델의
    자체 토크나이저로 직접 처리하기 위해 정규식으로만 문장 경계를 표시한다.
    """
    if data_files is None:
        data_files = DATA_FILES

    docs = []
    for newspaper, path in data_files.items():
        df = pd.read_csv(path, encoding="utf-8-sig")
        for _, row in df.iterrows():
            if not isinstance(row.get("text_cleaned"), str):
                continue
            sentences = _SENT_RE.split(row["text_cleaned"].strip()) or [row["text_cleaned"]]
            text = "\n".join(sentences)
            docs.append(Document(
                text=text,
                metadata={
                    "year":      str(row.get("year", "")),
                    "newspaper": newspaper,
                    "title":     str(row.get("title", "")),
                    "date":      str(row.get("date", "")),
                },
            ))
    return docs


def _get_chroma_collection(
    path: str = CHROMA_DB_PATH,
    collection_name: str = CHROMA_COLLECTION,
):
    client = chromadb.PersistentClient(path=path)
    collection = client.get_or_create_collection(collection_name)
    return client, collection


def load_or_build_index(
    chroma_path: str = CHROMA_DB_PATH,
    collection_name: str = CHROMA_COLLECTION,
) -> VectorStoreIndex:
    """ChromaDB 컬렉션이 있으면 로드, 없으면 빌드."""
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    Settings.embed_model = embed_model

    client, collection = _get_chroma_collection(chroma_path, collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)

    if collection.count() > 0:
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )

    return _build_index(vector_store)


def _build_index(vector_store: ChromaVectorStore) -> VectorStoreIndex:
    docs = load_documents()
    splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=20,
        paragraph_separator="\n",
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        transformations=[splitter],
        show_progress=True,
    )
