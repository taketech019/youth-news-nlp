import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
CHROMA_DB_PATH = str(BASE_DIR / "output" / "chroma_db")
CHROMA_COLLECTION = "chunyeon_news"
EMBED_MODEL_NAME = "ibm-granite/granite-embedding-97m-multilingual-r2"
NIM_API_KEY = os.environ.get("NVIDIA_NIM_API_KEY", "")
NIM_BASE_URL = os.environ.get("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
NIM_MODEL = os.environ.get("NVIDIA_NIM_MODEL", "moonshotai/kimi-k2.6")
TOP_K = 5
DATA_FILES = {
    "동아일보": str(BASE_DIR / "preprocessed_data" / "청년_동아일보_정제완료_EA13796.csv"),
    "한겨레":   str(BASE_DIR / "preprocessed_data" / "청년_한겨레_정제완료_EA10690.csv"),
    "한국일보": str(BASE_DIR / "preprocessed_data" / "청년_한국일보_정제완료_EA10813.csv"),
}
SYSTEM_PROMPT = (
    "당신은 국내 신문 기사를 기반으로 청년 이슈를 분석하는 어시스턴트입니다.\n"
    "주어진 기사 내용만을 근거로 답변하고, 반드시 출처(신문사, 연도, 제목)를 함께 제시하세요."
)
