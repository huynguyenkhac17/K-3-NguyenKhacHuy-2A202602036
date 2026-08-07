"""
Lab 11 — Configuration & API Key Setup
"""
import os


def setup_api_key():
    """Load Google API key from .env / environment, hoặc hỏi nếu chưa có.

    Tự nạp .env ở gốc repo trước: luồng README dặn dán key vào .env, nên
    ``python main.py --part X`` phải chạy được ngay mà không hỏi lại. Chỉ prompt
    khi chạy tương tác và thực sự thiếu key (part 4/HITL vốn không cần key).
    """
    if "GOOGLE_API_KEY" not in os.environ:
        try:
            from dotenv import load_dotenv
            from pathlib import Path

            # core/config.py → src → repo root
            load_dotenv(Path(__file__).resolve().parents[2] / ".env")
        except ImportError:
            pass

    if "GOOGLE_API_KEY" not in os.environ:
        import sys

        if sys.stdin and sys.stdin.isatty():
            os.environ["GOOGLE_API_KEY"] = input("Enter Google API Key: ")
        else:
            print(
                "Chưa có GOOGLE_API_KEY (đặt trong .env hoặc biến môi trường). "
                "Các phần cần LLM sẽ lỗi; part 4 (HITL) vẫn chạy offline."
            )
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "0"
    print("API key loaded.")


# Allowed banking topics (used by topic_filter)
ALLOWED_TOPICS = [
    "banking", "account", "transaction", "transfer",
    "loan", "interest", "savings", "credit",
    "deposit", "withdrawal", "balance", "payment",
    "tai khoan", "giao dich", "tiet kiem", "lai suat",
    "chuyen tien", "the tin dung", "so du", "vay",
    "ngan hang", "atm",
]

# Blocked topics (immediate reject)
BLOCKED_TOPICS = [
    "hack", "exploit", "weapon", "drug", "illegal",
    "violence", "gambling", "bomb", "kill", "steal",
]
