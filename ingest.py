import uuid
import chromadb
from fastapi import FastAPI, Request
import uvicorn

# ChromaDB クライアントの初期化
chroma_db_client = chromadb.PersistentClient()
collection = chroma_db_client.get_or_create_collection(name="logs")

app = FastAPI()

# ログ受信エンドポイント
@app.post("/ingest")
async def receive_log(request: Request):

    data = await request.json()

    for entry in data:
        ts = entry["date"] # ログのタイムスタンプ（UNIX timestamp 形式）を取得
        log = entry["log"] # ログ本文を取得

        # ChromaDBのメタデータにログのタイムスタンプを追加
        metadatas = []
        metadatas.append({"timestamp": ts})

        # ChromaDBにログを追加するための一意なIDを生成
        unique_id = str(uuid.uuid4())

        # ChromaDBにログを追加
        collection.add(
            documents=[log],
            ids=[unique_id],
            metadatas=metadatas,
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")