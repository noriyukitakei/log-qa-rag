import json
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st
import chromadb
from google import genai
from google.genai import types

load_dotenv(verbose=True)

# Azure OpenAI クライアントの初期化
llm_client = genai.Client()

# ChromaDB クライアントの初期化
chroma_db_client = chromadb.PersistentClient()
collection = chroma_db_client.get_or_create_collection(name="logs")

# クエリから日時レンジを抽出する関数
def extract_time_range_from_query(query: str):
    """
    ユーザーの日本語クエリから検索したい時間範囲を抽出する。タイムゾーンは考慮せずユーザーからリクエストのあった日時をUTCとして扱う。
    返り値: (start_ts, end_ts) いずれも UNIX timestamp (float) or None
    例:
      入力: 2025年1月14日の17時から18時に起きたOOM-killerの原因をまとめて教えて
      出力: (timestamp("2025-01-14T17:00:00+00:00"), timestamp("2025-01-14T18:00:00+00:00"))
    """
    system_prompt = """
あなたは日時抽出専用のアシスタントである。
ユーザーの日本語の質問文から、「検索したい時間範囲（開始と終了）」だけを抽出し、
次の JSON 形式で出力せよ。

{
  "start": "2025-01-14T17:00:00+00:00",
  "end":   "2025-01-14T18:00:00+00:00"
}

制約:
- 出力は JSON オブジェクト 1 個のみとする。
- 出力の前後に改行や説明文を付けてはならない。
- 「```」で囲むコードブロックや「json」という文字列を絶対に出力してはならない。
  - 悪い例: ```json\n{...}\n```
- 出力は必ず次のような 1 行の JSON だけにすること:
  {"start": "...","end": "..."}
- ISO8601 形式 (YYYY-MM-DDTHH:MM:SS) で出力すること。
- タイムゾーンは考慮せず、そのままローカル時刻でよい。つまり必ず +00:00 を付与すること(例: 2025-01-14T17:00:00+00:00)。
- もし終了時刻が明示されていない場合は、開始から1時間後を end としてよい。
- 日付が特定できない場合は、今から 24 時間以内の範囲を仮置きでよいが、
  その場合でも必ず上記 JSON フォーマットで出力せよ。
- JSON 以外の文字は一切出力してはならない。
    """.strip()

    # LLM に問い合わせて日時レンジを抽出
    # システムプロンプトを system ロールで渡し、ユーザーのクエリを user ロールで渡す
    resp = llm_client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=system_prompt),
        contents=query
    )

    content = resp.text

    # 抽出した JSON から日時をパースして UNIX timestamp に変換
    try:
        data = json.loads(content)
        start_str = data.get("start")
        end_str = data.get("end")
        start_ts = datetime.fromisoformat(start_str).timestamp() if start_str else None
        end_ts = datetime.fromisoformat(end_str).timestamp() if end_str else None
        return start_ts, end_ts
    except Exception:
        # 日付抽出に失敗したため、時間フィルタなしで検索
        return None, None
    
def search_logs(query: str) -> str:
    # ---- ユーザー質問例 ----
    # query = "2025年1月13日の11時ごろに起きたOOM-killerの原因をまとめて教えて"
    
    # ユーザーの質問に日時の指定がある場合、それを抽出してフィルタに利用
    start_ts, end_ts = extract_time_range_from_query(query)
    where_filter = None
    if start_ts is not None and end_ts is not None:
        where_filter = {
            "$and": [
                {"timestamp": {"$gte": start_ts}},
                {"timestamp": {"$lte": end_ts}}
            ]
        }
    
    # ChromaDBから関連ログを検索
    results = collection.query(
        query_texts=[query],
        n_results=5,  # 上位5件のみ
        where=where_filter,  # 日時フィルタを適用（取れなければ None のまま）
    )
    
    retrieved_logs = "\n".join(results["documents"][0]) if results["documents"] else ""
    
    # LLMに質問とログを渡して回答を生成
    prompt = f"""
    以下はサーバーログの一部である。このログをもとに質問に答えよ。
    
    ログ:
    {retrieved_logs}
    
    質問:
    {query}
    
    答えは簡潔にまとめてください。
    """
    
    # LLMに問い合わせて回答を取得
    response = llm_client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt
    )
    
    return response.text

# ここからは画面を構築するためのコード
# チャット履歴を初期化する。
if "history" not in st.session_state:
    st.session_state["history"] = []

# チャット履歴を表示する。
for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ユーザーが質問を入力したときの処理を記述する。
if prompt := st.chat_input("質問を入力してください"):

    # ユーザーが入力した質問を表示する。
    with st.chat_message("user"):
        st.write(prompt)

    # ユーザの質問をチャット履歴に追加する
    st.session_state.history.append({"role": "user", "content": prompt})

    # ユーザーの質問に対して回答を生成するためにsearch_logs関数を呼び出す。
    response = search_logs(prompt)

    # 回答を表示する。
    with st.chat_message("assistant"):
        st.write(response)

    # 回答をチャット履歴に追加する。
    st.session_state.history.append({"role": "assistant", "content": response})