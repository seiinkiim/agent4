# app.py
# --------------------------- Imports ---------------------------
import os
import re
import time
import random
from collections import deque

import pandas as pd
import streamlit as st

# RAG / LangChain (LLM은 사용하지 않음)
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory


# --------------------------- Streamlit 기본 ---------------------------
st.set_page_config(page_title="운동화 쇼핑 에이전트")
st.title("운동화 쇼핑 에이전트")
os.environ["OPENAI_API_KEY"] = st.secrets.get("OPENAI_API_KEY", "")

# 세션 상태
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "store" not in st.session_state:
    st.session_state["store"] = dict()
if "selected_question" not in st.session_state:
    st.session_state["selected_question"] = None
if "followup_step" not in st.session_state:
    st.session_state["followup_step"] = 0
if "seen_products" not in st.session_state:
    st.session_state["seen_products"] = set()  # "브랜드||제품명"
if "random_pool" not in st.session_state:
    st.session_state["random_pool"] = None  # 최초 로드 후 채움

SESSION_ID = "sneaker-chat"
CSV_PATH = "shoes_top12.csv"  # 파일명 맞게 변경 가능


# --------------------------- 후속질문(고정 6개) ---------------------------
followup_set_1 = {
    "Q1": "러닝이나 마라톤에서 더 좋은 기록을 내거나 오래 달릴 수 있도록 도움을 줄 수 있는 신발은 무엇인가요?",
    "Q2": "다른 러너들과 나란히 달릴 때, 나를 더 편안하고 자신감 있게 만드는 운동화가 있을까요?",
    "Q3": "새로운 코스를 달릴 때, 발걸음을 가볍게 만들어줄 신발은 어떤 신발일까요?",
}
followup_set_2 = {
    "Q4": "러닝 대회에서 피니시 라인을 통과할 때, 나를 더 돋보이게 만들어줄 신발이 있을까요?",
    "Q5": "깔끔하고 세련된 스타일의 운동화로 어떤 것이 있나요?",
    "Q6": "신발을 신을 때 ‘와, 이건 정말 내 신발이다! ’라고 느낄 만큼 만족을 주는 운동화가 있을까요?",
}


# --------------------------- 데이터/RAG 준비 ---------------------------
@st.cache_resource(show_spinner=True)
def load_product_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    # 구매링크 없는 행 제외
    df = df[df["구매링크"].notna() & (df["구매링크"].astype(str).str.strip() != "")].copy()
    df.reset_index(drop=True, inplace=True)
    return df

df_products = load_product_df(CSV_PATH)

# 세션 랜덤 풀 초기화
if st.session_state["random_pool"] is None:
    st.session_state["random_pool"] = deque(range(len(df_products)))
    random.shuffle(st.session_state["random_pool"])


@st.cache_resource(show_spinner=True)
def build_retriever(csv_path: str):
    loader = CSVLoader(csv_path, encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    splits = splitter.split_documents(docs)
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.from_documents(splits, embedding=embedding)
    # 후보 폭을 넉넉히 (CSV 내에서만 검색)
    return vs.as_retriever(search_kwargs={"k": 20})

retriever = build_retriever(CSV_PATH)


# --------------------------- 유틸 함수 ---------------------------
def stream_text(text: str, delay: float = 0.015):
    """단어 단위 스트리밍 출력 (LLM 아님, 로컬 텍스트용)"""
    ph = st.empty()
    buf = ""
    for chunk in re.split(r"(\s+)", text):  # 공백 보존
        buf += chunk
        ph.write(buf)
        time.sleep(delay)


def _norm(s: str) -> str:
    """공백/기호 제거 + 소문자화로 중복 키 정규화"""
    s = str(s).lower()
    s = re.sub(r"[\s\-\u2013_/\\.,()]+", "", s)
    return s

def product_key(brand: str, name: str) -> str:
    return f"{_norm(brand)}||{_norm(name)}"


def draw_random_products(n: int = 3) -> str:
    """세션 내 중복 없이 무작위 n개 추출해서 표시하고, seen에 등록 (CSV만)"""
    pool: deque = st.session_state["random_pool"]
    chosen_idx = []
    target = min(n, len(df_products))
    while len(chosen_idx) < target:
        if not pool:  # 전부 소진하면 다시 섞어서 순환
            pool.extend(range(len(df_products)))
            random.shuffle(pool)
        idx = pool.popleft()
        row = df_products.iloc[idx]
        key = product_key(row["브랜드"], row["제품명"])
        if key in st.session_state["seen_products"]:
            continue
        chosen_idx.append(idx)

    lines = []
    for i, idx in enumerate(chosen_idx, start=1):
        row = df_products.iloc[idx]
        st.session_state["seen_products"].add(product_key(row["브랜드"], row["제품명"]))
        lines.append(
            f"{i}. {row['브랜드']} {row['제품명']} | {row['가격']} | {row['제품설명']} | {row['구매링크']}"
        )
    return "\n".join(lines)


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state["store"]:
        st.session_state["store"][session_id] = ChatMessageHistory()
    return st.session_state["store"][session_id]


def build_query_from_history_and_input(
    history: BaseChatMessageHistory, user_input: str, max_turns: int = 4
) -> str:
    msgs = history.messages[-max_turns * 2 :] if hasattr(history, "messages") else []
    hist_text = []
    for m in msgs:
        role = getattr(m, "type", getattr(m, "role", ""))
        content = getattr(m, "content", "")
        if role in ("human", "user", "ai", "assistant"):
            hist_text.append(f"{role}: {content}")
    hist_blob = "\n".join(hist_text)
    return f"{hist_blob}\nuser: {user_input}\n\n요약 키워드: 운동 목적, 쿠션, 통풍, 경량/안정, 굽 높이, 브랜드 선호"


def filter_unseen_docs(docs):
    """세션에서 이미 본 상품 제거 (CSV 문서만 다룸)"""
    filtered = []
    for d in docs:
        t = d.page_content
        brand_m = re.search(r"브랜드\s*[:=]\s*(.+)", t)
        name_m = re.search(r"제품명\s*[:=]\s*(.+)", t)
        brand = brand_m.group(1).strip() if brand_m else ""
        name = name_m.group(1).strip() if name_m else ""
        if product_key(brand, name) not in st.session_state["seen_products"]:
            filtered.append(d)
    return filtered


def docs_to_rows(docs):
    """문서 -> 구조화 dict 리스트 (CSV만, 로컬 중복 제거)"""
    items = []
    local_keys = set()
    for d in docs:
        t = d.page_content

        def grab(field):
            m = re.search(rf"{field}\s*[:=]\s*(.+)", t)
            return m.group(1).strip() if m else ""

        brand = grab("브랜드")
        name = grab("제품명")
        price = grab("가격")
        desc = grab("제품설명")
        url = grab("구매링크")
        if not url:
            continue

        key = product_key(brand, name)
        if key in local_keys:
            continue
        local_keys.add(key)

        items.append({"brand": brand, "name": name, "price": price, "desc": desc, "url": url})
    return items


def topup_with_unseen(rows, need: int):
    """rows가 3개 미만이면 CSV에서 아직 안 본 제품(세션 기준)으로만 채움"""
    if need <= 0:
        return rows
    have_keys = {product_key(r["brand"], r["name"]) for r in rows}

    candidates = []
    for _, r in df_products.iterrows():
        key = product_key(r["브랜드"], r["제품명"])
        if key in have_keys or key in st.session_state["seen_products"]:
            continue
        candidates.append(
            {"brand": r["브랜드"], "name": r["제품명"], "price": r["가격"], "desc": r["제품설명"], "url": r["구매링크"]}
        )

    random.shuffle(candidates)
    rows.extend(candidates[:need])
    return rows


# ---------- 키워드 매핑 & 랭킹 ----------
def get_keywords(user_input: str):
    ui = str(user_input).lower()
    if ("가벼운" in ui) or ("경량" in ui) or ("light" in ui):
        return ["가벼운", "경량", "라이트", "light", "민첩", "빠른", "레이싱"]
    if ("무게감" in ui) or ("안정감" in ui) or ("지지" in ui) or ("support" in ui):
        return ["안정", "안정감", "지지", "서포트", "발목지지", "견고", "로커", "컨트롤"]
    if ("통풍" in ui) or ("메쉬" in ui) or ("메시" in ui) or ("breath" in ui):
        return ["통풍", "메쉬", "메시", "통기", "시원", "에어리"]
    if "쿠션" in ui or "충격" in ui or "부드럽" in ui:
        return ["쿠션", "충격", "흡수", "부드럽", "폭신", "소프트", "편안"]
    if ("평평" in ui) or ("플랫" in ui) or ("낮은 굽" in ui) or ("로우드롭" in ui):
        return ["평평", "플랫", "낮은굽", "로우드롭", "0mm", "4mm", "플랫솔"]
    if "굽" in ui or "드롭" in ui or "높" in ui:
        return ["굽", "드롭", "힐", "높이", "힐투토", "stack"]
    return []  # 자유 입력이면 키워드 랭킹 생략

def keyword_score(text: str, keywords: list[str]) -> int:
    if not keywords:
        return 0
    t = str(text).lower()
    score = 0
    for kw in keywords:
        k = kw.lower()
        score += t.count(k)
    return score

def rank_by_keywords(df: pd.DataFrame, keywords: list[str], exclude_keys: set[str], top_k: int = 12):
    """키워드 매칭 점수로 CSV 내 제품을 랭킹 (세션에서 이미 본 제품 제외)"""
    if not keywords:
        return []
    rows = []
    for _, r in df.iterrows():
        key = product_key(r["브랜드"], r["제품명"])
        if key in exclude_keys:
            continue
        if not isinstance(r.get("구매링크", ""), str) or not r["구매링크"].strip():
            continue
        name_desc = f"{r['브랜드']} {r['제품명']} {r['제품설명']}"
        sc = keyword_score(name_desc, keywords)
        if sc <= 0:
            continue
        rows.append({
            "brand": r["브랜드"], "name": r["제품명"], "price": r["가격"],
            "desc": r["제품설명"], "url": r["구매링크"], "score": sc
        })
    random.shuffle(rows)
    rows.sort(key=lambda x: x["score"], reverse=True)
    for r in rows:
        r.pop("score", None)
    return rows[:top_k]


def rows_to_output(rows):
    """dict 리스트 -> 최종 출력 문자열"""
    return "\n".join(
        f"{i}. {r['brand']} {r['name']} | {r['price']} | {r['desc']} | {r['url']}"
        for i, r in enumerate(rows, start=1)
    )


# --------------------------- 이전 대화 출력 ---------------------------
for role, msg in st.session_state["messages"]:
    st.chat_message(role).write(msg)


# --------------------------- 입력 ---------------------------
user_input = None
if st.session_state["selected_question"]:
    user_input = st.session_state["selected_question"]
    st.session_state["selected_question"] = None
else:
    tmp = st.chat_input("메시지를 입력해 주세요")
    if tmp:
        user_input = tmp


# --------------------------- 응답 처리 ---------------------------
if user_input:
    st.chat_message("user").write(user_input)
    st.session_state["messages"].append(("user", user_input))

    # 첫 요청: 무작위 3개 (세션 중복 방지) + 단어 단위 스트리밍 출력
    if user_input.strip() == "운동화 추천해줘" and st.session_state["followup_step"] == 0:
        random_reco = draw_random_products(3)
        with st.chat_message("assistant"):
            stream_text(random_reco, delay=0.015)
        st.session_state["messages"].append(("assistant", random_reco))
        st.session_state["followup_step"] = 1  # 패널 1세트 노출

    # 이후: CSV만 기반 추천 (키워드 랭킹 → 벡터검색 보충 → 미본 보충)
    else:
        # 0) 키워드 우선 랭킹
        keywords = get_keywords(user_input)
        exclude = set(st.session_state["seen_products"])
        rows = rank_by_keywords(df_products, keywords, exclude_keys=exclude, top_k=12)[:3]

        # 1) 부족하면 벡터검색(RAG)으로 보충
        if len(rows) < 3:
            history = get_session_history(SESSION_ID)
            query = build_query_from_history_and_input(history, user_input)
            rag_docs = retriever.get_relevant_documents(query)
            rag_docs_filtered = filter_unseen_docs(rag_docs)
            extra = []
            existing_keys = {product_key(r["brand"], r["name"]) for r in rows}
            for r in docs_to_rows(rag_docs_filtered):
                k = product_key(r["brand"], r["name"])
                if k not in existing_keys and k not in exclude:
                    extra.append(r)
                    existing_keys.add(k)
                if len(rows) + len(extra) >= 3:
                    break
            rows.extend(extra)

        # 2) 그래도 부족하면 CSV 미본(unseen)으로 최종 보충
        if len(rows) < 3:
            rows = topup_with_unseen(rows, 3 - len(rows))
        rows = rows[:3]  # 안전 가드

        # 3) seen 등록
        for r in rows:
            st.session_state["seen_products"].add(product_key(r["brand"], r["name"]))

        # 4) CSV 데이터만으로 최종 출력 + 스트리밍
        out_text = rows_to_output(rows)
        with st.chat_message("assistant"):
            stream_text(out_text, delay=0.015)
        st.session_state["messages"].append(("assistant", out_text))

        # 5) 패널 단계 진행(1세트 → 2세트 → 마지막 숨김 + 인증번호 안내)
        if st.session_state["followup_step"] == 1:
            st.session_state["followup_step"] = 2
        elif st.session_state["followup_step"] == 2:
            st.session_state["followup_step"] = 3
            code = f"{random.randint(0, 9999):04d}"
            end_msg = f"대화가 종료되었습니다. 인증번호는 {code} 입니다"
            st.chat_message("assistant").write(end_msg)
            st.session_state["messages"].append(("assistant", end_msg))


# --------------------------- 후속질문 패널 ---------------------------
def render_followup_panel(step: int):
    # step 1, 2일 때만 노출 (마지막엔 숨김)
    if step not in (1, 2):
        return
    st.markdown("### 이런 질문도 해보세요!")
    qset = followup_set_1 if step == 1 else followup_set_2
    for key, question in qset.items():
        col_q, col_btn = st.columns([8, 1])
        col_q.markdown(f"**{key}.** {question}")
        if col_btn.button("➕", key=f"btn_{key}"):
            st.session_state["selected_question"] = question
            st.rerun()

render_followup_panel(st.session_state["followup_step"])
