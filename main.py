# app.py
# --------------------------- Imports ---------------------------
import os
import re
import time
import random
from collections import deque

import pandas as pd
import streamlit as st

# RAG / LangChain
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_openai import ChatOpenAI  # ⬅️ LLM 사용

# --------------------------- App Config ---------------------------
st.set_page_config(page_title="운동화 쇼핑 에이전트")
st.title("운동화 쇼핑 에이전트")
os.environ["OPENAI_API_KEY"] = st.secrets.get("OPENAI_API_KEY", "")

# LLM 설정 (필요시 바꾸세요)
USE_LLM_DESC = True
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.4
LLM_MAX_TOKENS = 160  # 1~2문장

# 매 추천 출력 시 함께 보여줄 안내 문구
PREFACE = "네 알겠습니다. 귀하의 질문에 맞는 운동화를 추천해드리겠습니다."

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
    "Q1": "가벼운 운동화는 어떤 제품이 있나요?",
    "Q2": "무게감 있고 안정감 있는 제품은 무엇이 있나요?",
    "Q3": "통풍이 좋은 운동화 제품은 무엇이 있나요?",
}
followup_set_2 = {
    "Q4": "쿠션감이 좋은 제품은 무엇이 있나요?",
    "Q5": "평평한 운동화는 무엇이 있나요?",
    "Q6": "약간의 굽이 있는 제품은 무엇이 있나요?",
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
    return vs.as_retriever(search_kwargs={"k": 20})

retriever = build_retriever(CSV_PATH)

# --------------------------- 유틸 함수 ---------------------------
def stream_text(text: str, delay: float = 0.015):
    ph = st.empty()
    buf = ""
    for chunk in re.split(r"(\s+)", text):  # 공백 보존
        buf += chunk
        ph.markdown(buf)
        time.sleep(delay)

def _norm(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[\s\-\u2013_/\\.,()]+", "", s)
    return s

def product_key(brand: str, name: str) -> str:
    return f"{_norm(brand)}||{_norm(name)}"

def _md_link(url: str, label: str = "구매링크") -> str:
    u = str(url).strip()
    if not u:
        return ""
    if not re.match(r"^https?://", u, re.IGNORECASE):
        u = "http://" + u
    return f"[{label}]({u})"

def draw_random_products(n: int = 3) -> str:
    pool: deque = st.session_state["random_pool"]
    chosen_idx = []
    target = min(n, len(df_products))
    while len(chosen_idx) < target:
        if not pool:
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
        link_md = _md_link(row["구매링크"], "구매링크")
        # ✅ 가격 뒤 줄바꿈(마크다운 강제 개행: 공백 2개 + \n)
        lines.append(
            f"{i}. {row['브랜드']} {row['제품명']} | {row['가격']} |  \n {row['제품설명']} | {link_md}"
        )
    return "\n".join(lines)

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state["store"]:
        st.session_state["store"][session_id] = ChatMessageHistory()
    return st.session_state["store"][session_id]

def build_query_from_history_and_input(history: BaseChatMessageHistory, user_input: str, max_turns: int = 4) -> str:
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
    return []

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
    # ✅ 가격 뒤 줄바꿈(공백 2개 + \n) 적용
    return "\n".join(
        f"{i}. {r['brand']} {r['name']} | {r['price']} |  \n {r['desc']} | {_md_link(r['url'], '구매링크')}"
        for i, r in enumerate(rows, start=1)
    )

# ---------- 질문 맥락 맞춤 설명 생성 (LLM) ----------
def _get_llm():
    try:
        return ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE, max_tokens=LLM_MAX_TOKENS)
    except Exception:
        return None

LLM = _get_llm()

LLM_SYSTEM = (
    "당신은 러닝화 추천 설명을 작성하는 카피라이터입니다. "
    "다음 제품 정보와 기본 설명만을 근거로, 사용자 질문의 의도(예: 기록 단축/장거리, 통풍, 안정감 등)와 "
    "연결되게 한국어로 1~2문장 요약을 만듭니다. "
    "CSV에 없는 수치나 기능은 추측해 쓰지 마세요. 과장된 표현 대신 명료한 장점만 쓰고, 문장 끝에 마침표를 사용하세요."
)

def generate_contextual_descriptions(user_question: str, rows: list[dict]) -> list[dict]:
    if not USE_LLM_DESC or LLM is None or not os.environ.get("OPENAI_API_KEY"):
        return rows

    new_rows = []
    for r in rows:
        try:
            base_desc = str(r.get("desc", "")).strip()
            brand = r.get("brand", "")
            name = r.get("name", "")
            price = r.get("price", "")

            user_content = (
                f"[사용자 질문]\n{user_question}\n\n"
                f"[제품]\n브랜드: {brand}\n제품명: {name}\n가격: {price}\n\n"
                f"[기본 설명]\n{base_desc}\n\n"
                f"[작성]\n- 질문에 도움이 되는 포인트를 1~2문장으로 요약하세요.\n"
                f"- CSV 설명에 있는 정보만 사용하세요.\n"
                f"- 예: 기록 단축/지속주/통풍/안정감/쿠션/경량 중 해당하는 부분만 연결."
            )

            resp = LLM.invoke([{"role": "system", "content": LLM_SYSTEM},
                               {"role": "user", "content": user_content}])
            text = (getattr(resp, "content", "") or "").strip()
            if not text:
                text = base_desc
            text = re.sub(r"\s+", " ", text)
            r = {**r, "desc": text}
        except Exception:
            pass
        new_rows.append(r)
    return new_rows

# --------------------------- 이전 대화 출력 ---------------------------
for role, msg in st.session_state["messages"]:
    st.chat_message(role).write(msg)

# --------------------------- 입력 ---------------------------
def _hide_chat_input_css():
    st.markdown(
        "<style>div[data-testid='stChatInput']{display:none !important;}</style>",
        unsafe_allow_html=True,
    )

# ✅ 후속질문 패널이 보이는 동안에는 입력창 자체를 렌더링하지 않음
FOLLOWUP_ACTIVE = st.session_state["followup_step"] in (1, 2)

user_input = None
if st.session_state["selected_question"]:
    user_input = st.session_state["selected_question"]
    st.session_state["selected_question"] = None
else:
    if FOLLOWUP_ACTIVE:
        _hide_chat_input_css()  # 혹시 남아있는 입력창도 즉시 숨김
        user_input = None
    else:
        tmp = st.chat_input("메시지를 입력해 주세요", key="main_input")
        if tmp:
            user_input = tmp

# --------------------------- 응답 처리 ---------------------------
if user_input:
    # 사용자 말풍선은 항상 표시 (첫 추천 포함)
    st.chat_message("user").write(user_input)
    st.session_state["messages"].append(("user", user_input))

    # 첫 요청인지 판별
    is_first_trigger = (user_input.strip() == "운동화 추천해줘" and st.session_state["followup_step"] == 0)

    # 첫 요청: 프리페이스 제거하고 추천만 출력
    if is_first_trigger:
        random_reco = draw_random_products(3)
        combined = random_reco  # ✅ 프리페이스 제거
        with st.chat_message("assistant"):
            stream_text(combined, delay=0.015)
        st.session_state["messages"].append(("assistant", combined))
        st.session_state["followup_step"] = 1
        st.rerun()  # ▶ 패널이 바로 뜨는 턴에서 입력창 숨김

    else:
        # 0) 키워드 랭킹
        keywords = get_keywords(user_input)
        exclude = set(st.session_state["seen_products"])
        rows = rank_by_keywords(df_products, keywords, exclude_keys=exclude, top_k=12)[:3]

        # 1) 벡터검색 보충
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

        # 2) CSV 미본 보충
        if len(rows) < 3:
            rows = topup_with_unseen(rows, 3 - len(rows))
        rows = rows[:3]

        # 3) seen 등록
        for r in rows:
            st.session_state["seen_products"].add(product_key(r["brand"], r["name"]))

        # 4) LLM 요약 설명
        rows = generate_contextual_descriptions(user_input, rows)

        # 5) 출력 (이후 턴부터는 프리페이스 포함)
        out_text = rows_to_output(rows)
        combined = f"{PREFACE}\n\n{out_text}"
        with st.chat_message("assistant"):
            stream_text(combined, delay=0.015)
        st.session_state["messages"].append(("assistant", combined))

        # 6) 패널 단계 진행 및 종료 안내
        if st.session_state["followup_step"] == 1:
            st.session_state["followup_step"] = 2
            st.rerun()  # ▶ 두 번째 패널도 즉시 반영
        elif st.session_state["followup_step"] == 2:
            st.session_state["followup_step"] = 3
            code = "8172"  # 🔒 인증번호를 항상 8172로 고정
            end_msg = f"대화가 종료되었습니다. 인증번호는 {code} 입니다"
            st.chat_message("assistant").write(end_msg)
            st.session_state["messages"].append(("assistant", end_msg))

# --------------------------- 후속질문 패널 ---------------------------
def render_followup_panel(step: int):
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
