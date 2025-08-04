import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from utils import StreamHandler

# ---------------------- 설정 ----------------------
st.set_page_config(page_title="운동화 쇼핑 에이전트")
st.title("운동화 쇼핑 에이전트")

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# ---------------------- 상태 초기화 ----------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "store" not in st.session_state:
    st.session_state["store"] = dict()
if "selected_question" not in st.session_state:
    st.session_state["selected_question"] = None
if "followup_step" not in st.session_state:
    st.session_state["followup_step"] = 0

# ---------------------- 후속질문 ----------------------
followup_set_1 = {
    "Q1": "러닝 동호회 활동을 하면서 실력을 더 끌어올릴 수 있는 운동화가 있을까요?",
    "Q2": "장거리 러닝이나 기록 향상 같은 도전 과제를 도와줄 운동화가 있을까요?",
    "Q3": "러닝 동호회 사람들과 잘 어울릴 수 있는, 함께 신기 좋은 운동화가 있을까요?"
}
followup_set_2 = {
    "Q4": "가족이나 친구들과 같이 러닝할 때 어울리는 운동화가 있을까요?",
    "Q5": "러닝하면서도 스타일리시하고, 센스 있게 보일 수 있는 운동화를 원하시나요?",
    "Q6": "평소와는 다른 기분을 주는, 새로운 러닝 경험을 만들어 줄 운동화가 있을까요?"
}

# ---------------------- 대화 히스토리 ----------------------
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state["store"]:
        st.session_state["store"][session_id] = ChatMessageHistory()
    return st.session_state["store"][session_id]

# ---------------------- 이전 메시지 출력 ----------------------
for role, message in st.session_state["messages"]:
    st.chat_message(role).write(message)

# ---------------------- 입력 ----------------------
user_input = None
if st.session_state["selected_question"]:
    user_input = st.session_state["selected_question"]
    st.session_state["selected_question"] = None
elif tmp := st.chat_input("메시지를 입력해 주세요"):
    user_input = tmp

# ---------------------- 응답 처리 ----------------------
if user_input:
    st.chat_message("user").write(user_input)
    st.session_state["messages"].append(("user", user_input))

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        llm = ChatOpenAI(streaming=True, callbacks=[stream_handler])

        # 운동화 추천해줘 라는 특수 요청이면 전용 프롬프트 사용
        if user_input.strip() == "운동화 추천해줘":
            user_prompt = "운동화 3개를 추천해 주세요. 브랜드명, 모델명, 기능, 가격을 포함해서 리스트 형식으로 설명해 주세요."
        else:
            user_prompt = user_input

        prompt = ChatPromptTemplate.from_messages([
            ("system", """
당신은 운동화 쇼핑 에이전트입니다.
운동 목적, 기능, 착용감, 브랜드에 기반하여 사용자의 질문에 전문적으로 응답하세요.
추천 형식 규칙 (절대 위반 금지)
- 텍스트로만 출력 (이미지, 링크 등 금지)
- 반드시 다음과 같은 형식으로 리스트 3개를 출력하세요:
             
`- 1. [브랜드] [제품명] [가격] - [설명]`
`- 2. ...`
`- 3. ...`
             
- 각 줄은 하이픈(-)과 숫자 순번(1., 2., 3.)으로 시작해야 하며, 줄바꿈된 목록 형태여야 합니다.
- 각 운동화는 실제 브랜드명, 제품명, 가격과 함께 한 줄 설명을 포함해야 합니다.
- 설명에는 반드시 사용자가 언급한 기능 또는 조건이 포함되어야 합니다.
             
형식 예시 (참고용)

- 1. 나이키 에어줌 페가수스 40 129,000원 - 무게감이 있으며 통풍감이 좋고 쿠션감이 뛰어난 운동화입니다.

위 형식은 절대 변경하지 마세요.

---


"""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])

        chain = prompt | llm
        chain_with_memory = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="question",
            history_messages_key="history"
        )

        response = chain_with_memory.invoke(
            {"question": user_prompt},
            config={"configurable": {"session_id": "sneaker-chat"}},
        )
        msg = response.content
        st.session_state["messages"].append(("assistant", msg))

        # 후속질문 단계 설정
        if user_input.strip() == "운동화 추천해줘":
            st.session_state["followup_step"] = 1
        elif st.session_state["followup_step"] == 1:
            st.session_state["followup_step"] = 2
        elif st.session_state["followup_step"] == 2:
            st.session_state["followup_step"] = 3

# ---------------------- 후속질문 출력 ----------------------
if st.session_state["followup_step"] == 1:
    st.markdown("### 이런 질문도 해보세요!")
    for key, question in followup_set_1.items():
        col_q, col_btn = st.columns([8, 1])
        col_q.markdown(f"**{key}.** {question}")
        if col_btn.button("➕", key=f"btn_{key}"):
            st.session_state["selected_question"] = question
            st.rerun()
elif st.session_state["followup_step"] == 2:
    st.markdown("### 이런 질문도 해보세요!")
    for key, question in followup_set_2.items():
        col_q, col_btn = st.columns([8, 1])
        col_q.markdown(f"**{key}.** {question}")
        if col_btn.button("➕", key=f"btn_{key}"):
            st.session_state["selected_question"] = question
            st.rerun()
elif st.session_state["followup_step"] == 3:
    st.markdown("또 다른 추천이 필요하면 말씀해주세요!")
