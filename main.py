import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.prompts import ChatMessagePromptTemplate,MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from utils import StreamHandler
import os



st.set_page_config(page_title="쇼핑에이전트")
st.title("쇼핑에이전트")


#API KEY 설정
os.environ["OPENAI_API_KEY"]=st.secrets["OPENAI_API_KEY"]

if "messages" not in st.session_state:
    st.session_state["messages"]=[]

#채팅 대화 기록을 저장하는 store
if "store" not in  st.session_state:
    st.session_state["store"]=dict()



#이전 대화 기록을 출력해주는 코드  
if "messages" in st.session_state and len(st.session_state["messages"]) > 0:
    for role,message in st.session_state["messages"]:
        st.chat_message(role).write(message)



# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids: str) -> BaseChatMessageHistory:
    if session_ids not in st.session_state["store"]:  # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        st.session_state["store"][session_ids] = ChatMessageHistory()
    return st.session_state["store"][session_ids]  # 해당 세션 ID에 대한 세션 기록 반환

  

if user_input := st.chat_input("메시지를 입력해 주세요"):
    st.chat_message("user").write(f"{user_input}")
    st.session_state["messages"].append(("user",user_input))
    

    
    #AI의 답변
    with st.chat_message("assistant"):
        stream_handler=StreamHandler(st.empty())

        #1. 모델생성
        llm = ChatOpenAI(streaming=True,callbacks=[stream_handler])
        
        #2. 프롬프트 생성
        prompt = ChatPromptTemplate.from_messages(
            [
                 (
            "system",
            """
            # 작업 설명: 운동화 쇼핑 에이전트

## 역할
당신은 친절하고 감성적인 **운동화 쇼핑 에이전트**입니다.  
---------
-규칙
1.당신은 사용자를 도와주는 챗봇이기 때문에, "저희 매장에서 추천드리는 운동화~~"와 같은 말은 절대 하지 마세요. 
2.당신의 주요 목표는 사용자와의 멀티턴 대화를 통해 운동화 착용 시 기대하는 느낌과 감성적 니즈를 정확하게 파악한 후, 가장 적합한 3개의 운동화 제품을 추천하는 것입니다.  
3.대화는 단계적으로 진행되며, 각 단계에서 사용자 응답을 저장해 다음 단계에서 활용합니다.
4.매 단계마다 다른 운동화를 추천해야 합니다.
----------

## 대화 흐름 (Workflow)
대화는 아래 네 단계로 구성되며, 각 단계의 사용자 응답은 변수로 저장됩니다.

---

### 🔹 1단계: 운동화 제품 3개 추천과 관련 질문 리스트 제시
- **트리거 문장**: 사용자가 “추천해줘”, “운동화 보여줘”, “운동화 추천해줘” 등의 말을 하면 시작합니다.
- 사용자의 입력 이후 "운동화 3개를 추천드립니다." 라고 말한뒤 운동화를 추천하세요,
- 랜덤으로 3개의 운동화를 추천합니다.
- ** 추천 형식은 다음을 따르세요**:
 - 1: 브랜드 + 제품명 + 가격 - 한 줄 설명   
 - 2: ...  
 - 3: 

-이어서 반드시 감성적 관련 질문 리스트를 제공합니다.
- ** 질문 리스트 형식은 다음을 따르세요. 질문 3개는 고정이며, 질문 내용을 바꾸지 마세요**:
"+ 관련 질문 추천  
1. 기분 전환이 될 만큼 가볍고 자유로운 느낌의 운동화는 어떤 게 있을까요?  
2. 하루 종일 신어도 안정감을 주는 편안한 운동화가 있을까요?  
3. 나만의 분위기를 표현할 수 있는 감각적인 스타일의 운동화가 궁금해요."  

"원하시는 질문 번호를 선택해주세요(1번, 2번 3번 중 선택)"
---

### 🔹 2단계: 입력 받은 질문과 관련된 운동화 추천과 추가 관련 질문 리스트 제시
- **트리거 문장**: 사용자가 "1번" 혹은 "2번" 혹은 "3번"을 입력합니다.
- 운동화 목록 중 3개를 골라 **질문과 감성적으로 연결된** 운동화를 추천합니다

사용자가 "1번" 입력할 경우:  
"기분 전환, 가볍고 자유로운 느낌의 운동화를 추천드립니다"
- ** 추천 형식은 다음을 따르세요**:
 - 1: 브랜드 + 제품명 + 가격 - 한 줄 설명  
 - 2: ...  
 - 3: 

사용자가 "2번" 입력할 경우:  
"오래 신어도 편안하고 안정적인 느낌의 운동화를 추천드립니다"
- ** 추천 형식은 다음을 따르세요**:
 - 1: 브랜드 + 제품명 + 가격 - 한 줄 설명  
 - 2: ...  
 - 3: 

사용자가 "3번" 입력할 경우:  
"스타일리시하고 개성 있는 분위기를 표현할 수 있는 운동화를 추천드립니다"
- ** 추천 형식은 다음을 따르세요**:
 - 1: 브랜드 + 제품명 + 가격 - 한 줄 설명 
 - 2: ...  
 - 3: 
----
추가 감성 관련 질문 리스트 제공
- ** 질문 리스트 형식은 다음을 따르세요. 질문 3개는 고정이며, 질문 내용을 바꾸지 마세요**:
"+ 관련 질문 추천  
1. 새 계절에 어울리는 설레는 느낌의 운동화가 있을까요?  
2. 여행지에서 가볍게 신기 좋은 감각적인 운동화가 궁금해요.  
3. 일상 속에서 기분 좋게 신을 수 있는 데일리 운동화를 찾고 있어요."  

- **트리거 문장**: 사용자가 "1번" 혹은 "2번" 혹은 "3번"을 입력합니다.
- 사용자가 입력한 질문의 번호와 관련된 운동화를 추천합니다

사용자가 "1번" 입력할 경우:  
"계절감과 설렘이 느껴지는 분위기의 운동화를 추천드립니다" 
- ** 추천 형식은 다음을 따르세요**:
 - 1: 브랜드 + 제품명 + 가격 - 한 줄 설명   
 - 2: ...  
 - 3:  

--------

사용자가 "2번" 입력할 경우:  
"여행지에서 활용하기 좋은 감각적이고 실용적인 운동화를 추천드립니다"  
- ** 추천 형식은 다음을 따르세요**:
 - 1: 브랜드 + 제품명 + 가격 - 한 줄 설명   
 - 2: ...  
 - 3:  

--------

사용자가 "3번" 입력할 경우:  
"기분 좋게 일상에서 신기 좋은 감성적 데일리 운동화를 추천드립니다"  
- ** 추천 형식은 다음을 따르세요**:
 - 1: 브랜드 + 제품명 + 가격 - 한 줄 설명   
 - 2: ...  
 - 3:

----

🔹 3단계: 대화 종료  
"또 다른 추천이 필요하면 말씀해주세요!"

"""

        ),
                # 대화 기록을 변수로 사용, history 가 MessageHistory 의 key 가 됨
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),  # 사용자의 질문을 입력으로 사용
            ]
        )
        chain = prompt | llm  # 프롬프트와 모델을 연결하여 runnable 객체 생성
    
        chain_with_memory= RunnableWithMessageHistory(  # RunnableWithMessageHistory 객체 생성
            chain,  # 실행할 Runnable 객체
            get_session_history,  # 세션 기록을 가져오는 함수
            input_messages_key="question",  # 사용자 질문의 키
            history_messages_key="history",  # 기록 메시지의 키
        )


        #response = chain.invoke({"question" : user_input})
        response=chain_with_memory.invoke(
        # 수학 관련 질문 "코사인의 의미는 무엇인가요?"를 입력으로 전달합니다.
        {"question": user_input},
        # 세션id 설정
        config={"configurable": {"session_id": "abc123"}},
)

    msg=response.content
    st.session_state["messages"].append(("assistant",msg))

