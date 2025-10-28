# ============================== #
#            IMPORTS             #
# ============================== #
# API 불러오기
import os
from dotenv import load_dotenv

# 원본 파일 정리
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, UnstructuredHTMLLoader, Docx2txtLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.documents import Document

import datetime as dt
from zoneinfo import ZoneInfo
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.document_transformers import LongContextReorder
from langchain.retrievers.document_compressors import DocumentCompressorPipeline

# streamlit
import streamlit as st
from streamlit_lottie import st_lottie
import requests

# ============================== #
#        ENV & CONSTANTS         #
# ============================== #
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
KST = ZoneInfo("Asia/Seoul")

# ============================== #
#      SIDEBAR (선 정의 필요)     #
# ============================== #
with st.sidebar:
    st.title("🍀모두의 연구소")

    # 날짜 선택 (UI는 사이드바에 렌더되지만, 코드 순서는 상단이어도 OK)
    st.markdown("---")
    st.header("🗓️ 날짜 선택")
    today_default = dt.date.today()
    selected_date = st.date_input(
        "원하는 날짜를 선택하세요:",
        value=today_default,
        min_value=dt.date(2023, 1, 1),
        max_value=dt.date(2026, 12, 31),
        key="sidebar_date"
    )
    st.markdown("---")
    st.info(f"오늘은: **{selected_date}** 입니다.")

    # 관련 사이트
    st.markdown("---")
    st.header("🔗관련 사이트")
    st.link_button("모두의 연구소 홈페이지", "https://modulabs.co.kr")
    st.link_button("데싸 5기 노션 워크스페이스", "https://www.notion.so/New-5-25-07-07-26-01-08-New-23f2d25db62480828becc399aaa41877")
    st.link_button("데싸 5기 ZEP", "https://zep.us/play/8l5Vdo")
    st.link_button("LMS 홈페이지", "https://lms.aiffel.io/")

    # 첨부파일
    st.markdown("---")
    st.header("📄첨부파일")
    try:
        with open(r"C:\Users\user\Desktop\MODULABS\LangchainThon\Data\word\휴가신청서(데싸_5기).docx", 'rb') as file:
            st.download_button(
                label='휴가신청서 다운로드',
                data=file,
                file_name='휴가신청서.docx',
                mime='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            )
    except FileNotFoundError:
        st.warning(r"첨부파일 경로를 확인하세요: C:\Users\user\Desktop\MODULABS\LangchainThon\Data\word\휴가신청서(데싸_5기).docx")

# ============================== #
#         TEXT SPLITTER          #
# ============================== #
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=128,
    separators=["\n\n", "\n", " ", ""]
)

# ============================== #
#           LOAD FILES           #
# ============================== #
# PDF
PDF_PATH = r"C:\Users\user\Desktop\MODULABS\LangchainThon\Data\pdf\모두연 브랜딩북 정리.pdf"
docs_pdf = []
try:
    loader_pdf = PyPDFLoader(PDF_PATH)
    pages_pdf = loader_pdf.load()
    for d in pages_pdf:
        d.metadata["source_type"] = "pdf"
        d.metadata["source"] = os.path.basename(PDF_PATH)
    docs_pdf = text_splitter.split_documents(pages_pdf)
except Exception as e:
    st.error(f"PDF 로딩 실패: {e}")

# HTML
HTML_PATH = [
    r"C:\Users\user\Desktop\MODULABS\LangchainThon\Data\html\LMS oops 해결법.html",
    r"C:\Users\user\Desktop\MODULABS\LangchainThon\Data\html\LMS 아이펠 노트북이 아닙니다 에러.html",
    r"C:\Users\user\Desktop\MODULABS\LangchainThon\Data\html\LMS 이용시 발생하는 문제 해결법.html",
    r"C:\Users\user\Desktop\MODULABS\LangchainThon\Data\html\교육과정 중 취업 시.html",
    r"C:\Users\user\Desktop\MODULABS\LangchainThon\Data\html\데싸 5기 훈련 정보.html",
    r"C:\Users\user\Desktop\MODULABS\LangchainThon\Data\html\수강 중 고용 형태 관련 안내.html",
    r"C:\Users\user\Desktop\MODULABS\LangchainThon\Data\html\스터디를 만들고 싶은데 어떻게 해야 하나요.html",
    r"C:\Users\user\Desktop\MODULABS\LangchainThon\Data\html\오프닝 장소와 클로징 장소가 다릅니다.html",
    r"C:\Users\user\Desktop\MODULABS\LangchainThon\Data\html\인터넷이 불안정하여 출결 QR을 제대로 찍지 못하였습니다.html",
    r"C:\Users\user\Desktop\MODULABS\LangchainThon\Data\html\제적 가이드.html",
    r"C:\Users\user\Desktop\MODULABS\LangchainThon\Data\html\출결 및 공가에 대하여.html",
    r"C:\Users\user\Desktop\MODULABS\LangchainThon\Data\html\툴 세팅.html",
    r"C:\Users\user\Desktop\MODULABS\LangchainThon\Data\html\훈련 장려금 지급 확인.html",
    r"C:\Users\user\Desktop\MODULABS\LangchainThon\Data\html\훈련 참여 규칙.html",
]
docs_html = []
try:
    html_list = []
    for path in HTML_PATH:
        loader_html = UnstructuredHTMLLoader(path)
        pages_html = loader_html.load()
        for d in pages_html:
            d.metadata["source_type"] = "html"
            d.metadata["source"] = os.path.basename(path)
        html_list.extend(pages_html)
    docs_html = text_splitter.split_documents(html_list)
except Exception as e:
    st.error(f"HTML 로딩 실패: {e}")

# WORD
WORD_PATH = r"C:\Users\user\Desktop\MODULABS\LangchainThon\Data\word\휴가신청서(데싸_5기).docx"
docs_word = []
try:
    loader_word = Docx2txtLoader(WORD_PATH)
    pages_word = loader_word.load()
    for d in pages_word:
        d.metadata["source_type"] = "word"
        d.metadata["source"] = os.path.basename(WORD_PATH)
    docs_word = text_splitter.split_documents(pages_word)
except Exception as e:
    st.error(f"WORD 로딩 실패: {e}")

# CSV (동료/운영진)
CSV_PATH = [
    r"C:\Users\user\Desktop\MODULABS\LangchainThon\Data\csv\데싸 5기 동료들.csv",
    r"C:\Users\user\Desktop\MODULABS\LangchainThon\Data\csv\데싸 5기 운영진.csv"
]
docs_csv = []
try:
    csv_list = []
    for path in CSV_PATH:
        loader_csv = CSVLoader(path, encoding='cp949')
        pages_csv = loader_csv.load()
        for d in pages_csv:
            d.metadata["source_type"] = "csv"
            d.metadata["source"] = os.path.basename(path)
        csv_list.extend(pages_csv)
    docs_csv = text_splitter.split_documents(csv_list)
except Exception as e:
    st.error(f"CSV 로딩 실패: {e}")

# CSV (일정표 → 학생별 그룹 문서)
def create_grouped_documents(csv_path: str) -> list[Document]:
    try:
        df = pd.read_csv(csv_path, encoding='cp949')
    except Exception as e:
        st.error(f"일정표 CSV 로딩 실패: {e}")
        return []

    required_cols = ['이름', '사유', '날짜', '부재시간', '상태']
    if not all(col in df.columns for col in required_cols):
        st.error(f"필요 컬럼 누락: {required_cols}")
        return []

    df = df[required_cols].fillna('')
    documents = []
    grouped = df.groupby('이름')

    for name, group_df in grouped:
        record_strings = []
        for _, row in group_df.iterrows():
            record = (
                f"사유: {row['사유']}, "
                f"날짜: {row['날짜']}, "
                f"상태: {row['상태']}, "
                f"부재시간: {row['부재시간']}"
            )
            record_strings.append(record)

        full_records_text = "\n".join(record_strings)
        document = Document(
            page_content=f"학생 이름: {name}\n\n--- 전체 출결 기록 시작 ---\n{full_records_text}",
            metadata={'학생이름': name, '총기록수': len(group_df)}
        )
        documents.append(document)
    return documents

docs_attendance = []
attendance_path = r"C:\Users\user\Desktop\MODULABS\LangchainThon\Data\csv\데싸 5기 일정표.csv"
attendance_documents = create_grouped_documents(attendance_path)
docs_attendance = text_splitter.split_documents(attendance_documents) if attendance_documents else []

# ============================== #
#           VECTOR DB            #
# ============================== #
# 임베딩 및 인덱싱
vectorstore = Chroma.from_documents(docs_html, OpenAIEmbeddings(model='text-embedding-3-large'))
if docs_word: vectorstore.add_documents(docs_word)
if docs_csv: vectorstore.add_documents(docs_csv)
if docs_pdf: vectorstore.add_documents(docs_pdf)
if docs_attendance: vectorstore.add_documents(docs_attendance)

# ============================== #
#            RETRIEVER           #
# ============================== #

# Reranking 이전 base 
base_retriever = vectorstore.as_retriever(
    search_type="mmr", 
    search_kwargs={"lambda_mult": 0.4, "fetch_k": 96, "k": 48}
)

# Rerank
reranker = CohereRerank(
    model="rerank-multilingual-v3.0",    
    top_n=10                              
)

# Reorder
reorder = LongContextReorder()

# Rerank + Reorder
compressor = DocumentCompressorPipeline(transformers=[reranker, reorder])

upgraded_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor            
)

# ============================== #
#               LLM              #
# ============================== #
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ============================== #
#           PROMPTS/CHAINS       #
# ============================== #
contextualize_q_system_prompt = """
이전 대화가 있다면 참고하여,
사용자의 최신 질문을 독립적으로 이해 가능한 한 문장으로 바꿔주세요.
답변하지 말고 질문만 재작성하세요.
"""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm,
    upgraded_retriever,
    contextualize_q_prompt
)

qa_system_prompt = """
당신은 '모두의연구소(모두연)' 수강생들의 비서입니다.

현재 시간은 {today} (KST)입니다. 사용자의 '어제, 내일' 등의 표현은 {today}를 기준으로 파악하세요.
오늘의 날짜/요일은 {today_ko} / {weekday_ko} 입니다. 날짜 및 요일 관련 질문에는 추론하지 말고 반드시 이 값을 그대로 사용하세요.

제공된 문서 내용만을 근거로 답하세요. 근거가 없으면 '정보가 명확하지 않습니다. 운영매니저님이나 퍼실님께 문의해주세요.'라고만 대답하세요.
사용자 입력에 포함된 사실은 근거로 사용하지 마세요.

훈련장려금의 경우 주어진 단위 기간 일수의 80%이상을 출석해야만 금액이 지급됨을 명심하세요. 
최대 3문장으로 짧게 답변하세요.

{context}
"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_core_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# ============================== #
#       SESSION / HISTORY        #
# ============================== #
if "lc_store" not in st.session_state:
    st.session_state["lc_store"] = {}  

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    store = st.session_state["lc_store"]  
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_core_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# ============================== #
#           LOTTIE + UI          #
# ============================== #
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

snow_lottie_url = "https://assets2.lottiefiles.com/packages/lf20_1pxqjqps.json"
snow_animation = load_lottie_url(snow_lottie_url)
if snow_animation:
    st_lottie(snow_animation, speed=1, reverse=False, loop=True, quality="high", height=500, width=800, key="snow")

st.markdown(
    """
<div style="text-align: center;">
    <p style="font-size:25px;">
        안녕하세요! 저는 모두봇입니다.<br>즐거운 모두연 생활을 위한 정보를 제공합니다.😊
    </p>
</div>
""",
    unsafe_allow_html=True
)

if "session_id" not in st.session_state:
    st.session_state["session_id"] = "default"
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "무엇이든 물어보세요!"}]

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# ============================== #
#            CHAT LOOP           #
# ============================== #
if prompt_message := st.chat_input("질문을 입력해주세요 :)"):
    st.session_state["messages"].append({"role": "user", "content": prompt_message})
    st.chat_message("user").write(prompt_message)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # 사이드바에서 고른 날짜를 today로 동적 주입 (KST 00:00:00로 고정)
            today_override = f"{selected_date} 00:00:00"

            # selected_date가 이미 date 객체라고 가정
            weekday_names = ["월요일","화요일","수요일","목요일","금요일","토요일","일요일"]
            weekday_ko = weekday_names[selected_date.weekday()]
            today_ko = f"{selected_date.year}년 {selected_date.month}월 {selected_date.day}일"

            resp = conversational_rag_chain.invoke(
                {
                    "input": prompt_message,
                    "today": today_override,
                    "today_ko": today_ko,
                    "weekday_ko": weekday_ko,
                },
                config={"configurable": {"session_id": st.session_state["session_id"]}},
            )

            answer = resp if isinstance(resp, str) else resp.get("answer", "")
            st.write(answer)
            st.session_state["messages"].append({"role": "assistant", "content": answer})
