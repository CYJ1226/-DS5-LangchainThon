import os
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
import plotly.express as px
from streamlit_lottie import st_lottie
import requests
import datetime as dt

#오픈AI API 키 설정
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")


# ============================================== #

# Embedding - Chunk split
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=128,
    separators=["\n\n", "\n", " ", ""]             
)


# pdf 
from langchain_community.document_loaders import PyPDFLoader

PDF_PATH = r"C:\Users\user\Desktop\MODULABS\LangchainThon\Data\pdf\모두연 브랜딩북 정리.pdf"

loader_pdf = PyPDFLoader(PDF_PATH)
pages_pdf = loader_pdf.load()

for d in pages_pdf:
    d.metadata["source_type"] = "pdf"
    d.metadata["source"] = os.path.basename(PDF_PATH)

docs_pdf = text_splitter.split_documents(pages_pdf)


# html
from langchain_community.document_loaders import UnstructuredHTMLLoader

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

html_list = []

# 각 파일 로드 + Metadata 저장
for path in HTML_PATH:
    loader_html = UnstructuredHTMLLoader(path)
    pages_html = loader_html.load()

    for d in pages_html:
        d.metadata["source_type"] = "html"
        d.metadata["source"] = os.path.basename(path)

    html_list.extend(pages_html)  

docs_html = text_splitter.split_documents(html_list)


# word
from langchain_community.document_loaders import Docx2txtLoader

WORD_PATH = r"C:\Users\user\Desktop\MODULABS\LangchainThon\Data\word\휴가신청서(데싸_5기).docx"
loader_word = Docx2txtLoader(WORD_PATH)
pages_word = loader_word.load()

for d in pages_word:
    d.metadata["source_type"] = "word"
    d.metadata["source"] = os.path.basename(WORD_PATH)

docs_word = text_splitter.split_documents(pages_word)


# csv
from langchain_community.document_loaders.csv_loader import CSVLoader

CSV_PATH = [
    r"C:\Users\user\Desktop\MODULABS\LangchainThon\Data\csv\데싸 5기 동료들.csv",
    r"C:\Users\user\Desktop\MODULABS\LangchainThon\Data\csv\데싸 5기 운영진.csv"
]

csv_list = []

# 각 파일 로드
for path in CSV_PATH:
    loader_csv = CSVLoader(path, encoding = 'cp949')
    pages_csv = loader_csv.load()

    for d in pages_csv:
        d.metadata["source_type"] = "csv"
        d.metadata["source"] = os.path.basename(path)

    csv_list.extend(pages_csv)  

docs_csv = text_splitter.split_documents(csv_list)


# csv - calendar

from langchain_core.documents import Document
import pandas as pd

# 학생 이름을 기준 그룹화 함수
def create_grouped_documents(csv_path: str) -> list[Document]:
    """
    CSV 파일을 로드하여 학생 이름별로 출결 기록을 그룹화하고
    LangChain Document 객체 리스트로 반환합니다.

    Args:
        csv_path: 출결 CSV 파일의 경로.

    Returns:
        Document 객체 리스트. 각 Document는 한 학생의 전체 기록을 담습니다.
    """
    # 1. CSV 파일 로드
    df = pd.read_csv(csv_path, encoding='cp949')

    # 필요한 컬럼만 선택하고 NaN 값은 빈 문자열로 대체 (문자열 결합 시 오류 방지)
    required_cols = ['이름', '사유', '날짜', '부재시간', '상태']
    if not all(col in df.columns for col in required_cols):
        print(f"오류: CSV 파일에 필요한 컬럼 ({required_cols}) 중 일부가 누락되었습니다.")
        return []

    df = df[required_cols].fillna('')

    # Document 객체를 저장할 리스트
    documents = []

    # 2. '이름' 컬럼을 기준으로 그룹화
    grouped = df.groupby('이름')

    # 3. 각 학생 그룹을 하나의 긴 텍스트 Document로 변환
    for name, group_df in grouped:
        # 학생별 기록을 문자열로 변환 (날짜, 부재시간, 상태만 포함)
        # '이름' 컬럼은 메타데이터로 사용하기 때문에 텍스트 내용에서는 제외합니다.
        record_strings = []
        for index, row in group_df.iterrows():
            record = (
                f"사유: {row['사유']}, "
                f"날짜: {row['날짜']}, "
                f"상태: {row['상태']}, "
                f"부재시간: {row['부재시간']}"
            )
            record_strings.append(record)

        # 모든 기록을 줄 바꿈으로 연결하여 하나의 긴 텍스트 생성
        full_records_text = "\n".join(record_strings)

        # 최종 Document 객체 생성
        document = Document(
            page_content=(
                f"학생 이름: {name}\n\n"
                f"--- 전체 출결 기록 시작 ---\n"
                f"{full_records_text}"
            ),
            # 메타데이터에 핵심 정보 저장 (검색 시 활용 가능)
            metadata={'학생이름': name, '총기록수': len(group_df)}
        )
        documents.append(document)

    return documents

# 학생별 Document 리스트 생성
attendance_documents = create_grouped_documents(csv_path=r"C:\Users\user\Desktop\MODULABS\LangchainThon\Data\csv\데싸 5기 일정표.csv")

docs_attendance = text_splitter.split_documents(attendance_documents)



# VECTOR DB
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.document_transformers import LongContextReorder
from langchain.retrievers.document_compressors import DocumentCompressorPipeline

# ChromaDB 벡터 임베딩 후 저장
vectorstore = Chroma.from_documents(docs_html, OpenAIEmbeddings(model='text-embedding-3-large'))

vectorstore.add_documents(docs_word)
vectorstore.add_documents(docs_csv)
vectorstore.add_documents(docs_pdf)
vectorstore.add_documents(docs_attendance)



# RAG

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.retrievers.document_compressors import LLMChainExtractor
from zoneinfo import ZoneInfo

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


# LLM 모델 선언
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 현재 시간 선언
KST = ZoneInfo("Asia/Seoul")
today_str = dt.datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")

# 질문 프롬프트
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
    upgraded_retriever,             # Cohere API KEY 없는 경우 >> reordered_retriever
    contextualize_q_prompt
)


# 답변 프롬프트
qa_system_prompt = """

당신은 '모두의연구소(모두연)' 수강생들의 비서입니다.
현재 시간은 {today} (KST)입니다. 사용자의 '어제, 내일' 등의 표현은 {today}를 기준으로 파악하세요.
제공된 문서 내용만을 근거로 답하세요. 근거가 없으면 '정보가 명확하지 않습니다. 운영매니저님이나 퍼실님께 문의해주세요.'라고만 대답하세요.
사용자 입력에 포함된 사실은 근거로 사용하지 마세요.
최대 3문장으로 짧게 답변하세요.

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

qa_prompt = qa_prompt.partial(today=today_str)

# RAG - CHAIN

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


# RAG - SESSION

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

#채팅 세션별 기록 저장 위한 Dictionary 선언
store = {}

#주어진 session_id 값에 매칭되는 채팅 히스토리 가져오는 함수 선언
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


#RunnableWithMessageHistory 모듈로 rag_chain에 채팅 기록 세션별로 자동 저장 기능 추가
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# ============================================== #


# Streamlit UI(여기부터 스트림릿 코드이니 적용할 부분)
# st.markdown(
#     "<div style='text-align: center; color: #CD5C5C; font-size: 100px;'>모두봇</div>",
#     unsafe_allow_html=True
# )

# rag_chain = chaining()
# st.snow()


st.set_page_config(page_title="모두봇", layout="centered")

# Lottie 애니메이션 로딩 함수
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# 눈 내리는 Lottie 애니메이션 URL
snow_lottie_url = "https://assets2.lottiefiles.com/packages/lf20_1pxqjqps.json"
snow_animation = load_lottie_url(snow_lottie_url)

# Streamlit 앱 구성
# st.set_page_config(page_title="인사하는 로봇", layout="centered")
st.markdown(
    "<div style='text-align: center; color: #CD5C5C; font-size: 100px;'>모두봇</div>",
    unsafe_allow_html=True
)
st.title("❄️ 안녕하세요!! 모두봇입니다.")

# st.write("AI로봇이 인사를 합니다.")

# 애니메이션 표시
# col1, col2, col3 = st.columns([1, 2, 1])
# with col2:
st_lottie(
    snow_animation,
    speed=1,
    reverse=False,
    loop=True,
    quality="high",
    height=500,
    width=800,
    key="snow"
)

st.snow()
# st.title("ModuBot")
# st.header("💬 여러분의 모두봇입니다 📚")
rag_chain = conversational_rag_chain
# st.snow()

# st.data_input
st.markdown("---")
st.write("즐거운 모두연 생활을 위한 정보를 제공합니다.😊")

if "session_id" not in st.session_state:
    st.session_state["session_id"] = "default"
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "무엇이든 물어보세요!"}]

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt_message := st.chat_input("질문을 입력해주세요 :)"):
    st.session_state["messages"].append({"role": "user", "content": prompt_message})
    st.chat_message("user").write(prompt_message)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # conversational_rag_chain은 상단에서 정의되어 있다고 가정
            resp = conversational_rag_chain.invoke(
                {"input": prompt_message},
                config={"configurable": {"session_id": st.session_state["session_id"]}},
            )
            answer = resp if isinstance(resp, str) else resp.get("answer", "")
            st.write(answer)
            st.session_state["messages"].append({"role": "assistant", "content": answer})


with st.sidebar:
    st.title("⚙️모두의 연구소")
    st.markdown("---") # 구분선 추가
    
    menu = st.sidebar.selectbox("원하시는 정보를 선택하세요", ["모두연생활", "쉐밸그투", "학습"])

# 선택된 메뉴에 따라 다른 콘텐츠 출력
    if menu == "모두연생활":
        st.header("📘 모두연생활")
        st.write("콘텐츠를 여기에 표시합니다.")
    elif menu == "쉐벨그투":
        st.header("📗 쉐밸그투")
        st.write("콘텐츠를 여기에 표시합니다.")
    elif menu == "학습":
        st.header("📙 학습")
        st.write("콘텐츠를 여기에 표시합니다.")
    elif menu == "출결관리":
        st.header("📙 출결")
        st.write("콘텐츠를 여기에 표시합니다.")

# menu = st.sidebar.selectbox("파일을 선택하세요", ["모두연생활", "쉐벨그투", "학습", '출결관리'])

# # 과목별 PDF 파일 경로 설정 (예시: 로컬 또는 URL)
# pdf_files = {
#     "모두연생활": "data/korean.pdf",
#     "쉐벨그투": "data/english.pdf",
#     "학습": "data/math.pdf",
#     "출결관리": "data/math.pdf"
# }

# # 선택된 과목에 따라 PDF 표시
# st.header(f"📘 {menu} 를 위해 원하는 자료")

# pdf_path = pdf_files.get(menu)

# try:
#     with open(pdf_path, "rb") as f:
#         pdf_bytes = f.read()
#         st.download_button(
#             label=f"{menu} PDF 다운로드",
#             data=pdf_bytes,
#             file_name=f"{menu}.pdf",
#             mime="application/pdf"
#         )
#         st.write("PDF 파일을 다운로드하여 확인하세요.")
# except FileNotFoundError:
#     st.warning("PDF 파일이 존재하지 않습니다. 경로를 확인해주세요.")

# PDF 다운로드
    pdf_files = {
        "모두연생활": "data/korean.pdf",
        "쉐밸그투": "data/english.pdf",
        "학습": "data/math.pdf",
        "출결관리": "data/math.pdf"
    }

    selected_file = st.selectbox("파일을 선택하세요", list(pdf_files.keys()))
    pdf_path = pdf_files.get(selected_file)

    try:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
            st.download_button(
                label=f"{selected_file} PDF 다운로드",
                data=pdf_bytes,
                file_name=f"{selected_file}.pdf",
                mime="application/pdf"
            )
            st.write("PDF 파일을 다운로드하여 확인하세요.")
    except FileNotFoundError:
        st.warning("PDF 파일이 존재하지 않습니다. 경로를 확인해주세요.")

# 링크
# st.sidebar.markdown("[모두연 홈](https://https://modulabs.co.kr)")
    st.markdown("[모두연 홈](https://modulabs.co.kr)")
    st.markdown("<a href='https://biz.modulabs.co.kr/event' target='_blank'>페이지 열기</a>", unsafe_allow_html=True)


# 날짜 선택 (dt 별칭 사용)
    st.header("🗓️ 날짜 선택")
    st.markdown("---")
    today = dt.date.today()  # ← 여기!
    selected_date = st.date_input(
        "원하는 날짜를 선택하세요:",
        value=today,
        min_value=dt.date(2023, 1, 1),
        max_value=dt.date(2026, 12, 31),
        key="sidebar_date"
        )
    st.markdown("---")
    st.info(f"오늘은: **{selected_date}입니다.**")




# # 또는 HTML로 새 창 열기
# st.sidebar.markdown(
#     "<a href='https://biz.modulabs.co.kr/event' target='_blank'> 페이지 열기</a>",
#     unsafe_allow_html=True
# )


# # --- 페이지 설정 ---
# st.set_page_config(
#     page_title="사이드바 달력 예제",
#     layout="wide" # 넓은 레이아웃으로 설정하여 사이드바와 메인 컨텐츠 영역을 확보
# )

# # --- 사이드바 달력 표시 ---

# # 1. 사이드바 시작
# st.sidebar.header("🗓️ 날짜 선택")
# st.sidebar.markdown("---")

# # 2. st.date_input을 사용하여 날짜 입력 위젯 (달력) 표시
# # 기본값으로 오늘 날짜를 설정합니다.
# today = datetime.date.today()
# selected_date = st.sidebar.date_input(
#     label="원하는 날짜를 선택하세요:",
#     value=today,
#     min_value=datetime.date(2023, 1, 1), # 최소 날짜 설정
#     max_value=datetime.date(2026, 12, 31) # 최대 날짜 설정
# )

# st.sidebar.markdown("---")
# st.sidebar.info(f"오늘은: **{selected_date}입니다.**")

# # --- 메인 컨텐츠 영역 ---
# # st.title("Streamlit 사이드바 달력 예제")
# # st.write(f"사이드바에서 선택하신 날짜는 **{selected_date}** 입니다.")
# # st.write("이 날짜를 기반으로 메인 앱의 내용을 동적으로 업데이트할 수 있습니다.")



 # 출석 현황 그래프 예시     작업한 내용과 연계가 가능하다면 구현이 될듯   안되면 삭제할 부분

# data = {
#     "이름": ["김민수", "이영희", "박철수", "최지우", "정우성"],
#     "출석 횟수": [18, 20, 15, 19, 17]
# }

# df = pd.DataFrame(data)

# # 제목
# st.title("📊 출석 현황")

# # 데이터 테이블 표시
# # st.subheader("출석 데이터")
# # st.dataframe(df)

# # 막대 그래프 시각화
# fig = px.bar(df, x="이름", y="출석 횟수", color="이름",
#              title="학생별 출석 결과",
#              labels={"이름": "학생 이름", "출석 횟수": "출석 수"},
#              height=400)

# st.plotly_chart(fig)



