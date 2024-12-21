import streamlit as st
import random
import json
from operator import rshift
from langchain_community.retrievers import WikipediaRetriever
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import BaseOutputParser, output_parser


st.set_page_config(
    page_title="QuizGPT",
    page_icon="🎯",
)

st.title("QuizGPT")

difficulty_options = ["Easy", "Medium", "Hard"]

difficulty = st.selectbox("Select quiz difficulty:", difficulty_options)

function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-1106",
    streaming=True,
    callbacks=[
        StreamingStdOutCallbackHandler()
    ]
).bind(
    function_call={
        "name": "create_quiz",
    },
    functions=[
        function,
    ],
)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)
    
questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""
        You are a helpful assistant that is role playing as a teacher.
            
        Based ONLY on the following context make 10 questions to test the user's knowledge about the text.
        The questions should be of {difficulty.lower()} difficulty.
        Each question should have 4 answers, three of them must be incorrect and one should be correct.
            
        Use (o) to signal the correct answer.
            
        Question examples:
            
        Question: What is the color of the ocean?
        Answers: Red|Yellow|Green|Blue(o)
            
        Question: What is the capital or Georgia?
        Answers: Baku|Tbilisi(o)|Manila|Beirut
            
        Question: When was Avatar released?
        Answers: 2007|2001|2009(o)|1998
            
        Question: Who was Julius Caesar?
        Answers: A Roman Emperor(o)|Painter|Actor|Model
            
        Your turn!
            
        Context: {{context}}
        """,
        )
    ]
)

questions_chain = questions_prompt | llm




@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs

@st.cache_data(show_spinner="Making quiz....")
def run_quiz_chain(_docs, topic, difficulty):  # 서명을 만들지 않게 해줌 (streamlit 한정문제)
    chain = {"context" : format_docs} | questions_prompt | llm
    response = chain.invoke(_docs)
    arguments = json.loads(response.additional_kwargs["function_call"]["arguments"])
    for index in range(len(arguments["questions"])):
        random.shuffle(arguments["questions"][index]["answers"])
    return arguments

@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(topic)
    return docs
    


with st.sidebar:
    docs = None
    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikipedia Article",
        ),
    )
    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx , .txt or .pdf file",
            type=["pdf", "txt", "docx"],
        )
        if file:
            docs = split_file(file)
            st.write(docs)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            docs = wiki_search(topic)
         


if not docs:
    st.markdown(
        """
    Welcome to QuizGPT.
                
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )
else:
  response = run_quiz_chain(docs, topic if topic else file.name, difficulty)
  cnt = 0
  chk = 0
  with st.form("questions_form"):
    for idx, question in enumerate(response["questions"]):
        st.write(question["question"])
        chk += 1
        value = st.radio(
            f"Select an option {idx}.",
            [answer["answer"] for answer in question["answers"]],
            key=f"{idx}_radio",
            index=None
        )
        if {"answer": value, "correct": True} in question["answers"]:
            st.success("Correct!")
            cnt += 1
        elif value is not None:
            st.error("Wrong!")
    button = st.form_submit_button()
    if chk == cnt:
        st.balloons()