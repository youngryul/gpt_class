import streamlit as st
from langchain.prompts import PromptTemplate

st.title("Hello world!")

st.write(PromptTemplate) #PromptTemplate에 있는 property를 다 볼 수 있다.

st.selectbox(
    "Choose your model",
    (
        "GPT-3",
        "GPT-4"
    )
)

# 데이터가 변경되면 모든 페이지의 내용이 새로고침 된다.