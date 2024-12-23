# from langchain.document_loaders import AsyncChromiumLoader
# from langchain.document_transformers import Html2TextTransformer
from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
import streamlit as st

from main import question

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🎄",
)

st.title("SiteGPT")

llm = ChatOpenAI(
    temperature = 0.1,
)

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!

    Question: {question}
"""
)

choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def get_answers(inputs):
    docs = inputs['docs']
    question = inputs['question']
    answers_chain = answers_prompt | llm
    # answers = []
    # for doc in docs:
    #     result = answers_chain.invoke({
    #         "question": question,
    #         "context" : doc.page_content
    #     })
    #     answers.append(result.content)
    # st.write(answers)
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }

def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )

def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (str(soup.get_text())
            .replace("\n", "")
            .replace("\xa0", "")
            .replace("CloseSearch Submit Blog", "")
            )

@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    spliiter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200
    )
    loader = SitemapLoader(
        url,
        # filter_urls=[r"^(.*\/blog\/).*",],
        parsing_function=parse_page
    )
    # 스크롤링 시 빠르게 하면 차단 당하기 때문에 넣어줌
    loader.requests_per_second = 1
    docs = loader.load_and_split(text_splitter=spliiter)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vector_store.as_retriever()

# html2text_transformer = Html2TextTransformer()

with st.sidebar:
    url = st.text_input("Write down a URL", placeholder = "https://example.com")

if url:
    # loader = AsyncChromiumLoader([url])
    # docs = loader.load()
    # st.write(docs)
    # transformed = html2text_transformer.transform_documents(docs)
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL")
    else:
        retriever = load_website(url)
        query = st.text_input("Ask a question to the website.")
        if query:
            chain = {
                "docs" : retriever,
                "question" : RunnablePassthrough(),
            } | RunnableLambda(get_answers) | RunnableLambda(choose_answer)

            result = chain.invoke(query)
            st.markdown(result.content.replace("$", "\$"))
