from langchain_community.document_loaders import SitemapLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
import streamlit as st

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🎄",
)

st.title("SiteGPT")



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


@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    loader = SitemapLoader(
        url,
        filter_urls=[r"^(.*\/ai-gateway\/).*",r"^(.*\/workers-ai\/).*",r"^(.*\/vectorize\/).*"]
    )
    loader.requests_per_second = 1
    docs = loader.load()
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings(openai_api_key=openai_api_key))
    return vector_store.as_retriever()


with st.sidebar:
    url = st.text_input("Write down a URL", placeholder = "https://example.com")
    openai_api_key = st.text_input("Input your OpenAI API Key")

if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL")
    else:
        if not openai_api_key:
            st.error("Please input your OpenAI API Key on the sidebar")
        else:
            llm = ChatOpenAI(
                temperature=0.1,
                openai_api_key=openai_api_key,
            )
            retriever = load_website(url)
            query = st.text_input("Ask a question to the website.")
            if query:
                chain = {
                    "docs" : retriever,
                    "question" : RunnablePassthrough(),
                } | RunnableLambda(get_answers) | RunnableLambda(choose_answer)

                result = chain.invoke(query)
                st.markdown(result.content.replace("$", "\$"))
