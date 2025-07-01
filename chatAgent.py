import streamlit as st
import pandas as pd
import os
import pickle
import joblib
from langchain.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


# Handle API key
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")


from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)

st.subheader("Chat with your AI Assistant, Interview Bot!")

vector_store_path = "vectorstore.pkl"
with open(vector_store_path, "rb") as f:
    vectorstore = joblib.load(f)


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


prompt_template = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful AI assistant named Interview Bot..."), ("user", "{input}")]
)
chain = prompt_template | llm | StrOutputParser()


user_input = st.chat_input("Ask your question:")
if user_input and vectorstore is not None:
    st.session_state.messages.append({"role": "user", "content": user_input})

    retriever = vectorstore.as_retriever()
    context = "\n\n".join(doc.page_content for doc in retriever.get_relevant_documents(user_input))
    augmented_user_input = f"Context: {context}\n\nQuestion: {user_input}"


    response = chain.invoke({"input": augmented_user_input})
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)