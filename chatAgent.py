import streamlit as st
import pandas as pd
import os
import pickle
import joblib
from langchain.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Set your Google API key here or via environment variable
# Get your API key from: https://aistudio.google.com/app/apikey
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or "your-google-api-key-here"

if not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

st.set_page_config(layout="wide", page_title="AI Interview Bot")

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro-preview-03-25",  # Gemini 2.5 Pro  
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7
)

# Page header
st.title("🤖 AI Interview Bot")
st.markdown("Chat with your AI assistant powered by Google Gemini and your uploaded documents!")

# Load vector store
vector_store_path = "vectorstore.pkl"

if os.path.exists(vector_store_path):
    try:
        with open(vector_store_path, "rb") as f:
            vectorstore = pickle.load(f)
        st.success("✅ Knowledge base loaded successfully!")
        
        # Show vector store info
        if hasattr(vectorstore, 'index') and hasattr(vectorstore.index, 'ntotal'):
            st.info(f"📚 Knowledge base contains {vectorstore.index.ntotal} document chunks")
    except Exception as e:
        st.error(f"❌ Error loading knowledge base: {e}")
        vectorstore = None
else:
    st.warning("⚠️ No knowledge base found. Please run `streamlit run vector_store.py` first to create one.")
    vectorstore = None

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar with settings and info
with st.sidebar:
    st.subheader("🔧 Chat Settings")
    
    # Model settings
    with st.expander("⚙️ Model Configuration"):
        temperature = st.slider("Temperature (creativity)", 0.0, 1.0, 0.7, 0.1)
        max_docs = st.slider("Max documents to retrieve", 1, 10, 3)
        show_sources = st.checkbox("Show retrieved sources", value=True)
    
    st.markdown("---")
    
    # Chat controls
    st.subheader("💬 Chat Controls")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("🔄 Reload Knowledge Base"):
        st.rerun()
    
    st.markdown("---")
    
    # Instructions
    st.subheader("📋 How to Use")
    st.markdown("""
    1. **First time:** Run `vector_store.py` to upload documents
    2. **Ask questions** about your uploaded documents
    3. **View sources** to see which documents were used
    4. **Adjust settings** above to customize responses
    """)
    
    st.markdown("---")
    
    # Tips
    st.subheader("💡 Tips")
    st.markdown("""
    - Ask specific questions for better answers
    - Reference specific topics from your documents
    - Use follow-up questions to dive deeper
    - Check the sources to verify information
    """)

# Create the prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are an intelligent AI assistant called Interview Bot, powered by Google Gemini. You help users by answering questions based on the provided context from their uploaded documents.

Key Instructions:
- Use the provided context to answer questions accurately and comprehensively
- If the context doesn't contain relevant information, clearly state this and provide general knowledge if helpful
- Be conversational, helpful, and professional
- For interview-related questions, provide structured and detailed responses
- When discussing technical topics, break down complex concepts clearly
- Always be honest about the limitations of the provided context
- If asked about your capabilities, mention that you can help with questions about the uploaded documents

Context provided: {context}

Remember to be helpful and engaging while staying grounded in the provided information."""),
    ("human", "{input}")
])

# Create the chain with current temperature
chain = prompt_template | llm.with_config(configurable={"temperature": temperature}) | StrOutputParser()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show sources if they exist
        if message["role"] == "assistant" and "sources" in message and show_sources:
            with st.expander("📄 Sources Used"):
                for i, source in enumerate(message["sources"], 1):
                    st.write(f"**Source {i}:**")
                    st.write(source[:400] + "..." if len(source) > 400 else source)
                    if i < len(message["sources"]):
                        st.write("---")

# Chat input
user_input = st.chat_input("Ask me anything about your documents...")

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate assistant response
    with st.chat_message("assistant"):
        if vectorstore is not None:
            try:
                # Retrieve relevant documents
                retriever = vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": max_docs}
                )
                
                with st.spinner("🔍 Searching knowledge base..."):
                    relevant_docs = retriever.get_relevant_documents(user_input)
                
                # Create context from retrieved documents
                context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" 
                                     for i, doc in enumerate(relevant_docs)])
                
                # Generate response
                with st.spinner("🤔 Thinking..."):
                    response = chain.invoke({
                        "context": context,
                        "input": user_input
                    })
                
                # Display response
                st.markdown(response)
                
                # Prepare sources for storage
                sources = [doc.page_content for doc in relevant_docs] if relevant_docs else []
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "sources": sources
                })
                
                # Show sources immediately if enabled
                if show_sources and relevant_docs:
                    with st.expander("📄 Sources Used"):
                        for i, doc in enumerate(relevant_docs, 1):
                            st.write(f"**Source {i}:**")
                            content = doc.page_content
                            st.write(content[:400] + "..." if len(content) > 400 else content)
                            if i < len(relevant_docs):
                                st.write("---")
                
            except Exception as e:
                error_msg = f"❌ Error generating response: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_msg,
                    "sources": []
                })
                
        else:
            # Fallback response when no vectorstore is available
            fallback_response = """I don't have access to any documents right now. To get started:

1. Run `streamlit run vector_store.py` to upload and process your documents
2. Come back here to chat about them!

In the meantime, I can still help with general questions, but I won't have access to your specific documents."""
            
            st.markdown(fallback_response)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": fallback_response,
                "sources": []
            })

# Footer
st.markdown("---")
st.markdown("*Powered by Google Gemini 🧠 | Built with Streamlit 🚀*")