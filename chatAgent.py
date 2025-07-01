import streamlit as st
import pandas as pd
import os
import pickle
import joblib
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import langchain.docstore.document as docstore_document

# This patch is a temporary workaround for the loading error.
# It should be removed once the vector store is recreated.
docstore_document.Document.__slots__ = list(docstore_document.Document.__slots__) + ['__fields_set__']

# Handle API key
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

if not GOOGLE_API_KEY:
    st.error("❌ Please set your GOOGLE_API_KEY in Streamlit secrets")
    st.stop()

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

st.set_page_config(layout="wide")

# Use direct Google SDK for chat (avoids BaseCache issues)
@st.cache_resource
def init_gemini():
    try:
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-2.5-flash')
        # Test it
        test_response = model.generate_content("Hello")
        return model
    except Exception as e:
        st.error(f"❌ Failed to initialize Gemini: {e}")
        return None

# Initialize Gemini
gemini_model = init_gemini()
if not gemini_model:
    st.stop()

st.subheader("Chat with your AI Assistant, Interview Bot!")

@st.cache_resource
def load_vector_store():
    # This patch is a temporary workaround for the loading error.
    # It should be removed once the vector store is recreated.
    docstore_document.Document.__slots__ = list(docstore_document.Document.__slots__) + ['__fields_set__']
    
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        vectorstore = FAISS.load_local("vectorstore_faiss", embeddings, allow_dangerous_deserialization=True)
        return vectorstore, "✅ Vector store loaded successfully!"
    except FileNotFoundError:
        return None, "⚠️ No vector store found. Please run vector_store.py first to create one."
    except Exception as e:
        return None, f"❌ Error loading vector store: {e}"

vectorstore, status_message = load_vector_store()
st.info(status_message)

# Function to search vector store manually (since we can't use LangChain retriever)
def search_vectorstore(query, vectorstore, k=3):
    if not vectorstore or not hasattr(vectorstore, 'similarity_search'):
        return []
    
    try:
        # Use the vectorstore's similarity search directly
        docs = vectorstore.similarity_search(query, k=k)
        return docs
    except Exception as e:
        st.warning(f"⚠️ Search failed: {e}")
        return []

# Chat function using direct Gemini SDK
def chat_with_gemini(user_input, context=""):
    try:
        if context:
            prompt = f"""Your purpose is to answer questions about Youssef. To help other to get to know him

Context from documents:
{context}

User question: {user_input}

Instructions:
- Use the context to provide accurate and helpful answers
- If the context doesn't contain relevant information, say so politely
- Be conversational and professional
- For interview-related questions, provide structured responses"""
        else:
            prompt = f"""You are a helpful AI assistant named Interview Bot.

User question: {user_input}

Please provide a helpful response. Note: I don't have access to any uploaded documents right now."""

        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Error generating response: {str(e)}"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show sources if available
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📄 Sources Used"):
                for i, source in enumerate(message["sources"], 1):
                    st.write(f"**Source {i}:**")
                    st.write(source[:400] + "..." if len(source) > 400 else source)
                    if i < len(message["sources"]):
                        st.write("---")

# Chat input
user_input = st.chat_input("Ask your question:")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate assistant response
    with st.chat_message("assistant"):
        if vectorstore is not None:
            try:
                # Search for relevant documents
                with st.spinner("🔍 Searching documents..."):
                    relevant_docs = search_vectorstore(user_input, vectorstore, k=3)
                
                if relevant_docs:
                    # Create context from retrieved documents
                    context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" 
                                         for i, doc in enumerate(relevant_docs)])
                    sources = [doc.page_content for doc in relevant_docs]
                else:
                    context = ""
                    sources = []
                
                # Generate response using Gemini
                with st.spinner("🤔 Thinking..."):
                    response = chat_with_gemini(user_input, context)
                
                # Display response
                st.markdown(response)
                
                # Store response with sources
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "sources": sources
                })
                
                # Show sources immediately
                if sources:
                    with st.expander("📄 Sources Used"):
                        for i, source in enumerate(sources, 1):
                            st.write(f"**Source {i}:**")
                            st.write(source[:400] + "..." if len(source) > 400 else source)
                            if i < len(sources):
                                st.write("---")
                
            except Exception as e:
                error_msg = f"❌ Error processing request: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_msg,
                    "sources": []
                })
        else:
            # No vector store available
            with st.spinner("🤔 Thinking..."):
                response = chat_with_gemini(user_input)
            
            st.markdown(response)
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "sources": []
            })

# Sidebar with info
with st.sidebar:
    st.subheader("📊 Status")
    
    if gemini_model:
        st.success("✅ Gemini AI connected")
    else:
        st.error("❌ Gemini AI failed")
    
    if vectorstore:
        if hasattr(vectorstore, 'index') and hasattr(vectorstore.index, 'ntotal'):
            st.success(f"✅ Vector store: {vectorstore.index.ntotal} chunks")
        else:
            st.success("✅ Vector store loaded")
    else:
        st.warning("⚠️ No vector store")
    
    st.markdown("---")
    st.subheader("🔧 Controls")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("🔄 Reload Vector Store"):
        st.cache_resource.clear()
        st.rerun()
    
    st.markdown("---")
    st.subheader("💡 How It Works")
    st.markdown("""
    **Hybrid Architecture:**
    - 🔍 **Vector search:** Uses LangChain FAISS
    - 💬 **Chat:** Direct Google SDK
    - 📚 **Documents:** Processed by vector_store.py
    
    This avoids BaseCache issues while keeping full functionality!
    """)
