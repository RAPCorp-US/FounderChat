import streamlit as st
import os
import sys

# Page config first
st.set_page_config(layout="wide", page_title="AI Interview Bot")

# Handle API key
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

if not GOOGLE_API_KEY:
    st.error("❌ Please set your GOOGLE_API_KEY in Streamlit secrets")
    st.info("Go to Settings → Secrets in your app and add:")
    st.code("GOOGLE_API_KEY = 'your-actual-api-key-here'")
    st.stop()

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Import with proper error handling
@st.cache_resource
def import_and_init_models():
    try:
        # Import libraries with retry logic
        import time
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Core imports
                from langchain_core.output_parsers import StrOutputParser
                from langchain_core.prompts import ChatPromptTemplate
                
                # Google AI imports with error handling
                from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
                
                # Force rebuild if needed
                if hasattr(ChatGoogleGenerativeAI, 'model_rebuild'):
                    ChatGoogleGenerativeAI.model_rebuild()
                if hasattr(GoogleGenerativeAIEmbeddings, 'model_rebuild'):
                    GoogleGenerativeAIEmbeddings.model_rebuild()
                
                break
                
            except Exception as e:
                if attempt < max_retries - 1:
                    st.warning(f"Import attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(1)
                else:
                    raise e
        
        # Initialize LLM with conservative settings
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-pro",
                google_api_key=GOOGLE_API_KEY,
                temperature=0.7,
                convert_system_message_to_human=True  # For compatibility
            )
            model_name = "Gemini Pro"
        except Exception as e:
            st.warning(f"Gemini Pro failed: {e}")
            try:
                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    google_api_key=GOOGLE_API_KEY,
                    temperature=0.7,
                    convert_system_message_to_human=True
                )
                model_name = "Gemini 1.5 Flash"
            except Exception as e2:
                raise Exception(f"All models failed. Pro: {e}, Flash: {e2}")
        
        # Test the model
        test_response = llm.invoke("Hello")
        
        return llm, StrOutputParser(), ChatPromptTemplate, model_name, True
        
    except Exception as e:
        return None, None, None, str(e), False

# Initialize models
with st.spinner("🔄 Initializing AI models..."):
    llm, parser, prompt_template_class, result, success = import_and_init_models()

if not success:
    st.error(f"❌ Failed to initialize models: {result}")
    st.info("This might be a temporary issue. Please try refreshing the page.")
    st.stop()
else:
    st.success(f"✅ Successfully loaded: {result}")

# Page header
st.title("🤖 AI Interview Bot")
st.markdown("Chat with your AI assistant powered by Google Gemini!")

# Load vector store with fallback methods
@st.cache_resource
def load_vectorstore():
    try:
        from langchain.vectorstores import FAISS
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        import joblib
        
        # Try different loading methods
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        
        # Method 1: FAISS native
        if os.path.exists("vectorstore_faiss"):
            vectorstore = FAISS.load_local("vectorstore_faiss", embeddings, allow_dangerous_deserialization=True)
            return vectorstore, "FAISS native"
        
        # Method 2: Joblib
        if os.path.exists("vectorstore_data.pkl"):
            from langchain.schema import Document
            data = joblib.load("vectorstore_data.pkl")
            documents = [Document(page_content=c, metadata=m) for c, m in zip(data["documents"], data["metadatas"])]
            vectorstore = FAISS.from_documents(documents, embeddings)
            return vectorstore, "reconstructed"
        
        # Method 3: Legacy pickle
        if os.path.exists("vectorstore.pkl"):
            with open("vectorstore.pkl", "rb") as f:
                vectorstore = joblib.load(f)
            return vectorstore, "legacy"
            
        return None, "not found"
        
    except Exception as e:
        return None, f"error: {e}"

vectorstore, load_status = load_vectorstore()

if vectorstore:
    st.success(f"✅ Knowledge base loaded ({load_status})")
    if hasattr(vectorstore, 'index') and hasattr(vectorstore.index, 'ntotal'):
        st.info(f"📚 Contains {vectorstore.index.ntotal} document chunks")
else:
    st.warning(f"⚠️ No knowledge base found ({load_status})")
    st.info("Please run the vector store creation script first.")

# Sidebar settings
with st.sidebar:
    st.subheader("🔧 Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    max_docs = st.slider("Max documents", 1, 10, 3)
    show_sources = st.checkbox("Show sources", value=True)
    
    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message and show_sources:
            with st.expander("📄 Sources"):
                for i, source in enumerate(message["sources"], 1):
                    st.write(f"**Source {i}:**")
                    st.write(source[:300] + "..." if len(source) > 300 else source)

# Create prompt template
prompt_template = prompt_template_class.from_messages([
    ("system", """You are an intelligent AI assistant called Interview Bot. You help users by answering questions based on the provided context.

Instructions:
- Use the context to answer questions accurately
- If context doesn't contain relevant info, say so politely
- Be conversational and helpful
- For interviews, provide structured responses

Context: {context}"""),
    ("human", "{input}")
])

# Create chain
chain = prompt_template | llm.with_config(configurable={"temperature": temperature}) | parser

# Chat input
user_input = st.chat_input("Ask me anything about your documents...")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):
        if vectorstore:
            try:
                with st.spinner("🔍 Searching..."):
                    retriever = vectorstore.as_retriever(search_kwargs={"k": max_docs})
                    docs = retriever.get_relevant_documents(user_input)
                
                context = "\n\n".join([f"Doc {i+1}: {doc.page_content}" for i, doc in enumerate(docs)])
                
                with st.spinner("🤔 Thinking..."):
                    response = chain.invoke({"context": context, "input": user_input})
                
                st.markdown(response)
                
                # Store response
                sources = [doc.page_content for doc in docs]
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "sources": sources
                })
                
                # Show sources
                if show_sources and docs:
                    with st.expander("📄 Sources"):
                        for i, doc in enumerate(docs, 1):
                            st.write(f"**Source {i}:**")
                            st.write(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg, "sources": []})
        else:
            fallback = "I don't have access to any documents. Please upload documents first using the vector store creation script."
            st.markdown(fallback)
            st.session_state.messages.append({"role": "assistant", "content": fallback, "sources": []})

# Footer
st.markdown("---")
st.markdown("*Powered by Google Gemini 🧠 | Deployment-Safe Version 🚀*")