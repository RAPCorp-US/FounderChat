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

# Fix BaseCache import issue specifically
@st.cache_resource
def import_and_fix_langchain():
    try:
        # First, import and fix the BaseCache issue
        from langchain_core.caches import BaseCache
        from langchain_core.globals import set_llm_cache
        
        # Import other core components
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        
        # Now import Google components after BaseCache is defined
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        
        # Explicitly rebuild the models with BaseCache now available
        ChatGoogleGenerativeAI.model_rebuild()
        GoogleGenerativeAIEmbeddings.model_rebuild()
        
        return ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings, StrOutputParser, ChatPromptTemplate, None
        
    except Exception as e:
        return None, None, None, None, str(e)

# Import everything
with st.spinner("🔄 Fixing imports and initializing..."):
    ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings, StrOutputParser, ChatPromptTemplate, error = import_and_fix_langchain()

if error:
    st.error(f"❌ Import failed: {error}")
    st.info("Falling back to direct Google AI SDK...")
    
    # Fallback to direct SDK
    try:
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        
        def direct_chat(prompt, context=""):
            try:
                model = genai.GenerativeModel('gemini-pro')
                if context:
                    full_prompt = f"Context: {context}\n\nQuestion: {prompt}"
                else:
                    full_prompt = prompt
                response = model.generate_content(full_prompt)
                return response.text
            except Exception as e:
                return f"Error: {str(e)}"
        
        st.success("✅ Using direct Google AI SDK (fallback mode)")
        use_direct_mode = True
        
    except Exception as e2:
        st.error(f"❌ Both LangChain and direct SDK failed: {e2}")
        st.stop()
        
else:
    st.success("✅ LangChain imports successful with BaseCache fix")
    use_direct_mode = False

# Initialize LLM only if LangChain worked
llm = None
if not use_direct_mode:
    @st.cache_resource
    def init_llm():
        try:
            # Try most stable model first
            return ChatGoogleGenerativeAI(
                model="gemini-pro",
                google_api_key=GOOGLE_API_KEY,
                temperature=0.7
            ), "Gemini Pro"
        except Exception as e:
            try:
                return ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    google_api_key=GOOGLE_API_KEY,
                    temperature=0.7
                ), "Gemini 1.5 Flash"
            except Exception as e2:
                return None, f"Both models failed: {e}, {e2}"

    llm, model_status = init_llm()
    if llm:
        st.success(f"✅ Model loaded: {model_status}")
    else:
        st.error(f"❌ Model initialization failed: {model_status}")
        use_direct_mode = True

# Page header
st.title("🤖 AI Interview Bot")
if use_direct_mode:
    st.markdown("Chat with your AI assistant using Google Gemini (Direct SDK Mode)")
else:
    st.markdown("Chat with your AI assistant using Google Gemini (LangChain Mode)")

# Load vector store (simplified for compatibility)
@st.cache_resource
def load_simple_docs():
    try:
        import joblib
        
        documents = []
        
        # Try joblib format first
        if os.path.exists("vectorstore_data.pkl"):
            data = joblib.load("vectorstore_data.pkl")
            for content, meta in zip(data["documents"], data["metadatas"]):
                documents.append({"content": content, "metadata": meta})
            return documents, f"Loaded {len(documents)} documents"
        
        # Try legacy pickle
        elif os.path.exists("vectorstore.pkl"):
            with open("vectorstore.pkl", "rb") as f:
                import pickle
                vectorstore = pickle.load(f)
            
            if hasattr(vectorstore, 'docstore'):
                docs = list(vectorstore.docstore._dict.values())
                for doc in docs:
                    documents.append({"content": doc.page_content, "metadata": doc.metadata})
                return documents, f"Loaded {len(documents)} documents from legacy store"
        
        return [], "No documents found"
        
    except Exception as e:
        return [], f"Error loading: {str(e)}"

documents, doc_status = load_simple_docs()
st.info(f"📚 Documents: {doc_status}")

# Simple search function
def search_documents(query, docs, max_results=3):
    if not docs:
        return []
    
    query_lower = query.lower()
    results = []
    
    for doc in docs:
        content_lower = doc["content"].lower()
        # Simple scoring based on keyword matches
        score = sum(1 for word in query_lower.split() if word in content_lower)
        if score > 0:
            results.append((score, doc))
    
    results.sort(reverse=True, key=lambda x: x[0])
    return [doc for score, doc in results[:max_results]]

# Sidebar
with st.sidebar:
    st.subheader("🔧 Settings")
    if not use_direct_mode:
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    max_docs = st.slider("Max documents", 1, 10, 3)
    show_sources = st.checkbox("Show sources", value=True)
    
    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.subheader("📊 Status")
    if use_direct_mode:
        st.write("🤖 Mode: Direct SDK")
    else:
        st.write("🤖 Mode: LangChain")
    st.write(f"📚 Documents: {len(documents)}")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message and show_sources and message["sources"]:
            with st.expander("📄 Sources"):
                for i, source in enumerate(message["sources"], 1):
                    st.write(f"**Source {i}:**")
                    st.write(source[:300] + "..." if len(source) > 300 else source)

# Chat input
user_input = st.chat_input("Ask me anything about your documents...")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):
        try:
            # Search for relevant documents
            relevant_docs = search_documents(user_input, documents, max_docs)
            
            if relevant_docs:
                context = "\n\n".join([f"Document {i+1}:\n{doc['content']}" 
                                     for i, doc in enumerate(relevant_docs)])
                sources = [doc['content'] for doc in relevant_docs]
            else:
                context = ""
                sources = []
            
            # Generate response based on mode
            if use_direct_mode:
                # Direct SDK mode
                with st.spinner("🤔 Thinking..."):
                    if context:
                        full_prompt = f"""You are an AI assistant. Answer based on the context provided.

Context from documents:
{context}

User question: {user_input}

Provide a helpful response based on the context."""
                    else:
                        full_prompt = f"You are an AI assistant. User question: {user_input}"
                    
                    response = direct_chat(full_prompt)
            else:
                # LangChain mode
                if not ChatPromptTemplate or not llm:
                    response = "❌ LangChain components not properly initialized"
                else:
                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", "You are an AI assistant. Use the provided context to answer questions. Context: {context}"),
                        ("human", "{input}")
                    ])
                    
                    with st.spinner("🤔 Thinking..."):
                        if StrOutputParser:
                            chain = prompt_template | llm.with_config(configurable={"temperature": temperature}) | StrOutputParser()
                            response = chain.invoke({"context": context, "input": user_input})
                        else:
                            response = llm.invoke(f"Context: {context}\n\nQuestion: {user_input}")
                            if hasattr(response, 'content'):
                                response = response.content
            
            # Display response
            st.markdown(response)
            
            # Store response
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "sources": sources
            })
            
            # Show sources
            if show_sources and sources:
                with st.expander("📄 Sources"):
                    for i, source in enumerate(sources, 1):
                        st.write(f"**Source {i}:**")
                        st.write(source[:300] + "..." if len(source) > 300 else source)
                        
        except Exception as e:
            error_msg = f"❌ Error generating response: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant", 
                "content": error_msg,
                "sources": []
            })

# Footer
st.markdown("---")
mode_text = "Direct SDK" if use_direct_mode else "LangChain"
st.markdown(f"*Powered by Google Gemini 🧠 | {mode_text} Mode 🚀*")