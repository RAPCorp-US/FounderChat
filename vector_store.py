from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import pandas as pd
import streamlit as st
import pickle
import os

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

if not GOOGLE_API_KEY:
    st.error("❌ Please set your GOOGLE_API_KEY in Streamlit secrets")
    st.stop()

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

st.set_page_config(layout="wide", page_title="Document Vector Store Creator")
st.title("📚 Document Vector Store Creator")

# Initialize embeddings
@st.cache_resource
def get_embeddings():
    try:
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=GOOGLE_API_KEY
        )
    except Exception as e:
        st.error(f"❌ Failed to initialize embeddings: {e}")
        return None

document_embedder = get_embeddings()
if not document_embedder:
    st.stop()

# Create docs directory
DOCS_DIR = os.path.abspath("./uploaded_docs")
if not os.path.exists(DOCS_DIR):
    os.makedirs(DOCS_DIR)

# Function to get current files
def get_uploaded_files():
    try:
        return os.listdir(DOCS_DIR)
    except:
        return []

# Sidebar for file upload
with st.sidebar:
    st.subheader("📤 Upload Documents")
    
    # Show current files
    current_files = get_uploaded_files()
    if current_files:
        st.success(f"📁 {len(current_files)} files uploaded:")
        for file in current_files:
            st.write(f"• {file}")
    else:
        st.info("No files uploaded yet.")
    
    st.markdown("---")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Choose files to upload:",
        accept_multiple_files=True,
        type=['txt', 'pdf', 'docx', 'md', 'py', 'json', 'csv', 'html']
    )
    
    # Save uploaded files
    if uploaded_files:
        if st.button("💾 Save Files", type="primary"):
            saved_count = 0
            for uploaded_file in uploaded_files:
                try:
                    file_path = os.path.join(DOCS_DIR, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.read())
                    saved_count += 1
                    st.success(f"✅ Saved: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"❌ Failed to save {uploaded_file.name}: {e}")
            
            if saved_count > 0:
                st.rerun()  # Refresh to show new files
    
    # Manual refresh
    if st.button("🔄 Refresh File List"):
        st.rerun()

# Main processing area
st.markdown("---")
st.subheader("⚙️ Vector Store Options")

# Radio button for existing vs new
use_existing = st.radio(
    "Choose an option:",
    ["Use existing vector store (if available)", "Create new vector store"]
)

# Settings
if use_existing == "Create new vector store":
    chunk_size = st.slider("Chunk size for text splitting", 500, 3000, 2000, 100)
    chunk_overlap = st.slider("Chunk overlap", 0, 500, 200, 50)

# Check what files are available for processing
files_to_process = get_uploaded_files()
vector_store_path = "vectorstore.pkl"
vector_store_exists = os.path.exists(vector_store_path)

# Process based on selection
if st.button("🚀 Process Documents", type="primary"):
    if use_existing == "Use existing vector store (if available)" and vector_store_exists:
        try:
            with open(vector_store_path, "rb") as f:
                vectorstore = pd.read_pickle(f)
            st.success("✅ Existing vector store loaded successfully!")
            
            # Show info about loaded store
            if hasattr(vectorstore, 'index') and hasattr(vectorstore.index, 'ntotal'):
                st.info(f"📈 Vector store contains {vectorstore.index.ntotal} document chunks")
        except Exception as e:
            st.error(f"❌ Error loading existing vector store: {e}")
    
    elif use_existing == "Create new vector store" or not vector_store_exists:
        if not files_to_process:
            st.warning("⚠️ Please upload some documents first!")
        else:
            try:
                # Load documents
                with st.spinner("📖 Loading documents..."):
                    raw_documents = DirectoryLoader(DOCS_DIR).load()
                
                if not raw_documents:
                    st.error("❌ No documents could be loaded. Please check your file formats.")
                else:
                    st.success(f"📚 Loaded {len(raw_documents)} documents")
                    
                    # Split documents
                    with st.spinner("✂️ Splitting documents into chunks..."):
                        text_splitter = CharacterTextSplitter(
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap
                        )
                        documents = text_splitter.split_documents(raw_documents)
                    
                    st.success(f"📄 Created {len(documents)} document chunks")
                    
                    # Create vector store
                    with st.spinner("🧠 Creating embeddings and vector store..."):
                        vectorstore = FAISS.from_documents(documents, document_embedder)
                    
                    # Save vector store
                    with st.spinner("💾 Saving vector store..."):
                        vectorstore.save_local("vectorstore_faiss")
                        st.success("✅ Vector store saved as vectorstore_faiss folder")
                    
                    st.success("🎉 Vector store created and saved successfully!")
                    st.info(f"📈 Vector store contains {len(documents)} document chunks")
                    
                    # Show sample chunks
                    with st.expander("👀 Sample Document Chunks"):
                        for i, doc in enumerate(documents[:3]):
                            st.write(f"**Chunk {i+1}:**")
                            st.write(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                            if i < 2:
                                st.write("---")
                            
            except Exception as e:
                st.error(f"❌ Error creating vector store: {e}")
                st.exception(e)

# Show current status
st.markdown("---")
st.subheader("📊 Current Status")

if vector_store_exists:
    st.success("✅ Vector store file exists")
    try:
        # Try to get info about the vector store
        with open(vector_store_path, "rb") as f:
            vectorstore = pd.read_pickle(f)
        if hasattr(vectorstore, 'index') and hasattr(vectorstore.index, 'ntotal'):
            st.info(f"📈 Contains {vectorstore.index.ntotal} document chunks")
    except:
        st.warning("⚠️ Vector store file exists but may be corrupted")
else:
    st.info("ℹ️ No vector store found yet")

# Instructions
st.markdown("---")
st.markdown("### 📋 Next Steps")
st.markdown("""
1. **Upload your documents** using the sidebar file uploader
2. **Click "Save Files"** to store them
3. **Choose to create new or use existing** vector store
4. **Click "Process Documents"** to create the embeddings
5. **Run the chat interface:** `streamlit run chatAgent.py`

**Note:** Make sure to set your Google API key in Streamlit secrets as `GOOGLE_API_KEY`.
""")