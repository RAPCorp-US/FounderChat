from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import pandas as pd
import streamlit as st
import pickle
import os

# Set your Google API key here or via environment variable
# Get your API key from: https://aistudio.google.com/app/apikey
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or "your-google-api-key-here"

if not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

st.set_page_config(layout="wide", page_title="Document Vector Store Creator")

# Initialize Gemini components
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # Free tier model
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7
)

# Initialize embeddings using Gemini embedding model
document_embedder = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",  # Using stable model for Windows compatibility
    google_api_key=GOOGLE_API_KEY
)

st.title("📚 Document Vector Store Creator")
st.markdown("Upload documents to create a searchable knowledge base for your AI assistant using Google Gemini.")

# Sidebar for file upload
with st.sidebar:
    st.subheader("📤 Upload Documents")
    
    # Create docs directory
    DOCS_DIR = os.path.abspath("./uploaded_docs")
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
    
    # Display current files
    current_files = os.listdir(DOCS_DIR) if os.path.exists(DOCS_DIR) else []
    if current_files:
        st.markdown("**📁 Current files:**")
        for file in current_files:
            st.write(f"• {file}")
    else:
        st.info("No files uploaded yet.")
    
    st.markdown("---")
    
    # File upload form
    with st.form("upload-form", clear_on_submit=True):
        uploaded_files = st.file_uploader(
            "Choose files to upload:",
            accept_multiple_files=True,
            type=['txt', 'pdf', 'docx', 'md', 'py', 'json', 'csv', 'html']
        )
        submitted = st.form_submit_button("📁 Upload Files")
    
    if uploaded_files and submitted:
        for uploaded_file in uploaded_files:
            try:
                # Save uploaded file
                file_path = os.path.join(DOCS_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())
                st.success(f"✅ {uploaded_file.name} uploaded successfully!")
            except Exception as e:
                st.error(f"❌ Error uploading {uploaded_file.name}: {e}")
    
    st.markdown("---")
    st.subheader("📋 Supported File Types")
    st.markdown("""
    - `.txt` - Text files
    - `.pdf` - PDF documents  
    - `.docx` - Word documents
    - `.md` - Markdown files
    - `.py` - Python files
    - `.json` - JSON files
    - `.csv` - CSV files
    - `.html` - HTML files
    """)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔧 Vector Store Configuration")
    
    # Vector store options
    use_existing_vector_store = st.radio(
        "Choose an option:",
        ["Create new vector store", "Use existing vector store (if available)"]
    )
    
    # Advanced settings in expander
    with st.expander("⚙️ Advanced Settings"):
        chunk_size = st.slider("Chunk size for text splitting", 500, 3000, 2000, 100)
        chunk_overlap = st.slider("Chunk overlap", 0, 500, 200, 50)
        st.info("Smaller chunks = more precise retrieval, larger chunks = more context")

with col2:
    st.subheader("📊 Statistics")
    
    # File statistics
    if current_files:
        total_size = sum(os.path.getsize(os.path.join(DOCS_DIR, f)) for f in current_files)
        st.metric("Total Files", len(current_files))
        st.metric("Total Size", f"{total_size / 1024:.1f} KB")
    else:
        st.metric("Total Files", 0)

# Vector store creation/loading
vector_store_path = "vectorstore.pkl"
vector_store_exists = os.path.exists(vector_store_path)

st.markdown("---")

if st.button("🚀 Process Documents", type="primary"):
    if use_existing_vector_store == "Use existing vector store (if available)" and vector_store_exists:
        try:
            # Try loading with FAISS method first
            if os.path.exists("vectorstore_faiss"):
                vectorstore = FAISS.load_local("vectorstore_faiss", document_embedder, allow_dangerous_deserialization=True)
                st.success("✅ Existing vector store loaded successfully using FAISS method!")
            
            # Try alternative method
            elif os.path.exists("vectorstore_data.pkl"):
                import joblib
                vectorstore_data = joblib.load("vectorstore_data.pkl")
                
                # Recreate the vectorstore
                from langchain.docstore.in_memory import InMemoryDocstore
                from langchain.schema import Document
                
                # Recreate documents
                documents = [
                    Document(page_content=content, metadata=meta) 
                    for content, meta in zip(vectorstore_data["documents"], vectorstore_data["metadatas"])
                ]
                
                # Recreate vectorstore from documents
                vectorstore = FAISS.from_documents(documents, document_embedder)
                st.success("✅ Existing vector store loaded successfully using alternative method!")
            
            # Try legacy pickle method
            else:
                with open(vector_store_path, "rb") as f:
                    vectorstore = pickle.load(f)
                st.success("✅ Existing vector store loaded successfully using legacy method!")
            
            # Display info about the loaded vector store
            if hasattr(vectorstore, 'index') and hasattr(vectorstore.index, 'ntotal'):
                st.info(f"📈 Vector store contains {vectorstore.index.ntotal} document chunks")
                
        except Exception as e:
            st.error(f"❌ Error loading existing vector store: {e}")
            st.info("Creating a new vector store instead...")
            use_existing_vector_store = "Create new vector store"
    
    if use_existing_vector_store == "Create new vector store" or not vector_store_exists:
        if not current_files:
            st.warning("⚠️ Please upload some documents first!")
        else:
            try:
                # Load documents
                with st.spinner("📖 Loading documents..."):
                    loader = DirectoryLoader(DOCS_DIR)
                    raw_documents = loader.load()
                
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
                    with st.spinner("🧠 Creating embeddings and vector store (this may take a few minutes)..."):
                        vectorstore = FAISS.from_documents(documents, document_embedder)
                    
                    # Save vector store using FAISS built-in methods (Windows compatible)
                    with st.spinner("💾 Saving vector store..."):
                        try:
                            # Save FAISS index and documents separately
                            vectorstore.save_local("vectorstore_faiss")
                            
                            # Also save metadata for loading
                            import json
                            metadata = {
                                "chunk_size": chunk_size,
                                "chunk_overlap": chunk_overlap,
                                "num_documents": len(documents),
                                "embedding_model": "models/embedding-001"
                            }
                            with open("vectorstore_metadata.json", "w") as f:
                                json.dump(metadata, f)
                                
                        except Exception as e:
                            st.error(f"❌ Error saving with FAISS method: {e}")
                            # Fallback: Save without the embeddings object
                            st.info("🔄 Trying alternative save method...")
                            
                            # Extract just the index and texts
                            vectorstore_data = {
                                "index": vectorstore.index,
                                "docstore": vectorstore.docstore,
                                "index_to_docstore_id": vectorstore.index_to_docstore_id,
                                "documents": [doc.page_content for doc in documents],
                                "metadatas": [doc.metadata for doc in documents]
                            }
                            
                            import joblib
                            joblib.dump(vectorstore_data, "vectorstore_data.pkl")
                            
                            # Save a flag to indicate we used this method
                            with open("vectorstore_method.txt", "w") as f:
                                f.write("alternative")
                    
                    st.success("🎉 Vector store created and saved successfully!")
                    st.info(f"📈 Vector store contains {len(documents)} document chunks")
                    
                    # Show sample chunks
                    with st.expander("👀 Sample Document Chunks"):
                        for i, doc in enumerate(documents[:3]):
                            st.write(f"**Chunk {i+1}:**")
                            st.write(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                            st.write("---")
                            
            except Exception as e:
                st.error(f"❌ Error creating vector store: {e}")
                st.exception(e)

# Instructions
st.markdown("---")
st.markdown("### 📋 Next Steps")
st.markdown("""
1. **Upload your documents** using the sidebar file uploader
2. **Process the documents** by clicking the "Process Documents" button above
3. **Run the chat interface** by executing `streamlit run chatAgent.py`
4. **Start chatting** with your AI assistant about the uploaded documents!

**Note:** Make sure to set your Google API key in the code or as an environment variable `GOOGLE_API_KEY`.
Get your free API key from: https://aistudio.google.com/app/apikey
""")