🤖 AI Chat Bot with Google Gemini
A Streamlit-based chat application that lets you chat with your documents using Google's Gemini AI model.
🚀 Quick Start
1. Get Your API Key

Visit Google AI Studio
Create a free Google account if you don't have one
Generate an API key (it's free!)

2. Setup

Install dependencies:
bashpip install -r requirements.txt

Set your API key:

Option A: Set environment variable
bashexport GOOGLE_API_KEY="your-api-key-here"

Option B: Edit the code files and replace "your-google-api-key-here" with your actual key



3. Create Your Knowledge Base

Run the vector store creator:
bashstreamlit run vector_store.py

Upload your documents using the sidebar
Click "Process Documents" to create the knowledge base

4. Start Chatting

Run the chat interface:
bashstreamlit run chatAgent.py

Ask questions about your uploaded documents!

📁 Supported File Types

.txt - Text files
.pdf - PDF documents
.docx - Word documents
.md - Markdown files
.py - Python files
.json - JSON files
.csv - CSV files
.html - HTML files

🔧 Features

Smart Document Search: Uses FAISS vector similarity search
Context-Aware Responses: Gemini provides answers based on your documents
Source Attribution: See which documents were used for each answer
Customizable Settings: Adjust creativity, retrieval count, etc.
Chat History: Maintains conversation context
Free to Use: Google Gemini has a generous free tier

📋 Files Overview

chatAgent.py - Main chat interface
vector_store.py - Document upload and processing
requirements.txt - Python dependencies
devcontainer.json - Development container configuration

🛠️ How It Works

Document Processing: Your files are split into chunks and converted to embeddings
Vector Storage: Embeddings are stored in a FAISS vector database
Query Processing: When you ask a question, relevant document chunks are retrieved
AI Response: Gemini generates an answer based on the retrieved context

💡 Tips for Better Results

Ask specific questions rather than general ones
Reference topics you know are in your documents
Use follow-up questions to dive deeper into topics
Check the sources to verify information accuracy

🔒 Privacy & Security

Your documents are processed locally
Only your questions and retrieved context are sent to Google's API
Your full documents are never sent to external services
API keys should be kept secure and never shared

🆘 Troubleshooting
"No knowledge base found"

Run vector_store.py first to upload and process documents

"Error loading knowledge base"

Delete vectorstore.pkl and recreate it with vector_store.py

"API key error"

Make sure your Google API key is set correctly
Verify the key is active at Google AI Studio

"No documents could be loaded"

Check that your file types are supported
Ensure files aren't corrupted or password-protected

🎯 Next Steps

Add more document types by modifying the loaders
Implement different chunking strategies for better retrieval
Add conversation memory for multi-turn dialogues
Integrate with other Google AI models for specialized tasks


Happy chatting! 🚀