import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
# import tempfile
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import re
import google.generativeai as genai
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set page config
st.set_page_config(
    page_title="PDF RAG Assistant - Streamlit",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None
if 'gemini_configured' not in st.session_state:
    st.session_state.gemini_configured = False

def configure_gemini_from_env():
    """Configure Gemini with API key from .env file"""
    try:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            st.error("GEMINI_API_KEY not found in .env file")
            return False
        
        genai.configure(api_key=api_key)
        st.session_state.gemini_configured = True
        return True
    except Exception as e:
        st.error(f"Failed to configure Gemini: {e}")
        return False

def configure_gemini_from_input(api_key):
    """Configure Gemini with API key from user input"""
    try:
        genai.configure(api_key=api_key)
        st.session_state.gemini_configured = True
        return True
    except Exception as e:
        st.error(f"Failed to configure Gemini: {e}")
        return False

def process_pdfs(pdf_files):
    """Extract text from PDF files"""
    text = ""
    for pdf_file in pdf_files:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def chunk_text(text, chunk_size=500, chunk_overlap=100):
    """Split text into chunks for embedding"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def setup_embedding_model():
    """Initialize the embedding model"""
    if st.session_state.embedding_model is None:
        with st.spinner("Loading embedding model..."):
            model = SentenceTransformer('all-MiniLM-L6-v2')
            st.session_state.embedding_model = model
    return st.session_state.embedding_model

def create_vector_store(chunks, embedding_model):
    """Create FAISS vector store from text chunks"""
    embeddings = embedding_model.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings.astype('float32'))
    return index

def search_similar_chunks(question, vector_index, chunks, embedding_model, k=3):
    """Search for similar chunks using the vector index"""
    question_embedding = embedding_model.encode([question])
    faiss.normalize_L2(question_embedding)
    scores, indices = vector_index.search(question_embedding.astype('float32'), k)
    
    relevant_chunks = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(chunks):
            relevant_chunks.append({
                'content': chunks[idx],
                'score': float(score)
            })
    
    return relevant_chunks

def augment_with_gemini(question, relevant_chunks):
    """
    AUGMENTATION STEP: Use Gemini Pro to generate answers based on retrieved context
    """
    if not relevant_chunks:
        return "I couldn't find relevant information in your notes to answer this question."
    
    # Combine retrieved chunks into context
    context = "\n\n".join([chunk['content'] for chunk in relevant_chunks])
    
    # Create a prompt that AUGMENTS Gemini with our retrieved context
    prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided context from the user's personal notes.

CONTEXT FROM USER'S NOTES:
{context}

USER'S QUESTION: {question}

IMPORTANT INSTRUCTIONS:
1. Answer the question using ONLY the information from the context provided above
2. If the context doesn't contain enough information to answer the question, clearly state what information is missing
3. Do not use any external knowledge or make assumptions beyond what's in the context
4. Provide a clear, concise, and helpful answer based strictly on the notes
5. If relevant, cite which parts of the notes you're drawing from

Please provide your answer:"""
    
    try:
        # Initialize the Gemini model
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Generate content
        response = model.generate_content(prompt)
        
        return response.text
        
    except Exception as e:
        return f"Error using Gemini: {str(e)}\n\nFallback response based on your notes:\n\n{context}"

def augment_with_simple_llm(question, relevant_chunks):
    """
    Fallback augmentation when Gemini is not available
    """
    if not relevant_chunks:
        return "I couldn't find relevant information in your notes to answer this question."
    
    context = "\n\n".join([chunk['content'] for chunk in relevant_chunks])
    
    # Simple template-based generation
    answer = f"""Based on your notes regarding "{question}":

"""
    
    # Create a simple summary
    sentences = context.split('. ')
    key_sentences = sentences[:3]
    
    for i, sentence in enumerate(key_sentences):
        if sentence.strip():
            answer += f"• {sentence.strip()}.\n"
    
    answer += f"\nThis information was synthesized from {len(relevant_chunks)} relevant sections of your notes."
    
    return answer

def generate_rag_answer(question, relevant_chunks, use_gemini=True):
    """
    Proper RAG: Retrieve -> Augment with Gemini -> Generate
    """
    if use_gemini and st.session_state.gemini_configured:
        return augment_with_gemini(question, relevant_chunks)
    else:
        return augment_with_simple_llm(question, relevant_chunks)

# Auto-configure Gemini from .env on app start
if not st.session_state.gemini_configured:
    if configure_gemini_from_env():
        st.session_state.gemini_auto_configured = True

# UI Components
st.title("🤖 PDF RAG Assistant with Streamlit & Gemini")
st.markdown("**Retrieval-Augmented Generation System: Powered by Streamlit, SentenceTransformers, FAISS & Google Gemini**")

# Sidebar for configuration
with st.sidebar:
    st.header("🔧 Configuration")
    
    # Display Gemini status
    if st.session_state.gemini_configured:
        if hasattr(st.session_state, 'gemini_auto_configured') and st.session_state.gemini_auto_configured:
            st.success("✅ Gemini Pro: AUTO-CONFIGURED from .env file")
        else:
            st.success("✅ Gemini Pro: MANUALLY CONFIGURED")
    else:
        st.warning("🤖 Gemini Pro: INACTIVE")
        
        st.subheader("Manual Gemini Setup (Optional)")
        gemini_api_key = st.text_input(
            "Or enter Gemini API Key manually:",
            type="password",
            help="Get your API key from: https://aistudio.google.com/app/apikey"
        )
        
        if gemini_api_key:
            if st.button("Configure Gemini Manually"):
                if configure_gemini_from_input(gemini_api_key):
                    st.success("✅ Gemini Pro configured successfully!")
                else:
                    st.error("❌ Failed to configure Gemini")
    
    st.divider()
    
    st.header("1. Upload PDFs")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="Upload one or more PDF files containing your notes"
    )
    
    if uploaded_files:
        st.success(f"📄 Uploaded {len(uploaded_files)} PDF file(s)")
    
    st.header("2. Process Notes")
    if st.button("Process PDFs", disabled=not uploaded_files, type="primary"):
        with st.spinner("Processing your PDFs..."):
            text = process_pdfs(uploaded_files)
            
            if not text.strip():
                st.error("No text could be extracted from the PDFs.")
            else:
                chunks = chunk_text(text)
                st.session_state.chunks = chunks
                
                embedding_model = setup_embedding_model()
                st.session_state.vector_store = create_vector_store(chunks, embedding_model)
                st.session_state.processed = True
                
                st.success(f"✅ Processed {len(chunks)} text chunks!")
                st.info(f"📊 Total text extracted: {len(text):,} characters")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("🚀 RAG Pipeline Status")
    if st.session_state.processed:
        st.success("✅ **Retrieval Ready**: Your notes are embedded and searchable")
        
        if st.session_state.gemini_configured:
            st.success("🤖 **Augmentation Ready**: Gemini Pro will generate intelligent answers")
        else:
            st.warning("⚡ **Augmentation Ready**: Simple augmentation (Gemini not configured)")
        
        st.success("💬 **Generation Ready**: System can produce final answers")
        
        # Show RAG pipeline visualization
        st.subheader("🔁 RAG Pipeline Flow:")
        st.markdown("""
        1. **Retrieval** → Find relevant chunks from your notes ✅
        2. **Augmentation** → Enhance Gemini with retrieved context ✅  
        3. **Generation** → Produce final answer using augmented context ✅
        """)
        
        # Show statistics
        st.subheader("📊 Knowledge Base Stats")
        st.write(f"• Number of text chunks: {len(st.session_state.chunks)}")
        if st.session_state.vector_store:
            st.write(f"• Vector dimension: {st.session_state.vector_store.d}")
        st.write(f"• Gemini Pro: {'✅ Active' if st.session_state.gemini_configured else '❌ Inactive'}")
        
    else:
        st.info("📤 Upload PDFs and click 'Process PDFs' to initialize the RAG pipeline")

with col2:
    st.header("💬 Ask Questions")
    
    question = st.text_area(
        "Ask a question about your notes:",
        placeholder="e.g., Summarize the main concepts from my machine learning notes...\nWhat are the key points about neural networks?\nExplain the important ideas from chapter 3...",
        disabled=not st.session_state.processed,
        height=100
    )
    
    if question and st.session_state.processed:
        with st.spinner("Executing RAG pipeline with Gemini..."):
            # STEP 1: RETRIEVAL
            embedding_model = setup_embedding_model()
            relevant_chunks = search_similar_chunks(
                question, 
                st.session_state.vector_store, 
                st.session_state.chunks, 
                embedding_model,
                k=4  # Get more chunks for better context
            )
            
            # STEP 2 & 3: AUGMENTATION + GENERATION with Gemini
            use_gemini = st.session_state.gemini_configured
            answer = generate_rag_answer(question, relevant_chunks, use_gemini=use_gemini)
            
            # Display the final generated answer
            st.subheader("🤖 Generated Answer:")
            st.markdown("---")
            st.write(answer)
            st.markdown("---")
            
            # Show the augmentation process
            with st.expander("🔍 View Detailed RAG Process"):
                st.markdown("### 1. Retrieval Step")
                st.write(f"Found **{len(relevant_chunks)}** relevant chunks from your notes")
                
                st.markdown("### 2. Augmentation Step")
                if use_gemini:
                    st.success("Gemini Pro was augmented with the retrieved context")
                else:
                    st.warning("Simple augmentation used (Gemini not configured)")
                
                st.markdown("### 3. Retrieved Context Used:")
                total_context_length = 0
                for i, chunk_info in enumerate(relevant_chunks):
                    st.markdown(f"**Chunk {i+1}** (Similarity: `{chunk_info['score']:.3f}`):")
                    st.write(chunk_info['content'])
                    st.caption(f"Length: {len(chunk_info['content'])} characters")
                    st.divider()
                    total_context_length += len(chunk_info['content'])
                
                st.info(f"Total context provided to LLM: {total_context_length:,} characters")

# Instructions for .env setup
with st.expander("🔧 .env File Setup Instructions"):
    st.markdown("""
    ### How to Set Up Your .env File
    
    1. **Create a `.env` file** in the same directory as your Python script
    2. **Add your Gemini API key** to the file:
    ```
    GEMINI_API_KEY=your_actual_gemini_api_key_here
    ```
    3. **Save the file** - The app will automatically load it on startup
    
    ### Required Installation:
    ```bash
    pip install streamlit PyPDF2 sentence-transformers faiss-cpu numpy google-generativeai python-dotenv
    ```
    
    ### Your .env file should look like:
    ```env
    # .env file
    GEMINI_API_KEY=AIzaSyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    ```
    
    ⚠️ **Important**: Add `.env` to your `.gitignore` file to keep your API key secure!
    """)

# Example questions
with st.expander("💡 Example Questions to Try"):
    st.markdown("""
    After processing your PDFs, try asking:
    
    - **Summarization**: "What are the main topics covered in my notes?"
    - **Explanation**: "Explain the key concepts from chapter 2"
    - **Comparison**: "What are the differences between method A and method B?"
    - **Specific Queries**: "What does the document say about [specific topic]?"
    - **Application**: "How can I apply these concepts in practice?"
    
    *The system will find relevant information from your notes and Gemini will generate coherent answers.*
    """)