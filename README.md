# 🤖 Personal Notes RAG Assistant with Gemini Pro

A powerful **Retrieval-Augmented Generation (RAG)** application that transforms your PDF notes into an intelligent question-answering system powered by Google's Gemini API.

## Features

✨ **Smart Document Processing**
- Upload and process multiple PDF files
- Intelligent text chunking with sentence-aware boundaries
- Automatic text extraction and preprocessing

🔍 **Advanced Retrieval**
- Vector embeddings using SentenceTransformer (`all-MiniLM-L6-v2`)
- FAISS-based similarity search for fast retrieval
- Retrieves top-4 most relevant chunks from your notes

🤖 **Gemini-Powered Augmentation**
- Context-aware answer generation with Gemini 1.5 Flash
- Strictly adheres to your notes (no hallucination)
- Fallback simple augmentation when Gemini is unavailable

💬 **Interactive Interface**
- Clean, intuitive Streamlit UI
- Real-time RAG pipeline visualization
- Detailed view of retrieved context and similarity scores
- Knowledge base statistics and status dashboard

## How RAG Works

```
User Question
     ↓
1. RETRIEVAL: Search vector store for similar chunks
     ↓
2. AUGMENTATION: Enhance Gemini with retrieved context
     ↓
3. GENERATION: Generate accurate, grounded answers
     ↓
Generated Answer (Based ONLY on your notes)
```

## Installation

### Requirements
- Python 3.8+
- pip (Python package manager)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/rag-notes-assistant.git
   cd rag-notes-assistant
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your API key**
   
   Create a `.env` file in the project root:
   ```env
   GEMINI_API_KEY=your_actual_gemini_api_key_here
   ```
   
   Get your free API key from: https://aistudio.google.com/app/apikey

4. **Add .env to .gitignore** (important for security!)
   ```bash
   echo ".env" >> .gitignore
   ```

## Usage

Run the Streamlit app:
```bash
streamlit run rag_app.py
```

The app will open in your browser at `http://localhost:8501`

### Step-by-Step Guide

1. **Upload PDFs**: Use the sidebar to upload one or more PDF files containing your notes
2. **Process Notes**: Click "Process PDFs" to extract text and create embeddings
3. **Ask Questions**: Type your question in the main area and get instant answers
4. **View Details**: Expand the "View Detailed RAG Process" section to see how the system works

### Example Questions

- "What are the main topics covered in my notes?"
- "Explain the key concepts from chapter 2"
- "What are the differences between method A and method B?"
- "Summarize the important ideas about [topic]"
- "How can I apply these concepts in practice?"

## Architecture

### Components

| Component | Purpose |
|-----------|---------|
| `process_pdfs()` | Extracts text from PDF files using PyPDF2 |
| `chunk_text()` | Intelligently splits text into 500-char chunks |
| `setup_embedding_model()` | Loads SentenceTransformer for embeddings |
| `create_vector_store()` | Creates FAISS index with L2-normalized vectors |
| `search_similar_chunks()` | Retrieves k=4 most relevant chunks using cosine similarity |
| `augment_with_gemini()` | Constructs prompt and calls Gemini API |
| `generate_rag_answer()` | Orchestrates the full RAG pipeline |

### Technology Stack

- **Streamlit**: Interactive web interface
- **PyPDF2**: PDF text extraction
- **SentenceTransformers**: Text embeddings (384 dimensions)
- **FAISS**: Vector similarity search (IndexFlatIP)
- **Google Generative AI**: Gemini 1.5 Flash model
- **python-dotenv**: Environment variable management
- **NumPy**: Numerical operations

## Configuration

### Environment Variables

```env
GEMINI_API_KEY=your_api_key_here  # Required for Gemini integration
```

### Tunable Parameters

Edit these in the code to customize behavior:

```python
# In chunk_text()
chunk_size=500          # Size of each text chunk
chunk_overlap=100       # Overlap between chunks (unused in current implementation)

# In search_similar_chunks()
k=4                     # Number of chunks to retrieve
```

## Performance Tips

1. **For large PDFs**: Reduce `chunk_size` to 300-400 for better granularity
2. **For complex queries**: Increase `k` to 5-6 to get more context
3. **For speed**: The embedding model loads only once and caches in session state
4. **API costs**: Gemini 1.5 Flash is free tier friendly; monitor usage at https://aistudio.google.com/billing

## Limitations & Future Improvements

### Current Limitations
- ⚠️ Chunking doesn't implement overlap (parameter unused)
- ⚠️ No multi-language support
- ⚠️ PDF extraction may fail on scanned/image-based PDFs
- ⚠️ No persistent storage of embeddings (regenerated each session)

### Planned Improvements
- [ ] Persistent vector store storage
- [ ] Support for more file formats (DOCX, TXT, Web URLs)
- [ ] Hybrid retrieval (BM25 + semantic)
- [ ] Multi-document comparison
- [ ] Custom model selection
- [ ] Conversation history tracking
- [ ] Export answers to PDF/Markdown

## Troubleshooting

### Issue: "GEMINI_API_KEY not found in .env file"
**Solution**: Ensure your `.env` file has `GEMINI_API_KEY=your_key_here` and is in the same directory as `rag_app.py`

### Issue: "404 models/gemini-2.5-flash is not found"
**Solution**: Update the model name in `augment_with_gemini()` to an available model like `gemini-1.5-flash`

### Issue: "No text could be extracted from the PDFs"
**Solution**: Your PDFs might be scanned images. Try extracting text first using an OCR tool.

### Issue: FAISS index errors
**Solution**: Ensure you've processed at least one PDF before asking questions. The vector store must be created first.

## Security Considerations

🔒 **API Key Safety**
- Never commit `.env` to version control
- Always use `.gitignore` to exclude `.env`
- Rotate your API key regularly
- Use environment variables in production

## API Costs

- **Gemini 1.5 Flash**: Free tier available (limited requests)
- **SentenceTransformer**: Free, runs locally
- **FAISS**: Free, runs locally

Monitor your Gemini API usage: https://aistudio.google.com/billing

## License

MIT License - feel free to use this project for personal and commercial use.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues, questions, or suggestions, please open a GitHub issue.

---

**Built with ❤️ using Streamlit, SentenceTransformers, FAISS, and Google Gemini Pro**
