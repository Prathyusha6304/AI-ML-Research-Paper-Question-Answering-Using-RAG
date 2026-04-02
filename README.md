# 📄AI-ML-Research-Paper-Question-Answering-Using-RAG
A RAG-based system that answers questions from uploaded AI/ML research papers.

## 📌 Description
A RAG-based system that answers questions from AI/ML research papers.  
Users can upload a PDF research paper and ask questions, and the system retrieves relevant information and generates accurate answers.

---

## 🚀 Features
- 📄 Upload AI/ML research papers (PDF)
- ❓ Ask questions based on the uploaded document
- 🔍 Retrieves relevant content using embeddings
- 🤖 Generates accurate answers using LLM (Gemini/OpenAI)
- ⚡ Fast and simple RAG pipeline implementation

---

## 🧠 How It Works (RAG Pipeline)
1. Upload research paper (PDF)  
2. Extract text from document  
3. Split text into chunks  
4. Convert text into embeddings  
5. Store embeddings in vector database  
6. User asks a question  
7. Retrieve relevant chunks  
8. Generate answer using LLM  

---

## 🛠️ Tech Stack
- Python  
- LangChain  
- Vector Database (FAISS / Chroma)  
- Google Gemini API / OpenAI API  
- Streamlit / CLI  
